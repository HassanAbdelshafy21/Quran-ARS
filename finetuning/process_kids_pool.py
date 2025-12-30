import os
import glob
import json
import torch
import librosa
import numpy as np
import jiwer
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import soundfile as sf
import sys
import shutil

# Force UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# --- CONFIGURATION ---
# Source -> Output -> Label
TARGET_DIRS = [
    ("data/dataset_cache/audio/downloaded_mp3s", "Minshawi_Child_Extracted", "Minshawi_Child"),
    ("data/dataset_cache/audio/Al_Husayni_Al_Azazi_Children_mp3s", "Azazi_Child_Extracted", "Azazi_Child")
]

BASE_OUTPUT = "data/dataset_cache/audio"
METADATA_FILE = "data/kids_metadata.jsonl"

# USES V4 MODEL (Checkpoint 5000) for best alignment
MODEL_ID = "finetuning/checkpoints_v4/checkpoint-5000"
BASE_MODEL_NAME = "tarteel-ai/whisper-base-ar-quran" # For processor fallback

# Logic Params
SIMILARITY_THRESHOLD = 0.45  # < 0.45 WER = Perfect Match (Teacher then Child)
MIN_CHUNK_DURATION = 2.0     # Ignore tiny noises
MAX_CHUNK_DURATION = 30.0    # Whisper limit
BATCH_SIZE = 8               # Process chunks in batches for speed

def normalize_text(text):
    import re
    # Standardize Arabic
    text = re.sub(r'[إأٱآ]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    # Filter to only letters
    val_chars = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي ا"
    out = ""
    for char in text:
        if char in val_chars: out += char
    out = re.sub(r'\s+', ' ', out)
    return out.strip()

def load_model(device):
    print(f"Loading Model: {MODEL_ID}...")
    from peft import PeftModel, PeftConfig
    
    # Load Base
    base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)
    
    # Load Adapter
    try:
        model = PeftModel.from_pretrained(base_model, MODEL_ID)
        model = model.merge_and_unload() # Merge for inference speed
    except Exception as e:
        print(f"Warning: Could not load LoRA adapter ({e}). Using Base model.")
        model = base_model

    model.to(device)
    model.eval()
    if device == "cuda":
        model.half()
        
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_NAME)
    return model, processor

def process_file_intervals(file_path):
    """
    Loads file, runs VAD (silence splitting), returns list of chunks.
    Does NOT transcribe yet.
    """
    try:
        # Load full audio
        y, sr = librosa.load(file_path, sr=16000)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None, None

    # Split by silence (VAD)
    try:
        # top_db=40 is conservative to catch breaths
        intervals = librosa.effects.split(y, top_db=40, frame_length=2048, hop_length=512)
    except:
        return None, None, None

    valid_chunks = []
    
    # Process intervals to ensure they are within Whisper limits
    for (start, end) in intervals:
        chunk_len_samples = end - start
        chunk_dur = chunk_len_samples / sr
        
        if chunk_dur < MIN_CHUNK_DURATION:
            continue
            
        if chunk_dur > MAX_CHUNK_DURATION:
            # Sub-split long chunks simply by hard cut for now (or skip)
            # Better: Recursive split? For now, we just chop.
            curr = start
            while curr < end:
                sub_end = min(curr + int(MAX_CHUNK_DURATION * sr), end)
                valid_chunks.append((curr, sub_end))
                curr = sub_end
        else:
            valid_chunks.append((start, end))
            
    return y, sr, valid_chunks

def transcribe_batch(model, processor, audio, chunks, device):
    """
    Transcribes a list of (start, end) tuples from the audio array in batches.
    """
    results = []
    
    # Prepare batch
    batch_input = []
    
    for (s, e) in chunks:
        segment = audio[s:e]
        batch_input.append(segment)
        
    # Run Inference in Batches
    for i in range(0, len(batch_input), BATCH_SIZE):
        batch_segments = batch_input[i : i + BATCH_SIZE]
        
        try:
            # Feature extraction
            input_features = processor(
                batch_segments, 
                sampling_rate=16000, 
                return_tensors="pt",
                padding=True
            ).input_features.to(device)
            
            if device == "cuda":
                input_features = input_features.half()
            
            # Generate
            with torch.no_grad():
                # Force Arabic
                forced_ids = processor.get_decoder_prompt_ids(language="ar", task="transcribe")
                model.config.forced_decoder_ids = forced_ids
                
                generated_ids = model.generate(input_features, max_new_tokens=128)
            
            # Decode
            transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            for j, text in enumerate(transcriptions):
                idx = i + j
                start, end = chunks[idx]
                dur = (end - start) / 16000
                
                results.append({
                    "start": start,
                    "end": end,
                    "duration": dur,
                    "text": text,
                    "norm": normalize_text(text),
                    "audio": batch_segments[j] # Keep ref for saving
                })
                
        except Exception as e:
            print(f"Batch Error: {e}")
            # Fallback? Skip.
            continue
            
    return results

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")
    
    # Load V4 Model
    model, processor = load_model(device)
    
    # Clean Metadata if needed? No, append or clean manually. 
    # Let's verify if file exists, if so count lines.
    if os.path.exists(METADATA_FILE):
        print(f"Appending to existing {METADATA_FILE}")
    
    total_extracted_global = 0

    for (src_path, dst_folder, reciter_label) in TARGET_DIRS:
        print(f"\n--- Processing {reciter_label} in {src_path} ---")
        
        dst_full = os.path.join(BASE_OUTPUT, dst_folder)
        os.makedirs(dst_full, exist_ok=True)
        
        files = sorted(glob.glob(os.path.join(src_path, "*.mp3")))
        print(f"Found {len(files)} files.")
        
        for file_path in tqdm(files, desc=reciter_label):
            filename = os.path.basename(file_path)
            
            # 1. Load & VAD
            y, sr, chunk_intervals = process_file_intervals(file_path)
            if not chunk_intervals:
                continue
                
            # 2. Transcribe All Chunks (Batched)
            segments = transcribe_batch(model, processor, y, chunk_intervals, device)
            
            # 3. Separation Logic (Compare i vs i+1)
            # Looking for Teacher -> Child repetition
            i = 0
            while i < len(segments) - 1:
                curr = segments[i]
                next_seg = segments[i+1]
                
                # Check text validity
                if len(curr['norm']) < 5 or len(next_seg['norm']) < 5:
                    i += 1
                    continue
                
                # Compare
                wer = jiwer.wer(curr['norm'], next_seg['norm'])
                
                if wer < SIMILARITY_THRESHOLD:
                    # MATCH FOUND!
                    # next_seg is the Child (echo)
                    
                    # Construct Output Name
                    base_name = os.path.splitext(filename)[0]
                    out_name = f"{base_name}_seq{i:03d}_child.mp3"
                    out_path = os.path.join(dst_full, out_name)
                    
                    # Save Audio
                    sf.write(out_path, next_seg['audio'], sr)
                    
                    # Save Metadata
                    entry = {
                        "audio_path": out_path,
                        "text": next_seg['text'],
                        "norm_text": next_seg['norm'],
                        "duration": next_seg['duration'],
                        "reciter": reciter_label,
                        "source_file": filename,
                        "wer_match": wer
                    }
                    
                    with open(METADATA_FILE, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        
                    total_extracted_global += 1
                    i += 2 # Skip pair (Teacher, Child)
                
                else:
                    # No match, advance onestep (maybe Teacher was noise, or Child missed)
                    i += 1
                    
        print(f"Finished {reciter_label}. Total extracted so far: {total_extracted_global}")

if __name__ == "__main__":
    main()
