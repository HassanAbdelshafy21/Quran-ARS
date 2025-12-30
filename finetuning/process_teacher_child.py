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

# Config
# We process one root at a time or loop specific folders
TARGET_DIRS = [
    # Source Folder -> Output Folder Name -> Reciter Label
    ("data/dataset_cache/audio/downloaded_mp3s", "Minshawi_Child_Extracted", "Minshawi_Child"),
    ("data/dataset_cache/audio/Al_Husayni_Al_Azazi_Children_mp3s", "Azazi_Child_Extracted", "Azazi_Child")
]

BASE_OUTPUT = "data/dataset_cache/audio"
METADATA_FILE = "data/kids_metadata.jsonl"
BASE_MODEL = "tarteel-ai/whisper-base-ar-quran"
SIMILARITY_THRESHOLD = 0.4 # Strict match for pattern detection

def normalize_text(text):
    import re
    text = re.sub(r'[إأٱآ]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    val_chars = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي ا"
    out = ""
    for char in text:
        if char in val_chars: out += char
    out = re.sub(r'\s+', ' ', out)
    return out.strip()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL).to(device)
    model.eval()
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ar", task="transcribe")
    if device == "cuda": model.half()

    for (src_path, dst_folder, reciter_label) in TARGET_DIRS:
        print(f"Processing {src_path} -> {dst_folder}...")
        
        dst_full = os.path.join(BASE_OUTPUT, dst_folder)
        os.makedirs(dst_full, exist_ok=True)
        
        files = sorted(glob.glob(os.path.join(src_path, "*.mp3")))
        
        total_extracted = 0
        
        for file_path in tqdm(files, desc=f"Scanning {dst_folder}"):
            filename = os.path.basename(file_path)
            # Skip huge files for now to avoid hang
            if os.path.getsize(file_path) > 100 * 1024 * 1024: 
                print(f"Skipping {filename} (Too large)")
                continue
                
            y, sr = None, 16000
            try:
                # Use librosa load as verified in debug script
                y, sr = librosa.load(file_path, sr=16000)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
            
            # Split
            try:
                intervals = librosa.effects.split(y, top_db=45, frame_length=2048, hop_length=512)
            except Exception as e:
                print(f"Error splitting {filename}: {e}")
                continue
            
            chunks_data = []
            
            # 1. Transcribe ALL chunks first
            for (start, end) in intervals:
                chunk = y[start:end]
                dur = len(chunk)/sr
                if dur < 1.0: continue # Skip noise
                
                # Sefeguard: Skip chunks > 30s to prevent OOM / Model Errors
                if dur > 30.0:
                    print(f"DEBUG: Skipping oversized chunk {dur:.2f}s")
                    continue
                
                print(f"DEBUG: Processing chunk {start}-{end} ({dur:.2f}s)")
                try:
                    input_feat = processor(chunk, sampling_rate=16000, return_tensors="pt").input_features.to(device)
                    if device == "cuda": input_feat = input_feat.half()
                    
                    with torch.no_grad():
                        gen = model.generate(input_feat, max_new_tokens=80)
                    
                    text = processor.batch_decode(gen, skip_special_tokens=True)[0]
                    norm = normalize_text(text)
                    print(f"DEBUG: Transcribed: {norm[:20]}...")
                    
                    chunks_data.append({
                        "audio": chunk,
                        "text": text,
                        "norm": norm,
                        "duration": dur,
                        "start": start,
                        "end": end
                    })
                except Exception as e:
                     print(f"Error processing chunk: {e}")
                     continue
                
            # 2. Analyze Pattern: Look for Duplicates
            # Teacher(A) ... Child(A)
            i = 0
            while i < len(chunks_data) - 1:
                curr = chunks_data[i]
                next_chunk = chunks_data[i+1]
                
                # Check similarity
                if not curr['norm'] or not next_chunk['norm']:
                    i += 1
                    continue
                    
                wer = jiwer.wer(curr['norm'], next_chunk['norm'])
                
                # Logic: If text matches, it's a repetition.
                if wer < SIMILARITY_THRESHOLD:
                    # Match! 
                    surah_id = filename.split('.')[0] # e.g. 001
                    seq_id = i + 1 # rough sequence
                    out_name = f"{surah_id}_seq{seq_id:03d}.mp3"
                    out_path = os.path.join(dst_full, out_name)
                    
                    sf.write(out_path, next_chunk['audio'], sr)
                    
                    entry = {
                        "audio_path": out_path,
                        "text": next_chunk['text'], 
                        "reciter": reciter_label,
                        "duration": next_chunk['duration'],
                        "source_file": filename
                    }
                    
                    with open(METADATA_FILE, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    
                    total_extracted += 1
                    i += 2 # Skip both
                else:
                    i += 1
                    
        print(f"Extracted {total_extracted} segments from {src_path}")

if __name__ == "__main__":
    main()
