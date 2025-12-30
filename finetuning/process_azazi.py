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

# Config - ONLY AZAZI
TARGET_DIRS = [
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

    # Clean previous Azazi entries
    if os.path.exists(METADATA_FILE):
        print("Cleaning old Azazi entries...")
        temp = METADATA_FILE + ".tmp"
        with open(METADATA_FILE, 'r', encoding='utf-8') as fin, open(temp, 'w', encoding='utf-8') as fout:
            for line in fin:
                try:
                    d = json.loads(line)
                    if "Azazi" not in d.get('reciter', ''):
                        fout.write(line)
                except:
                    fout.write(line)
        shutil.move(temp, METADATA_FILE)

    for (src_path, dst_folder, reciter_label) in TARGET_DIRS:
        print(f"Processing {src_path} -> {dst_folder}...")
        
        dst_full = os.path.join(BASE_OUTPUT, dst_folder)
        os.makedirs(dst_full, exist_ok=True)
        
        files = sorted(glob.glob(os.path.join(src_path, "*.mp3")))
        
        total_extracted = 0
        
        for file_path in tqdm(files, desc=f"Scanning {dst_folder}"):
            filename = os.path.basename(file_path)
            
            # Load Audio (Librosa) - NO SIZE LIMIT
            try:
                # 3GB RAM is fine for 274MB MP3.
                y, sr = librosa.load(file_path, sr=16000)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
            
            # Split by silence
            try:
                intervals = librosa.effects.split(y, top_db=45, frame_length=2048, hop_length=512)
            except:
                continue
            
            chunks_data = []
            
            for (start, end) in intervals:
                # SAFEGUARD: Chunk must be < 30s for Whisper
                chunk_len = end - start
                max_len = 30 * sr
                
                # Sub-split logic
                sub_intervals = []
                if chunk_len > max_len:
                    curr = start
                    while curr < end:
                        sub_end = min(curr + max_len, end)
                        sub_intervals.append((curr, sub_end))
                        curr = sub_end
                else:
                    sub_intervals = [(start, end)]
                    
                for (s_start, s_end) in sub_intervals:
                    chunk = y[s_start:s_end]
                    dur = len(chunk)/sr
                    if dur < 1.0: continue 
                    
                    try:
                        input_feat = processor(chunk, sampling_rate=16000, return_tensors="pt").input_features.to(device)
                        if device == "cuda": input_feat = input_feat.half()
                        
                        with torch.no_grad():
                            gen = model.generate(input_feat, max_new_tokens=80)
                        
                        text = processor.batch_decode(gen, skip_special_tokens=True)[0]
                        norm = normalize_text(text)
                        
                        chunks_data.append({
                            "audio": chunk,
                            "text": text,
                            "norm": norm,
                            "duration": dur,
                            "start": s_start, # original index
                            "end": s_end
                        })
                    except Exception as e:
                        print(f"Error checking chunk in {filename}: {e}")
                
            # Analyze Pattern
            i = 0
            while i < len(chunks_data) - 1:
                curr = chunks_data[i]
                next_chunk = chunks_data[i+1]
                
                if not curr['norm'] or not next_chunk['norm']:
                    i += 1
                    continue
                    
                wer = jiwer.wer(curr['norm'], next_chunk['norm'])
                
                # If match -> Teacher (curr) + Child (next)
                if wer < SIMILARITY_THRESHOLD:
                    surah_id = filename.split('.')[0] 
                    # Use unique ID based on start sample to prevent overwrite if many segments
                    out_name = f"{surah_id}_{next_chunk['start']}.mp3"
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
                    i += 2 
                else:
                    i += 1
                    
        print(f"Extracted {total_extracted} segments from {src_path}")

if __name__ == "__main__":
    main()
