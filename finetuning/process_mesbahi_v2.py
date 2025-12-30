import os
import glob
import json
import torch
import librosa
import numpy as np
import soundfile as sf
import sys
import re
import difflib
from tqdm import tqdm
from datasets import load_from_disk
from transformers import pipeline

# Force UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Config for Mesbahi
SRC_DIR = "data/dataset_cache/audio/Ahmed_Al_Mesbahi_mp3s"
DST_DIR = "data/dataset_cache/audio/Mesbahi_Segmented"
DATASET_PATH = "data/quran_dataset"
BASE_MODEL = "tarteel-ai/whisper-base-ar-quran"
METADATA_FILE = "data/kids_metadata.jsonl" 

def normalize_text(text):
    text = re.sub(r'[إأٱآ]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text) 
    val_chars = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي ا"
    out = ""
    for char in text:
        if char in val_chars:
            out += char
    out = re.sub(r'\s+', ' ', out)
    return out.strip()

def load_ground_truth():
    print("Loading Ground Truth from dataset...")
    try:
        if os.path.exists("data/quran_dataset_kids"):
             ds = load_from_disk("data/quran_dataset_kids")
        else:
             ds = load_from_disk(DATASET_PATH)
    except:
        return {}

    gt_map = {}
    for row in tqdm(ds):
        key = (row['surah'], row['ayah'])
        if key not in gt_map:
            gt_map[key] = row['text']
    return gt_map

def main():
    os.makedirs(DST_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize Pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=BASE_MODEL,
        device=device,
        chunk_length_s=30,
        return_timestamps="word", 
    )

    gt_map = load_ground_truth()
    
    audio_files = sorted(glob.glob(os.path.join(SRC_DIR, "*.mp3")))
    
    print(f"Processing {len(audio_files)} files...")
    
    processed_count = 0
    
    for file_path in tqdm(audio_files, desc="Processing Surahs"):
        filename = os.path.basename(file_path)
        try:
            surah_num = int(filename.split('.')[0])
        except: continue
        
        # Load Surah Text
        surah_ayahs = []
        ayah_idx = 1
        while (surah_num, ayah_idx) in gt_map:
            surah_ayahs.append({
                "ayah": ayah_idx, 
                "text": normalize_text(gt_map[(surah_num, ayah_idx)]),
                "orig_text": gt_map[(surah_num, ayah_idx)]
            })
            ayah_idx += 1
            
        if not surah_ayahs:
            continue

        # Move audio load up to bypass ffmpeg in transformers
        try:
            y, sr = librosa.load(file_path, sr=16000)
        except Exception as e:
             print(f"Skipping {filename} due to load error: {e}")
             continue

        print(f"Processing {filename} (Surah {surah_num}, {len(surah_ayahs)} Ayahs)...")
        
        # Run Inference
        try:
            # Pass numpy array to bypass ffmpeg
            result = pipe(y, return_timestamps="word")
            chunks = result.get('chunks', []) 
        except Exception as e:
            print(f"Failed to transcribe {filename}: {e}")
            continue
            
        print(f"Transcribed {len(chunks)} words/segments.")
        
        current_ayah_idx = 0
        word_idx = 0
             
        while current_ayah_idx < len(surah_ayahs) and word_idx < len(chunks):
            target_ayah = surah_ayahs[current_ayah_idx]
            target_words = target_ayah['text'].split()
            
            if not target_words: 
                current_ayah_idx += 1
                continue
                
            start_time = chunks[word_idx]['timestamp'][0]
            collected_text = ""
            
            best_ratio = 0
            best_end_idx = word_idx
            
            for i in range(word_idx, min(word_idx + 200, len(chunks))):
                w_text = normalize_text(chunks[i]['text'])
                collected_text += w_text
                
                joined_target = target_ayah['text'].replace(" ", "")
                
                # Check similarity
                ratio = difflib.SequenceMatcher(None, collected_text, joined_target).ratio()
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_end_idx = i
                
                if ratio > 0.85 and len(collected_text) >= len(joined_target):
                    break
            
            if best_ratio > 0.7:
                # Accepted
                end_time = chunks[best_end_idx]['timestamp'][1]
                
                if start_time is None: start_time = 0.0
                if end_time is None: end_time = start_time + 5.0
                
                s_sample = int(start_time * sr)
                e_sample = int(end_time * sr)
                segment = y[s_sample:e_sample]
                
                if len(segment)/sr > 1.0:
                    out_name = f"{surah_num:03d}{target_ayah['ayah']:03d}_{current_ayah_idx}.mp3"
                    out_path = os.path.join(DST_DIR, out_name)
                    sf.write(out_path, segment, sr)
                    
                    entry = {
                        "audio_path": out_path,
                        "text": target_ayah['orig_text'],
                        "surah": surah_num,
                        "ayah": target_ayah['ayah'],
                        "reciter": "Ahmed_Al_Mesbahi_Child",
                        "duration": len(segment)/sr
                    }
                    with open(METADATA_FILE, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        
                    processed_count += 1
                
                word_idx = best_end_idx + 1
                current_ayah_idx += 1
            else:
                word_idx += 1
                if word_idx > len(chunks) - 5: break
                
    print(f"Done. Processed {processed_count} segments.")

if __name__ == "__main__":
    main()
