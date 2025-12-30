import os
import glob
import json
import torch
import librosa
import numpy as np
import jiwer
from tqdm import tqdm
from datasets import load_from_disk
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import soundfile as sf
import sys
import shutil
import re

# Force UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Config for Mesbahi
SRC_DIR = "data/dataset_cache/audio/Ahmed_Al_Mesbahi_mp3s"
DST_DIR = "data/dataset_cache/audio/Mesbahi_Segmented"
DATASET_PATH = "data/quran_dataset"
BASE_MODEL = "tarteel-ai/whisper-base-ar-quran"
METADATA_FILE = "data/kids_metadata.jsonl" # Append to same metadata file
SIMILARITY_THRESHOLD = 0.5 

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
        print(f"Error loading dataset from {DATASET_PATH}")
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
    
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL).to(device)
    model.eval()
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ar", task="transcribe")
    if device == "cuda":
        model.half()

    gt_map = load_ground_truth()
    
    audio_files = sorted(glob.glob(os.path.join(SRC_DIR, "*.mp3")))
    
    # DEBUG FILTER
    args = sys.argv
    if len(args) > 1:
        target = args[1]
        print(f"DEBUG MODE: Filtering for {target}")
        audio_files = [f for f in audio_files if target in f]

    print(f"Processing {len(audio_files)} files...")
    
    processed_count = 0
    
    for file_path in tqdm(audio_files, desc="Processing Surahs"):
        filename = os.path.basename(file_path) # e.g. 001.mp3
        print(f"Starting {filename}...")
        
        try:
            surah_num = int(filename.split('.')[0])
        except:
            continue
            
        surah_ayahs = []
        ayah_idx = 1
        while (surah_num, ayah_idx) in gt_map:
            surah_ayahs.append({
                "ayah": ayah_idx, 
                "text": gt_map[(surah_num, ayah_idx)]
            })
            ayah_idx += 1
            
        if not surah_ayahs:
            print(f"No text found for Surah {surah_num}")
            continue

        try:
            y, sr = librosa.load(file_path, sr=16000)
            print(f"Loaded {filename}. Duration: {len(y)/sr:.2f}s")
        except Exception as e:
            print(f"Skipping {filename} due to load error: {e}")
            continue
        
        intervals = librosa.effects.split(y, top_db=40, frame_length=2048, hop_length=512) # Lower DB to 40?
        print(f"Split {filename} into {len(intervals)} chunks.")
        
        cursor_gt = 0 
        
        for (start, end) in intervals:
            chunk_len = end - start
            max_len = 30 * sr
            
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
                chunk_dur = len(chunk)/sr
                
                if chunk_dur < 1.0: continue
                
                input_features = processor(chunk, sampling_rate=16000, return_tensors="pt").input_features.to(device)
                if device == "cuda": input_features = input_features.half()
                
                with torch.no_grad():
                    gen_ids = model.generate(input_features, max_new_tokens=100)
                
                transcription = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
                norm_trans = normalize_text(transcription)
                
                best_match = None
                best_wer = 100.0
                best_offset = -1
                best_merged_count = 1
                
                for offset in range(10): # Look ahead more?
                    if cursor_gt + offset >= len(surah_ayahs): break
                    
                    current_target_text = ""
                    for merge_count in range(1, 4):
                        if cursor_gt + offset + merge_count - 1 >= len(surah_ayahs): break
                        
                        added_ayah = surah_ayahs[cursor_gt + offset + merge_count - 1]
                        current_target_text += " " + added_ayah['text']
                        
                        norm_target = normalize_text(current_target_text)
                        
                        if not norm_trans or not norm_target: wer = 1.0
                        else: wer = jiwer.wer(norm_target, norm_trans)
                        
                        if wer < best_wer:
                            best_wer = wer
                            best_offset = offset
                            best_merged_count = merge_count
                
                if best_wer < SIMILARITY_THRESHOLD:
                    matched_ayahs = []
                    final_text = ""
                    for k in range(best_merged_count):
                        ma = surah_ayahs[cursor_gt + best_offset + k]
                        matched_ayahs.append(ma)
                        final_text += ma['text'] + " "
                    
                    final_text = final_text.strip()
                    start_ayah_num = matched_ayahs[0]['ayah']
                    end_ayah_num = matched_ayahs[-1]['ayah']
                    
                    out_name = f"{surah_num:03d}{start_ayah_num:03d}-{end_ayah_num:03d}_{s_start}.mp3"
                    out_path = os.path.join(DST_DIR, out_name)
                    
                    sf.write(out_path, chunk, sr)
                    
                    entry = {
                        "audio_path": out_path,
                        "text": final_text,
                        "surah": surah_num,
                        "ayah": start_ayah_num,
                        "reciter": "Ahmed_Al_Mesbahi_Child",
                        "duration": chunk_dur
                    }
                    
                    with open(METADATA_FILE, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    
                    cursor_gt += (best_offset + best_merged_count)
                    processed_count += 1
                    
                    if processed_count % 10 == 0:
                         print(f"Matched {start_ayah_num}-{end_ayah_num} (WER: {best_wer:.2f})")
                else:
                    pass
                    
                if cursor_gt >= len(surah_ayahs):
                    break 
            
            if cursor_gt >= len(surah_ayahs):
                break
 
    print(f"Processing Complete. {processed_count} Mesbahi segments added.")

if __name__ == "__main__":
    main()
