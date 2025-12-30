import os
import glob
import json
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import sys
import shutil

# Force UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Config
AUDIO_DIR = "data/dataset_normalized/audio"
METADATA_FILE = "data/normalized_metadata.jsonl"
FINAL_METADATA_FILE = "data/final_metadata.jsonl"

TARGET_MAX_SEC = 30.0
TARGET_MIN_SEC = 25.0

MODEL_ID = "tarteel-ai/whisper-base-ar-quran"

def load_model():
    print("Loading Whisper for re-alignment...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    model.eval()
    if device == "cuda": model.half()
    return model, processor, device

def transcribe(model, processor, audio_array, sr, device):
    input_features = processor(audio_array, sampling_rate=sr, return_tensors="pt").input_features.to(device)
    if device == "cuda": input_features = input_features.half()
    with torch.no_grad():
        gen_ids = model.generate(input_features, max_new_tokens=100)
    return processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

def split_and_transcribe(file_path, original_reciter, model, processor, device, counter_start):
    try:
        y, sr = librosa.load(file_path, sr=16000)
    except:
        return [], counter_start
        
    duration = len(y) / sr
    
    # If already compliant, return as is (but we need to re-verify text? No, original is trusted if we have it)
    # But this function is only called for > 30s.
    
    # Logic: Split into chunks of max 30s.
    # We use silent splitting.
    intervals = librosa.effects.split(y, top_db=40)
    
    new_batches = []
    current_audio = []
    current_len = 0
    
    # Process intervals to form ~25-30s chunks
    for start, end in intervals:
        seg = y[start:end]
        seg_len = len(seg)
        
        # If segment itself is > 30s, we force split it
        if seg_len/sr > TARGET_MAX_SEC:
            # Force split big segment
            max_s = int(TARGET_MAX_SEC * sr)
            curr = 0
            while curr < seg_len:
                sub = seg[curr: min(curr+max_s, seg_len)]
                # Add to buffer or process directly?
                # Direct process to ensure size
                if len(sub)/sr > 1.0: # Ignore tiny noise
                    # Transcribe
                    # Note: We can't combine this easily without more buffering logic. 
                    # Simpler: Just save it.
                    out_name = f"rebatch_{counter_start:06d}_{original_reciter}.mp3"
                    out_path = os.path.join(AUDIO_DIR, out_name)
                    sf.write(out_path, sub, 16000)
                    text = transcribe(model, processor, sub, sr, device)
                    new_batches.append({
                        "audio_path": out_path,
                        "text": text,
                        "duration": len(sub)/sr,
                        "reciter": original_reciter
                    })
                    counter_start += 1
                curr += max_s
            continue

        if (current_len + seg_len)/sr <= TARGET_MAX_SEC:
            current_audio.append(seg)
            current_len += seg_len
        else:
            # Flush current
            if current_len > 0:
                full_seg = np.concatenate(current_audio)
                out_name = f"rebatch_{counter_start:06d}_{original_reciter}.mp3"
                out_path = os.path.join(AUDIO_DIR, out_name)
                sf.write(out_path, full_seg, 16000)
                text = transcribe(model, processor, full_seg, sr, device)
                new_batches.append({
                    "audio_path": out_path,
                    "text": text,
                    "duration": len(full_seg)/sr,
                    "reciter": original_reciter
                })
                counter_start += 1
            
            # Start new
            current_audio = [seg]
            current_len = seg_len
            
    # Flush final
    if current_len > 0:
        full_seg = np.concatenate(current_audio)
        out_name = f"rebatch_{counter_start:06d}_{original_reciter}.mp3"
        out_path = os.path.join(AUDIO_DIR, out_name)
        sf.write(out_path, full_seg, 16000)
        text = transcribe(model, processor, full_seg, sr, device)
        new_batches.append({
            "audio_path": out_path,
            "text": text,
            "duration": len(full_seg)/sr,
            "reciter": original_reciter
        })
        counter_start += 1
        
    return new_batches, counter_start

def main():
    model, processor, device = load_model()
    
    print("Scanning metadata...")
    entries = []
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except: pass
                
    files_to_fix = [e for e in entries if e['duration'] > TARGET_MAX_SEC]
    good_files = [e for e in entries if e['duration'] <= TARGET_MAX_SEC and e['duration'] >= 5.0] # Keep existing good ones
    
    print(f"Total Files: {len(entries)}")
    print(f"Files to Fix (>30s): {len(files_to_fix)}")
    print(f"Good Files: {len(good_files)}")
    
    if not files_to_fix:
        print("No files to fix!")
        return

    # Prepare specific output list
    final_metadata = good_files.copy()
    
    rebatch_counter = 0
    
    for item in tqdm(files_to_fix, desc="Fixing Oversized Files"):
        path = item['audio_path']
        reciter = item['reciter']
        
        # Remove old file from disk (optional, or keep as backup? We overwrite in list, but file remains)
        # We will generate NEW files.
        # We should delete the OLD file to save space? 
        # Yes, delete old large file.
        
        new_items, rebatch_counter = split_and_transcribe(path, reciter, model, processor, device, rebatch_counter)
        
        # Add new items
        final_metadata.extend(new_items)
        
        # Delete old file
        try:
            os.remove(path)
        except: pass
        
    # Write Final Metadata
    print("Writing final metadata...")
    with open(FINAL_METADATA_FILE, 'w', encoding='utf-8') as f:
        for item in final_metadata:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    # Overwrite original metadata
    shutil.move(FINAL_METADATA_FILE, METADATA_FILE)
    print("Done.")

if __name__ == "__main__":
    main()
