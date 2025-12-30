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

# --- CONFIGURATION V5 ---
CACHE_ROOT = "data/dataset_cache/audio"
OUTPUT_DIR = "data/dataset_v5/audio"
METADATA_OUT = "data/v5_metadata.jsonl"

TARGET_MIN_SEC = 25.0
TARGET_MAX_SEC = 30.0

MODEL_ID = "tarteel-ai/whisper-base-ar-quran" # For smart splitting fallback

def load_model():
    print("Loading Whisper for splitting long files...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        processor = WhisperProcessor.from_pretrained(MODEL_ID)
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
        model.eval()
        if device == "cuda": model.half()
        return model, processor, device
    except:
        return None, None, None

def transcribe(model, processor, audio_array, sr, device):
    if not model: return ""
    input_features = processor(audio_array, sampling_rate=sr, return_tensors="pt").input_features.to(device)
    if device == "cuda": input_features = input_features.half()
    with torch.no_grad():
        gen_ids = model.generate(input_features, max_new_tokens=100)
    return processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

def smart_split(f, text, model, processor, device):
    # Same fallback logic for huge files, just in case
    try:
        y, sr = librosa.load(f, sr=16000)
    except:
        return []

    intervals = librosa.effects.split(y, top_db=40) 
    valid_chunks = []
    current_chunk = []
    current_len = 0
    
    for start, end in intervals:
        seg = y[start:end]
        seg_len = len(seg)
        
        # If single segment is huge (rare for Ayah clips, but possible in Mesbahi)
        if seg_len/sr > TARGET_MAX_SEC:
            # Force Split
            max_s = int(TARGET_MAX_SEC * sr)
            curr = 0
            while curr < seg_len:
                sub = seg[curr: min(curr+max_s, seg_len)]
                if len(sub)/sr > 1.0:
                     valid_chunks.append({
                        "audio": sub,
                        "text": "", 
                        "duration": len(sub)/sr
                    })
                curr += max_s
            continue
            
        if (current_len + seg_len)/sr <= TARGET_MAX_SEC:
            current_chunk.append(seg)
            current_len += seg_len
        else:
            if current_len/sr > 1.0:
                full_seg = np.concatenate(current_chunk)
                valid_chunks.append({
                    "audio": full_seg,
                    "text": "", # Text lost in blind split, acceptable for rare fallback
                    "duration": len(full_seg)/sr
                })
            current_chunk = [seg]
            current_len = seg_len
            
    if current_len/sr > 1.0:
        full_seg = np.concatenate(current_chunk)
        valid_chunks.append({
            "audio": full_seg,
            "text": "",
            "duration": len(full_seg)/sr
        })
            
    return valid_chunks

def load_all_files():
    # Only need to scan the cache if we don't have paths in metadata
    # But our metadata has absolute paths usually.
    return []

def output_batch(batch, count):
    combined_audio = []
    combined_text = ""
    reciters = set()
    
    for (obj, dur, txt) in batch:
        if "path" in obj:
            try:
                y, sr = librosa.load(obj['path'], sr=16000)
                combined_audio.append(y)
            except: pass
        elif "audio" in obj:
            # Already loaded numpy array
            combined_audio.append(obj['audio'])
        
        combined_text += txt + " "
        if "reciter" in obj: reciters.add(obj['reciter'])
    
    if not combined_audio: return False

    try:
        final_audio = np.concatenate(combined_audio)
    except: return False
    
    combined_text = combined_text.strip()
    reciter_label = list(reciters)[0] if len(reciters) == 1 else "Mixed"
    
    # Filename
    safe_reciter = str(reciter_label).replace(" ", "_").replace("/", "-")
    out_name = f"v5_batch_{count:06d}_{safe_reciter}.mp3"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    
    sf.write(out_path, final_audio, 16000)
    
    entry = {
        "audio_path": out_path,
        "text": combined_text,
        "duration": len(final_audio)/16000.0,
        "reciter": reciter_label,
        "source_files": [obj.get('path', 'mem') for obj,_,_ in batch]
    }
    
    with open(METADATA_OUT, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
    return True

def main():
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if os.path.exists(METADATA_OUT):
        os.remove(METADATA_OUT)
        
    model, processor, device = load_model()
    
    # 1. Load Metadata Sources
    meta_map = {}
    print("Loading Metadata Sources...")
    
    # Priority 1: New Child Data
    if os.path.exists("data/kids_metadata.jsonl"):
        print(" - Loading data/kids_metadata.jsonl (Child)")
        with open("data/kids_metadata.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    d['is_child'] = True
                    # Key by audio path
                    meta_map[d['audio_path']] = d
                except: pass
                
    # Priority 2: Adult Data (V4 Cleaned)
    if os.path.exists("data/cleaned_metadata.jsonl"):
        print(" - Loading data/cleaned_metadata.jsonl (Adults)")
        with open("data/cleaned_metadata.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    # Filter out old child data from V4 to avoid duplication/confusion?
                    # "Ahmed_Al_Mesbahi_Child" and "Junaid_Child" are in kids_metadata now?
                    # Actually kids_metadata has NEW extraction.
                    # V4 Cleaned might have duplicates.
                    # Safe overlap Strategy: Use filename as key.
                    # If duplicate key, Child metadata wins (loaded first? No, dict overwrrite).
                    # Let's load Adults ONLY if not child reciter?
                    reciter = d.get('reciter', '')
                    if "Child" in reciter and d['audio_path'] in meta_map:
                        continue # Already loaded better version
                    
                    meta_map[d['audio_path']] = d
                except: pass

    # Convert to List
    processable_files = []
    print(f"Total Unique Files to Process: {len(meta_map)}")
    
    for path, meta in meta_map.items():
        reciter = meta.get('reciter', 'Unknown')
        
        # Sort Key: Reciter -> Path
        processable_files.append({
            "path": path,
            "reciter": reciter,
            "text": meta.get('text', ''),
            "duration": meta.get('duration', 0.0),
            "sort_key": (reciter, path)
        })
        
    # Sort to group by reciter
    processable_files.sort(key=lambda x: x['sort_key'])
    
    batch_buffer = [] 
    current_batch_dur = 0.0
    output_count = 0
    
    for item in tqdm(processable_files, desc="Batching V5"):
        f = item['path']
        reciter = item['reciter']
        text = item['text']
        dur = item['duration']
        
        if not text: continue
        
        # Verify duration definition vs reality?
        # Trust metadata for speed, but handle errors
        try:
           # If metadata duration is suspiciously 0 or None
           if not dur:
               dur = librosa.get_duration(path=f)
        except: continue
        
        # Logic: Flush-Before-Overflow (Strict 30s)
        
        # If single file is huge (>30), we must split
        if dur > TARGET_MAX_SEC:
            chunks = smart_split(f, text, model, processor, device)
            for ch in chunks:
                c_dur = ch['duration']
                c_text = ch['text']
                chunk_obj = {"audio": ch['audio'], "reciter": reciter}
                
                # Check fit
                if current_batch_dur + c_dur > TARGET_MAX_SEC:
                    output_batch(batch_buffer, output_count)
                    output_count += 1
                    batch_buffer = []
                    current_batch_dur = 0.0
                
                batch_buffer.append((chunk_obj, c_dur, c_text))
                current_batch_dur += c_dur
                
                # If specifically hit target after add
                if current_batch_dur >= TARGET_MIN_SEC:
                     output_batch(batch_buffer, output_count)
                     output_count += 1
                     batch_buffer = []
                     current_batch_dur = 0.0
            continue
            
        # Normal Concatenation
        # Check if adding this file would exceed Max
        if current_batch_dur + dur > TARGET_MAX_SEC:
            # FLUSH NOW
            if batch_buffer:
                output_batch(batch_buffer, output_count)
                output_count += 1
            batch_buffer = []
            current_batch_dur = 0.0
        
        # Add to buffer
        chunk_obj = {"path": f, "reciter": reciter}
        batch_buffer.append((chunk_obj, dur, text))
        current_batch_dur += dur
        
        # Check if we hit target range (25-30)
        if current_batch_dur >= TARGET_MIN_SEC:
            # Since we checked limit above, we know it is <= 30.0 (unless single file was exactly 30+)
            output_batch(batch_buffer, output_count)
            output_count += 1
            batch_buffer = []
            current_batch_dur = 0.0

    # Flush Final
    if batch_buffer:
        output_batch(batch_buffer, output_count)
        output_count += 1
        
    print(f"V5 Generation Complete. Created {output_count} batches.")

if __name__ == "__main__":
    main()
