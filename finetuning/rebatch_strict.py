import os
import glob
import json
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import sys
import shutil

# Config
INPUT_DIR = "data/dataset_normalized/audio"
OUTPUT_DIR = "data/dataset_final/audio"
METADATA_IN = "data/normalized_metadata.jsonl"
METADATA_OUT = "data/final_metadata.jsonl"

TARGET_MIN = 25.0
TARGET_MAX = 30.0

def load_all_batches():
    # Load metadata to get order/text
    batches = []
    if os.path.exists(METADATA_IN):
        with open(METADATA_IN, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    batches.append(json.loads(line))
                except: pass
    return batches

def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(METADATA_OUT):
        os.remove(METADATA_OUT)
        
    batches = load_all_batches()
    print(f"Loaded {len(batches)} batches.")
    
    # Continuous Stream Buffer
    audio_buffer = np.array([])
    text_buffer = ""
    reciter_buffer = set()
    
    count = 0
    
    # Iterate
    for batch in tqdm(batches):
        path = batch['audio_path']
        text = batch['text']
        reciter = batch['reciter']
        
        try:
            y, sr = librosa.load(path, sr=16000)
        except: continue
        
        # Append to buffer
        if len(audio_buffer) == 0:
            audio_buffer = y
        else:
            audio_buffer = np.concatenate([audio_buffer, y])
            
        text_buffer += " " + text
        reciter_buffer.add(reciter)
        
        # Process Buffer
        while len(audio_buffer)/16000 > TARGET_MAX:
             # Find split point between 25 and 30
             # Scan silence in window [25, 30]
             # If no silence, force split at 30? Or scan [20, 30]?
             
             # Let's verify buffer size
             dur = len(audio_buffer)/16000
             
             # Only split if we have enough for 2 batches? OR if > Max?
             # User wants NO FILE > 30.
             # So if > 30, we must split.
             
             # Search for silence in [20s, 30s] to be safe
             search_start = int(20 * 16000)
             search_end = int(30 * 16000)
             
             # If buffer is huge, just look at the first 30s
             window = audio_buffer[search_start:search_end]
             
             # RMS energy
             # We want min energy point
             frame_len = 512
             hop = 128
             rms = librosa.feature.rms(y=window, frame_length=frame_len, hop_length=hop)[0]
             
             min_idx = np.argmin(rms)
             # Convert rms frame to sample index
             sample_offset = min_idx * hop
             split_point = search_start + sample_offset
             
             # Cut
             chunk = audio_buffer[:split_point]
             remainder = audio_buffer[split_point:]
             
             # Update buffers
             audio_buffer = remainder
             
             # Text? We don't know where text splits perfectly.
             # But this is ASR training. Partial text or approximate text is okay-ish?
             # Actually, splitting mid-sentence breaks alignment.
             # BUT we already merged text in the previous step.
             # For now, we will assign the FULL 'text_buffer' to the chunk, and clear it?
             # NO.
             # This re-slicer destroys text alignment.
             # CRITICAL FLAW.
             # We cannot re-slice blindly without Whisper alignment.
             
             # Stop. 
             # Re-slicing without re-aligning destroys the dataset (Audio != Text).
             pass
             
    print("Optimization: Cannot blindly re-slice without losing text alignment.")

if __name__ == "__main__":
    pass
