
import os
import librosa
import soundfile as sf
import random
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

DATASET_PATH = "data/quran_dataset"
OUTPUT_FILE = "chipmunk_sample.wav"

def generate_sample():
    print("Loading dataset...")
    dataset = load_from_disk(DATASET_PATH)
    
    # 1. Find a file in our target range (25-30s)
    candidates = []
    print("Searching for a 25-30s file...")
    for i, item in enumerate(tqdm(dataset)):
        # Check duration
        audio_path = item["audio"]
        try:
            d = librosa.get_duration(path=audio_path)
            if 25.0 <= d <= 30.0:
                candidates.append(item)
                if len(candidates) > 5: # Just need a few
                    break
        except:
            continue
            
    if not candidates:
        print("No files found between 25-30s!")
        return

    # Pick one
    target = candidates[0]
    print(f"Selected: {target['audio']}")
    
    # 2. Load and Shift
    print("Applying Chipmunk Effect (Pitch Shift up 3 semitones)...")
    y, sr = librosa.load(target['audio'], sr=16000)
    
    # Apply shift
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=3.0)
    
    # 3. Save
    sf.write(OUTPUT_FILE, y_shifted, sr)
    print(f"File saved to: {os.path.abspath(OUTPUT_FILE)}")
    print("Play this file to hear the effect.")

if __name__ == "__main__":
    generate_sample()
