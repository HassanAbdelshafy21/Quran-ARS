import librosa
import os
import numpy as np
from pydub import AudioSegment

FILE = r"data/dataset_cache/audio/downloaded_mp3s/001.mp3"

print(f"Testing file: {FILE}")

try:
    print("Attempting librosa.load...")
    y, sr = librosa.load(FILE, sr=16000)
    print(f"Librosa success. Shape: {y.shape}")
except Exception as e:
    print(f"Librosa failed: {e}")

try:
    print("Attempting pydub load...")
    audio = AudioSegment.from_file(FILE)
    print(f"Pydub success. Duration: {len(audio)/1000}s")
    
    # Convert to standard numpy for verification
    data = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        data = data.reshape((-1, 2))
    print(f"Pydub numpy shape: {data.shape}")
except Exception as e:
    print(f"Pydub failed: {e}")
