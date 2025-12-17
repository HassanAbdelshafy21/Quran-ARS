
import os
import torch
import torchaudio
import librosa
import soundfile as sf

# Path to a sample file
audio_path = "data/dataset_cache/audio/001001.mp3"

print(f"Testing audio loading for: {audio_path}")
print(f"File exists: {os.path.exists(audio_path)}")

print("\n--- Testing Torchaudio ---")
print(f"Torchaudio version: {torchaudio.__version__}")
# print(f"Torchaudio backends: {torchaudio.list_audio_backends()}")

try:
    waveform, sample_rate = torchaudio.load(audio_path)
    print(f"Success! Loaded with torchaudio. Shape: {waveform.shape}, SR: {sample_rate}")
except Exception as e:
    print(f"Torchaudio failed: {e}")

print("\n--- Testing Librosa ---")
print(f"Librosa version: {librosa.__version__}")
try:
    y, sr = librosa.load(audio_path, sr=16000)
    print(f"Success! Loaded with librosa. Shape: {y.shape}, SR: {sr}")
except Exception as e:
    print(f"Librosa failed: {e}")

print("\n--- Testing SoundFile directly ---")
try:
    data, samplerate = sf.read(audio_path)
    print(f"Success! Loaded with soundfile. Shape: {data.shape}, SR: {samplerate}")
except Exception as e:
    print(f"SoundFile failed: {e}")
