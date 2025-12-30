import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa
import numpy as np
import os
import sys

# Constants
BASE_MODEL = "tarteel-ai/whisper-base-ar-quran"
FILE_PATH = "data/dataset_cache/audio/Muhammad_Taha_Al_Junaid_mp3s/001.mp3"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Model
    print("Loading model...")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL).to(device)
    model.eval()

    # Load Audio
    print(f"Loading {FILE_PATH}...")
    audio, sr = librosa.load(FILE_PATH, sr=16000)
    
    # Silence Splitting
    print("Splitting via silence...")
    # top_db=40 usually good for studio. Junaid is studio.
    intervals = librosa.effects.split(audio, top_db=45, frame_length=2048, hop_length=512)
    
    print(f"Found {len(intervals)} chunks.")
    
    results = []
    
    for idx, (start, end) in enumerate(intervals):
        chunk_audio = audio[start:end]
        duration = len(chunk_audio) / sr
        
        # Skip tiny blips
        if duration < 1.0:
            continue
            
        # Transcribe
        inputs = processor(chunk_audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        if device == "cuda":
            model.half()
            input_features = input_features.half()
            
        with torch.no_grad():
            generated_ids = model.generate(input_features, max_new_tokens=100)
        
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"Chunk {idx+1}: {duration:.2f}s -> {text}")

if __name__ == "__main__":
    main()
