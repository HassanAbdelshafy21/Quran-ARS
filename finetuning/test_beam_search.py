import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, GenerationConfig
import librosa
import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

MODEL_ID = "finetuning/checkpoints_v5/checkpoint-30000"
TEST_FILE = "finetuning/test_samples/test 4.mp4" # At-Tin

def run_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} on {device}...")
    
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    processor = WhisperProcessor.from_pretrained("tarteel-ai/whisper-base-ar-quran")
    
    if device == "cuda":
        model.half()
        
    print(f"Loading audio {TEST_FILE}...")
    # Use moviepy fallback just in case librosa fails with mp4
    try:
        y, sr = librosa.load(TEST_FILE, sr=16000)
    except:
        from moviepy import AudioFileClip
        clip = AudioFileClip(TEST_FILE)
        clip.write_audiofile("temp.wav", fps=16000, logger=None)
        y, sr = librosa.load("temp.wav", sr=16000)

    inputs = processor(y, sampling_rate=sr, return_tensors="pt").input_features.to(device)
    if device == "cuda":
        inputs = inputs.half()

    # 1. Greedy Search (Baseline)
    print("\n--- Greedy Search (Beam=1) ---")
    
    # Set direct config to bypass argument validation hell
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ar", task="transcribe")
    model.config.suppress_tokens = []
    
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=400)
    print(processor.batch_decode(out, skip_special_tokens=True)[0])

    # 2. Beam Search (Better Thinking)
    print("\n--- Beam Search (Beam=5) ---")
    
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=400, num_beams=5, early_stopping=True)
    print(processor.batch_decode(out, skip_special_tokens=True)[0])

if __name__ == "__main__":
    run_test()
