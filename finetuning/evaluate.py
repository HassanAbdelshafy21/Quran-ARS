import os
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, GenerationConfig
from datasets import load_from_disk
from peft import PeftModel
import jiwer
from tqdm import tqdm
import re
import numpy as np
import sys
import glob
import random
import librosa
import gc

# Try to import moviepy for MP4 fallback
try:
    from moviepy import AudioFileClip
except ImportError:
    AudioFileClip = None

# Force UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Constants
BASE_MODEL = "tarteel-ai/whisper-base-ar-quran"
CHECKPOINTS_DIR = "finetuning/checkpoints"
DATASET_PATH = "data/quran_dataset"
BATCH_SIZE = 16

def load_audio_file(audio_path):
    """Loads audio using librosa, falling back to moviepy for MP4s."""
    try:
        # Try direct load
        audio_array, sr = librosa.load(audio_path, sr=16000)
        return audio_array
    except Exception as e:
        # If fallback is available and it's likely a video format
        if AudioFileClip and audio_path.lower().endswith(('.mp4', '.m4a', '.mov')):
            try:
                # Convert to temporary wav
                temp_wav = "temp_audio_extract.wav"
                clip = AudioFileClip(audio_path)
                clip.write_audiofile(temp_wav, fps=16000, logger=None)
                # Load the temp wav
                audio_array, sr = librosa.load(temp_wav, sr=16000)
                # Clean up
                try:
                    os.remove(temp_wav)
                except:
                    pass
                return audio_array
            except Exception as mp_err:
                print(f"MoviePy failed for {audio_path}: {mp_err}")
                raise e # Raise original librosa error
        else:
            raise e

def clean_memory():
    torch.cuda.empty_cache()
    gc.collect()

def main():
    TEST_DIR = "finetuning/test_samples"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    
    # scan files
    audio_files = glob.glob(os.path.join(TEST_DIR, "*"))
    # Filter for audio types
    audio_files = [f for f in audio_files if f.lower().endswith(('.mp3', '.wav', '.ogg', '.mp4', '.m4a'))]
    audio_files.sort()
    
    if not audio_files:
        print(f"No audio files found in {TEST_DIR}")
        return

    print(f"Found {len(audio_files)} test files.")
    
    # Prepare pseudo-dataset structure for our function
    # The function expects a dataset object with "audio" key, but we can pass a list of dicts or just modify the function.
    # Actually, generate_predictions expects a dataset object that is sliceable.
    # Let's create a simple list of dicts and modify generate_predictions to handle it.
    
    # Or cleaner: just rewrite the loop here since it's small data (7 files).
    
    models_to_test = [
        ("Base Model", None),
        ("Old LoRA", "KheemP/whisper-base-quran-lora"),
        ("New Model (40k)", os.path.join(CHECKPOINTS_DIR, "checkpoint-40000"))
    ]
    
    results = {f: {} for f in audio_files}
    
    for model_name, adapter_path in models_to_test:
        print(f"\n--- Loading {model_name} ---")
        model = WhisperForConditionalGeneration.from_pretrained(
            BASE_MODEL, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device
        )
        
        if adapter_path:
            try:
                model = PeftModel.from_pretrained(model, adapter_path)
                print(f"Loaded adapter: {adapter_path}")
            except Exception as e:
                print(f"Error loading adapter: {e}")
                continue
                
        model.eval()
        model.generation_config = GenerationConfig.from_model_config(model.config)
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="ar", task="transcribe")
        model.config.forced_decoder_ids = forced_decoder_ids
        model.generation_config.forced_decoder_ids = forced_decoder_ids
        
        print(f"Transcribing {len(audio_files)} files...")
        for audio_path in tqdm(audio_files):
            try:
                # Load
                audio_array = load_audio_file(audio_path)
                features = processor.feature_extractor(audio_array, sampling_rate=16000).input_features[0]
                
                input_features = torch.tensor(np.array([features])).to(device)
                if device == "cuda":
                    input_features = input_features.half()
                
                with torch.no_grad():
                    generated_ids = model.generate(input_features, max_new_tokens=400)
                
                transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                results[audio_path][model_name] = transcription
                
            except Exception as e:
                results[audio_path][model_name] = f"ERROR: {e}"
        
        # Cleanup
        del model
        clean_memory()

    # Print Report
    print("\n" + "="*80)
    print("CUSTOM FILE EVALUATION REPORT")
    print("="*80)
    
    for audio_path in audio_files:
        filename = os.path.basename(audio_path)
        print(f"\nFile: {filename}")
        print("-" * 40)
        for model_name, _ in models_to_test:
            text = results[audio_path].get(model_name, "N/A")
            print(f"{model_name:<15}: {text}")

if __name__ == "__main__":
    main()
