
import os
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizerFast
from datasets import load_from_disk
from peft import PeftModel
import jiwer
from tqdm import tqdm
import re
import numpy as np
import sys

# Force UTF-8 output for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Constants
BASE_MODEL = "tarteel-ai/whisper-base-ar-quran"
ADAPTER_PATH = "finetuning/checkpoints/checkpoint-7720" # Final checkpoint
DATASET_PATH = "data/quran_dataset"
BATCH_SIZE = 16

def normalize_text(text):
    # Basic normalization if needed. 
    # For now, we assume the model output tries to match Uthmani or Emlaey.
    # We might want to remove diacritics for a fairer comparison if the model struggles with them.
    # Simple regex to remove Tashkeel (diacritics) for basic WER:
    text = re.sub(r'[\u064B-\u0652\u06D6-\u06ED]', '', text)
    return text.strip()

def evaluate():
    print("Loading processor and model...")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    
    # Load Base Model
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL, 
        torch_dtype=torch.float32, # CPU usually needs float32
        device_map="cpu"
    )
    
    # Load LoRA Adapter
    print(f"Loading adapter from {ADAPTER_PATH}")
    try:
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        print("Adapter loaded successfully.")
    except Exception as e:
        print(f"Error loading adapter: {e}")
        return

    model.eval()
    
    print("Loading dataset...")
    dataset = load_from_disk(DATASET_PATH)
    
    # Taking a subset for quick evaluation (e.g. 10 random samples)
    dataset = dataset.select(range(10))
    print(f"Evaluating on {len(dataset)} samples...")

    references = []
    predictions = []

    print("Running inference...")
    
    import librosa
    
    for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
        batch = dataset[i:i+BATCH_SIZE]
        audio_paths = batch["audio"]
        texts = batch["text"]
        
        input_features_list = []
        valid_indices = []
        
        for idx, path in enumerate(audio_paths):
            try:
                # Load audio
                y, sr = librosa.load(path, sr=16000)
                # Feature extraction
                features = processor.feature_extractor(y, sampling_rate=16000).input_features[0]
                input_features_list.append(features)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
        
        if not input_features_list:
            continue
            
        input_features = torch.tensor(np.array(input_features_list)).to(model.device).to(torch.float32)

        # Set language and task via forced_decoder_ids on the config
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="ar", task="transcribe")
        model.config.forced_decoder_ids = forced_decoder_ids

        with torch.no_grad():
            generated_ids = model.generate(input_features)
        
        transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        for idx, trans in enumerate(transcriptions):
            orig_idx = valid_indices[idx]
            ref = texts[orig_idx]
            
            # optional normalization
            # trans = normalize_text(trans)
            # ref = normalize_text(ref)
            
            predictions.append(trans)
            references.append(ref)
            
            if i == 0 and idx < 3:
                print(f"\nRef: {ref}")
                print(f"Pred: {trans}")
                
    wer = jiwer.wer(references, predictions)
    print(f"\nWord Error Rate (WER): {wer:.4f}")
    
    # Normalized WER
    norm_refs = [normalize_text(r) for r in references]
    norm_preds = [normalize_text(p) for p in predictions]
    norm_wer = jiwer.wer(norm_refs, norm_preds)
    print(f"Normalized WER (No Tashkeel): {norm_wer:.4f}")

if __name__ == "__main__":
    evaluate()
