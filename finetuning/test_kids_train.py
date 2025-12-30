import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import json
import os
import random
import librosa
import sys

# Force UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

METADATA_FILE = "data/v5_metadata.jsonl"
MODEL_ID = "finetuning/checkpoints_v5/checkpoint-30000"

def load_samples():
    print(f"Loading metadata from {METADATA_FILE}...")
    samples = {
        "minshawi_child": [],
        "azazi_child": [],
        "mesbahi": []
    }
    
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            reciter = r.get('reciter', '').lower()
            
            # Simple heuristic mapping based on how we built V5
            if "minshawi" in reciter and "repeat" in r['audio_path'].lower(): 
                 # Minshawi Repeat was the target
                 samples["minshawi_child"].append(r)
            elif "azazi" in reciter:
                samples["azazi_child"].append(r)
            elif "mesbahi" in reciter:
                samples["mesbahi"].append(r)
            # Note: Explicitly ignoring Husary/AbdulBasit (Adults)
                
    print(f"Found: {len(samples['minshawi_child'])} Minshawi, {len(samples['azazi_child'])} Azazi, {len(samples['mesbahi'])} Mesbahi")
    
    # Pick 2 random from each
    final_list = []
    for key in samples:
        if samples[key]:
            picked = random.sample(samples[key], 2)
            for p in picked:
                p['source'] = key
                final_list.append(p)
                
    return final_list

def run_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} on {device}...")
    
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    processor = WhisperProcessor.from_pretrained("tarteel-ai/whisper-base-ar-quran")
    
    # Force Arabic
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ar", task="transcribe")
    model.config.suppress_tokens = []
    
    if device == "cuda":
        model.half()
        
    test_samples = load_samples()
    
    print("\n--- Verifying Training Data (Kids) ---")
    
    for sample in test_samples:
        path = sample['audio_path']
        truth = sample['text']
        source = sample['source']
        
        print(f"\n[{source.upper()}] Path: {os.path.basename(path)}")
        print(f"Truth: {truth}")
        
        # Transcribe
        try:
            y, sr = librosa.load(path, sr=16000)
            inputs = processor(y, sampling_rate=sr, return_tensors="pt").input_features.to(device)
            if device == "cuda": inputs = inputs.half()
            
            with torch.no_grad():
                out = model.generate(inputs, max_new_tokens=400)
            pred = processor.batch_decode(out, skip_special_tokens=True)[0]
            
            print(f"Model: {pred}")
            
        except Exception as e:
            print(f"Error processing {path}: {e}")

if __name__ == "__main__":
    run_test()
