import argparse
import os
import sys
import io
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import librosa

# Force UTF-8 for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def transcribe_comparison(audio_path, base_model_name="tarteel-ai/whisper-base-ar-quran", adapter_path="finetuning/checkpoints/full-run-v2/checkpoint-10000"):
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return

    print(f"\nLoading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = WhisperProcessor.from_pretrained(base_model_name)(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda").half()

    # 1. Inference with Base Model ONLY
    print(f"Loading BASE model: {base_model_name}")
    model = WhisperForConditionalGeneration.from_pretrained(base_model_name, device_map="auto", torch_dtype=torch.float16)
    model.eval()
    
    print("Transcribing with Base Model...")
    with torch.no_grad():
        base_ids = model.generate(inputs, max_new_tokens=225)
    base_text = WhisperProcessor.from_pretrained(base_model_name).batch_decode(base_ids, skip_special_tokens=True)[0]

    # 2. Inference with LoRA Adapter
    print(f"Loading LoRA adapter: {adapter_path}")
    try:
        model = PeftModel.from_pretrained(model, adapter_path)
    except Exception as e:
        print(f"Error loading adapter: {e}")
        return
    
    print("Transcribing with Fine-Tuned Model...")
    with torch.no_grad():
        lora_ids = model.generate(inputs, max_new_tokens=225)
    lora_text = WhisperProcessor.from_pretrained(base_model_name).batch_decode(lora_ids, skip_special_tokens=True)[0]

    return base_text, lora_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the audio file")
    
    args = parser.parse_args()
    
    print("-" * 50)
    base, lora = transcribe_comparison(args.audio_path)
    
    print("\n" + "=" * 50)
    print("COMPARISON RESULTS")
    print("=" * 50)
    print(f"[Base Model]:\n{base}")
    print("-" * 20)
    print(f"[Fine-Tuned]:\n{lora}")
    print("=" * 50 + "\n")
