import re
import jiwer
from datasets import load_from_disk
import sys

# Force UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

ds = load_from_disk("data/quran_dataset")
sample_text = ds[0]['text'] # Bismillah

def normalize_text(text):
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'[إأٱآ]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    return text.strip()

whisper_out = "بسم الله الرحمن الرحيم"
dataset_raw = sample_text
dataset_norm = normalize_text(dataset_raw)
whisper_norm = normalize_text(whisper_out)

print(f"Dataset Raw: {dataset_raw}")
print(f"Dataset Norm: {dataset_norm}")
print(f"Whisper Out:  {whisper_out}")
print(f"Whisper Norm: {whisper_norm}")
print(f"WER: {jiwer.wer(dataset_norm, whisper_norm)}")

# Test chars
print("--- Hex Dump ---")
print("Dataset Norm Hex:", dataset_norm.encode('utf-8').hex())
print("Whisper Norm Hex:", whisper_norm.encode('utf-8').hex())
