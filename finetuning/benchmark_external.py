"""
Benchmark: V5-30k Whisper vs. Conformer-CTC-Arabic-ASR
Compares transcription quality on child recitation samples.
"""
import sys
sys.path.insert(0, ".")  # Allow importing from backend/
sys.stdout.reconfigure(encoding='utf-8')

import os
import librosa

# --- Configuration ---
TEST_FILES = [
    "finetuning/test_samples/test 3.ogg",  # Surah Al-Hadid (Child)
    "finetuning/test_samples/test 6.mp4",  # Surah Al-Kafirun (Child)
]

# Our Model Path (Absolute)
OUR_MODEL_PATH = r"E:\Coding\Quran-ARS\finetuning\checkpoints\checkpoint-30000"

# --- Load Our V5-30k Model using our existing loader ---
print("Loading V5-30k Whisper Model...")
from backend.core.model_loader import QuranModel
our_model = QuranModel(OUR_MODEL_PATH)
print("V5-30k Loaded.\n")

# --- Try Loading Their Conformer CTC Model ---
nemo_available = False
their_model = None
try:
    import nemo.collections.asr as nemo_asr
    print("Loading Conformer-CTC-Arabic-ASR from Hugging Face...")
    their_model = nemo_asr.models.EncDecCTCModel.from_pretrained("MostafaAhmed98/Conformer-CTC-Arabic-ASR")
    their_model.eval()
    print("Conformer CTC Loaded.\n")
    nemo_available = True
except Exception as e:
    print(f"[WARNING] Could not load NeMo model: {e}\n")

# --- Helper: Transcribe with Our Model ---
def transcribe_whisper(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    return our_model.transcribe(audio, sampling_rate=16000)

# --- Helper: Transcribe with Their Model ---
def transcribe_nemo(audio_path):
    if not nemo_available:
        return "[NeMo not available]"
    transcription = their_model.transcribe([audio_path])
    return transcription[0]

# --- Run Benchmark ---
print("="*60)
print("BENCHMARK: V5-30k (Ours) vs. Conformer CTC (Theirs)")
print("="*60 + "\n")

for test_file in TEST_FILES:
    if not os.path.exists(test_file):
        print(f"[SKIP] File not found: {test_file}")
        continue
    
    print(f"File: {test_file}")
    print("-" * 40)
    
    # Our Model
    our_result = transcribe_whisper(test_file)
    print(f"[V5-30k]     : {our_result}")
    
    # Their Model
    their_result = transcribe_nemo(test_file)
    print(f"[Conformer]  : {their_result}")
    
    print()

print("="*60)
print("Benchmark Complete.")
