import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa
import os
import glob
from tqdm import tqdm
from peft import PeftModel, PeftConfig
import sys
import jiwer
import re
import time

# Add path for QuranDB
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.quran_db.core import QuranDB

# Force UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Config
TEST_DIR = "finetuning/test_samples"
OUTPUT_FILE = "finetuning/benchmark_v5_final.md"

# Defined Models
# Defined Models
MODELS = {
    # "Whisper Base (OpenAI)": "openai/whisper-base",
    "Tarteel Base": "tarteel-ai/whisper-base-ar-quran",
    "V3 (50k)": "finetuning/checkpoints/checkpoint-50000",
    "V5 (30k)": "finetuning/checkpoints_v5/checkpoint-30000",
    "V5 (40k)": "finetuning/checkpoints_v5/checkpoint-40000",
    "V5 (50k)": "finetuning/checkpoints_v5/checkpoint-50000",
}

# Ground Truth Mapping: Filename -> Surah Number
GT_MAPPING = {
    "test 4.mp4": 95,  # At-Tin
    "test 5.mp4": 112, # Al-Ikhlas
    "test 6.mp4": 109, # Al-Kafirun
    "sanity_mesbahi.mp3": "أَلَآ إِنَّهُمۡ هُمُ ٱلۡمُفۡسِدُونَ وَلَٰكِن لَّا يَشۡعُرُونَ ﰋ وَإِذَا قِيلَ لَهُمۡ ءَامِنُواْ كَمَآ ءَامَنَ ٱلنَّاسُ قَالُوٓاْ أَنُؤۡمِنُ كَمَآ ءَامَنَ ٱلسُّفَهَآءُۗ أَلَآ إِنَّهُمۡ هُمُ ٱلسُّفَهَآءُ وَلَٰكِن لَّا يَعۡلَمُونَ ﰌ"
}

def normalize_text(text):
    import re
    # Keep only basic Arabic letters 0621-064A and spaces
    # Also normalize Alefs to bare Alif
    # Normalize Teh Marbuta to Ha
    
    # 1. Normalize chars
    text = re.sub(r'[إأٱآ]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text) # Optional: Alif Maqsura to Ya (common in Whisper)
    
    # 2. Remove non-letters (diacritics, symbols, punctuation, tatweel)
    # Range 0621-064A includes Hamza, Alif...Ya. 
    # But we want to exclude others.
    # Easiest: Keep specific set.
    val_chars = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي ا"
    
    out = ""
    for char in text:
        if char in val_chars:
            out += char
            
    # Collapse spaces
    out = re.sub(r'\s+', ' ', out)
    return out.strip()

def get_ground_truth(db, filename):
    fname = os.path.basename(filename)
    if fname in GT_MAPPING:
        val = GT_MAPPING[fname]
        if isinstance(val, str):
            return val
            
        sura_no = val
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT aya_text FROM quran WHERE sura_no = ? ORDER BY aya_no", (sura_no,))
        rows = cursor.fetchall()
        full_text = " ".join([r['aya_text'] for r in rows])
        conn.close()
        return full_text
    return None

def load_model(model_id, device):
    print(f"Loading {model_id}...")
    try:
        start_time = time.time()
        
        # Check if it's a PEFT model (LoRA)
        is_peft = False
        if "lora" in model_id.lower() or "kheemp" in model_id.lower():
            is_peft = True
        elif os.path.exists(os.path.join(model_id, "adapter_config.json")):
            is_peft = True
            
        if is_peft:
            peft_config = PeftConfig.from_pretrained(model_id)
            base_model_path = peft_config.base_model_name_or_path
            
            print(f"  base model: {base_model_path}")
            base_model = WhisperForConditionalGeneration.from_pretrained(base_model_path).to(device)
            model = PeftModel.from_pretrained(base_model, model_id)
            processor = WhisperProcessor.from_pretrained(base_model_path)
            model = model.merge_and_unload() # Optional: Merge for speed
        else:
            model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
            processor = WhisperProcessor.from_pretrained(model_id)
            
        model.eval()
        
        if device == "cuda":
            model.half()
            
        print(f"  Loaded in {time.time() - start_time:.2f}s")
        return model, processor
    except Exception as e:
        print(f"Failed to load {model_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def transcribe(model, processor, audio_path, device):
    # Load separate to avoid pipeline issues
    try:
        y, sr = librosa.load(audio_path, sr=16000)
    except Exception as e:
        # Try moviepy fallback
        try:
            from moviepy import AudioFileClip
            temp = "temp_bench.wav"
            clip = AudioFileClip(audio_path)
            clip.write_audiofile(temp, fps=16000, logger=None)
            y, sr = librosa.load(temp, sr=16000)
            os.remove(temp)
        except Exception as e2:
            return f"[Error Loading: {e} / {e2}]"

    # Preprocess
    features = processor(y, sampling_rate=sr, return_tensors="pt").input_features.to(device)
    if device == "cuda":
        features = features.half()
        
    # Generate
    # Force Arabic
    forced_ids = processor.get_decoder_prompt_ids(language="ar", task="transcribe")
    
    # Handle forced_decoder_ids for newer transformers
    model.config.forced_decoder_ids = forced_ids
    if hasattr(model, "generation_config"):
        model.generation_config.forced_decoder_ids = forced_ids

    with torch.no_grad():
        gen_ids = model.generate(features, max_new_tokens=400) # Increased token limit for full surah
        
    transcript = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return transcript

def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    db = QuranDB()
    
    # Get test files
    files = glob.glob(os.path.join(TEST_DIR, "*"))
    # Filter for interesting ones if desired, or all
    # files = [f for f in files if "test 4" in f or "test 5" in f or "test 6" in f]
    files = sorted(files)
    
    results = {}
    
    # Run Models
    for model_name, model_id in MODELS.items():
        print(f"\n--- Benchmarking {model_name} ---")
        model, processor = load_model(model_id, device)
        if not model: continue
        
        results[model_name] = {}
        
        for f in tqdm(files):
            fname = os.path.basename(f)
            # Only process files we have GT for or all? User wants WER. 
            # If we don't have GT, WER is N/A.
            
            print(f"Processing {fname}...")
            transcription = transcribe(model, processor, f, device)
            
            # Metric Calculation
            ground_truth = get_ground_truth(db, f)
            metrics = ""
            wer_val = None
            cer_val = None
            
            if ground_truth:
                ref_norm = normalize_text(ground_truth)
                hyp_norm = normalize_text(transcription)
                
                if ref_norm:
                    wer_val = jiwer.wer(ref_norm, hyp_norm)
                    cer_val = jiwer.cer(ref_norm, hyp_norm)
                    metrics = f"WER: {wer_val:.2f} | CER: {cer_val:.2f}"
                else:
                    metrics = "Ref Empty (Norm)"
            else:
                metrics = "N/A"
                
            results[model_name][fname] = {
                "text": transcription,
                "metrics": metrics,
                "wer": wer_val
            }
            
        del model
        del processor
        torch.cuda.empty_cache()
        
    # Generate Report
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("# Benchmark Comparison v2 (With WER)\n\n")
        
        for file_path in files:
            fname = os.path.basename(file_path)
            f.write(f"## File: {fname}\n")
            
            # Show Ground Truth if available
            gt = get_ground_truth(db, file_path)
            if gt:
                f.write(f"**Ground Truth:** {gt}\n\n")
            
            f.write("| Model | WER | Transcription |\n")
            f.write("|---|---|---|\n")
            
            for model_name in MODELS.keys():
                if model_name in results and fname in results[model_name]:
                    data = results[model_name][fname]
                    # Clean text for table - remove newlines, pipes
                    clean_text = data['text'].replace("\n", " ").replace("|", "-")
                    f.write(f"| {model_name} | {data['metrics']} | {clean_text} |\n")
            
            f.write("\n---\n")
            
    print(f"Done! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_benchmark()
