import librosa
import numpy as np
import soundfile as sf
import os
import sys

# Force UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def smart_segment(filename, output_dir):
    print(f"Loading {filename}...")
    try:
        y, sr = librosa.load(filename, sr=16000)
    except Exception as e:
        print(f"Error loading file using librosa: {e}")
        # Try moviepy fallback
        from moviepy import AudioFileClip
        clip = AudioFileClip(filename)
        clip.write_audiofile("temp_vad.wav", fps=16000, logger=None)
        y, sr = librosa.load("temp_vad.wav", sr=16000)

    print(f"Audio Duration: {len(y)/sr:.2f}s")
    
    # 1. Detect Non-Silent Intervals (VAD)
    # top_db=25 implies silence is 25dB below peak (Adjustable for noise)
    # frame_length/hop_length defines "resolution" of detection
    intervals = librosa.effects.split(y, top_db=25, frame_length=2048, hop_length=512)
    
    print(f"Detected {len(intervals)} speech segments (Breath Groups).")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Export Chunks
    # In production, we would stitch these until they reach ~30s, then cut at the nearest silence.
    # Here we just dump them to prove we found the silences.
    
    for i, (start, end) in enumerate(intervals):
        # Add a tiny padding (0.1s)
        pad = int(0.1 * sr)
        start = max(0, start - pad)
        end = min(len(y), end + pad)
        
        chunk = y[start:end]
        duration = len(chunk) / sr
        
        # Filter tiny blips just in case
        if duration < 0.2: continue
            
        out_name = os.path.join(output_dir, f"segment_{i:02d}.wav")
        sf.write(out_name, chunk, sr)
        print(f"  ✅ Segment {i}: {duration:.2f}s (Saved to {out_name})")

if __name__ == "__main__":
    # Test on the At-Tin file (Child Recitation)
    TEST_FILE = "finetuning/test_samples/test 4.mp4" 
    OUT_DIR = "finetuning/vad_output"
    smart_segment(TEST_FILE, OUT_DIR)
