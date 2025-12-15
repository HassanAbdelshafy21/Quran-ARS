
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests
from src.quran_db.core import QuranDB
from src.utils.audio import preprocess_audio
from src.asr.transcriber import ASRTranscriber

import argparse

def main():
    parser = argparse.ArgumentParser(description="Verify Quran ASR Integration")
    parser.add_argument("--surah", type=int, default=1, help="Surah number (1-114)")
    parser.add_argument("--ayah", type=int, default=1, help="Ayah number")
    args = parser.parse_args()

    print(f"=== Integration Verification: Phases 1 to 4 ===")
    print(f"Target: Surah {args.surah}, Ayah {args.ayah}")
    
    # 1. Quran DB Verification
    print("\n[1/4] Verifying Quran Database...")
    db = QuranDB()
    ayah_obj = db.get_ayah(args.surah, args.ayah)
    if ayah_obj:
        print(f"✅ Retrieved Ayah {args.surah}:{args.ayah}: {ayah_obj.aya_text}")
    else:
        print(f"❌ Failed to retrieve Ayah {args.surah}:{args.ayah}")
        return

    # 2. Audio Download (Sample)
    # Format: 001001.mp3 -> leading zeros, 3 digits for surah, 3 digits for ayah
    file_id = f"{args.surah:03d}{args.ayah:03d}"
    print(f"\n[2/4] Downloading Sample Audio ({file_id}.mp3)...")
    audio_url = f"https://everyayah.com/data/Abdul_Basit_Mujawwad_128kbps/{file_id}.mp3"
    raw_audio_path = f"data/sample_{file_id}.mp3"
    
    if not os.path.exists(raw_audio_path):
        try:
            r = requests.get(audio_url)
            if r.status_code == 200:
                with open(raw_audio_path, 'wb') as f:
                    f.write(r.content)
                print(f"✅ Downloaded sample to {raw_audio_path}")
            else:
                print(f"❌ Failed to download audio (Status {r.status_code})")
                return
        except Exception as e:
            print(f"❌ Failed to download audio: {e}")
            return
    else:
        print(f"✅ Sample audio already exists at {raw_audio_path}")

    # 3. Audio Preprocessing
    print("\n[3/4] Preprocessing Audio...")
    try:
        processed_path = preprocess_audio(raw_audio_path)
        print(f"✅ Processed audio saved to {processed_path}")
    except Exception as e:
        print(f"❌ Audio processing failed: {e}")
        return

    # 4. ASR Transcription
    print("\n[4/4] Running ASR Model...")
    try:
        # Initialize transcriber
        transcriber = ASRTranscriber()
        print("✅ Model loaded.")
        
        # Transcribe
        result = transcriber.transcribe(processed_path)
        transcribed_text = result["text"]
        print(f"🎤 Transcription: {transcribed_text}")
        
        # Comparison
        print("\n--- Comparison ---")
        print(f"Original: {ayah_obj.aya_text}")
        print(f"ASR Output: {transcribed_text}")
        
        if transcribed_text.strip() == ayah_obj.aya_text.strip():
            print("🎉 Exact Match!")
        else:
            print("⚠️ Not exact match (Expected for MVP/LoRA). Check tashkeel/spelling.")
            
    except Exception as e:
        print(f"❌ ASR failed: {e}")
        return

if __name__ == "__main__":
    main()
