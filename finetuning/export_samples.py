import json
import shutil
import os

METADATA_IN = "data/cleaned_metadata.jsonl"
EXPORT_DIR = "finetuning/v4_samples_export"
METADATA_OUT = os.path.join(EXPORT_DIR, "samples_metadata.jsonl")

def main():
    print("Exporting diverse samples...")
    if not os.path.exists(EXPORT_DIR):
        os.makedirs(EXPORT_DIR)
        
    # Clear existing audio files
    for f in os.listdir(EXPORT_DIR):
        if f.endswith(".mp3") or f.endswith(".jsonl"):
            try: os.remove(os.path.join(EXPORT_DIR, f))
            except: pass

    seen_reciters = set()
    copied = 0
    target_count = 5
    
    with open(METADATA_IN, 'r', encoding='utf-8') as f_in, \
         open(METADATA_OUT, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if copied >= target_count: break
            
            try:
                data = json.loads(line)
                reciter = data.get('reciter', 'Unknown')
                
                if reciter in seen_reciters:
                    continue
                    
                audio_path = data.get('audio_path') or data.get('audio')
                if not audio_path: continue

                # Normalize path sep
                audio_path = audio_path.replace("\\", "/")
                
                if os.path.exists(audio_path):
                    fname = os.path.basename(audio_path)
                    dest = os.path.join(EXPORT_DIR, fname)
                    shutil.copy2(audio_path, dest)
                    
                    # Update path in metadata for portability
                    data['audio'] = fname
                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                    
                    print(f"Copied {fname} ({reciter})")
                    seen_reciters.add(reciter)
                    copied += 1
                else:
                    # print(f"File missing: {audio_path}")
                    pass
            except Exception as e:
                print(f"Error: {e}")
                
    print(f"Exported {copied} diverse samples to {EXPORT_DIR}")

if __name__ == "__main__":
    main()
