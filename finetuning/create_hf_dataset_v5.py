from datasets import Dataset, Audio
import json
import os
import sys

# Force UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

METADATA_FILE = "data/v5_metadata.jsonl"
OUTPUT_DIR = "data/quran_dataset_v5"

def main():
    print(f"Loading V5 Metadata from {METADATA_FILE}...")
    
    data = []
    if not os.path.exists(METADATA_FILE):
        print(f"Error: {METADATA_FILE} not found.")
        return

    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                data.append({
                    "audio": record['audio_path'],
                    "text": record['text'],
                    "duration": record['duration'],
                    "reciter": record['reciter'],
                    "id": os.path.basename(record['audio_path'])
                })
            except Exception as e:
                print(f"Skipping line: {e}")

    print(f"Loaded {len(data)} samples.")
    
    if not data:
        print("No data found.")
        return

    print("Creating Hugging Face Dataset...")
    ds = Dataset.from_list(data)
    
    # Cast Audio Column
    print("Casting Audio column...")
    # ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    print(f"Saving to {OUTPUT_DIR}...")
    ds.save_to_disk(OUTPUT_DIR)
    
    print("Done. V5 Dataset is ready for training.")

if __name__ == "__main__":
    main()
