import json
import os
import shutil

METADATA_FILE = "data/kids_metadata.jsonl"
TEMP_FILE = "data/kids_metadata.jsonl.tmp"

if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, 'r', encoding='utf-8') as f_in, open(TEMP_FILE, 'w', encoding='utf-8') as f_out:
        kept = 0
        removed = 0
        for line in f_in:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                # Remove Minshawi_Child entries to allow fresh restart without duplicates
                if data.get("reciter") == "Minshawi_Child":
                    removed += 1
                else:
                    f_out.write(line)
                    kept += 1
            except:
                pass # skip bad lines
    
    shutil.move(TEMP_FILE, METADATA_FILE)
    print(f"Removed {removed} Minshawi_Child entries. Kept {kept} others.")
else:
    print("Metadata file not found.")
