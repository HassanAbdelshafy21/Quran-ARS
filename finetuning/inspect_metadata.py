import json
import os

metadata_path = "data/cleaned_metadata.jsonl"

print(f"--- Inspecting {metadata_path} ---")
try:
    with open(metadata_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5: break
            data = json.loads(line)
            print(f"Line {i}:")
            print(f"  Reciter: {data.get('reciter')}")
            print(f"  Text: {data.get('text')}")
            print("-" * 20)
except Exception as e:
    print(f"Error reading metadata: {e}")
