import json
import sys

# Force UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

target_file = "data/dataset_normalized/audio\\batch_003305_Ahmed_Al_Mesbahi_Child.mp3"
# Normalize slashes for comparison
target_file = target_file.replace("\\", "/")

with open('data/normalized_metadata.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if target_file in data['audio_path'].replace("\\", "/"):
            print(data['text'])
            break
