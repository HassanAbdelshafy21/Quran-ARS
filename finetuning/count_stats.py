import json
import collections

meta_file = "data/normalized_metadata.jsonl"
counts = collections.Counter()
durations = []

with open(meta_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            d = json.loads(line)
            r = d.get('reciter', 'Unknown')
            counts[r] += 1
            durations.append(d.get('duration', 0))
        except: pass

print("Reciter Counts:")
for k, v in counts.items():
    print(f"{k}: {v}")

if durations:
    print(f"\nTotal Files: {len(durations)}")
    print(f"Min Duration: {min(durations):.2f}")
    print(f"Max Duration: {max(durations):.2f}")
    print(f"Avg Duration: {sum(durations)/len(durations):.2f}")
else:
    print("No data found.")
