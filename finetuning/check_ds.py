from datasets import load_from_disk
import sys

# Force UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

ds = load_from_disk("data/quran_dataset")
print(ds)
# Print first sample to see structure
# We expect 'chapter_id', 'verse_id', 'text'
print(ds[0])
