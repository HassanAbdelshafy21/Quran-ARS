import json
import re
import os
from tqdm import tqdm

INPUT_FILE = "data/normalized_metadata.jsonl"
OUTPUT_FILE = "data/cleaned_metadata.jsonl"

def normalize_text(text):
    if not text: return ""
    
    # 1. Unite Alefs
    text = re.sub(r'[إأٱآ]', 'ا', text)
    
    # 2. Unite Teh Marbuta -> Ha
    text = re.sub(r'ة', 'ه', text)
    
    # 3. Alif Maqsura -> Ya
    text = re.sub(r'ى', 'ي', text)
    
    # 4. Filter non-basic chars (Diacritics, Tatweel, Symbols, Ligatures)
    # Allowed: Basic Arabic Letters + Space
    # Range 0621-064A covers most, but specific allows precise control.
    valid_chars = set("ابتثجحخدذرزسشصضطظعغفقكلمنهوي ا")
    
    out = []
    for char in text:
        if char in valid_chars:
            out.append(char)
            
    normalized = "".join(out)
    
    # 5. Collapse multiple spaces
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

def main():
    print(f"Cleaning text from {INPUT_FILE} -> {OUTPUT_FILE}...")
    
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        
    count = 0
    skipped = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        
        for line in tqdm(fin):
            try:
                data = json.loads(line)
                original_text = data.get('text', "")
                
                clean_text = normalize_text(original_text)
                
                if not clean_text:
                    skipped += 1
                    continue
                    
                # Update text field
                data['text'] = clean_text
                
                # Write back
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                count += 1
                
            except Exception as e:
                print(f"Error processing line: {e}")
                
    print(f"Done! Processed {count} lines. Skipped {skipped} empty lines.")

if __name__ == "__main__":
    main()
