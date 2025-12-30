import json
import re
import os

# --- Benchmark Normalization Logic (Copied from verify_bench.py) ---
def normalize_text(text):
    # Keep only basic Arabic letters 0621-064A and spaces
    # Also normalize Alefs to bare Alif
    # Normalize Teh Marbuta to Ha
    
    # 1. Normalize chars
    text = re.sub(r'[إأٱآ]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text) 
    
    # 2. Remove non-letters (diacritics, symbols, punctuation, tatweel)
    val_chars = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي ا"
    
    out = ""
    for char in text:
        if char in val_chars:
            out += char
            
    # Collapse spaces
    out = re.sub(r'\s+', ' ', out)
    return out.strip()
# ----------------------------------------------------------------

METADATA_PATH = "data/normalized_metadata.jsonl"
REPORT_PATH = "audit_report.md"

def analyze():
    print("Starting Audit...")
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f.readlines()]
    except Exception as e:
        print(f"Failed to read metadata: {e}")
        return

    # Sample a few random lines (e.g. 5)
    samples = lines[:5] + lines[1000:1005] if len(lines) > 1000 else lines[:5]
    
    with open(REPORT_PATH, "w", encoding="utf-8") as rep:
        rep.write("# Dataset Normalization Audit\n\n")
        
        rep.write("## 1. Character Check\n")
        all_text = "".join([l['text'] for l in lines[:500]]) # Check first 500 lines
        unique_chars = sorted(list(set(all_text)))
        
        rep.write(f"**Unique Characters found in Training Data (First 500 lines):**\n")
        rep.write(f"`{' '.join(unique_chars)}`\n\n")
        
        rep.write("| Char | Hex | Status in Normalizer |\n")
        rep.write("|---|---|---|\n")
        
        val_chars = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي ا"
        
        for c in unique_chars:
            hex_val = f"{ord(c):04x}"
            # Simulate logic
            temp = c
            temp = re.sub(r'[إأٱآ]', 'ا', temp)
            temp = re.sub(r'ة', 'ه', temp)
            temp = re.sub(r'ى', 'ي', temp)
            
            status = "KEPT" if temp in val_chars else "REMOVED"
            rep.write(f"| {c} | {hex_val} | {status} |\n")
            
        rep.write("\n## 2. Sample Normalization\n")
        for i, s in enumerate(samples):
            raw = s['text']
            norm = normalize_text(raw)
            rep.write(f"### Sample {i+1}\n")
            rep.write(f"**Raw:** `{raw}`\n")
            rep.write(f"**Normalized:** `{norm}`\n")
            
            # Check length ratio
            if len(norm) == 0:
                 rep.write("**WARNING: Normalized text is EMPTY!**\n")
            elif len(norm) < len(raw) * 0.5:
                 rep.write(f"**WARNING: Significant reduction ({len(norm)}/{len(raw)})**\n")
            rep.write("\n---\n")
            
    print(f"Audit complete. Saved to {REPORT_PATH}")

if __name__ == "__main__":
    analyze()
