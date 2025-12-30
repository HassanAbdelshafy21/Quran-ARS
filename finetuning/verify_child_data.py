import os
import glob
from collections import Counter

AUDIO_DIR = "data/dataset_normalized/audio"

def main():
    print(f"Scanning {AUDIO_DIR}...")
    files = glob.glob(os.path.join(AUDIO_DIR, "*.mp3"))
    print(f"Total Files Found: {len(files)}")
    
    counts = Counter()
    child_files = []
    
    for f in files:
        # Filename format: batch_000123_Reciter_Name.mp3
        # Or rebatch_000123_Reciter_Name.mp3
        base = os.path.basename(f)
        parts = base.replace(".mp3", "").split("_")
        
        # Heuristic to find Reciter Name
        # Usually it's after the 2nd underscore?
        # batch_000123_Reciter_Name -> Reciter_Name is parts[2:]
        # But Reciter Names can have underscores.
        
        # Let's extract known names
        name = base
        if "Junaid" in name: 
            counts["Junaid"] += 1
            child_files.append(base)
        elif "Mesbahi" in name: 
            counts["Mesbahi"] += 1
            child_files.append(base)
        elif "Azazi" in name: 
            counts["Azazi"] += 1
            child_files.append(base)
        elif "Minshawi_Child" in name:
            counts["Minshawi_Child"] += 1
            child_files.append(base)
        elif "Minshawy_Child" in name: # Spelling variant
            counts["Minshawi_Child"] += 1
            child_files.append(base)
        elif "Husary" in name:
            counts["Husary_Adult"] += 1
        elif "Minshawy_Adult" in name:
            counts["Minshawy_Adult"] += 1
        elif "AbdulBasit" in name:
            counts["AbdulBasit"] += 1
        else:
            counts["Other"] += 1
            
    print("\n--- Reciter Statistics (On Disk) ---")
    for k, v in counts.items():
        print(f"{k}: {v}")
        
    print(f"\nTotal Child Files: {len(child_files)}")
    
    if len(child_files) > 0:
        print("CONFIRMED: Child data is present on disk.")
    else:
        print("WARNING: NO CHILD DATA FOUND.")

if __name__ == "__main__":
    main()
