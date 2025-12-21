
import os
import librosa
from datasets import load_from_disk
from tqdm import tqdm
import numpy as np

DATASET_PATH = "data/quran_dataset"

def analyze():
    if not os.path.exists(DATASET_PATH):
        print("Dataset not found.")
        return

    print("Loading dataset...")
    dataset = load_from_disk(DATASET_PATH)
    print(f"Dataset Loaded. Total samples: {len(dataset)}")
    
    print("Calculating durations...")
    
    stats = {}
    
    for item in tqdm(dataset):
        reciter = item["reciter"]
        if reciter not in stats:
            stats[reciter] = {"total": 0, "over_30": 0, "between_25_30": 0, "under_25": 0, "duration_under_25": 0.0}
            
        stats[reciter]["total"] += 1
        
        try:
            audio_path = item["audio"]
            d = librosa.get_duration(path=audio_path)
            
            if d > 30:
                stats[reciter]["over_30"] += 1
            elif d > 25:
                stats[reciter]["between_25_30"] += 1
                total_duration_kept_30 += d
            else:
                stats[reciter]["under_25"] += 1
                stats[reciter]["duration_under_25"] += d
                total_duration_kept_30 += d

        except Exception as e:
            print(f"Error reading {item['audio']}: {e}")
            
    print(f"\n--- Statistics by Reciter (Detailed) ---")
    
    total_kept_25 = 0
    total_kept_30 = 0
    total_duration_30 = 0.0 # Seconds
    
    total_samples = len(dataset)
    
    for r, s in stats.items():
        total = s["total"]
        over_30 = s["over_30"]
        between = s["between_25_30"]
        under_25 = s["under_25"]
        
        avg_under_25 = s["duration_under_25"] / under_25 if under_25 > 0 else 0
        
        print(f"{r}: Total {total}")
        print(f"  > 30s:   {over_30} ({(over_30/total)*100:.1f}%)")
        print(f"  25s-30s: {between} ({(between/total)*100:.1f}%)")
        print(f"  < 25s:   {under_25} ({(under_25/total)*100:.1f}%) [Avg Dur: {avg_under_25:.2f}s]")

        
    print(f"\n--- Summary ---")
    print(f"Total Original: {total_samples}")
    # Recalculate kept counts
    total_kept_30 = 0
    for r, s in stats.items():
        total_kept_30 += (s["between_25_30"] + s["under_25"])
        
    print(f"Kept if Limit=30s: {total_kept_30} Files (Drop {(1 - total_kept_30/total_samples)*100:.1f}%)")
    print(f"Total Duration (Kept <30s): {total_duration_kept_30/3600:.2f} Hours")
    print(f"\n--- Child Data Analysis (Minshawy_Teacher_128kbps) ---")
    child_durations = []
    for item in tqdm(dataset):
        if item["reciter"] == "Minshawy_Teacher_128kbps":
            try:
                d = librosa.get_duration(path=item["audio"])
                child_durations.append(d)
            except:
                pass

    if child_durations:
        print(f"Child Samples: {len(child_durations)}")
        print(f"Child Min: {np.min(child_durations):.2f}s")
        print(f"Child Max: {np.max(child_durations):.2f}s")
        print(f"Child Avg: {np.mean(child_durations):.2f}s")
    else:
        print("No child data found.")

if __name__ == "__main__":
    analyze()
