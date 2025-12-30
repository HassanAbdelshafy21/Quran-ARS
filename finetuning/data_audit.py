import os
import glob
from tqdm import tqdm
import librosa
import multiprocessing

DATA_ROOT = "data/dataset_cache/audio"

# Known categories
CATEGORIES = {
    # Ready (Ayah-level)
    "Husary_128kbps": ("Adult", "Ready"),
    "Abdul_Basit_Murattal_192kbps": ("Adult", "Ready"),
    "Minshawy_Murattal_128kbps": ("Adult", "Ready"),
    "Minshawy_Teacher_128kbps": ("Adult", "Ready"), # The teacher part
    "Minshawy_Child_Only": ("Child", "Ready"), 
    "Junaid_Segmented": ("Child", "Ready"),
    "Mesbahi_Segmented": ("Child", "Ready"),
    "Minshawi_Child_Extracted": ("Child", "Ready"),
    "Azazi_Child_Extracted": ("Child", "Ready"),
    
    # Mixed (Teacher+Child Repetition) -> Needs Separation
    "Al_Husayni_Al_Azazi_Children_mp3s": ("Mix", "Raw"),
    "downloaded_mp3s": ("Mix", "Raw"),
    
    # Pure Child (Raw)
    "Muhammad_Taha_Al_Junaid_mp3s": ("Child", "Raw"),
    "Ahmed_Al_Mesbahi_mp3s": ("Child", "Raw"), # User confirmed
}

def get_duration(f):
    try:
        return librosa.get_duration(path=f)
    except:
        return 0

def get_dir_stats(path, folder_name):
    files = glob.glob(os.path.join(path, "*.mp3")) + glob.glob(os.path.join(path, "*.wav"))
    count = len(files)
    size_mb = sum(os.path.getsize(f) for f in files) / (1024*1024)
    
    total_duration = 0
    estimated = False
    
    # Estimate if > 500 files to save time
    if count > 500:
        sample = files[:50] # Sample first 50
        sample_dur = sum(get_duration(f) for f in sample)
        if len(sample) > 0:
            avg_dur = sample_dur / len(sample)
            total_duration = avg_dur * count
        estimated = True
    else:
        total_duration = sum(get_duration(f) for f in files)
        estimated = False
        
    return {
        "count": count,
        "size_mb": size_mb,
        "duration_sec": total_duration,
        "estimated": estimated
    }

def print_row(name, type_, status, count, duration, size, estimated):
    dur_str = f"{duration/3600:.2f} hrs"
    est_mark = "*" if estimated else " "
    print(f"{name:<35} | {type_:<7} | {status:<6} | {count:>6} | {dur_str:>9}{est_mark} | {size:>7.2f} MB")

if __name__ == "__main__":
    print(f"{'Folder Name':<35} | {'Type':<7} | {'Status':<6} | {'Count':>6} | {'Duration':>10} | {'Size':>7}   ")
    print("-" * 95)
    
    totals = {"Child": 0, "Adult": 0, "Ready": 0, "Raw": 0, "Mix": 0}
    
    subdirs = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])
    
    for d in subdirs:
        full_path = os.path.join(DATA_ROOT, d)
        stats = get_dir_stats(full_path, d)
        
        cat = CATEGORIES.get(d, ("Unknown", "Unk"))
        type_, status = cat
        
        # Aggregate logic for Ready data only (approx)
        if status == "Ready":
            totals["Ready"] += stats["duration_sec"]
            if type_ == "Child": totals["Child"] += stats["duration_sec"]
            if type_ == "Adult": totals["Adult"] += stats["duration_sec"]
        
        print_row(d, type_, status, stats["count"], stats["duration_sec"], stats["size_mb"], stats["estimated"])

    print("-" * 95)
    print(f"\nSummary (Ready for Training Data Only):")
    print(f"Total Adult Duration: {totals['Adult']/3600:.2f} hrs")
    print(f"Total Child Duration: {totals['Child']/3600:.2f} hrs")
    print(f"Total Ready Duration: {totals['Ready']/3600:.2f} hrs")
