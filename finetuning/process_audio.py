
import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import glob

# Constants
INPUT_DIR = "data/dataset_cache/audio/Minshawy_Teacher_128kbps"
OUTPUT_DIR = "data/dataset_cache/audio/Minshawy_Child_Only"

def process_file(file_path):
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=None)
        
        # Primary attempt with top_db=20
        intervals = librosa.effects.split(y, top_db=20, frame_length=2048, hop_length=512)
        
        # Filter short segments (noise) - less than 0.5s
        min_samples = int(0.5 * sr)
        valid_intervals = [i for i in intervals if (i[1] - i[0]) > min_samples]
        
        # Fallback if splits not found
        if len(valid_intervals) < 2:
            # Try more aggressive split (louder threshold)
            intervals = librosa.effects.split(y, top_db=25, frame_length=2048, hop_length=512)
            valid_intervals = [i for i in intervals if (i[1] - i[0]) > min_samples]
            
            if len(valid_intervals) < 2:
                 # Try less aggressive (in case noise is quiet but speech is soft?) - Unlikely for this issue.
                 # Actually if we see 1 big segment, we need LOWER top_db (more aggressive silence detection)
                 intervals = librosa.effects.split(y, top_db=15, frame_length=2048, hop_length=512)
                 valid_intervals = [i for i in intervals if (i[1] - i[0]) > min_samples]

        if len(valid_intervals) < 2:
            return False, f"Could not isolate 2 major segments. Found {len(valid_intervals)}"
            
        # Strategy: The Teacher and Student are providing the bulk of the content.
        # Find the 2 longest segments.
        # Calculate durations
        interval_durations = [(i[1] - i[0], i) for i in valid_intervals]
        
        # Sort by duration desc
        interval_durations.sort(key=lambda x: x[0], reverse=True)
        
        # Take top 2
        top_2 = interval_durations[:2]
        
        # Sort by time (start index) to find order
        top_2.sort(key=lambda x: x[1][0])
        
        # Teacher is first, Student is second
        student_interval = top_2[1][1]
        
        # Verify student segment is substantial?
        # print(f"  Teacher len: {top_2[0][0]/sr:.2f}s, Student len: {top_2[1][0]/sr:.2f}s")
        
        # Extract
        y_student = y[student_interval[0]:student_interval[1]]
        
        # Save
        filename = os.path.basename(file_path)
        out_path = os.path.join(OUTPUT_DIR, filename)
        sf.write(out_path, y_student, sr)
        
        return True, "Success"

    except Exception as e:
        return False, str(e)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    files = glob.glob(os.path.join(INPUT_DIR, "*.mp3"))
    print(f"Found {len(files)} files in {INPUT_DIR}")
    
    success = 0
    skipped = 0
    
    for f in tqdm(files):
        ok, msg = process_file(f)
        if ok:
            success += 1
        else:
            skipped += 1
            # print(f"Skipped {os.path.basename(f)}: {msg}")
            
    print(f"Processed: {success}, Skipped: {skipped}")

if __name__ == "__main__":
    main()
