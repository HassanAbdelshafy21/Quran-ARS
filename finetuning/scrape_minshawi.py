
import os
import requests
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import glob

# Constants
# New candidate item from search results
BASE_URL = "https://archive.org/download/Al-Mushaf_Al-Mualim_For_children_riwayat_Hafs_An_Assem_recited_by_Mohamed_Siddiq_El-Minshawi/"
DIRECT_BASE_URL = None # Will be set by list_files logic

OUTPUT_DIR = "data/dataset_cache/audio/Minshawy_Teacher_128kbps"
REFERENCE_DIR = "data/dataset_cache/audio/Minshawy_Murattal_128kbps"

def download_file(url, filepath):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    return False

def get_reference_durations(surah_num):
    """
    Get the duration of each ayah in the reference reciter's folder.
    Returns a list of (ayah_num, duration_sec).
    """
    surah_str = f"{surah_num:03d}"
    pattern = os.path.join(REFERENCE_DIR, f"{surah_str}*.mp3")
    files = sorted(glob.glob(pattern))
    
    durations = []
    for fpath in files:
        # File name format: 001001.mp3 (SurahAyah)
        fname = os.path.basename(fpath)
        ayah_str = fname[3:6]
        try:
            dur = librosa.get_duration(path=fpath)
            durations.append((int(ayah_str), dur))
        except Exception as e:
            print(f"Error reading ref file {fpath}: {e}")
            
    return durations

def list_files_and_pattern():
    global DIRECT_BASE_URL
    import time
    time.sleep(1)
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36'}
    
    items = [BASE_URL.split("/")[-2], "Al-Mushaf_Al-Mualim_For_children_riwayat_Hafs_An_Assem_recited_by_Mohamed_Siddiq_El-Minshawi", "Mohamed_Siddiq_El-Minshawi_Mushaf_Mualim"]
    
    for item in items:
        if not item: continue
        print(f"Checking metadata for Item: '{item}'")
        api_url = f"https://archive.org/metadata/{item}"
        
        try:
            r = requests.get(api_url, headers=headers)
            print(f"API Status: {r.status_code}")
            if r.status_code == 200:
                data = r.json()
                keys = list(data.keys())
                print(f"API Keys: {keys}")
                
                if 'files' in data and len(data['files']) > 0:
                    files = [f['name'] for f in data['files'] if f['name'].endswith('.mp3')]
                    server = data.get('d1') or data.get('d2')
                    directory = data.get('dir')
                    print(f"Direct Server: {server}, Directory: {directory}")
                    
                if server and directory:
                    DIRECT_BASE_URL = f"https://{server}{directory}/"
                    print(f"Using Direct URL: {DIRECT_BASE_URL}")
                    return files
        except Exception as e:
            print(f"Error querying API for {item}: {e}")
            
    # Fallback: Try downloading the _files.xml directly
    # We need the correct item name for this.
    current_item = BASE_URL.split("/")[-2]
    xml_url = f"https://archive.org/download/{current_item}/{current_item}_files.xml"
    print(f"Attempting to download metadata XML: {xml_url}")
    try:
        r = requests.get(xml_url, headers=headers)
        if r.status_code == 200:
             content = r.text
             # primitive parsing
             # <file name="001.mp3" source="original">
             import re
             files = re.findall(r'file name="([^"]+\.mp3)"', content)
             print(f"Found {len(files)} MP3s in XML. Examples: {files[:5]}")
             
             # If successful, we don't have direct server info, 
             # so we must rely on the redirector (which was 503ing).
             # BUT sometimes knowing the exact filename helps?
             # No, if redirector is 503, exact filename won't help.
             # We NEED the direct server.
             # Does XML contain server info? No.
             
             # Wait, can we guess the server from the XML response headers?
             # archive.org often redirects or serves from a d1 node.
             # Let's check r.url or r.history
             if hasattr(r, 'url'):
                 print(f"XML Final URL: {r.url}") # e.g. https://ia800505.us.archive.org/.../file.xml
                 # Extract server base
                 # https://ia800505.us.archive.org/24/items/Mujawwad_Minshawi_Child_Repeat/file.xml
                 if "/items/" in r.url:
                     base = r.url.split("/items/")[0] + "/items/" + r.url.split("/items/")[1].split("/")[0] + "/"
                     DIRECT_BASE_URL = base
                     print(f"inferred DIRECT_BASE_URL: {DIRECT_BASE_URL}")

             return files
    except Exception as e:
        print(f"XML download failed: {e}")

    print("All API attempts failed.")
    return []

def process_surah(surah_num, known_files=None):
    surah_str = f"{surah_num:03d}"
    
    # Try multiple patterns
    patterns = [
        f"{surah_str}.mp3",
        f"{surah_str}.Opus",  # Archive sometimes lists case sensitive
        f"{surah_num}.mp3",
        f"{surah_str}_Minshawi_Child_Repeat.mp3",
        f"{surah_str}_Child_Repeat.mp3",
        f"{surah_str}_P.mp3" # Sometimes P for practice?
    ]
    
    # If we have a scraped list, prioritize it
    target_file = None
    if known_files:
        for f in known_files:
            if f.startswith(surah_str):
                target_file = f
                break
    
    if target_file:
         patterns.insert(0, target_file)

    final_url = None
    first_base = DIRECT_BASE_URL if DIRECT_BASE_URL else BASE_URL
    for p in patterns:
        url = f"{first_base}{p}"
        print(f"Checking {url}...")
        try:
           # Use GET with stream=True to avoid downloading body but check headers/status
           # HEAD often blocked or 403 on some servers
           r = requests.get(url, stream=True, allow_redirects=True)
           print(f"Status: {r.status_code}")
           if r.status_code == 200:
               final_url = url
               print(f"Found match: {url}")
               r.close()
               break
           r.close()
        except Exception as check_exc:
           print(f"Check failed: {check_exc}")
        
        # DEBUG: Break after one pattern check to avoid log spam
        break
           
    if not final_url:
        print(f"Could not find valid URL for Surah {surah_num}")
        return

    local_mp3 = os.path.join(OUTPUT_DIR, f"{surah_str}_full.mp3")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Download
    if not os.path.exists(local_mp3):
        print(f"Downloading Surah {surah_num} from {final_url}...")
        success = download_file(final_url, local_mp3)
        if not success:
            print(f"Failed to download Surah {surah_num}")
            return

    # 2. Load Audio
    print(f"Loading Surah {surah_num} audio...")
    y, sr = librosa.load(local_mp3, sr=16000)
    
    # 3. Get Guidelines (Reference Durations of Teacher Alone)
    # The Muallim audio is Teacher + Student.
    # Usually: Teacher Recites (T) + Pause + Student Recites (S) + Pause.
    # We want to extract 'S'.
    # This is HARD without silence detection.
    # Simple Heuristic: 
    #   The total duration for an Ayah in Muallim is roughly 2.2x to 2.5x the teacher's duration.
    #   Teacher (1.0) + Student (1.0) + Pauses (0.5).
    #   But we need to synchronize.
    
    # BETTER APPROACH FOR NOW:
    # Just cut the whole file into Ayah chunks using the *Relative* durations?
    # No, that will drift.
    # We need silence detection.
    
    # Let's try `librosa.effects.split` to find non-silent chunks.
    # Then group chunks into Ayahs based on count?
    
    print(f"Splitting Surah {surah_num}...")
    # top_db=30 is a reasonable default for silence
    intervals = librosa.effects.split(y, top_db=40, frame_length=2048, hop_length=512)
    
    # Ref durations to know how many ayahs we expect
    ref_durations = get_reference_durations(surah_num)
    expected_ayahs = len(ref_durations)
    
    print(f"Found {len(intervals)} non-silent intervals. Expected {expected_ayahs} Ayahs.")
    
# Surah Ayah counts (Juz Amma + Fatiha)
AYAH_COUNTS = {
    1: 7, 78: 40, 79: 46, 80: 42, 81: 29, 82: 19, 83: 36, 84: 25, 85: 22, 86: 17, 87: 19,
    88: 26, 89: 30, 90: 20, 91: 15, 92: 21, 93: 11, 94: 8, 95: 8, 96: 19, 97: 5, 98: 8,
    99: 8, 100: 11, 101: 11, 102: 8, 103: 3, 104: 9, 105: 5, 106: 4, 107: 7, 108: 3,
    109: 6, 110: 3, 111: 5, 112: 4, 113: 5, 114: 6
}

def process_everyayah_surah(surah_num):
    # For EveryAyah: Download 001001.mp3, 001002.mp3...
    surah_str = f"{surah_num:03d}"
    # Flat structure for dataset_gen compatibility
    out_surah_dir = OUTPUT_DIR 
    os.makedirs(out_surah_dir, exist_ok=True)
    
    ayah_count = AYAH_COUNTS.get(surah_num, 286) # Default max if unknown, but better to be safe
    
    print(f"Downloading Surah {surah_num} ({ayah_count} Ayahs) from EveryAyah...")
    
    success_count = 0
    for ayah in range(1, ayah_count + 1):
        ayah_str = f"{ayah:03d}"
        filename = f"{surah_str}{ayah_str}.mp3"
        url = f"{BASE_URL}{filename}"
        local_path = os.path.join(out_surah_dir, f"{surah_str}{ayah_str}.mp3")
        
        if not os.path.exists(local_path):
             # Simple download
             try:
                 r = requests.get(url, stream=True)
                 if r.status_code == 200:
                     with open(local_path, 'wb') as f:
                         for chunk in r.iter_content(chunk_size=4096):
                             f.write(chunk)
                     success_count += 1
                 else:
                     print(f"Failed {url}: {r.status_code}")
             except Exception as e:
                 print(f"Error downloading {url}: {e}")
        else:
            success_count += 1
            
    print(f"Surah {surah_num}: Downloaded {success_count}/{ayah_count} Ayahs.")

def process_surah(surah_num, known_files=None):
    if "everyayah" in BASE_URL:
        process_everyayah_surah(surah_num)
        return

    surah_str = f"{surah_num:03d}"
    
    # ... Legacy Full Surah Logic ...
    # (Kept for reference but likely unused if we stick to EA)
    pass

def main():
    known_files = []
    
    # Try EveryAyah URLs first (Best Case)
    ea_bases = [
        "http://www.everyayah.com/data/Al_Minshawy_-_Murattal_-_with_Child_128kbps/",
        "http://www.everyayah.com/data/Minshawy_Teacher_128kbps/", 
        "http://www.everyayah.com/data/Al-Minshawy_Teacher_128kbps/",
        "http://www.everyayah.com/data/Minshawy_Child_128kbps/",
        "http://www.everyayah.com/data/Minshawy_Muallim_128kbps/",
        "http://www.everyayah.com/data/Al_Minshawy_Muallim_128kbps/",
        # Mp3Quran Fallback
        "https://server10.mp3quran.net/minshawi/Al-Mushaf-Al-Muallim/",
        "https://server8.mp3quran.net/minshawi/Al-Mushaf-Al-Muallim/",
        "https://server6.mp3quran.net/minshawi/Al-Mushaf-Al-Muallim/"
    ]
    
    found_ea = False
    for base in ea_bases:
        print(f"Checking Base: {base}")
        try:
             # Heuristic: If "everyayah", check 001001.mp3. Else check 001.mp3
             if "everyayah" in base:
                 test_url = f"{base}001001.mp3"
             else:
                 test_url = f"{base}001.mp3"
                 
             # Mp3Quran sometimes needs User-Agent
             headers = {'User-Agent': 'Mozilla/5.0'}
             r = requests.head(test_url, headers=headers, allow_redirects=True)
             print(f"Status for {test_url}: {r.status_code}")
             
             if r.status_code == 200:
                 print(f"FOUND SOURCE: {base}")
                 global BASE_URL
                 BASE_URL = base
                 global DIRECT_BASE_URL
                 DIRECT_BASE_URL = base
                 found_ea = True
                 break
        except Exception as e:
            print(f"Check failed: {e}")
            
    if not found_ea:
        known_files = list_files_and_pattern()
        
    target_surahs = [1] + list(range(78, 115))
    print(f"Starting download for {len(target_surahs)} Surahs (Juz Amma)...")
    
    for s in target_surahs:
        process_surah(s)

if __name__ == "__main__":
    main()
