
import os
import requests
from datasets import Dataset, Audio
import sys

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quran_db.core import QuranDB

class QuranDatasetGenerator:
    def __init__(self, output_dir="data/dataset_cache", reciters=None):
        self.output_dir = output_dir
        self.audio_base_dir = os.path.join(output_dir, "audio")
        # Default to correct identifiers for: Husary (Hafs), Abdul Basit (Murattal), Minshawy (Murattal), Minshawi (Muallim/Child)
        self.reciters = reciters or ["Husary_128kbps", "Abdul_Basit_Murattal_192kbps", "Minshawy_Murattal_128kbps", "Minshawy_Teacher_128kbps"]
        self.db = QuranDB()
        os.makedirs(self.audio_base_dir, exist_ok=True)

    def download_ayah_audio(self, sura_no, aya_no, reciter):
        """Downloads audio for a specific Ayah if not present."""
        reciter_dir = os.path.join(self.audio_base_dir, reciter)
        os.makedirs(reciter_dir, exist_ok=True)
        
        file_name = f"{sura_no:03d}{aya_no:03d}.mp3"
        if reciter == "Minshawy_Teacher_128kbps":
             # Use locally processed files if available
             local_processed_path = os.path.join(self.output_dir, "audio", "Minshawy_Child_Only", file_name)
             if os.path.exists(local_processed_path):
                 return local_processed_path
        
        url = f"https://everyayah.com/data/{reciter}/{file_name}"
        path = os.path.join(reciter_dir, file_name)
        
        if not os.path.exists(path):
            print(f"Downloading {file_name} for {reciter}...")
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    with open(path, 'wb') as f:
                        f.write(r.content)
                else:
                    print(f"Failed to download {url} (Status: {r.status_code})")
                    return None
            except Exception as e:
                print(f"Error downloading {url}: {e}")
                return None
        return path

    def create_dataset(self):
        """Creates a HF dataset with multiple reciters, handling specific ranges for each."""
        from tqdm import tqdm
        
        data = []
        FULL_QURAN_SURAHS = list(range(1, 115))
        JUZ_AMMA_AND_FATIHA = [1] + list(range(78, 115))

        for reciter in self.reciters:
            print(f"Processing {reciter}...")
            
            if reciter == "Minshawy_Teacher_128kbps":
                target_surahs = JUZ_AMMA_AND_FATIHA
                print(f"  - Logic: Juz Amma + Fatiha")
            else:
                target_surahs = FULL_QURAN_SURAHS
                print(f"  - Logic: Full Quran")
            
            for s in tqdm(target_surahs, desc=f"Surahs for {reciter}"):
                 # Iterate ayahs until we hit the limit for the surah
                 for a in range(1, 300):
                     ayah_obj = self.db.get_ayah(s, a)
                     if not ayah_obj:
                         break
                     
                     text = ayah_obj.aya_text
                     audio_path = self.download_ayah_audio(s, a, reciter)
                     
                     if audio_path:
                        data.append({
                            "audio": audio_path,
                            "text": text,
                            "surah": s,
                            "ayah": a,
                            "reciter": reciter,
                            "id": f"{reciter}_{s}_{a}"
                        })

        if not data:
            print("No data collected.")
            return None

        ds = Dataset.from_list(data)
        return ds

if __name__ == "__main__":
    gen = QuranDatasetGenerator()
    ds = gen.create_dataset()
    
    if ds:
        print(ds)
        ds.save_to_disk(os.path.join(gen.output_dir, "quran_dataset"))
        print(f"Saving dataset to {os.path.join(gen.output_dir, 'quran_dataset')}...")
        print("Done.")
    else:
        # Default to Juz Amma + Fatiha (what we scraped)
        target_surahs = [1] + list(range(78, 115))
        ayahs_to_fetch = []
        print("Generating dataset for Juz Amma (Surahs 78-114 + 1)...")
        for s in target_surahs:
            # We don't know exact ayah count easily without DB, but DB calls are cheap
            # Loop a safe amount
             for a in range(1, 300):
                    if gen.db.get_ayah(s, a):
                        ayahs_to_fetch.append((s, a))
                    else:
                        break
        ds = gen.create_dataset(ayahs_to_fetch=ayahs_to_fetch)
    
    if ds:
        print(ds)
        save_path = "data/quran_dataset"
        print(f"Saving dataset to {save_path}...")
        ds.save_to_disk(save_path)
        print("Done.")
