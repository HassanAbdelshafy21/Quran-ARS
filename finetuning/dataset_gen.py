
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
        # Default to correct identifiers for: Husary (Hafs), Abdul Basit (Murattal), Minshawy (Murattal)
        self.reciters = reciters or ["Husary_128kbps", "Abdul_Basit_Murattal_192kbps", "Minshawy_Murattal_128kbps"]
        self.db = QuranDB()
        os.makedirs(self.audio_base_dir, exist_ok=True)

    def download_ayah_audio(self, sura_no, aya_no, reciter):
        """Downloads audio for a specific Ayah if not present."""
        reciter_dir = os.path.join(self.audio_base_dir, reciter)
        os.makedirs(reciter_dir, exist_ok=True)
        
        file_name = f"{sura_no:03d}{aya_no:03d}.mp3"
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

    def create_dataset(self, ayahs_to_fetch=None, full_quran=False):
        """Creates a HF dataset with multiple reciters."""
        if full_quran:
            print("Generating FULL Quran dataset for all reciters...")
            ayahs_to_fetch = []
            for s in range(1, 115):
                for a in range(1, 300):
                    if self.db.get_ayah(s, a):
                        ayahs_to_fetch.append((s, a))
                    else:
                        break
        elif not ayahs_to_fetch:
            print("No ayahs specified. Defaulting to Al-Fatiha.")
            ayahs_to_fetch = [(1, i) for i in range(1, 8)]

        data = []
        print(f"Processing {len(ayahs_to_fetch)} ayahs for {len(self.reciters)} reciters...")
        
        for sura, aya in ayahs_to_fetch:
            # Get text once (same for all reciters)
            ayah_obj = self.db.get_ayah(sura, aya)
            if not ayah_obj:
                continue
            text = ayah_obj.aya_text

            for reciter in self.reciters:
                audio_path = self.download_ayah_audio(sura, aya, reciter)
                if audio_path:
                    data.append({
                        "audio": audio_path,
                        "text": text,
                        "surah": sura,
                        "ayah": aya,
                        "reciter": reciter,
                        "id": f"{reciter}_{sura}_{aya}"
                    })
        
        if not data:
            print("No data collected.")
            return None

        ds = Dataset.from_list(data)
        return ds

if __name__ == "__main__":
    gen = QuranDatasetGenerator()
    
    is_full = "--full" in sys.argv
    ds = gen.create_dataset(full_quran=is_full)
    
    if ds:
        print(ds)
        save_path = "data/quran_dataset"
        print(f"Saving dataset to {save_path}...")
        ds.save_to_disk(save_path)
        print("Done.")
