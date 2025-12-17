
import os
import requests
from datasets import Dataset, Audio
import sys

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quran_db.core import QuranDB

class QuranDatasetGenerator:
    def __init__(self, output_dir="data/dataset_cache", reciter="Husary_128kbps"):
        # Husary is the standard for Hafs. Using HTTPS to avoid ISP blocking.
        self.output_dir = output_dir
        self.audio_dir = os.path.join(output_dir, "audio")
        self.reciter = reciter
        self.db = QuranDB()
        os.makedirs(self.audio_dir, exist_ok=True)

    def download_ayah_audio(self, sura_no, aya_no):
        """Downloads audio for a specific Ayah if not already present."""
        file_name = f"{sura_no:03d}{aya_no:03d}.mp3"
        url = f"https://everyayah.com/data/{self.reciter}/{file_name}"
        path = os.path.join(self.audio_dir, file_name)
        
        if not os.path.exists(path):
            print(f"Downloading {file_name} as {path}...")
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
        """
        Creates a HuggingFace dataset.
        ayahs_to_fetch: list of tuples (sura_no, aya_no)
        full_quran: if True, downloads the entire Quran (Hafs)
        """
        if full_quran:
            print("Generating FULL Quran dataset (Hafs)...")
            ayahs_to_fetch = []
            # There are 114 Surahs. We can iterate and query DB until we hit None for an Ayah.
            for s in range(1, 115):
                for a in range(1, 300): # Max ayahs in a sura is 286 (Baqarah)
                    if self.db.get_ayah(s, a):
                        ayahs_to_fetch.append((s, a))
                    else:
                        break
        elif not ayahs_to_fetch:
            print("No ayahs specified. Defaulting to Al-Fatiha (Sura 1).")
            ayahs_to_fetch = [(1, i) for i in range(1, 8)]

        data = []
        print(f"Processing {len(ayahs_to_fetch)} ayahs...")
        
        for sura, aya in ayahs_to_fetch:
            audio_path = self.download_ayah_audio(sura, aya)
            if audio_path:
                # Retrieve text from local DB
                ayah_obj = self.db.get_ayah(sura, aya)
                if ayah_obj:
                    # You might want to use aya_text_emlaey or aya_text depending on normalization needs
                    # Whisper usually expects normalized or standard text. 
                    # For now using 'aya_text' (Uthmani) but 'aya_text_emlaey' might be better for ASR.
                    # Let's include both for flexibility.
                    text = ayah_obj.aya_text 
                    
                    data.append({
                        "audio": audio_path,
                        "text": text,
                        "surah": sura,
                        "ayah": aya,
                        "id": f"{sura}_{aya}"
                    })
        
        if not data:
            print("No data collected.")
            return None

        # Create HF Dataset
        ds = Dataset.from_list(data)
        # We will load audio manually in training loop to avoid dependency hell with torchcodec/ffmpeg on Windows
        # ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        
        return ds

if __name__ == "__main__":
    gen = QuranDatasetGenerator()
    
    is_full = "--full" in sys.argv
    ds = gen.create_dataset(full_quran=is_full)
    
    if ds:
        print(ds)
        # print("Sample entry:", ds[0]) # Avoid unicode error in Windows terminal
        save_path = "data/quran_dataset"
        print(f"Saving dataset to {save_path}...")
        ds.save_to_disk(save_path)
        print("Done.")
