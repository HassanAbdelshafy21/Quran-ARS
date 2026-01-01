import requests

# Configuration
API_URL = "http://localhost:8000/grade_recitation"
FILE_PATH = "finetuning/test_samples/test 3.ogg" # Updated to Test 3
# Correct Text for Test 3 (Surah Al-Hadid 57:1-2)
TARGET_TEXT = "سَبَّحَ لِلَّهِ مَا فِي السَّمَاوَاتِ وَالْأَرْضِ وَهُوَ الْعَزِيزُ الْحَكِيمُ لَهُ مُلْكُ السَّمَاوَاتِ وَالْأَرْضِ يُحْيِي وَيُمِيتُ وَهُوَ عَلَى كُلِّ شَيْءٍ قَدِيرٌ"

def test_api():
    print(f"Sending {FILE_PATH} to {API_URL}...")
    
    with open(FILE_PATH, 'rb') as f:
        
        try:
            response = requests.post(
                API_URL, 
                files={"file": f}, 
                data={
                    "target_ayah": TARGET_TEXT,
                    "surah_num": 57
                    # "ayah_num": 1 # Full Surah mode
                }
            )
            
            if response.status_code == 200:
                print("\n[SUCCESS]")
                # Save to file to avoid Encoding Errors
                import json
                with open("response_log.json", "w", encoding="utf-8") as f:
                    json.dump(response.json(), f, indent=2, ensure_ascii=False)
                print("Saved response to response_log.json")
            else:
                print(f"\n[ERROR] {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"Failed to connect: {e}")

if __name__ == "__main__":
    test_api()
