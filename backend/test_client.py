import requests

API_URL = "http://localhost:8000/grade_recitation"
FILE_PATH = "finetuning/test_samples/test 6.mp4" # Al-Kafirun

# Correct Text for Test 6 (Surah Al-Kafirun)
TARGET_TEXT = "قُلْ يَا أَيُّهَا الْكَافِرُونَ لَا أَعْبُدُ مَا تَعْبُدُونَ وَلَا أَنْتُمْ عَابِدُونَ مَا أَعْبُدُ وَلَا أَنَا عَابِدٌ مَا عَبَدْتُمْ وَلَا أَنْتُمْ عَابِدُونَ مَا أَعْبُدُ لَكُمْ دِينُكُمْ وَلِيَ دِينِ"

def test_api():
    print(f"Sending {FILE_PATH} to {API_URL}...")
    
    with open(FILE_PATH, 'rb') as f:
        files = {'file': f}
        data = {'target_ayah': TARGET_TEXT}
        
        try:
            response = requests.post(API_URL, files=files, data=data)
            
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
