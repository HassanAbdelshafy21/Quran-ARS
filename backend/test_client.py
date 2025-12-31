import requests

API_URL = "http://localhost:8000/grade_recitation"
FILE_PATH = "finetuning/test_samples/test 4.mp4" # Run from project root

# Correct Text for Test 4
TARGET_TEXT = "وَٱلتِّينِ وَٱلزَّيۡتُونِ وَطُورِ سِينِينَ وَهَٰذَا ٱلۡبَلَدِ ٱلۡأَمِينِ"

def test_api():
    print(f"Sending {FILE_PATH} to {API_URL}...")
    
    with open(FILE_PATH, 'rb') as f:
        files = {'file': f}
        data = {'target_ayah': TARGET_TEXT}
        
        try:
            response = requests.post(API_URL, files=files, data=data)
            
            if response.status_code == 200:
                print("\n[SUCCESS]")
                print(response.json())
            else:
                print(f"\n[ERROR] {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"Failed to connect: {e}")

if __name__ == "__main__":
    test_api()
