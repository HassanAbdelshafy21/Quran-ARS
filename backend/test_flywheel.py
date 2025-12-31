import requests
import os
import time

API_URL = "http://localhost:8000"
FILE_PATH = "finetuning/test_samples/test 6.mp4" 
TARGET_TEXT = "قُلْ يَا أَيُّهَا الْكَافِرُونَ"

def test_flywheel():
    print("1. Sending Audio to Grade...")
    with open(FILE_PATH, "rb") as f:
        response = requests.post(
            f"{API_URL}/grade_recitation",
            files={"file": f},
            data={"target_ayah": TARGET_TEXT}
        )
    
    data = response.json()
    req_id = data.get("request_id")
    print(f"   Received Request ID: {req_id}")
    
    if not req_id:
        print("FAIL: No Request ID returned")
        return

    print("2. Simulating User Report ('I said it right!')...")
    report_payload = {
        "request_id": req_id,
        "user_comment": "I swear I said 'Qul' correctly!"
    }
    
    resp2 = requests.post(f"{API_URL}/report_issue", json=report_payload)
    print(f"   Report Status: {resp2.status_code}")
    print(f"   Response: {resp2.json()}")
    
    # 3. Verify Server Side
    golden_path = f"data/golden_negatives/{req_id}.json"
    if os.path.exists(golden_path):
        print("SUCCESS: Metadata file found in golden_negatives!")
    else:
        print(f"FAIL: {golden_path} not found.")

if __name__ == "__main__":
    test_flywheel()
