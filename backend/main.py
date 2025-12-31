from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import os
import sys

# Add current dir to path to find 'core'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.grader import QuranGrader
from core.segmenter import AudioSegmenter
from core.model_loader import QuranModel

# Initialize App
app = FastAPI(title="Quran ASR Kids API", version="1.0.0")

# Global State (Loaded on Startup)
MODEL_CHECKPOINT = "finetuning/checkpoints_v5/checkpoint-30000"
model = None
grader = None
segmenter = None

@app.on_event("startup")
def load_resources():
    global model, grader, segmenter
    print("Initializing Backend Resources...")
    
    # 1. Load V5 Model (Heavy)
    try:
        model = QuranModel(MODEL_CHECKPOINT)
    except Exception as e:
        print(f"CRITICAL: Failed to load model from {MODEL_CHECKPOINT}. {e}")
        # In dev, we might start anyway, but in prod we should fail.
        
    # 2. Load Logic
    grader = QuranGrader()
    segmenter = AudioSegmenter()
    print("Backend Ready! 🚀")

import uuid
from pydantic import BaseModel

# Additional Dirs
TEMP_STORAGE = "temp_storage"
GOLDEN_DATASET = "data/golden_negatives"
os.makedirs(TEMP_STORAGE, exist_ok=True)
os.makedirs(GOLDEN_DATASET, exist_ok=True)

class ReportRequest(BaseModel):
    request_id: str
    user_comment: str

@app.post("/grade_recitation")
async def grade_recitation(
    file: UploadFile = File(...), 
    target_ayah: str = Form(...) 
):
    """
    Main Endpoint: Returns Grade + Request ID for reporting.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1. Generate Request ID & Save to Cache
    request_id = str(uuid.uuid4())
    file_ext = file.filename.split('.')[-1]
    cached_path = os.path.join(TEMP_STORAGE, f"{request_id}.{file_ext}")
    
    with open(cached_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        print(f"Processing Request {request_id} ({file.filename})")
        
        # 2. Segment & Transcribe
        segments = segmenter.segment_file(cached_path)
        full_transcript = []
        for seg in segments:
            text = model.transcribe(seg['audio_data'])
            if text: full_transcript.append(text)
                
        final_text = " ".join(full_transcript)
        
        # 3. Grade
        result = grader.grade(final_text, target_ayah)
        
        response = {
            "request_id": request_id, # Key for Flywheel
            "status": "success",
            "user_recitation": final_text,
            "expected_recitation": target_ayah,
            "passed": result['passed'],
            "accuracy": result['accuracy'],
            "mistakes": result['mistakes'],
            "segments_processed": len(segments)
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"Error: {e}")
        # Clean up on error only
        if os.path.exists(cached_path): os.remove(cached_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/report_issue")
async def report_issue(report: ReportRequest):
    """
    Data Flywheel Endpoint:
    Moves the cached audio to 'Golden Negatives' for future training.
    """
    # 1. Find the file in temp storage
    # We don't know the extension, so check likely ones or glob
    import glob
    search_pattern = os.path.join(TEMP_STORAGE, f"{report.request_id}.*")
    matches = glob.glob(search_pattern)
    
    if not matches:
        raise HTTPException(status_code=404, detail="Audio file not found (expired or invalid ID)")
    
    src_file = matches[0]
    filename = os.path.basename(src_file)
    dst_file = os.path.join(GOLDEN_DATASET, filename)
    
    # 2. Move file (or Copy to be safe)
    shutil.copy2(src_file, dst_file)
    
    # 3. Save Metadata
    meta_file = os.path.join(GOLDEN_DATASET, f"{report.request_id}.json")
    metadata = {
        "request_id": report.request_id,
        "user_comment": report.user_comment,
        "audio_file": filename,
        "timestamp": "2025-12-31" # In prod use time.time()
    }
    import json
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        
    print(f"REPORTED: Saved Golden Negative {report.request_id}")
    
    return {"status": "success", "message": "Thank you! This helps improve the model."}

if __name__ == "__main__":
    # Dev Server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
