"""
Quran ASR Kids API — Self-Contained Delivery Server
====================================================
Run: python main.py
Docs: http://localhost:8000/docs
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import os
import sys
import uuid
import glob
import json
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

# Add current dir to path to find 'core'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.grader import QuranGrader
from core.segmenter import AudioSegmenter
from core.model_loader import QuranModel
from core.tts import generate_feedback_audio

# Initialize App
app = FastAPI(
    title="Quran ASR Kids API",
    version="2.0.0",
    description="AI system to listen, transcribe, and grade children's Quran recitation."
)

# --- Configuration ---
# Model checkpoint path (relative to this file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CHECKPOINT = os.path.join(BASE_DIR, "model", "checkpoint-30000")

# Global State
model = None
grader = None
segmenter = None

# Directories
TEMP_STORAGE = os.path.join(BASE_DIR, "temp_storage")
os.makedirs(TEMP_STORAGE, exist_ok=True)

# Mount Static Files for Audio Feedback
app.mount("/audio", StaticFiles(directory=TEMP_STORAGE), name="audio")

@app.on_event("startup")
def load_resources():
    global model, grader, segmenter
    print("=" * 50)
    print("Initializing Quran ASR Backend...")
    print(f"Model Path: {MODEL_CHECKPOINT}")
    print("=" * 50)
    try:
        model = QuranModel(MODEL_CHECKPOINT)
    except Exception as e:
        print(f"⚠️  CRITICAL: Failed to load model: {e}")
        print("   The API will return 503 for /grade_recitation")
    grader = QuranGrader()
    segmenter = AudioSegmenter()
    print("Backend Ready! 🚀")
    print(f"API Docs: http://localhost:8000/docs")
    print("=" * 50)

class ReportRequest(BaseModel):
    request_id: str
    user_comment: str

@app.post("/grade_recitation", 
          summary="Grade a child's Quran recitation",
          description="Upload audio file + expected text → get transcription, grading, per-word analysis, timestamps, and feedback audio.")
async def grade_recitation(
    request: Request,
    file: UploadFile = File(..., description="Audio file (MP3, WAV, OGG, M4A)"),
    target_ayah: str = Form(..., description="Expected Quranic text"),
    surah_num: int = Form(default=None, description="Surah number (1-114) — enables Sheikh reference audio"),
    ayah_num: int = Form(default=None, description="Specific Ayah number — uses single ayah reference")
):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    # 1. Generate Request ID & Save to Cache
    request_id = str(uuid.uuid4())
    file_ext = file.filename.split('.')[-1]
    cached_path = os.path.join(TEMP_STORAGE, f"{request_id}.{file_ext}")
    
    with open(cached_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        print(f"Processing Request {request_id} ({file.filename})")
        
        # 2. Segment & Transcribe (with timestamps)
        segments = segmenter.segment_file(cached_path)
        full_transcript = []
        all_word_timestamps = []
        segment_time_offset = 0.0
        
        for seg in segments:
            result = model.transcribe(seg['audio_data'])
            text = result['text']
            word_ts = result.get('words', [])
            
            if text:
                full_transcript.append(text)
                
                # Adjust timestamps with segment offset
                for wt in word_ts:
                    adjusted = {
                        "word": wt["word"],
                        "start": round(wt["start"] + segment_time_offset, 2) if wt["start"] is not None else None,
                        "end": round(wt["end"] + segment_time_offset, 2) if wt["end"] is not None else None,
                    }
                    all_word_timestamps.append(adjusted)
                
            segment_time_offset += seg.get('duration', 0)
                
        final_text = " ".join(full_transcript)
        
        # 3. Grade (with per-word analysis and character-level errors)
        grade_result = grader.grade(final_text, target_ayah)
        
        # 4. Merge timestamps into word details
        words_detail = grade_result.get('words', [])
        
        ts_index = 0
        for wd in words_detail:
            if wd['word'] is not None and ts_index < len(all_word_timestamps):
                wd['timestamp_start'] = all_word_timestamps[ts_index].get('start')
                wd['timestamp_end'] = all_word_timestamps[ts_index].get('end')
                ts_index += 1
            else:
                wd['timestamp_start'] = None
                wd['timestamp_end'] = None
        
        # 5. Generate TTS Feedback
        feedback_filename = f"feedback_{request_id}.mp3"
        feedback_path = os.path.join(TEMP_STORAGE, feedback_filename)
        mistakes = grade_result['mistakes'] if grade_result['mistakes'] else []
        
        await generate_feedback_audio(mistakes, feedback_path)
        
        base_url = str(request.base_url).rstrip("/")
        feedback_url = f"{base_url}/audio/{feedback_filename}"
        
        # 6. Generate Reference Audio URL (Sheikh Minshawi)
        ref_url = None
        if surah_num:
            s_str = f"{surah_num:03d}"
            if grade_result['passed']:
                ref_url = None
            else:
                if ayah_num:
                    a_str = f"{ayah_num:03d}"
                    ref_url = f"https://everyayah.com/data/Minshawy_Mujawwad_192kbps/{s_str}{a_str}.mp3"
                else:
                    ref_url = f"https://server10.mp3quran.net/minsh/{s_str}.mp3"
        
        # 7. Build Response
        response = {
            "request_id": request_id,
            "status": "success",
            "user_recitation": final_text,
            "expected_recitation": target_ayah,
            "passed": grade_result['passed'],
            "accuracy": grade_result['accuracy'],
            "raw_score": grade_result['raw_score'],
            "mistakes": grade_result['mistakes'],
            "words": words_detail,
            "feedback_audio": feedback_url,
            "reference_audio": ref_url,
            "segments_processed": len(segments)
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(cached_path): os.remove(cached_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/report_issue",
          summary="Report a grading issue",
          description="Save problematic audio for future model retraining.")
async def report_issue(report: ReportRequest):
    golden_dir = os.path.join(BASE_DIR, "data", "golden_negatives")
    os.makedirs(golden_dir, exist_ok=True)
    
    search_pattern = os.path.join(TEMP_STORAGE, f"{report.request_id}.*")
    matches = glob.glob(search_pattern)
    
    if not matches:
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    src_file = None
    for m in matches:
        if "feedback_" not in os.path.basename(m):
            src_file = m
            break
            
    if not src_file:
        raise HTTPException(status_code=404, detail="Original audio file not found")

    filename = os.path.basename(src_file)
    dst_file = os.path.join(golden_dir, filename)
    shutil.copy2(src_file, dst_file)
    
    meta_file = os.path.join(golden_dir, f"{report.request_id}.json")
    metadata = {
        "request_id": report.request_id,
        "user_comment": report.user_comment,
        "audio_file": filename
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        
    return {"status": "success", "message": "Reported."}

@app.get("/health", summary="Health check")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "2.0.0"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
