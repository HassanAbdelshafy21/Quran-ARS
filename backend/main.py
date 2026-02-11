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
from core.tts import generate_feedback_audio # Import async function

# Initialize App
app = FastAPI(title="Quran ASR Kids API", version="2.0.0")

# Global State
MODEL_CHECKPOINT = "finetuning/checkpoints_v5/checkpoint-30000"
model = None
grader = None
segmenter = None

# Directories
TEMP_STORAGE = "temp_storage"
GOLDEN_DATASET = "data/golden_negatives"
os.makedirs(TEMP_STORAGE, exist_ok=True)
os.makedirs(GOLDEN_DATASET, exist_ok=True)

# Mount Static Files for Audio Feedback
app.mount("/audio", StaticFiles(directory=TEMP_STORAGE), name="audio")

@app.on_event("startup")
def load_resources():
    global model, grader, segmenter
    print("Initializing Backend Resources...")
    try:
        model = QuranModel(MODEL_CHECKPOINT)
    except Exception as e:
        print(f"CRITICAL: Failed to load model from {MODEL_CHECKPOINT}. {e}")
    grader = QuranGrader()
    segmenter = AudioSegmenter()
    print("Backend Ready! 🚀")

class ReportRequest(BaseModel):
    request_id: str
    user_comment: str

@app.post("/grade_recitation")
async def grade_recitation(
    request: Request,
    file: UploadFile = File(...), 
    target_ayah: str = Form(...),
    surah_num: int = Form(default=None),
    ayah_num: int = Form(default=None)
):
    """
    Main Endpoint: Returns Grade + Per-Word Analysis + Timestamps + Feedback Audio.
    
    Request (Multipart Form):
      - file: Audio file (MP3, WAV, OGG, M4A)
      - target_ayah: Expected Quranic text (or Ayah ID)
      - surah_num: Surah number (1-114)
      - ayah_num: Specific Ayah number (optional)
    
    Response includes:
      - Overall accuracy score
      - Per-word analysis with error types and character-level errors
      - Word-level timestamps
      - TTS feedback audio URL
      - Sheikh reference audio URL (when failed)
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
        
        # Map timestamps to graded words (best-effort matching)
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
        
        # Generate Audio (Async)
        await generate_feedback_audio(mistakes, feedback_path)
        
        # Use absolute URL for client convenience
        base_url = str(request.base_url).rstrip("/")
        feedback_url = f"{base_url}/audio/{feedback_filename}"
        
        # 6. Generate Reference Audio URL (Minshawi)
        ref_url = None
        if surah_num:
            s_str = f"{surah_num:03d}"
            
            # Conditional Reference Audio:
            # If passed, we don't need to send the Sheikh audio (as per user rule).
            if grade_result['passed']:
                ref_url = None
            else:
                 if ayah_num:
                    # Single Ayah (EveryAyah)
                    a_str = f"{ayah_num:03d}"
                    ref_url = f"https://everyayah.com/data/Minshawy_Mujawwad_192kbps/{s_str}{a_str}.mp3"
                 else:
                    # Full Surah (MP3Quran)
                    ref_url = f"https://server10.mp3quran.net/minsh/{s_str}.mp3"
        
        # 7. Build Enhanced Response
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
            "reference_audio": ref_url, # Will be None if passed
            "segments_processed": len(segments)
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"Error: {e}")
        if os.path.exists(cached_path): os.remove(cached_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/report_issue")
async def report_issue(report: ReportRequest):
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
    dst_file = os.path.join(GOLDEN_DATASET, filename)
    shutil.copy2(src_file, dst_file)
    
    meta_file = os.path.join(GOLDEN_DATASET, f"{report.request_id}.json")
    metadata = {
        "request_id": report.request_id,
        "user_comment": report.user_comment,
        "audio_file": filename,
        "timestamp": "2025-12-31"
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        
    print(f"REPORTED: Saved Golden Negative {report.request_id}")
    return {"status": "success", "message": "Reported."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
