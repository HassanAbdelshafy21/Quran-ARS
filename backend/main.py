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

@app.post("/grade_recitation")
async def grade_recitation(
    file: UploadFile = File(...), 
    target_ayah: str = Form(...) 
):
    """
    Main Endpoint: 
    1. Receives Audio + Target Text (Ayah).
    2. Segments Audio (VAD).
    3. Transcribes each segment.
    4. Joins text and Grades it.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1. Save File Temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        print(f"Processing File: {file.filename}")
        
        # 2. Segment Audio (To avoid broken words)
        # For a short ayah request, we might not need heavy segmentation, 
        # but for 15m audio we do. Let's assume this endpoint handles chunks or full recitations.
        # If the file is huge, this call blocks. In prod, use Celery/Queue.
        segments = segmenter.segment_file(temp_path)
        
        full_transcript = []
        
        # 3. Transcribe Segments
        for seg in segments:
            # We already have audio data in memory if we used the segmenter correctly,
            # but our current segmenter returns data or paths. 
            # Let's use the 'audio_data' from segmenter directly if available
            audio_chunk = seg['audio_data'] 
            
            text = model.transcribe(audio_chunk)
            if text:
                full_transcript.append(text)
                
        final_text = " ".join(full_transcript)
        print(f"Transcript: {final_text}")
        
        # 4. Grade
        # Note: If valid input is 15 minutes, 'target_ayah' must be a long string of Surah.
        # Ideally user sends "Surah ID" and we fetch text. 
        # For now, we trust the Client sends the text.
        
        result = grader.grade(final_text, target_ayah)
        
        response = {
            "status": "success",
            "transcription": final_text,
            "passed": result['passed'],
            "accuracy": result['accuracy'],
            "mistakes": result['mistakes'], # List of missed/wrong words
            "segments_processed": len(segments)
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    # Dev Server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
