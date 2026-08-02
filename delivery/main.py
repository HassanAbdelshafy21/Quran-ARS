"""
Quran ASR Kids API — Self-Contained Delivery Server
====================================================
Run: python main.py
Docs: http://localhost:8000/docs
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, BackgroundTasks, Header
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import os
import sys
import uuid
import glob
import json
import asyncio
import tempfile
import logging
import hmac
import httpx
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
from fastapi.staticfiles import StaticFiles

# Add current dir to path to find 'core'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.grader import QuranGrader
from core.segmenter import AudioSegmenter
from core.namaa_model import NamaaModel
from core.harakat_grader import grade_harakat
from core.tts import generate_feedback_audio
from core.quran_db import QuranDB

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("quran-asr")

# Initialize App
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_resources()   # runs once at startup (defined below)
    yield


app = FastAPI(
    title="Quran ASR Kids API",
    version="2.0.0",
    description="AI system to listen, transcribe, and grade children's Quran recitation.",
    lifespan=lifespan,
)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Env config for the async integration (see AI-Integration-Spec-AR.md)
AI_API_KEY = os.getenv("AI_API_KEY", "")


def _valid_api_key(api_key: str | None) -> bool:
    """Constant-time check of the shared secret. Empty AI_API_KEY never authenticates.

    Sent as X-AI-API-Key (not Authorization) because RunPod Serverless's own gateway
    auth occupies the Authorization header on the load-balancer endpoint.
    """
    if not AI_API_KEY or not api_key:
        return False
    return hmac.compare_digest(api_key, AI_API_KEY)


# Public, absolute base URL for audio links. localhost is useless to a phone. (§6.4)
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")
MODEL_VERSION = "namaa-cohere-speech-tashkeel-2b"

# Global State
model = None
grader = None
segmenter = None
quran_db = None

# Directories.
# TEMP_STORAGE_DIR must point at PERSISTENT storage: feedback_*.mp3 is served at /audio/ and the
# backend references those URLs for 30 days. On RunPod Serverless the container disk is ephemeral
# (wiped on scale-to-zero) and not shared between workers, so set this to the network volume:
#   TEMP_STORAGE_DIR=/runpod-volume/temp_storage
TEMP_STORAGE = os.getenv("TEMP_STORAGE_DIR") or os.path.join(BASE_DIR, "temp_storage")
os.makedirs(TEMP_STORAGE, exist_ok=True)

# Mount Static Files for Audio Feedback
app.mount("/audio", StaticFiles(directory=TEMP_STORAGE), name="audio")


def load_resources():
    global model, grader, segmenter, quran_db
    print("=" * 50)
    print("Initializing Quran ASR Backend (NAMAA)...")
    print("=" * 50)
    try:
        model = NamaaModel()
    except Exception as e:
        print(f"⚠️  CRITICAL: Failed to load model: {e}")
        print("   The API will return 503 for /grade_recitation")
    grader = QuranGrader()
    segmenter = AudioSegmenter()
    try:
        quran_db = QuranDB()
    except Exception as e:
        print(f"⚠️  Quran DB not loaded: {e} (/api/evaluate ayah lookup will fail)")
    print("Backend Ready! 🚀")
    print(f"API Docs: http://localhost:8000/docs")
    print("=" * 50)


# =====================================================================
# Shared grading pipeline — used by both /grade_recitation and /api/evaluate
# =====================================================================
async def run_grading(audio_path, target_ayah, surah_num=None, ayah_num=None, uid=None):
    """Segment -> transcribe -> grade -> TTS -> reference. Returns a result dict.

    Does not build public URLs (callers do, since the base URL differs) and does
    not delete audio_path (callers own the lifecycle).
    """
    if not model:
        raise RuntimeError("Model not loaded")
    uid = uid or str(uuid.uuid4())

    # 1. Segment & Transcribe (with timestamps)
    segments = segmenter.segment_file(audio_path)
    full_transcript = []
    all_word_timestamps = []
    segment_time_offset = 0.0

    for seg in segments:
        result = model.transcribe(seg['audio_data'])
        text = result['text']
        word_ts = result.get('words', [])
        if text:
            full_transcript.append(text)
            for wt in word_ts:
                all_word_timestamps.append({
                    "word": wt["word"],
                    "start": round(wt["start"] + segment_time_offset, 2) if wt["start"] is not None else None,
                    "end": round(wt["end"] + segment_time_offset, 2) if wt["end"] is not None else None,
                })
        segment_time_offset += seg.get('duration', 0)

    final_text = " ".join(full_transcript)

    # 2. Grade (per-word analysis + character-level errors)
    grade_result = grader.grade(final_text, target_ayah)
    words_detail = grade_result.get('words', [])

    # 2b. Harakat/tajweed grading (NAMAA's diacritics are acoustic, so this reflects
    # what the learner actually pronounced). Only correctly-recited words are checked;
    # tajweed tolerances (waqf, shadda, implicit-sukun) keep false-rejections low. Non-fatal.
    try:
        harakat = grade_harakat(final_text, target_ayah)
    except Exception as e:
        print(f"Harakat grading failed (non-critical): {e}")
        harakat = {"words": [], "harakat_errors": [], "checked": 0, "wrong": 0}

    # 3. Merge timestamps into word details
    ts_index = 0
    for wd in words_detail:
        if wd['word'] is not None and ts_index < len(all_word_timestamps):
            wd['timestamp_start'] = all_word_timestamps[ts_index].get('start')
            wd['timestamp_end'] = all_word_timestamps[ts_index].get('end')
            ts_index += 1
        else:
            wd['timestamp_start'] = None
            wd['timestamp_end'] = None

    # 4. Generate TTS Feedback (filename keyed on uid; caller builds the URL)
    feedback_filename = f"feedback_{uid}.mp3"
    feedback_path = os.path.join(TEMP_STORAGE, feedback_filename)
    mistakes = grade_result['mistakes'] if grade_result['mistakes'] else []
    await generate_feedback_audio(mistakes, feedback_path)

    # 5. Reference audio (Sheikh Minshawi) — only when the child fails
    ref_url = None
    if surah_num and not grade_result['passed']:
        s_str = f"{surah_num:03d}"
        if ayah_num:
            a_str = f"{ayah_num:03d}"
            ref_url = f"https://everyayah.com/data/Minshawy_Mujawwad_192kbps/{s_str}{a_str}.mp3"
        else:
            ref_url = f"https://server10.mp3quran.net/minsh/{s_str}.mp3"

    return {
        "uid": uid,
        "user_recitation": final_text,
        "user_recitation_diacritized": final_text,   # NAMAA output is the actual diacritized recitation
        "harakat_checked": harakat["checked"],
        "harakat_errors": harakat["harakat_errors"],
        "expected_recitation": target_ayah,
        "passed": grade_result['passed'],
        "accuracy": grade_result['accuracy'],
        "raw_score": grade_result['raw_score'],
        "mistakes": grade_result['mistakes'],
        "words": words_detail,
        "feedback_filename": feedback_filename,
        "reference_audio": ref_url,
        "segments_processed": len(segments),
    }


class ReportRequest(BaseModel):
    request_id: str
    user_comment: str


# =====================================================================
# Existing sync endpoint (unchanged behavior) — for demo / testing
# =====================================================================
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

    request_id = str(uuid.uuid4())
    file_ext = file.filename.split('.')[-1]
    cached_path = os.path.join(TEMP_STORAGE, f"{request_id}.{file_ext}")

    with open(cached_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        print(f"Processing Request {request_id} ({file.filename})")
        r = await run_grading(cached_path, target_ayah, surah_num, ayah_num, uid=request_id)

        base_url = str(request.base_url).rstrip("/")
        response = {
            "request_id": r["uid"],
            "status": "success",
            "user_recitation": r["user_recitation"],
            "user_recitation_diacritized": r["user_recitation_diacritized"],
            "expected_recitation": r["expected_recitation"],
            "passed": r["passed"],
            "accuracy": r["accuracy"],
            "raw_score": r["raw_score"],
            "mistakes": r["mistakes"],
            "harakat_checked": r["harakat_checked"],
            "harakat_errors": r["harakat_errors"],
            "words": r["words"],
            "feedback_audio": f"{base_url}/audio/{r['feedback_filename']}",
            "reference_audio": r["reference_audio"],
            "segments_processed": r["segments_processed"],
        }
        return JSONResponse(content=response)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(cached_path):
            os.remove(cached_path)
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# New async endpoint for the "Quran Yutla" backend integration
# =====================================================================
class EvaluateRequest(BaseModel):
    audioUrl: str
    surahNumber: int
    surahName: Optional[str] = None
    fromAyah: int
    toAyah: int
    userId: int
    recitationId: int
    webhookUrl: str
    webhookSecret: str


@app.post("/api/evaluate",
          summary="Async recitation evaluation (returns a jobId immediately)",
          description="Accepts a job, replies instantly with a jobId, processes in the background, and POSTs the result to the caller's webhook.")
async def evaluate_async(
    req: EvaluateRequest,
    background_tasks: BackgroundTasks,
    x_ai_api_key: str = Header(None, alias="X-AI-API-Key"),
):
    # Auth (§3.3)
    if not _valid_api_key(x_ai_api_key):
        return JSONResponse(
            status_code=401,
            content={"status": "error", "message": "Invalid API key", "code": "AUTH_FAILED"},
        )

    job_id = str(uuid.uuid4())
    background_tasks.add_task(_process_and_callback, job_id, req)
    # Returns immediately — must be < 2s (§3.2)
    return {"status": "processing", "jobId": job_id, "estimatedTime": 30}


async def _process_and_callback(job_id: str, req: EvaluateRequest):
    audio_path = None
    try:
        # 1. Download the audio from cloud storage
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as c:
            resp = await c.get(req.audioUrl)
            resp.raise_for_status()

        suffix = os.path.splitext(req.audioUrl.split("?")[0])[1] or ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=TEMP_STORAGE) as f:
            f.write(resp.content)
            audio_path = f.name

        # 2. Resolve the target text for the ayah range (§6.2 + §6.3)
        if not quran_db:
            raise RuntimeError("Quran DB not loaded")
        target_text = quran_db.get_ayah_range(req.surahNumber, req.fromAyah, req.toAyah)

        # 3. Grade (same pipeline as /grade_recitation)
        r = await run_grading(
            audio_path, target_text,
            surah_num=req.surahNumber, ayah_num=req.fromAyah, uid=job_id,
        )

        words = r["words"]
        payload = {
            "jobId": job_id,
            "recitationId": req.recitationId,
            "userId": req.userId,
            "status": "success",
            "data": {
                # 🔴 Critical scale conversion 0.0–1.0 -> 0–100 (§6.1)
                "overallScore": round(r["accuracy"] * 100, 2),
                "passed": r["passed"],
                "totalWords": len(words),
                "correctWords": sum(1 for w in words if w["is_correct"]),
                "incorrectWords": sum(1 for w in words if not w["is_correct"]),
                "userRecitation": r["user_recitation"],
                # The learner's ACTUAL diacritized recitation (NAMAA acoustic output) — show this
                "userRecitationDiacritized": r["user_recitation_diacritized"],
                "expectedRecitation": r["expected_recitation"],
                # Tajweed/harakat feedback: for correctly-recited words, where the vowel differed
                "harakatChecked": r["harakat_checked"],
                "harakatErrors": [
                    {
                        "word": e["word"],
                        "expectedWord": e["expected_word"],
                        "details": [
                            {"letter": d["letter"], "got": d["got"], "expected": d["expected"]}
                            for d in e["details"]
                        ],
                    }
                    for e in r["harakat_errors"]
                ],
                "words": [
                    {
                        "word": w["word"],
                        "expected": w["expected"],
                        "isCorrect": w["is_correct"],
                        "score": w["score"],
                        "errorType": w["error_type"],
                        "errorTypeAr": w["error_type_ar"],
                        "charErrors": [
                            {
                                "type": ce["type"],
                                "typeEn": ce["type_en"],
                                "position": ce["position"],
                                "got": ce.get("got"),
                                "expected": ce.get("expected"),
                            }
                            for ce in w.get("char_errors", [])
                        ],
                        "timestampStart": w["timestamp_start"],
                        "timestampEnd": w["timestamp_end"],
                    }
                    for w in words
                ],
                "errors": r["mistakes"],
                "errorSummary": {
                    "substitution": sum(1 for w in words if w["error_type"] == "substitution"),
                    "deletion": sum(1 for w in words if w["error_type"] == "deletion"),
                    "insertion": sum(1 for w in words if w["error_type"] == "insertion"),
                },
                "feedbackAudio": f"{PUBLIC_BASE_URL}/audio/{r['feedback_filename']}",
                "referenceAudio": r["reference_audio"],
                "segmentsProcessed": r["segments_processed"],
                "requestId": job_id,
                "modelVersion": MODEL_VERSION,
            },
        }

    except Exception as e:
        log.exception("evaluation failed for job %s", job_id)
        payload = {
            "jobId": job_id,
            "recitationId": req.recitationId,
            "userId": req.userId,
            "status": "error",
            "message": str(e)[:500],
        }
    finally:
        # Delete only the downloaded source audio. The feedback_*.mp3 must persist
        # (served at /audio, referenced for 30 days by the backend — §6.4).
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)

    await _send_webhook(req.webhookUrl, req.webhookSecret, payload)


async def _send_webhook(url: str, secret: str, payload: dict, attempts: int = 3):
    """POST result to the backend webhook. Retry 3x on 5xx/timeout only (§4.5)."""
    headers = {
        "Authorization": f"Bearer {secret}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=30) as c:
        for i in range(attempts):
            try:
                resp = await c.post(url, json=payload, headers=headers)
                if resp.status_code < 400:
                    log.info("webhook delivered: %s", payload["jobId"])
                    return
                if 400 <= resp.status_code < 500:
                    # Bad request/auth — retrying won't help (§4.4)
                    log.error("webhook rejected %s: %s", resp.status_code, resp.text[:300])
                    return
                log.warning("webhook attempt %d got %s", i + 1, resp.status_code)
            except Exception as e:
                log.warning("webhook attempt %d failed: %s", i + 1, e)
            if i < attempts - 1:
                await asyncio.sleep(2 ** i)  # 1s, 2s, 4s
    log.error("webhook FAILED after %d attempts: %s", attempts, payload["jobId"])


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
        "status": "ok",
        "model_loaded": model is not None,
        "version": "2.0.0"
    }


@app.get("/ping", summary="RunPod Serverless worker health check")
async def ping():
    return {"status": "healthy" if model is not None else "initializing"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # RunPod Serverless (load balancer) injects PORT
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
