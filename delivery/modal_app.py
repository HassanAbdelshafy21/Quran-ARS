"""Modal deployment for Quran-ARS.

Architecture (deliberate — this is what makes Modal cheap AND fast here):

    ┌── web()  ── CPU only, tiny image, always warm ────────────────────────┐
    │   /health, /ping          -> instant                                  │
    │   /api/evaluate           -> auth, validate, SPAWN the GPU job,       │
    │                              return {"status":"processing", jobId}    │
    └───────────────────────────────┬───────────────────────────────────────┘
                                    │ .spawn()
    ┌───────────────────────────────▼───────────────────────────────────────┐
    │  Grader (GPU)  — loads NAMAA once per container (@modal.enter),       │
    │  downloads the audio, grades words + harakat, uploads the feedback    │
    │  mp3 to R2, then POSTs the webhook. Scales to zero when idle.         │
    └───────────────────────────────────────────────────────────────────────┘

Why the split: the API never waits on a GPU, so callers get their `jobId` in milliseconds even
when no GPU worker is running. The GPU is billed only while a recitation is actually being graded
— which is the whole point at ~250 evaluations/day.

Feedback audio MUST go to object storage here: Modal containers are ephemeral, so anything written
to local disk disappears. Configure the S3_* variables (see core/storage.py) in the Modal secret.

Deploy:
    pip install modal && modal token new
    modal secret create quran-ars \
        AI_API_KEY=... S3_BUCKET=... S3_ACCESS_KEY_ID=... S3_SECRET_ACCESS_KEY=... \
        S3_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com \
        S3_PUBLIC_BASE_URL=https://cdn.quranyutla.com
    modal deploy modal_app.py
"""
import modal

MODEL_ID = "NAMAA-Space/Cohere-Speech-Tashkeel-2B"
SECRET = modal.Secret.from_name("quran-ars")


def _download_model():
    """Baked into the image at build time so containers never download at runtime."""
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration
    AutoProcessor.from_pretrained(MODEL_ID)
    CohereAsrForConditionalGeneration.from_pretrained(MODEL_ID)


# Heavy image: model + torch + the app code. Used only by the GPU class.
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch", "torchaudio",
        "transformers>=5.4.0", "accelerate>=1.0.0", "sentencepiece", "protobuf", "safetensors",
        "fastapi", "uvicorn", "python-multipart", "httpx",
        "librosa>=0.10.1", "soundfile", "moviepy", "jiwer", "edge-tts", "requests", "boto3",
    )
    .run_function(_download_model)
    .add_local_dir("core", remote_path="/root/core")
    .add_local_dir("data", remote_path="/root/data")
    .add_local_file("main.py", remote_path="/root/main.py")
)

# Light image for the always-warm web layer — no model, no torch, so cold starts are ~1 s.
web_image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi[standard]")

app = modal.App("quran-ars")


@app.cls(
    image=gpu_image,
    gpu="A10G",              # ~6 GB needed; A10G/L4 both fine. bf16 required (no T4).
    secrets=[SECRET],
    scaledown_window=60,     # keep a warm container 60 s after a job (cheap, helps bursts)
    timeout=600,             # a single grading job may take up to 10 min
    max_containers=3,
)
class Grader:
    @modal.enter()
    def setup(self):
        """Runs once per container: load NAMAA + graders into memory."""
        import main
        main.load_resources()
        self.main = main
        print("Grader container ready.")

    @modal.method()
    def grade(self, job_id: str, payload: dict):
        """Full pipeline for one recitation, then POST the webhook. Reuses main.py."""
        import asyncio
        req = self.main.EvaluateRequest(**payload)
        asyncio.run(self.main._process_and_callback(job_id, req))
        return {"jobId": job_id, "done": True}


@app.function(image=web_image, secrets=[SECRET], scaledown_window=300, min_containers=1)
@modal.asgi_app()
def web():
    """Public HTTP API. CPU-only and kept warm, so it always answers immediately."""
    import hmac
    import os
    import uuid

    from fastapi import FastAPI, Header, Response
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    AI_API_KEY = os.getenv("AI_API_KEY", "")
    api = FastAPI(title="Quran ASR API (Modal)", version="2.0.0")

    class EvaluateRequest(BaseModel):
        audioUrl: str
        surahNumber: int
        surahName: str | None = None
        fromAyah: int
        toAyah: int
        userId: int | None = None
        recitationId: int | None = None
        webhookUrl: str
        webhookSecret: str

    def _valid(key: str | None) -> bool:
        # Constant-time; an unset AI_API_KEY never authenticates.
        return bool(AI_API_KEY and key and hmac.compare_digest(key, AI_API_KEY))

    @api.get("/health")
    def health():
        # The web layer is stateless; model readiness lives in the GPU containers.
        return {"status": "ok", "service": "web", "version": "2.0.0"}

    @api.get("/ping")
    def ping():
        return Response(status_code=200)

    @api.post("/api/evaluate")
    def evaluate(req: EvaluateRequest, x_ai_api_key: str = Header(None, alias="X-AI-API-Key")):
        if not _valid(x_ai_api_key):
            return JSONResponse(
                status_code=401,
                content={"status": "error", "message": "Invalid API key", "code": "AUTH_FAILED"},
            )
        job_id = str(uuid.uuid4())
        # Hand the job to a GPU container and return immediately. .spawn() is fire-and-forget:
        # the job runs to completion independently of this HTTP request, which is exactly what
        # our async contract needs (no risk of the work dying when the response is sent).
        Grader().grade.spawn(job_id, req.model_dump())
        return {"status": "processing", "jobId": job_id, "estimatedTime": 30}

    return api
