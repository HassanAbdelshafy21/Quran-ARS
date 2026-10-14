"""Modal deployment that replaces the LIVE app in place.

Production URL:  https://yutlaquran--quran-asr-fastapi-app.modal.run
                 = workspace "yutlaquran" + app "quran-asr" + function "fastapi_app"

The app name and function name below MUST stay exactly "quran-asr" / "fastapi_app" — then
`modal deploy` creates a new version of the existing app, the URL does not change, and the
Railway backend needs no changes. (modal_app.py is a different, newer layout that deploys a
separate app called "quran-ars" with a different URL.)

Deploy (logged in to Modal as the yutlaquran workspace), from inside delivery/:
    modal deploy modal_quran_asr.py
"""
import modal

# ---- Must match the live app (Modal dashboard: Apps -> quran-asr -> fastapi_app, and Secrets) ----
SECRET_NAME = "quran-asr-secrets"   # the secret the live app uses (holds AI_API_KEY etc.)
GPU = "T4"                  # same GPU as the live fastapi_app function
# --------------------------------------------------------------------------------------------------

MODEL_ID = "NAMAA-Space/Cohere-Speech-Tashkeel-2B"


def _download_model():
    """Baked into the image at build time so containers never download at runtime."""
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration
    AutoProcessor.from_pretrained(MODEL_ID)
    CohereAsrForConditionalGeneration.from_pretrained(MODEL_ID)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch", "torchaudio",
        "transformers>=5.4.0", "accelerate>=1.0.0", "sentencepiece", "protobuf", "safetensors",
        "fastapi", "uvicorn", "python-multipart", "httpx",
        "librosa>=0.10.1", "soundfile", "moviepy", "jiwer", "edge-tts", "requests", "boto3",
    )
    .run_function(_download_model)
    .add_local_dir("core", remote_path="/app/core")
    .add_local_dir("data", remote_path="/app/data")
    .add_local_file("main.py", remote_path="/app/main.py")
)

app = modal.App("quran-asr")


@app.function(
    image=image,
    gpu=GPU,
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    timeout=900,              # long recitations (5+ min audio) + webhook delivery
    scaledown_window=300,
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def fastapi_app():
    import os, sys
    os.chdir("/app")
    sys.path.insert(0, "/app")
    from contextlib import asynccontextmanager
    import main

    # Load the model BEFORE serving, so the first request on a cold container never sees
    # "Model not loaded". main.py's own lifespan also loads it (in a background thread, for
    # RunPod) — replace it with a no-op so the model isn't loaded twice into GPU memory.
    main.load_resources()

    @asynccontextmanager
    async def _no_reload(_app):
        yield

    main.app.router.lifespan_context = _no_reload
    return main.app
