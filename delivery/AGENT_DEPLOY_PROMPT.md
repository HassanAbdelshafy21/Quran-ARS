# Deploy Prompt — give this to your DevOps/coding agent

> Copy everything below the line and give it to your agent, together with this `delivery/` folder.
> It contains everything needed to deploy and integrate the Quran recitation-grading AI service.
> No further input from the AI team is required.

---

You are a DevOps engineer. Deploy the **Quran-ARS** recitation-grading service contained in this
`delivery/` folder onto our GPU server and wire it to our backend. Work only from the files here;
do not invent endpoints or change the model.

## What the service is
A FastAPI app that scores a user's Quran recitation. It uses a **single model**,
`NAMAA-Space/Cohere-Speech-Tashkeel-2B`, which transcribes the recitation **with diacritics
(tashkeel)** acoustically, and from that it grades:
- **words** (memorization: which words were correct/wrong/skipped), and
- **harakat** (tajweed: whether the short vowels were pronounced correctly).

It exposes an **async** endpoint: you POST a job, it replies instantly with a `jobId`, processes
in the background, and POSTs the result to a **webhook** you provide.

## Hardware & software requirements (hard requirements)
- **A CUDA GPU is required** (a 2B model; CPU is not viable). **NVIDIA T4 (16 GB) / L4 / A10G** is
  plenty — the model uses ~5–6 GB VRAM in bf16.
- **~15 GB disk** (the model is ~5 GB), **8 GB+ system RAM**, **CUDA 12.x** driver on the host,
  NVIDIA Container Toolkit installed (so Docker can see the GPU: `--gpus all`).
- The provided `Dockerfile` uses CUDA 12.1, Python 3.11, `transformers>=5.4`, torch cu121, and
  **bakes the model into the image** at build time (so runtime needs no internet). **bf16 is
  required** — do not switch to fp16.

## Environment variables (only two)
- `AI_API_KEY` — a shared secret. Callers of `/api/evaluate` must send `X-AI-API-Key:
  <AI_API_KEY>` (plain value, no `Bearer` prefix — the `Authorization` header is reserved for
  RunPod Serverless's own gateway auth). Pick a strong value and give it to our app backend.
- `PUBLIC_BASE_URL` — the public URL where **this service** is reachable (e.g.
  `https://ai.example.com`). It is used to build links to generated feedback audio, so it must be
  the externally-reachable address of this service.

## Deploy (Docker — recommended)
```bash
cd delivery
# create the env file the compose file reads
printf 'AI_API_KEY=%s\nPUBLIC_BASE_URL=%s\n' "<STRONG_SECRET>" "https://ai.example.com" > .env

docker compose up -d --build      # first build is large/slow (~5 GB model bakes in)
docker compose logs -f            # wait for "NAMAA loaded." (~10–30 s after build)
curl http://localhost:8000/health # expect {"status":"ok","model_loaded":true}
```
(If you prefer plain Docker, see `DEPLOYMENT_GUIDE.md` for the `docker run --gpus all -e AI_API_KEY=... -e PUBLIC_BASE_URL=... -p 8000:8000` form, plus Nginx+TLS, systemd, and cloud-provider guides.)

Put a reverse proxy (Nginx) with TLS in front, forwarding 443 → 8000. Keep port 8000 internal.

## The API you must integrate (async)
**Request** — our app backend calls:
```http
POST /api/evaluate
X-AI-API-Key: <AI_API_KEY>
Content-Type: application/json

{
  "audioUrl": "https://<our-storage>/recording.webm",   // must be reachable from THIS server
  "surahNumber": 112, "surahName": "الإخلاص",
  "fromAyah": 1, "toAyah": 4,
  "userId": 42, "recitationId": 1337,
  "webhookUrl": "https://api.example.com/webhook/ai",    // must be reachable from THIS server
  "webhookSecret": "<shared-with-webhook>"
}
```
**Immediate response** (< 2 s): `{ "status": "processing", "jobId": "…", "estimatedTime": 30 }`.

**Then** the service POSTs to your `webhookUrl` with `Authorization: Bearer <webhookSecret>`:
```json
{
  "jobId": "…", "recitationId": 1337, "userId": 42, "status": "success",
  "data": {
    "overallScore": 87.5, "passed": true,
    "totalWords": 4, "correctWords": 4, "incorrectWords": 0,
    "userRecitation": "…",                    // bare text
    "userRecitationDiacritized": "…",         // the learner's ACTUAL recitation WITH tashkeel — show this
    "expectedRecitation": "…",
    "harakatChecked": 4,                       // # correctly-recited words checked for tajweed
    "harakatErrors": [                          // where a vowel was mispronounced (empty = clean)
      { "word": "دِينَكُمْ", "expectedWord": "دِينُكُمْ",
        "details": [ { "letter": "ن", "got": "fatha", "expected": "damma" } ] }
    ],
    "words": [ { "word": "…", "expected": "…", "isCorrect": true, "score": 1.0,
                 "errorType": null, "errorTypeAr": null, "charErrors": [],
                 "timestampStart": null, "timestampEnd": null } ],
    "errors": ["…arabic feedback…"],
    "feedbackAudio": "<PUBLIC_BASE_URL>/audio/feedback_<jobId>.mp3",
    "referenceAudio": "https://everyayah.com/...mp3",   // only when the user fails
    "modelVersion": "namaa-cohere-speech-tashkeel-2b"
  }
}
```
On failure it POSTs `{ ..., "status":"error", "message":"…" }`. Webhook delivery retries 3×
(1s→2s→4s) on 5xx/timeout only. **The `data` object contains ALL detail** (5 top-level keys), so a
strict `forbidNonWhitelisted` validator is fine.

Full contract, error codes, and acceptance checklist: **`BACKEND_HANDOFF.md`**.

## Verify it works (a test client is included)
```bash
# async — the real integration (needs the server running + a PUBLIC audio URL it can download):
python test_client.py async \
  --audio-url https://<our-storage>/clip.webm \
  --surah 112 --from-ayah 1 --to-ayah 4 \
  --api-key "$AI_API_KEY" \
  --webhook-host <host-the-server-can-reach-this-script-on>   # e.g. host.docker.internal in Docker

# sync — quick smoke test with a LOCAL file (no webhook needed):
python test_client.py sync --file <local.wav> --surah 112 --text "قُلْ هُوَ ٱللَّهُ أَحَدٌ ..."
```
The client prints the score, the diacritized recitation, and the harakat errors, and saves the raw
JSON. Success = you see a webhook payload with `"status":"success"` and a populated `data`.

## Operational notes (do these)
- **Feedback audio retention:** the service writes `feedback_*.mp3` into `temp_storage/` (served at
  `/audio/`). **Keep them ≥ 30 days.** Do NOT add a 1-hour temp-cleanup cron that deletes
  `feedback_*`.
- **Networking checklist:** (a) the server must be able to **download `audioUrl`**; (b) the server
  must be able to **reach `webhookUrl`**; (c) `PUBLIC_BASE_URL` must be where our app can fetch the
  feedback audio. Open egress accordingly.
- **Health & autostart:** wire `GET /health` into your load balancer; `restart: unless-stopped`
  (compose) or a systemd unit (see `DEPLOYMENT_GUIDE.md`) keeps it up.
- **Scaling:** one GPU handles many recitations (async). To scale, run more replicas behind the
  proxy; each needs its own GPU.

## If something fails
- `model_loaded:false` / slow first start → the model is still loading; wait, check
  `docker compose logs`. If it re-downloads at runtime, ensure the build step baked it in.
- `401` on `/api/evaluate` → `AI_API_KEY` mismatch between caller and the service env.
- Webhook never arrives → the server can't reach `webhookUrl` (firewall), or it returned non-2xx.
- CUDA/`no GPU` errors → NVIDIA Container Toolkit not installed, or you forgot `--gpus all` /
  the compose `deploy.resources` GPU reservation.
- Garbled Arabic output → you changed the model to fp16; it **must** run bf16 (the provided
  Dockerfile already does).

Deliverables: the service reachable over HTTPS, `GET /health` green, and a successful end-to-end
`test_client.py async` run producing a webhook with real `data`. Report the public URL and the
`AI_API_KEY` you set (share the key securely).
