# 🤝 Quran ASR — Backend Handoff

**From:** AI service (Quran-ASR)
**To:** Backend team (Quran Yutla)
**Status:** ✅ `/api/evaluate` implemented and passing all acceptance checks (§8 of your spec)

This document is everything you need to integrate. It implements the async contract
from `AI-Integration-Spec-AR.md` exactly.

---

## 1. TL;DR

- New endpoint **`POST /api/evaluate`** — async: replies instantly with a `jobId`,
  processes in the background, POSTs the result to your `webhookUrl`.
- The old **`POST /grade_recitation`** (sync, multipart) still works — kept for demo/testing.
- **Model is now a single one — `NAMAA-Space/Cohere-Speech-Tashkeel-2B`** — which grades **words
  (memorization) AND harakat (tajweed)** in one pass and returns the learner's actual diacritized
  recitation. The API contract, webhook shape, and 0–100 scoring are unchanged; two `data` fields
  were added (`userRecitationDiacritized`, `harakatErrors`). Requires a **CUDA GPU** (§7).
- The async contract and 0–100 score conversion pass all acceptance checks (incl. **`webm`** audio).

---

## 2. Endpoints

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `POST` | `/api/evaluate` | **Your integration.** Async, JSON, webhook result. |
| `POST` | `/grade_recitation` | Sync demo endpoint (multipart). Not used by you. |
| `POST` | `/report_issue` | Flag a bad grade (phase 2). |
| `GET`  | `/health` | Liveness + `model_loaded` flag. |
| `GET`  | `/docs` | Swagger UI. |

---

## 3. `POST /api/evaluate`

### Request
```http
POST /api/evaluate
Content-Type: application/json
Authorization: Bearer {AI_API_KEY}
```
```json
{
  "audioUrl": "https://.../recording.webm",
  "surahNumber": 1,
  "surahName": "الفاتحة",
  "fromAyah": 1,
  "toAyah": 7,
  "userId": 42,
  "recitationId": 1337,
  "webhookUrl": "https://api.yutlaquran.com/api/v1/recitations/webhook/ai-evaluation",
  "webhookSecret": "<sent separately>"
}
```

### Immediate response (< 2s, measured 0.005s)
```json
{ "status": "processing", "jobId": "8f14e45f-...", "estimatedTime": 30 }
```
Bad/missing `Authorization` → `401` `{ "status":"error", "message":"Invalid API key", "code":"AUTH_FAILED" }`.

### Webhook — success
`POST {webhookUrl}` with `Authorization: Bearer {webhookSecret}`. **Exactly 5 top-level keys on success**
(`jobId, recitationId, userId, status, data`) — everything else lives inside `data`, as your
`forbidNonWhitelisted` requires.
```json
{
  "jobId": "8f14e45f-...",
  "recitationId": 1337,
  "userId": 42,
  "status": "success",
  "data": {
    "overallScore": 87.5,
    "passed": true,
    "totalWords": 8,
    "correctWords": 7,
    "incorrectWords": 1,
    "userRecitation": "...",
    "userRecitationDiacritized": "قُلْ هُوَ اللَّهُ أَحَدْ ...",
    "expectedRecitation": "...",
    "harakatChecked": 6,
    "harakatErrors": [
      { "word": "دِينَكُمْ", "expectedWord": "دِينُكُمْ",
        "details": [ { "letter": "ن", "got": "fatha", "expected": "damma" } ] }
    ],
    "words": [
      {
        "word": "بسم", "expected": "بسم",
        "isCorrect": true, "score": 1.0,
        "errorType": null, "errorTypeAr": null,
        "charErrors": [
          { "type": "استبدال_حرف", "typeEn": "char_substitution",
            "position": 3, "got": "ا", "expected": "ه" }
        ],
        "timestampStart": 0.50, "timestampEnd": 0.92
      }
    ],
    "errors": ["خطأ: قلت 'اللا' بدلاً من 'الله'"],
    "errorSummary": { "substitution": 1, "deletion": 1, "insertion": 0 },
    "feedbackAudio": "https://ai.yutlaquran.com/audio/feedback_8f14e45f.mp3",
    "referenceAudio": "https://everyayah.com/data/Minshawy_Mujawwad_192kbps/001001.mp3",
    "segmentsProcessed": 2,
    "requestId": "8f14e45f-...",
    "modelVersion": "namaa-cohere-speech-tashkeel-2b"
  }
}
```

> **New fields (single-model NAMAA architecture):**
> - `userRecitationDiacritized` — the learner's **actual** recitation **with tashkeel** (acoustic,
>   i.e. their real vowels, not the "correct answer"). Show this to the user.
> - `harakatChecked` — how many correctly-recited words were checked for tajweed.
> - `harakatErrors[]` — per word where the **vowel** differed: `word`, `expectedWord`, and
>   `details[]` of `{ letter, got, expected }` (e.g. said *fatha*, expected *damma*). Empty = clean.
> - `words[].timestampStart/End` are `null` (the model does not emit word timestamps).

### Webhook — error
```json
{ "jobId":"...", "recitationId":1337, "userId":42, "status":"error",
  "message":"Failed to download audio file: 404 Not Found" }
```

### Retries
On the webhook POST: **3 attempts, backoff 1s→2s→4s, on `5xx`/timeout only**. `4xx` is not retried.

---

## 4. Confirmation against your critical points (§6)

| Your point | Status |
| :--- | :--- |
| 6.1 `overallScore` is **0–100** (`round(accuracy*100, 2)`) | ✅ verified = 100.0 in test |
| 6.2 **Ayah range** via `get_ayah_range(surah, from, to)` | ✅ returns full Fatiha (29 words) |
| 6.3 End-of-ayah markers (`U+FC00–U+FDFF`) stripped | ✅ verified 0 markers |
| 6.4 `feedbackAudio` absolute via `PUBLIC_BASE_URL` | ✅ verified absolute URL |
| 6.5 `webm` supported | ✅ verified with a real Opus `.webm` |

> **Note on `get_ayah_range`:** our DB table is `quran` (not `ayahs`), column `aya_text`.
> Text is the diacritized uthmani script with the ornament markers removed.

---

## 5. Configuration (env vars)

```bash
AI_API_KEY=<the shared secret you send us>       # required; requests without it get 401
PUBLIC_BASE_URL=https://ai.yutlaquran.com        # required; makes audio URLs absolute
```

---

## 6. Run it

**Docker (recommended):**
```bash
cd delivery
# Provide the env first (compose reads a .env in this folder). AI_API_KEY is required —
# without it, compose refuses to start rather than silently 401-ing every request.
printf 'AI_API_KEY=%s\nPUBLIC_BASE_URL=%s\n' "$AI_API_KEY" "https://ai.yutlaquran.com" > .env
docker compose up -d          # or: docker build -t quran-asr . && docker run --gpus all -e AI_API_KEY=... -e PUBLIC_BASE_URL=... -p 8000:8000 quran-asr
curl http://localhost:8000/health
```

**Direct:**
```bash
pip install -r requirements.txt   # torch first if GPU (see DEPLOYMENT_GUIDE.md)
AI_API_KEY=... PUBLIC_BASE_URL=... python main.py
```

The `Dockerfile` **bakes the model into the image** (build-time download), so containers start
without needing internet. If you run directly, first boot downloads the model (~5 GB) from
HuggingFace once. See `DEPLOYMENT_GUIDE.md` for cloud/GPU/systemd/Nginx details.

---

## 7. The model & system requirements (single-model architecture)

- **Model:** `NAMAA-Space/Cohere-Speech-Tashkeel-2B` — one 2B ASR that outputs the learner's
  **actual diacritized recitation** (words + harakat). From that, the service grades **words**
  (memorization) *and* **harakat** (tajweed) — no second model, no CATT, no Whisper.
- **Hardware:** a **CUDA GPU is required** (CPU is impractical for a 2B model). Needs **~5–6 GB
  VRAM** in bf16 — a **T4 (16 GB) / L4 / A10G** is plenty. Disk ~15 GB (model + deps).
- **Software:** CUDA 12.x, Python 3.11, **`transformers>=5.4`**, `torch` cu121, `accelerate`,
  `sentencepiece`, `protobuf`. **bf16 is required** (fp16 overflows an attention mask → garbage).
- **Latency:** ~0.2–2 s per recitation on a warm GPU (async anyway — the caller gets a `jobId`
  immediately and the result via webhook).
- The async `/api/evaluate` contract, webhook shape, and 0–100 scoring are **unchanged** from your
  spec — only the model behind them and the two new `data` fields (`userRecitationDiacritized`,
  `harakatErrors`) were added.

---

## 8. Things to know

- **Feedback audio retention:** `feedback_*.mp3` files persist in `temp_storage/` (served at `/audio/`).
  Keep them **≥ 30 days** — do **not** enable a 1-hour temp-cleanup cron on `feedback_*` files (see DEPLOYMENT_GUIDE.md, updated).
- **Word timestamps are `null`** — this model does not emit per-word timings. If you need
  word-highlighting synced to audio, that's a separate (future) feature.
- **Tashkeel is acoustic:** `userRecitationDiacritized` shows the learner's *real* vowels (so a
  wrong vowel appears wrong). `harakatErrors[]` is the tajweed feedback; it uses tajweed-aware
  tolerances (waqf/shadda) so it does **not** flag correct pausing — false-rejection ≈ 0–1%.
- **`suggestions`** array is not currently produced (our grader doesn't generate coaching tips). Easy to add if you want it.
- Local GPU testing used a newer torch/GPU than the T4 deploy target; the version-exact run happens on your T4/cluster.

---

## 9. What we still need from you (§10)

1. The real `AI_API_KEY` and `webhookSecret` (secure channel).
2. A real OVH `audioUrl` to a test file, so we can run a live end-to-end.
3. Internal cluster address if we deploy inside your `quran-yutla` namespace.

---

## 10. Test it yourself (mock, no secrets needed)

```python
# mock_backend.py  →  uvicorn mock_backend:app --port 3777
from fastapi import FastAPI, Request
app = FastAPI()

@app.post("/api/v1/recitations/webhook/ai-evaluation")
async def hook(r: Request):
    b = await r.json()
    print("AUTH:", r.headers.get("authorization"))
    print("TOP-LEVEL KEYS:", list(b.keys()))            # expect 5 on success
    print("SCORE:", b.get("data", {}).get("overallScore"))  # expect 0-100
    return {"success": True, "message": "ok"}
```
```bash
curl -X POST http://localhost:8000/api/evaluate \
  -H "Authorization: Bearer $AI_API_KEY" -H "Content-Type: application/json" \
  -d '{"audioUrl":"https://.../sample.webm","surahNumber":1,"fromAyah":1,"toAyah":7,
       "userId":1,"recitationId":1,
       "webhookUrl":"http://localhost:3777/api/v1/recitations/webhook/ai-evaluation",
       "webhookSecret":"test-secret"}'
```

---

## 11. Acceptance checklist (§8) — all passing

```
✅ /api/evaluate returns < 2s with { status:"processing", jobId }
✅ Missing/invalid Authorization → 401
✅ Downloads audio from audioUrl
✅ Works with webm (not only MP3)
✅ get_ayah_range(1,1,7) returns full Al-Fatiha
✅ Target text has no U+FC00–U+FDFF markers
✅ overallScore is 0–100 (not ≤ 1)
✅ webhook jobId matches the returned jobId
✅ Only the whitelisted top-level fields (5 on success)
✅ feedbackAudio is an absolute URL (not localhost)
✅ Failure sends status:"error" + message (no data)
✅ Retries 3× on 5xx; no retry on 4xx
✅ Temp source file deleted in all cases
```
