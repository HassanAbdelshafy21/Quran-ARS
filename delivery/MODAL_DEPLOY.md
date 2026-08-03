# Deploying Quran-ARS on Modal — step by step

> Chosen over RunPod Serverless after the backend's cost analysis: at ~250 evaluations/day Modal is
> both **cheaper** and has **better UX**, because its fast cold starts let the GPU scale down
> aggressively without every request paying a 1–3 minute model load.
>
> **End result:** a public HTTPS endpoint (`https://<org>--quran-ars-web.modal.run`) whose API
> answers in milliseconds, with the GPU billed only while a recitation is actually being graded.

---

## 1. The architecture (why this is cheap *and* fast)

```
web()   — CPU only, tiny image, kept warm (min_containers=1)
          /health, /ping        -> instant
          /api/evaluate         -> auth -> SPAWN gpu job -> {"status":"processing", jobId}
              │
              │  .spawn()  (fire-and-forget; runs to completion independently)
              ▼
Grader  — GPU (A10G), loads NAMAA once per container (@modal.enter)
          download audio -> grade words + harakat -> upload mp3 to R2 -> POST webhook
          scales to zero 60 s after the last job
```

Two consequences that matter:

- **The API never waits on a GPU.** Callers get their `jobId` immediately even when no GPU is
  running — so the cold start is invisible to the request, and only delays the webhook.
- **The GPU is billed only while grading.** At 250 evaluations/day (~10 s each) that's ~21 GPU-hours
  a month instead of paying for idle time.

> This is the fix for the problem that made RunPod expensive: there, a long idle timeout was needed
> to keep the background task alive, which meant paying for a mostly-idle GPU all day. Modal's
> `.spawn()` makes the job a first-class unit of work, so it can't be killed by the HTTP response
> finishing, and the idle window can be short.

---

## 2. Prerequisites

- [ ] A **Modal account** ([modal.com](https://modal.com)) — includes free monthly credits.
- [ ] **Cloudflare R2** bucket (or any S3-compatible storage) for feedback audio — **required**,
      because Modal containers are ephemeral (§4).
- [ ] Python locally: `pip install modal && modal token new`
- [ ] A strong `AI_API_KEY` (share with the backend):
      `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`

---

## 3. Create the R2 bucket (feedback audio)

1. Cloudflare dashboard → **R2 → Create bucket**, e.g. `quran-ars-audio`.
2. **R2 → Manage API tokens → Create** (Object Read & Write) → note the **Access Key ID**,
   **Secret Access Key**, and your **Account ID**.
3. Expose it publicly on your domain: bucket → **Settings → Public access → Connect custom domain**
   → `cdn.quranyutla.com`. (The backend already planned this domain for audio.)

You now have:

| Value | Example |
|---|---|
| `S3_ENDPOINT_URL` | `https://<ACCOUNT_ID>.r2.cloudflarestorage.com` |
| `S3_BUCKET` | `quran-ars-audio` |
| `S3_PUBLIC_BASE_URL` | `https://cdn.quranyutla.com` |

---

## 4. Create the Modal secret

All configuration is injected as environment variables from one Modal secret named
**`quran-ars`** (referenced by `modal_app.py`):

```bash
modal secret create quran-ars \
  AI_API_KEY='<your-strong-secret>' \
  S3_BUCKET='quran-ars-audio' \
  S3_ACCESS_KEY_ID='<r2-access-key>' \
  S3_SECRET_ACCESS_KEY='<r2-secret>' \
  S3_ENDPOINT_URL='https://<ACCOUNT_ID>.r2.cloudflarestorage.com' \
  S3_PUBLIC_BASE_URL='https://cdn.quranyutla.com' \
  S3_PREFIX='feedback/'
```

> **Why R2 is mandatory here:** Modal wipes the container disk on scale-down, so a
> `feedback_*.mp3` written locally would vanish. With these variables set, the service uploads each
> file and returns a `https://cdn.quranyutla.com/feedback/...` URL in `feedbackAudio`. If the
> variables are missing the code still runs (it falls back to a local path) — but those URLs will
> 404 on Modal, so don't skip this.

---

## 5. Deploy

```bash
cd delivery
modal deploy modal_app.py
```

The first deploy builds the image and **bakes the 5 GB model into it** (10–25 min). Subsequent
deploys reuse the cached layers and take seconds.

Modal prints the public URL, e.g.:

```
✓ Created web => https://<your-org>--quran-ars-web.modal.run
```

That URL is what the backend sets as `AI_SERVICE_URL`.

---

## 6. Verify

```bash
BASE=https://<your-org>--quran-ars-web.modal.run
KEY=<your AI_API_KEY>

# 1) health — should answer instantly (CPU container, always warm)
curl $BASE/health
# {"status":"ok","service":"web","version":"2.0.0"}

# 2) auth is enforced
curl -s -X POST $BASE/api/evaluate -H "X-AI-API-Key: wrong" \
  -H "Content-Type: application/json" -d '{"audioUrl":"https://x/a.mp3","surahNumber":112,"fromAyah":1,"toAyah":1,"webhookUrl":"https://x","webhookSecret":"s"}'
# {"status":"error","message":"Invalid API key","code":"AUTH_FAILED"}

# 3) full end-to-end (webhook.site gives you a URL that captures the callback)
curl -s -X POST $BASE/api/evaluate \
  -H "X-AI-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{
    "audioUrl": "https://everyayah.com/data/Husary_128kbps/112002.mp3",
    "surahNumber": 112, "surahName": "الإخلاص", "fromAyah": 2, "toAyah": 2,
    "userId": 1, "recitationId": 1,
    "webhookUrl": "https://webhook.site/<YOUR-ID>", "webhookSecret": "test-secret"
  }'
# {"status":"processing","jobId":"...","estimatedTime":30}
```

✅ Success = the webhook arrives at webhook.site with `"status":"success"`, and its
`data.feedbackAudio` is a **`https://cdn.quranyutla.com/feedback/...`** URL that plays in a browser.

Logs: `modal app logs quran-ars` (look for `Grader container ready.`).

---

## 7. What it costs

Billed per second, only while containers run.

| Component | When it runs | Rate (approx) |
|---|---|---|
| `web` (CPU, warm) | always (`min_containers=1`) | ~$0.03–0.07 / GB-hr → **~$5–10/mo** |
| `Grader` (A10G GPU) | only while grading | ~$0.000306/s → **~$1.10/hr** |

At **250 evaluations/day** (~10 s each → ~21 GPU-hours/month) plus the 60 s scale-down window:

- GPU: **~$25–45/month**
- Always-warm web container: **~$5–10/month**
- R2 storage: **~$1/month**
- **Total ≈ $30–55/month** (vs ~$130–320 on RunPod Serverless at the same load)

**Cheaper still:** set `min_containers=0` on `web()` to drop the ~$5–10 — the API then cold-starts
in ~1–2 s (tiny image, no model). Worth it if a 1–2 s first response is acceptable.

---

## 8. Tuning

| Knob | Where | Effect |
|---|---|---|
| `scaledown_window=60` | `@app.cls` on `Grader` | how long a warm GPU waits for the next job. Raise to 120–300 during a demo burst; lower to ~10 for minimum cost. |
| `min_containers=1` | `@app.function` on `web` | keeps the API instant. Set 0 to save ~$5–10/mo at the cost of a ~1–2 s cold start. |
| `gpu="A10G"` | `@app.cls` | `"L4"` is cheaper/slower, `"A100"` faster. Needs **bf16** support — do not use T4. |
| `max_containers=3` | `@app.cls` | concurrency ceiling; raise with traffic. |
| `timeout=600` | `@app.cls` | max seconds for one grading job. |

---

## 9. Notes for the backend team

- **Base URL** = the `modal.run` URL from §5 (or your domain once the Worker is up — §10).
- **Auth header** is unchanged: `X-AI-API-Key: <AI_API_KEY>`.
- **Request/response and webhook payloads are identical** to the documented contract
  (`BACKEND_HANDOFF.md` / `BACKEND_GUIDE_AR.md`) — the platform change is invisible to you.
- **`webhookUrl` is still read per-request** from the body; nothing is hardcoded.
- **`feedbackAudio` is now a CDN URL** (`cdn.quranyutla.com/feedback/...`) instead of a service
  path — better for you: it's served straight from storage, never wakes a GPU, and persists.
- **Cold starts now only delay the webhook**, not the API response. The app should still show a
  pending state, but `/api/evaluate` itself always answers immediately.

---

## 10. Custom domain (`ai.quranyutla.com`)

Modal supports custom domains on paid plans (dashboard → your app → domains), which is simpler than
the Cloudflare Worker that RunPod required. Alternatively use the same Worker approach: route
`ai.quranyutla.com/*` → the `modal.run` URL. Either way, **audio does not flow through it** — it
comes from R2/`cdn.quranyutla.com` directly.

Because the backend keeps `AI_SERVICE_URL` in an env var, switching from the `modal.run` URL to
`ai.quranyutla.com` later is a one-variable change with no code impact.

---

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `AUTH_FAILED` on every call | `AI_API_KEY` in the Modal secret ≠ the header the backend sends. |
| Webhook never arrives | Check `modal app logs quran-ars` for the Grader job; make sure `webhookUrl` is publicly reachable. |
| `feedbackAudio` 404s | S3_* variables missing/wrong → the code fell back to a local path that doesn't exist on Modal. Re-check §4. |
| Slow first grading | Normal GPU cold start (model load). The API response is unaffected; only the webhook is delayed. |
| Garbled Arabic output | Model must run **bf16** — don't switch the GPU to T4 or force fp16. |
| Image build fails on `add_local_dir` | Run `modal deploy` from inside `delivery/` (paths are relative to it). |
