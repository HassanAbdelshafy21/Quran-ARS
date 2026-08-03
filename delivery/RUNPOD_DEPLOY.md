# Deploying Quran-ARS on RunPod Serverless — complete step-by-step

> Everything needed to get the recitation-grading service live on **RunPod Serverless**, today.
> Follow the steps in order. Commands are copy-paste ready; replace `<PLACEHOLDERS>`.
>
> **End result:** a public HTTPS endpoint `https://<ENDPOINT_ID>.api.runpod.ai` that your backend
> calls with `POST /api/evaluate`, that **scales to zero when idle** (you pay per second of use).

---

## 0. TL;DR — the 6 steps

1. **Build** the Docker image (model baked in, ~12–15 GB).
2. **Push** it to Docker Hub.
3. **Create a Network Volume** (so feedback audio survives worker restarts).
4. **Create a Serverless endpoint** — type **Load Balancer**, 16 GB GPU (testing), and set
   ⚠️ **Idle Timeout = 300 s** for a private beta (see §0.2, §8.1).
5. **Set `PUBLIC_BASE_URL`** to the endpoint URL you just got, and restart workers.
6. **Test** with curl + webhook.site, then hand the URL + key to the backend.

---

## 0.1 💸 Cheapest setup (testing / private beta) — use this

If this is for **testing**, configure it to cost as close to nothing as possible. Pure
scale-to-zero: you pay only for the seconds a request is actually being processed.

| Setting | Value | Note |
|---|---|---|
| **Active workers (`workersMin`)** | **0** | ⭐ the big one — nothing runs when idle |
| **Max workers** | **2** | 1 is fine for solo testing; 2 avoids queuing if two friends record at once |
| **Idle timeout** | **300 s** | ⭐ see §0.2 — keeps the worker warm *across a testing session* |
| **GPU** | **16 GB** tier | model needs ~6 GB. Prefer **A4000/A4500/RTX 4000 Ada**; avoid **T4** (no native bf16 → slow) |
| **FlashBoot** | **on** | free, cuts cold-start time |
| **Network volume** | **10 GB** (~$0.70/mo) | keeps feedback audio alive; skip only if you don't test `feedbackAudio` |
| Scheduled warm hours (§11) | **don't bother** | for real production traffic, not a private beta |

**Expected bill:** a private beta (a handful of users, a few dozen recitations/day) ≈
**$5–15/month**, plus ~$0.70 for the volume. Idle time is genuinely free at `workersMin = 0`.

> Delete the endpoint (and volume) when you're done testing to stop all charges.

### 0.2 👥 If real users (the client's friends) will try the app — read this

A private beta is **not** the same as developer testing: your testers will *feel* the cold start
instead of knowing to retry. Two cheap adjustments make a big difference:

**1. Set Idle Timeout to 300 s, not 60 s.** Testers arrive in *bursts* ("everyone try it tonight").
With a 60 s timeout the worker dies between two friends and **each of them pays a 1–3 min cold
start**. With 300 s the worker stays warm across the whole session, so only the *first* person
waits. Cost of those idle tails at beta volume is only a few dollars a month — easily worth it.

**2. Warm it manually before a scheduled demo.** If the client says "we're all testing at 8 pm",
run this ~5 minutes before and turn it off after — it costs about **$1 per hour**:

```bash
./runpod_scale.sh 1     # before the demo — no cold start for anyone
./runpod_scale.sh 0     # after — back to paying nothing
```

**⚠️ Requirement for the app/backend team:** because of cold starts, a result can take up to
**~3 minutes** on the first request after a quiet period. The app **must show a "جاري التصحيح…"
(processing) state and wait for the webhook** — it must not assume a result within 30 s or time
out. The API is asynchronous precisely so this is possible; if the app handles the pending state
properly, a cold start is merely *slow*, not *broken*.

---

## 0.3 🌐 Domain & integration requirements from the backend team

The final domain is **`quranyutla.com`** (not `yutlaquran.com`, which appeared in the old spec).
Their four requirements, and where each stands:

| # | Requirement | Status |
|---|---|---|
| 1 | `PUBLIC_BASE_URL=https://ai.quranyutla.com` | ⚠️ needs a proxy on RunPod — see §0.3.1 |
| 2 | Point `ai.quranyutla.com` at the service + SSL | ⚠️ same — §0.3.1 |
| 3 | Take `webhookUrl` from the **request body**, never hardcoded | ✅ **already correct** |
| 4 | Server must reach `https://cdn.quranyutla.com/...` to download audio | ✅ works — §0.3.2 |

**On #3 (their main worry) — already satisfied.** `webhookUrl` is a required field of the request
model and the service calls exactly what you send: `_send_webhook(req.webhookUrl, ...)`. There is
**no hardcoded domain anywhere in the service code** — the old domain only ever appeared in
documentation examples. Every request may use a different `webhookUrl`.

### 0.3.1 ⚠️ Custom domain + RunPod Serverless — read before promising `ai.quranyutla.com`

RunPod Serverless gives you a fixed URL: **`https://<ENDPOINT_ID>.api.runpod.ai`**. You **cannot
point `ai.quranyutla.com` straight at it with valid SSL** — RunPod serves the certificate for
`*.api.runpod.ai`, so a bare DNS record to that host fails TLS validation. Three options:

**Option A — use the RunPod URL for the beta (recommended now).**
Set `PUBLIC_BASE_URL=https://<ENDPOINT_ID>.api.runpod.ai` and give the backend that URL. Zero
extra cost/infra. Add the branded domain later when moving to a dedicated GPU server.

**Option B — Cloudflare Worker proxy (branded domain, ~free).**
`quranyutla.com` on Cloudflare → a small Worker on the route `ai.quranyutla.com/*` that forwards
to the RunPod endpoint. Cloudflare terminates SSL for your domain; the Worker calls RunPod over
its own valid HTTPS. Then `PUBLIC_BASE_URL=https://ai.quranyutla.com`. *(A plain proxied CNAME is
unreliable here because of SNI/host-header mismatch — use a Worker, not just a DNS record.)*

**Option C — tiny reverse-proxy VPS (~$5/mo).**
Nginx + Let's Encrypt for `ai.quranyutla.com`, `proxy_pass` → the RunPod endpoint. The most
conventional and predictable, at the cost of one small always-on box.

> **On a normal (non-serverless) GPU server**, requirement #1/#2 is trivial: point DNS at the
> server, run Nginx + certbot, set `PUBLIC_BASE_URL=https://ai.quranyutla.com`. The complication
> exists *only* because serverless endpoints live on the provider's domain.

### 0.3.2 CDN audio downloads (`cdn.quranyutla.com`)
The worker downloads `audioUrl` over the public internet; RunPod allows outbound traffic, so this
works with no extra configuration. Two things to confirm with the backend:

- the CDN objects must be **publicly readable** (or use pre-signed URLs) — RunPod worker IPs are
  **dynamic**, so an IP allow-list will not work;
- URLs must be reachable from outside your VPC/network (test with `curl -I <audioUrl>`).

### 0.3.3 ⚠️ Feedback audio on serverless (their separate point — valid)
They're right: on serverless the container disk is wiped, so `feedback_*.mp3` written to a local
`temp_storage/` would disappear. Two answers:

- **On RunPod (what we're doing now):** attach the **Network Volume** and set
  `TEMP_STORAGE_DIR=/runpod-volume/temp_storage` (§5/§6). Files then persist across worker
  restarts **with no code change**. ✅
- **If you later move to Modal / any provider without a shared volume:** the robust fix is to
  **upload `feedback_*.mp3` to object storage (S3 / Cloudflare R2 / Spaces)** and return that
  public URL in `feedbackAudio`. That is a **code change** (not yet implemented) — a small one in
  `run_grading()` where the file is written. Tell us before that migration and we'll add it.

> Also remember the trade-off in §8.3: while workers are scaled to zero, fetching a
> `feedbackAudio` URL served *by the endpoint* wakes a worker (slow). Object storage removes that
> problem too, which is a second good reason to move to it for production.

---

## 1. Why "Load Balancer" and not the normal queue endpoint

RunPod Serverless has two endpoint types:

| Type | How it works | Fits us? |
|---|---|---|
| **Queue** (default) | You write a `handler(job)` function; clients call `/run`, `/runsync`. | ❌ We already have a full REST API (`/api/evaluate`, `/health`, `/audio/...`). |
| **Load Balancer** ⭐ | RunPod exposes **your own HTTP server** directly at `https://<ID>.api.runpod.ai/<your-paths>`. | ✅ Exactly our FastAPI app. |

So we use a **Load Balancer (LB) endpoint**. The app is already prepared for it:

- listens on the injected `PORT` env var (`main.py` → `uvicorn.run(..., port=int(os.getenv("PORT", 8000)))`),
- exposes **`/ping`** implementing RunPod's health contract: **204** while the model loads,
  **200** once ready (so RunPod holds traffic until the worker can actually serve),
- loads the model in a **background thread** so the port opens immediately (RunPod can poll `/ping`
  during the 1–3 min model load instead of seeing a dead port),
- authenticates with **`X-AI-API-Key`**, because RunPod's own gateway uses the `Authorization`
  header on LB endpoints.

*(References: [RunPod LB overview](https://docs.runpod.io/serverless/load-balancing/overview).)*

---

## 2. Before you start — checklist

- [ ] A **RunPod account** with credits ([runpod.io](https://runpod.io)).
- [ ] A **Docker Hub** account (free) — or GHCR. RunPod pulls the image from there.
- [ ] **Docker installed** on a machine with good upload bandwidth (the image is ~12–15 GB).
- [ ] This repo cloned; you'll work inside `delivery/`.
- [ ] Decide your **`AI_API_KEY`** now — a strong random secret. Generate one:
      `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`
      You will give this same value to the backend team.

---

## 3. Step 1 — Build the Docker image

```bash
cd delivery
docker build -t <DOCKERHUB_USER>/quran-ars:v1 .
```

What happens: CUDA 12.1 base → Python 3.11 → torch cu121 → `requirements.txt` → **the 5 GB NAMAA
model is downloaded and baked into the image** → app code copied. Takes ~15–40 min the first time.

Verify locally before pushing (optional but recommended if you have an NVIDIA GPU):

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e AI_API_KEY=test-key-123 -e PUBLIC_BASE_URL=http://localhost:8000 \
  <DOCKERHUB_USER>/quran-ars:v1
# in another terminal — while loading you should get 204, then 200:
curl -o /dev/null -w "%{http_code}\n" http://localhost:8000/ping
curl http://localhost:8000/health     # -> {"status":"ok","model_loaded":true,...}
```

> **Low upload bandwidth?** The image is big because the model is baked in (best cold-start).
> Alternative: comment out the model pre-download `RUN` line in the `Dockerfile`, set
> `HF_HOME=/runpod-volume/hf` in the endpoint env, and let the model download once onto the
> Network Volume (§5). Image drops to ~7 GB, but the **first** cold start is slower.

---

## 4. Step 2 — Push to Docker Hub

```bash
docker login
docker push <DOCKERHUB_USER>/quran-ars:v1
```

Upload time depends on your connection (12 GB ≈ 20 min at 100 Mbps, ≈ 3 h at 10 Mbps). Keep the
repo **public** on Docker Hub, or add registry credentials in the RunPod endpoint settings.

> **Faster option to check:** RunPod can also **build from a GitHub repo** for you (no local build
> or upload). In the console: *Serverless → New Endpoint → import from GitHub*, point it at this
> repo with the Dockerfile path `delivery/Dockerfile`. If available to your account, this is the
> quickest path today since the repo is public.

---

## 5. Step 3 — Create a Network Volume (important)

Serverless workers have **ephemeral disks** — they are wiped when a worker scales down. The service
writes `feedback_<jobId>.mp3` files that the backend serves to users **for 30 days**, so they must
live on persistent storage.

1. RunPod console → **Storage → Network Volumes → New**.
2. Pick a **datacenter/region** (remember it — the endpoint must be in the **same** region).
3. Size: **10 GB** is plenty for testing (~$0.70/mo; audio is small). Name it e.g. `quran-ars-vol`.

It will be mounted inside the worker at **`/runpod-volume`**. We point the app at it with
`TEMP_STORAGE_DIR=/runpod-volume/temp_storage` (already supported by `main.py`).

---

## 6. Step 4 — Create the Serverless endpoint

Console → **Serverless → New Endpoint**.

### Settings

| Setting | Value | Why |
|---|---|---|
| **Endpoint type** | **Load Balancer** | we serve our own HTTP API (§1) |
| **Container image** | `<DOCKERHUB_USER>/quran-ars:v1` | what you pushed |
| **GPU** | **16 GB** tier for testing (24 GB for production) | model uses ~6 GB. Avoid **T4** — no native bf16 |
| **Expose HTTP port** | **8000** | our app's default port |
| **Container disk** | **25 GB** | must exceed the image size |
| **Network volume** | `quran-ars-vol` (same region) | persistent feedback audio (§5) |
| **Active workers** | **0** | scale to zero = pay only when used |
| **Max workers** | **2** for a private beta | raise later for concurrency |
| ⚠️ **Idle timeout** | **300 s** for a private beta (§0.2); 90 s for cost-only testing; never below 60 s | **critical** — see §8.1; also drives cost (§10) |
| **Execution timeout** | leave default (≥ 5 min) | our request returns in <2 s |

### Environment variables

| Name | Value |
|---|---|
| `AI_API_KEY` | your strong secret (from §2) |
| `TEMP_STORAGE_DIR` | `/runpod-volume/temp_storage` |
| `PUBLIC_BASE_URL` | *leave blank for now* — filled in Step 5 (RunPod URL, or `https://ai.quranyutla.com` if you add the §0.3.1 proxy) |
| `HF_HUB_OFFLINE` | `1` (model is baked in; avoids runtime downloads) |

> If you used the "don't bake the model" variant (§3), **omit `HF_HUB_OFFLINE`** and add
> `HF_HOME=/runpod-volume/hf` instead.

Create the endpoint. RunPod pulls the image (a few minutes the first time).

---

## 7. Step 5 — Set `PUBLIC_BASE_URL` (chicken-and-egg)

Once the endpoint exists you get its ID and URL:

```
https://<ENDPOINT_ID>.api.runpod.ai
```

The service needs to know its own public address so the `feedbackAudio` links it returns are
reachable by the mobile app. So:

1. Copy the endpoint URL.
2. Edit the endpoint → **Environment variables** → set
   `PUBLIC_BASE_URL = https://<ENDPOINT_ID>.api.runpod.ai`
   *(or `https://ai.quranyutla.com` **only if** you set up the proxy in §0.3.1 — it must be a URL
   the mobile app can actually fetch `feedbackAudio` from)*
3. **Save** — RunPod will restart the workers with the new value.

---

## 8. Critical gotchas (read these — they will bite you otherwise)

### 8.1 ⚠️ Idle Timeout must cover background processing
Our `/api/evaluate` **returns immediately** (`{"status":"processing","jobId":...}`) and then does the
real work (download audio → transcribe → grade → **POST the webhook**) in a background task.

RunPod decides a worker is idle based on **HTTP requests**, and our HTTP request finished in
~1 second. If **Idle Timeout is short (e.g. the 5 s default)**, RunPod can shut the worker down
**mid-processing** → the webhook is never sent and the result is silently lost.

**Fix: set Idle Timeout to 90 s** (never below 60 s). Grading takes ~5–30 s, so 90 s leaves a
comfortable margin. This is the single most important setting on the page.

Don't just set it huge: the idle tail is **billed** in scale-to-zero mode, so an over-long timeout
(e.g. 300 s) can triple your bill on sparse traffic (§10). 90 s is the sweet spot.

### 8.2 Cold starts — the backend must retry
With 0 active workers, the first request after idle must boot a container and load the 5 GB model:
**~1–3 minutes**. During that time RunPod's gateway may return an error rather than queueing.

**Tell the backend:** retry `/api/evaluate` at least **3 times with 5–10 s delays** on non-2xx.
(Already documented in `BACKEND_HANDOFF.md`.) To eliminate cold starts entirely, set
**Active workers = 1** — but then you pay 24/7 (see §10).

### 8.3 Feedback audio wakes a worker
`feedbackAudio` URLs point at `https://<ID>.api.runpod.ai/audio/...`. If all workers are asleep,
fetching that file triggers a cold start (slow for the user). Acceptable for beta; for production,
consider uploading feedback audio to object storage (S3/R2/Spaces) and returning that URL instead.

### 8.4 Two different headers — don't mix them up
- **Your app key:** `X-AI-API-Key: <AI_API_KEY>` (plain value, **no** `Bearer`).
- **RunPod's gateway** uses `Authorization: Bearer <RUNPOD_API_KEY>` on LB endpoints — that's why
  our key moved to a custom header.
- **The webhook** we send to the backend uses `Authorization: Bearer <webhookSecret>` — a third,
  unrelated header. Correct as-is.

### 8.5 Networking
The worker must be able to **download `audioUrl`** and **reach `webhookUrl`** over the public
internet. `webhook.site` and `everyayah.com` both work for testing.

---

## 9. Step 6 — Verify the deployment

Replace `<ENDPOINT_ID>` and `<AI_API_KEY>`.

**a) Health** (may take 1–3 min on the very first call — cold start):

```bash
curl https://<ENDPOINT_ID>.api.runpod.ai/health
# expect: {"status":"ok","model_loaded":true,"version":"2.0.0"}
```

**b) Auth rejects a wrong key:**

```bash
curl -s -X POST https://<ENDPOINT_ID>.api.runpod.ai/api/evaluate \
  -H "X-AI-API-Key: wrong" -H "Content-Type: application/json" \
  -d '{"audioUrl":"https://everyayah.com/data/Husary_128kbps/112002.mp3","surahNumber":112,"fromAyah":2,"toAyah":2,"userId":1,"recitationId":1,"webhookUrl":"https://example.com","webhookSecret":"s"}'
# expect: {"status":"error","message":"Invalid API key","code":"AUTH_FAILED"}
```

**c) Full end-to-end with a real webhook:**

1. Open **https://webhook.site** and copy your unique URL.
2. Fire the request:

```bash
curl -s -X POST https://<ENDPOINT_ID>.api.runpod.ai/api/evaluate \
  -H "X-AI-API-Key: <AI_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "audioUrl": "https://everyayah.com/data/Husary_128kbps/112002.mp3",
    "surahNumber": 112, "surahName": "الإخلاص",
    "fromAyah": 2, "toAyah": 2,
    "userId": 1, "recitationId": 1,
    "webhookUrl": "https://webhook.site/<YOUR-ID>",
    "webhookSecret": "test-secret"
  }'
# immediate: {"status":"processing","jobId":"...","estimatedTime":30}
```

3. Within ~10–40 s the **full result appears on webhook.site**, containing `data` with
   `overallScore`, `userRecitationDiacritized`, `harakatErrors`, `words`, `feedbackAudio`.

✅ **If step (c) delivers a webhook with `"status":"success"`, your deployment is done.**

---

## 10. What it will cost

RunPod Serverless bills **per second of active worker time** (plus a tiny charge for the network
volume). Read the exact per-second rate for your chosen GPU in the console — approximate 2026
ranges:

| GPU tier | ≈ per second | ≈ per hour |
|---|---|---|
| 16 GB (A4000/4500) | ~$0.00016–0.00020 | ~$0.60–0.72 |
| **24 GB (L4/A5000/3090)** ⭐ | ~$0.00019–0.00034 | ~$0.70–1.20 |
| 48 GB (A6000/L40S) | ~$0.00050+ | ~$1.80+ |

**Cost per recitation** ≈ *active seconds* × *per-second rate*. Our grading takes roughly **5–15 s**
of GPU time, so at ~$0.0003/s that's ≈ **$0.0015–0.005 per recitation**.

⚠️ **You also pay for the idle-timeout tail.** In scale-to-zero mode a worker stays alive for your
**Idle Timeout** after finishing a request, and that time **is billed**. So the real cost depends
heavily on whether requests **cluster** (sharing one warm worker + one tail) or arrive **isolated**
(each paying its own full tail). For 10,000 requests/month with a 120 s idle timeout:

| Traffic pattern | Billed GPU time | ~Cost |
|---|---|---|
| **Clustered** (requests close together) | ~30–50 h | **~$25–50** |
| **Spread out** (isolated requests) | ~360 h | **~$250** |

**Lever:** the background task only needs ~10–30 s after the HTTP response, so **Idle Timeout = 90 s**
is safe and roughly 3× cheaper than 300 s on sparse traffic. Don't go below ~60 s (§8.1).

> **"But it's serverless — why does it cost anything when idle?"** Serverless means RunPod manages
> the scaling, not that idle GPUs are free. You are billed for every second a worker is *running* —
> which includes the idle-timeout tail, and (if you set `workersMin ≥ 1`) the entire time that
> worker is held warm.

**Trust the dashboard, not this table:** RunPod bills real GPU-seconds and shows them in the
console. After a week of real traffic you'll know your actual number.

Plus **network volume** ≈ $0.07/GB/month (20 GB ≈ **$1.40/mo**).

**Notes**
- **Idle time is free** with Active workers = 0 — you only pay while a request is being processed
  (plus the idle-timeout tail you configured in §8.1).
- Setting **Active workers = 1** removes cold starts but costs the full hourly rate 24/7
  (~$500–870/mo) — only worth it at high, steady traffic. At that point a **dedicated L4 VM
  (~$300–580/mo)** is cheaper; see `docs/Deployment-and-Cost.md`.
- **No per-request AI fees** — the model is Apache-2.0 and self-hosted.

---

## 11. (OPTIONAL — not for testing) Scheduled warm hours, 10:00–17:00

> ⏭️ **Skip this section while testing.** Keep `workersMin = 0` (§0.1). This is a
> latency optimization for *real production traffic*; turning it on costs ~$100–200/month.
> Come back here only when users complain about first-request latency.

Great idea for killing cold starts during working hours while paying nothing overnight.
**RunPod has no built-in scheduler**, but its REST API lets you change the active-worker count,
so you drive it from **cron**. ([RunPod recommends exactly this](https://www.runpod.io/articles/guides/ai-on-a-schedule).)

The field you change is **`workersMin`** — that is the console's **"Active workers"** setting:

- `workersMin = 1` → one worker stays warm permanently → **no cold starts**, billed at the
  cheaper **active** rate.
- `workersMin = 0` → scale to zero → pay only per request, but the first request after a quiet
  period pays the 1–3 min cold start.

### 11.1 The script (included: `runpod_scale.sh`)

```bash
export RUNPOD_API_KEY=<from runpod.io → Settings → API Keys>
export RUNPOD_ENDPOINT_ID=<your endpoint id>

./runpod_scale.sh 1    # warm  (start of day)
./runpod_scale.sh 0    # cold  (end of day)
```

It calls `POST https://rest.runpod.io/v1/endpoints/{id}/update` with `{"workersMin": N}`
(partial updates are allowed, so nothing else is touched).

### 11.2 Option A — cron on a machine you control

On any always-on Linux box (your backend server, a small VPS), `crontab -e`:

```cron
# times are in the SERVER's timezone — check with:  timedatectl
0 10 * * *  cd /path/to/delivery && RUNPOD_API_KEY=xxx RUNPOD_ENDPOINT_ID=yyy ./runpod_scale.sh 1 >> /var/log/runpod_scale.log 2>&1
0 17 * * *  cd /path/to/delivery && RUNPOD_API_KEY=xxx RUNPOD_ENDPOINT_ID=yyy ./runpod_scale.sh 0 >> /var/log/runpod_scale.log 2>&1
```

⚠️ **Timezone**: cron uses the machine's local time. If the server runs UTC and you want Cairo
hours, convert (Cairo is UTC+2 in winter, **UTC+3 in summer**). Safest is to set the box to
`Africa/Cairo` (`sudo timedatectl set-timezone Africa/Cairo`) so the schedule follows DST by itself.

### 11.3 Option B — GitHub Actions (no server needed) ⭐

Since the repo is on GitHub, you can schedule it for free. Add
`.github/workflows/runpod-schedule.yml`:

```yaml
name: RunPod warm schedule
on:
  schedule:
    - cron: "0 7 * * *"    # 07:00 UTC -> 10:00 Cairo (summer) / 09:00 (winter)
    - cron: "0 15 * * *"   # 15:00 UTC -> 18:00 Cairo (summer) / 17:00 (winter)
  workflow_dispatch:        # lets you trigger it manually too
jobs:
  scale:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set active workers
        env:
          RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}
          RUNPOD_ENDPOINT_ID: ${{ secrets.RUNPOD_ENDPOINT_ID }}
        run: |
          # first cron of the day warms up, second cools down
          if [ "$(date -u +%H)" -lt 12 ]; then N=1; else N=0; fi
          chmod +x delivery/runpod_scale.sh && delivery/runpod_scale.sh "$N"
```

Add `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` under **repo → Settings → Secrets and variables →
Actions**. Note: GitHub cron is **UTC only** and can fire **5–15 min late** under load — fine here.
The UTC times above deliberately err on the side of warming *earlier* and cooling *later* so you're
never cold during working hours.

### 11.4 Is it actually worth it? (honest math)

7 h/day × 30 days = **210 h/month**. On a 24 GB GPU at the **active** rate
(~20–30 % cheaper than flex, roughly $0.00013–0.00027/s):

| Strategy | Billed GPU time | Cost / month | Cold starts |
|---|---|---|---|
| Scale-to-zero, **clustered** traffic | ~30–50 h | **~$25–50** | yes, after each gap |
| Scale-to-zero, **spread-out** traffic | ~360 h | **~$250** | yes, after each gap |
| **Warm 10:00–17:00 only** | 210 h | **~$100–200** | **none in working hours** |
| Active 24/7 | 730 h | ~$340–710 | none ever |

**The decision rule:**

- **Sparse traffic in bursts** (a few clusters per day, long quiet gaps) → **stay at
  `workersMin = 0`**. Cold starts only hit the first request of each burst, and the backend
  retries (§8.2).
- **Requests trickling all day long** → this is the case people get wrong. Scale-to-zero pays a
  full idle tail per isolated request *at the higher flex rate*, so **the warm schedule is often
  both cheaper and faster**. Active workers bill ~20–30 % below flex.

Note the counter-intuitive result: "serverless" is **not** automatically the cheapest option — it
wins on *bursty* traffic and loses on *thin, constant* traffic.

**My recommendation for today:** launch with `workersMin = 0`, watch real usage for a week, then
turn on the schedule if first-request latency actually bothers users. The script is ready whenever
you want it — flipping it on is one cron entry.

---

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Workers never become "ready"; 502s | Port mismatch. The endpoint's **Expose HTTP port** must be **8000** (or set env `PORT` to whatever you exposed). |
| `/health` works but no webhook arrives | **Idle Timeout too short** (§8.1) — raise to 90 s. Or the worker can't reach `webhookUrl`. |
| First request fails, later ones work | Normal **cold start** — implement retries (§8.2). |
| `401 AUTH_FAILED` | Wrong header or value. Must be `X-AI-API-Key: <AI_API_KEY>`, exactly matching the endpoint env var. |
| Feedback audio 404 after a while | `TEMP_STORAGE_DIR` not set to `/runpod-volume/temp_storage`, or no network volume attached (§5). |
| Model re-downloads every cold start | You didn't bake it in and/or `HF_HOME` isn't on the volume. Set `HF_HOME=/runpod-volume/hf`. |
| Garbled Arabic output | Someone switched the model to fp16 — it **must** run bf16 (the shipped code does). |
| Out-of-memory | GPU tier too small; use ≥ 16 GB (24 GB recommended). |

**Logs:** endpoint → **Workers → Logs**. Look for `NAMAA loaded.` (model ready) and any
`Harakat grading failed` / webhook errors.

---

## 13. Updating the service later

```bash
cd delivery
docker build -t <DOCKERHUB_USER>/quran-ars:v2 .
docker push <DOCKERHUB_USER>/quran-ars:v2
# console: edit endpoint -> change image tag to :v2 -> save (workers restart)
```

Always bump the tag (`:v2`, `:v3`) rather than reusing `:latest`, so rollback is just pointing the
endpoint back at the previous tag.

---

## 14. Hand-off to the backend team

Give them exactly three things:

1. **Base URL:** `https://<ENDPOINT_ID>.api.runpod.ai`
2. **API key:** the `AI_API_KEY` value (share securely), sent as header **`X-AI-API-Key`**
3. **The guide:** `BACKEND_GUIDE_AR.md` (Arabic) or `BACKEND_HANDOFF.md` (English full contract)

Remind them of: the **cold-start retry** requirement (§8.2) and that `webhookUrl` must be publicly
reachable.
