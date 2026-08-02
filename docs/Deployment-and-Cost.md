# Deployment & Server Cost Guide

> A plain-language guide to **where to run the AI service, which server is best, and how much it
> costs** — written so you can explain it to the client. Prices are approximate 2026 ranges; treat
> them as ballpark, not quotes.

---

## 1. In plain terms — what actually needs to run

The AI (the model that listens to a recitation and grades it) is a **2-billion-parameter neural
network**. Networks this size run on a **GPU** (the specialized chip used for AI), not a normal
CPU server. So we need **one machine with a GPU**. The service:

1. receives a recording (a URL to an audio file),
2. runs it through the model on the GPU (~a few seconds),
3. sends back the score + the diacritized recitation + tajweed feedback.

Everything is **self-hosted and open-source (Apache-2.0)** — meaning **there are no per-request AI
fees** (unlike paying Google/OpenAI per audio minute). You pay only for the GPU machine.

---

## 2. What the server needs (requirements)

| Requirement | Value | Why |
|---|---|---|
| **GPU** | NVIDIA with **≥ 8 GB** VRAM (we use ~6 GB) | to run the 2 B model in bf16 |
| GPU memory (recommended) | 16–24 GB | comfortable headroom |
| Disk | ~15 GB | the model is ~5 GB |
| System RAM | 8 GB+ | — |
| Software | CUDA 12.x, Docker + NVIDIA Container Toolkit | to run the container with the GPU |

A **CPU-only server will not work** (far too slow for a 2 B model). This is the one hard rule.

---

## 3. Which server — the two ways to pay

There are two pricing models. Pick based on **how much traffic** you expect.

### (A) Always-on GPU server — rent a machine 24/7
Best when you have **steady traffic**. You pay a flat hourly rate whether it's busy or idle.

### (B) Serverless / pay-per-use GPU — pay only while processing
Best when traffic is **low or bursty** (e.g. a new app). Because our API is **asynchronous** (the
caller doesn't wait), serverless "cold starts" don't matter — perfect fit. You pay per second of
actual GPU use, so an idle night costs ~nothing.

### GPU choices (both models use one of these)

| GPU | VRAM | Speed | Typical rent (always-on) | Notes |
|---|---|---|---|---|
| **NVIDIA L4** ⭐ | 24 GB | fast, modern | **~$0.40–0.80 /hr → ~$290–580 /mo** | best value; recommended |
| **NVIDIA T4** | 16 GB | slower (older) | ~$0.30–1.00 /hr → ~$220–720 /mo | cheapest; fine for low volume |
| **NVIDIA A10G** | 24 GB | very fast | ~$1.00 /hr → ~$730 /mo | if you need max speed |

**Where to rent them (2026):**
- **RunPod** — usually the cheapest (L4 ≈ $0.39/hr, spot even less); also has serverless GPU.
- **AWS** — `g6` = L4 (~$0.80/hr), `g4dn` = T4, `g5` = A10G.
- **Google Cloud / Lambda / Azure** — similar; GCP and Azure have free trial credits (see §7).
- **Modal / RunPod Serverless** — for the pay-per-use model.

---

## 4. How much it will cost — by usage

At roughly **~5 seconds of GPU time per recitation**, a single always-on **L4 can handle up to
~500,000 recitations per month**. Here's the cost at different scales:

| Monthly recitations | Serverless (pay-per-use) | Always-on **L4** |
|---|---|---|
| **10,000** (beta) | **~$25** | ~$290–580 (mostly idle) |
| **100,000** | ~$250 | **~$290–580** |
| **500,000** | ~$1,250 | **~$290–580** (one GPU) |
| **1,000,000** | ~$2,500 | ~$580–1,160 (two GPUs) |

**Rule of thumb:** below ~100–150k recitations/month, **serverless is cheaper**; above that, a
flat **always-on L4** wins. You switch models as you grow — no code change needed.

---

## 5. Our recommendation (the short answer for the client)

- **While testing:** **FREE** — use the Colab notebook (§7). $0.
- **At launch / beta (low, variable traffic):** **serverless GPU** (RunPod/Modal) → typically
  **~$0–50 / month**. Pay only for what you use.
- **At real usage (hundreds of thousands of recitations/month):** **one always-on NVIDIA L4** →
  **~$300–580 / month**, and it handles up to ~500k recitations.
- **To scale further:** add more GPU replicas behind a load balancer; each handles the same load.

**One GPU goes a very long way.** You do **not** need an expensive multi-GPU cluster to launch.

---

## 6. Other costs (small)

- **Storage & bandwidth:** negligible (the model is baked into the image once; audio is small).
- **Feedback audio:** stored ≥ 30 days on the server disk — a few GB at most.
- **No AI usage fees:** because the model is self-hosted and Apache-2.0, there is **zero
  per-request cost** — a major saving compared to paid speech APIs (which can be $0.01–0.02 per
  audio minute and would dominate the bill at scale).

---

## 7. Test it FREE before paying (do this first)

Prove everything works on a real GPU without spending anything:

1. **Google Colab (free T4 GPU)** — open [`delivery/Colab_Test.ipynb`](../delivery/Colab_Test.ipynb),
   set *Runtime → GPU*, *Run all*. It loads the model, gives a public URL, and runs the full
   sync + async(webhook) tests. **Cost: $0.**
2. **Google Cloud $300 free trial** (or Azure $200) — spin up an L4 VM and run the **actual Docker
   container** (`docker compose up`) exactly like production. Best "dress rehearsal" before buying;
   you won't be charged if you stay within the credit and delete the VM.

Only after this passes do you rent a paid server.

---

## 8. How to deploy (once you have the GPU box)

```bash
cd delivery
printf 'AI_API_KEY=%s\nPUBLIC_BASE_URL=%s\n' "<STRONG_SECRET>" "https://ai.example.com" > .env
docker compose up -d --build          # first build bakes the ~5 GB model into the image
curl http://localhost:8000/health     # -> {"status":"ok","model_loaded":true}
```

Put **Nginx + HTTPS** in front (port 443 → 8000), set the two env vars (`AI_API_KEY`,
`PUBLIC_BASE_URL`), and wire `GET /health` into your monitoring. Full, self-contained steps:
**[delivery/AGENT_DEPLOY_PROMPT.md](../delivery/AGENT_DEPLOY_PROMPT.md)** and
**[delivery/DEPLOYMENT_GUIDE.md](../delivery/DEPLOYMENT_GUIDE.md)**.

---

## 9. One-line summary (for the client)

> Testing is **free** (Colab). A small launch runs on **serverless GPU for ~$0–50/month**. Real
> usage (up to ~500k recitations/month) runs on **one NVIDIA L4 for ~$300–580/month**, with **no
> per-request AI fees** because the model is self-hosted. Start serverless, move to a dedicated L4
> as you grow.
