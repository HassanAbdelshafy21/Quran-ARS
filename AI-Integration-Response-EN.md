# Integration Response — AI → Backend

**From:** AI service (Quran-ASR)
**To:** Backend team (Quran Yutla)
**Re:** `AI-Integration-Verification-EN.md`
**Status:** ✅ Both questions answered with evidence · one scheduling item to agree on (T4 run)

---

## 1. TL;DR

- **Q1 (2xx vs 200):** We treat **`status < 400`** as success — your `201 Created` is accepted, **no retries on success**. Already in the code, nothing to change.
- **Q2 (mid-range scale):** Verified with a real 5-of-8 recitation → **`overallScore: 62.5`** (not `0.625`). Raw JSON below.
- **T4 version-exact run (your §8):** We **cannot** run it on our current hardware (Blackwell GPU, needs cu128; the deploy target is T4/cu118). Proposal in §4 — let's agree on a debugging window.
- Everything else acknowledged; we'll use the `AI_API_KEY` + rotated `webhookSecret` you send and discard any earlier values.

---

## 2. Q1 — Success is `status < 400` (your `201` is fine)

Our webhook sender treats anything below 400 as delivered and stops. 4xx is terminal (no retry); only 5xx/timeout retries 3× with 1s→2s→4s backoff.

```python
# delivery/main.py — _send_webhook()
if resp.status_code < 400:          # ✅ 201 Created lands here → success, no retry
    return
if 400 <= resp.status_code < 500:   # 401 / 404 / 400 → do not retry
    return
# 5xx or timeout → retry (up to 3 attempts)
```

So: **`201` = success (delivered once), `401`/`404`/`400` = no retry, `5xx`/timeout = retried.** Matches your §7 status semantics exactly.

---

## 3. Q2 — Mid-range score proof (`62.5`, not `0.625`)

We ran the **actual grader** on a deliberately flawed recitation — 5 correct words out of 8 — through the exact conversion used in `/api/evaluate` (`round(accuracy * 100, 2)`):

```
target : بسم الله الرحمن الرحيم الحمد لله رب العالمين      (8 words)
student: بسم الله الرحمن الرحيم الحمد كتاب قلم شمس         (5 correct, 3 wrong)

raw accuracy : 0.625
raw_score    : 5/8
overallScore : 62.5
```

The exact `data` fields the webhook would carry:

```json
{
  "overallScore": 62.5,
  "passed": false,
  "totalWords": 8,
  "correctWords": 5,
  "incorrectWords": 3
}
```

Scale sweep, in your verification style:

```
accuracy 0.625 -> overallScore 62.5
accuracy 0.400 -> overallScore 40.0
accuracy 0.875 -> overallScore 87.5
accuracy 1.000 -> overallScore 100.0
accuracy 0.000 -> overallScore 0.0
```

There is exactly **one** place this conversion happens in our code, and it is on the acceptance checklist — so a future regression to a 0–1 ratio would fail the check before it ships. We understand you removed your defensive ×100 net and that the scale is now load-bearing on our side; understood and owned.

---

## 4. T4 version-exact run (your §8) — the one open item

You're right that this is the most likely deploy-time surprise, and we want to be straight about it:

- All functional verification (grader, `/api/evaluate`, webhook, webm, 0–100) was done on a GPU, and the **dependency + code layer** was re-verified in a clean Python 3.10 install (no conflicts, app imports clean).
- **But** our dev GPU is Blackwell (needs torch cu128); it **cannot** run the T4/cu118 image. So the exact CUDA/torch/driver combo of the deploy target is genuinely unverified by us.

**Proposal (pick one):**
1. **Preferred:** give us a short-lived **T4 (or any cu118) slot** — or a first deploy into your `quran-yutla` namespace — and we run the version-exact smoke test (health + one real grade) as the very first step, before the joint test.
2. **Fallback:** run the version-exact check as the **opening move of the joint test**, and block out a **debugging window** for it. We'd rather flag it now than surprise the schedule.

Either way it's a small check (model loads on cu118 + one recitation graded), not a rebuild.

---

## 5. Acknowledgements (your §5–§7)

- **Secrets:** please send the `AI_API_KEY` + rotated `webhookSecret` over the secure channel; we'll load `AI_API_KEY` from env and use the per-request `webhookSecret` as-is. Any values exchanged earlier are treated as invalid.
- **We resolve the text ourselves** from `surahNumber` + `fromAyah` + `toAyah` (no `ayahsText` expected) — confirmed.
- **Duplicate webhooks / 60-min timeout+refund / fire-and-forget / absolute audio URL** — noted; our retry logic and idempotent `jobId` echo align with all of it.
- **Cluster address** `http://quran-asr.quran-yutla.svc.cluster.local:8000` works for us; keeping traffic off the public internet is preferred.

---

## 6. Next steps

| # | Item | Owner |
| :---: | :--- | :--- |
| 1 | Q1 + Q2 answered (this doc) | ✅ AI |
| 2 | Send `AI_API_KEY` + rotated `webhookSecret` | Backend |
| 3 | Send real OVH `.webm` test URL | Backend |
| 4 | Agree T4 approach (§4) — slot vs debugging window | Both |
| 5 | Joint end-to-end with a real child recitation | Both |

Steps 2 and 4 can move in parallel. Thanks for the thorough replay — glad it matched on the first pass.
