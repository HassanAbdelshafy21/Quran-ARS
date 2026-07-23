# Integration Verification — Backend → AI

**From:** Backend team (Quran Yutla)
**To:** AI service (Quran-ASR)
**Re:** `BACKEND_HANDOFF.md`
**Status:** ✅ Verified compatible — your documented payloads were replayed verbatim against a running backend

---

## 1. TL;DR

- We took the exact JSON from §3 of your handoff and POSTed it to a live backend. **Success and error paths both work end to end.**
- We made one change on our side as a direct result (§3) — no action needed from you, but please read it, because it makes the 0–100 contract load-bearing.
- **2 questions** we need answered before the joint test (§4).
- **3 items acknowledged**, no action needed (§5).
- **3 things we owe you** (§6).

Nothing in your implementation needs to change based on our testing.

---

## 2. What we verified

Replayed against a running backend + PostgreSQL, using your payloads character-for-character:

| Test | Result |
| :--- | :--- |
| Your success payload (§3) | ✅ recitation → `completed`, `evaluation_score = 87.50` |
| Arabic text inside `words[]` | ✅ preserved intact (`بسم`), no encoding damage |
| Nested `charErrors[]` | ✅ stored and queryable |
| `errorSummary` object | ✅ `{"deletion":1,"insertion":0,"substitution":1}` |
| `feedbackAudio` absolute URL | ✅ stored as sent |
| `suggestions` **absent** | ✅ no breakage — we don't read it |
| Your error payload (§3) | ✅ recitation → `failed`, student's session refunded |
| **Duplicate of the same webhook** (your retry) | ✅ **no double refund** — safe to retry |
| Bad / missing `Authorization` | ✅ `401` |
| Extra top-level field | ✅ `400` (as your §3 note expects) |

We store your entire `data` object verbatim in a JSONB column. Add fields freely — nothing inside `data` is validated or dropped.

---

## 3. Change we made on our side (FYI — affects you indirectly)

We previously had a **defensive auto-correction**: any `overallScore` below 1 was multiplied by 100, on the assumption it might be a 0–1 ratio.

Now that you've confirmed and tested the 0–100 scale, **we removed that correction**, because it was doing more harm than good:

```
Before:  you send 0.5  (meaning 0.5%)  →  we stored 50.00   ← 100× wrong
After:   you send 0.5                  →  we store  0.50    ← correct
```

This matters for long ayah ranges. A student who gets 1 word right out of 250 scores 0.4 — a legitimate value under 1 — and our old code would have recorded it as a 40% pass.

**What this means for you:** there is no longer any safety net on the scale. If a future change ever emits a 0–1 ratio, the values go into the database as-is and every score in the product becomes ~100× too small.

We kept a loud warning as the tripwire:

```
AI returned score 0.5 for recitation 10. Storing as-is (0-100 scale assumed).
If EVERY score looks like this, the AI service is sending a 0-1 ratio and must be fixed.
```

Verified after the change: `87.5→87.50` · `100→100.00` · `0.5→0.50` · `0→0.00` · `3.45→3.45`

---

## 4. Two questions — please answer before the joint test

### Q1. Do you treat **any 2xx** as success, or specifically `200`?

Our webhook returns **`201 Created`**, not `200` — that's the NestJS default for `POST`.

Your §3 says *"3 attempts on 5xx/timeout only; 4xx is not retried"*, which doesn't state what counts as success. If the check is `status == 200`, you'll retry every successful delivery three times.

Please confirm the condition is `status < 400` (or `2xx`).

### Q2. Can you verify the score scale with a **mid-range** result?

§4 records `overallScore` verified as `100.0` from a perfect recitation. A perfect recitation is the weakest possible evidence for a scale check, since `1.0` and `100.0` are the two values most likely to look plausible either way.

Could you run one deliberately flawed recitation — say 5 correct words out of 8 — and confirm the payload carries something like `62.5` and **not** `0.625`? A screenshot or the raw JSON line is enough.

---

## 5. Acknowledged — no action needed

**`suggestions` not produced.** Fine. Our backend never reads it; verified explicitly. We'll tell the mobile team not to assume the key exists. If it's ever added later, no backend change is required.

**Timestamps are approximate (evenly spread).** Understood and acceptable for word highlighting. We'll set expectations with the app team that it will drift on longer passages.

**Feedback audio retained ≥ 30 days.** Good — this lines up with our own retention: we delete recitation audio after 30 days via a nightly job, so the two windows match. No mismatch to resolve.

---

## 6. What we owe you (your §9)

| Item | Status |
| :--- | :--- |
| `AI_API_KEY` | ✅ Generated and configured on our side — sent to you separately, not in this document |
| `webhookSecret` | **Being rotated.** The value in the earlier spec draft was exposed in a chat thread; treat any secret you already have as invalid |
| Real OVH `audioUrl` to a `.webm` test file | Pending storage credentials on our side; will follow |
| Internal cluster address | Will confirm — likely `http://quran-asr.quran-yutla.svc.cluster.local:8000` if you deploy in our namespace, which keeps the traffic off the public internet entirely |

---

## 7. Our side's behaviour you can rely on

Worth knowing when you're debugging a live run:

- **We call you fire-and-forget.** The student's upload returns immediately; we never block on your response. A slow evaluation costs nothing on our side.
- **We send the audio as an absolute public URL** — you download it directly, no credentials needed.
- **We do not send `ayahsText`.** You resolve the text yourself from `surahNumber` + `fromAyah` + `toAyah`, as you've implemented.
- **If your callback never arrives**, we mark the recitation `failed` after **60 minutes** and refund the student's session automatically. So a dropped webhook degrades gracefully — it doesn't strand the student or cost them a paid session.
- **Duplicate webhooks are safe.** We only act on the first transition into a terminal state; a retry that arrives after we've already recorded the result is accepted and ignored. Retry freely.
- **We return `201`** on success (see Q1), `401` on a bad secret, `404` if `recitationId` + `jobId` don't match a known record, `400` if the body shape is wrong.
- **A `404` from us means the pairing is wrong**, not that the recitation vanished — check that the `jobId` in the webhook is byte-identical to the one you returned from `/api/evaluate`.

---

## 8. One risk we'd flag back to you

Your §8 notes that local testing used a newer torch/GPU than the T4 deployment target, with the version-exact run still pending.

That's the single most likely source of a surprise at deploy time — pinned `transformers==4.46.3` / `peft==0.19.1` constrain the library versions but not the CUDA/torch/driver combination. If you can get a T4 run done before the joint test, it would de-risk the schedule considerably. If it has to wait, we'd rather know now so we plan for a debugging window.

---

## 9. Proposed next steps

1. You answer **Q1** and **Q2** (§4) — both are quick.
2. We send `AI_API_KEY` + rotated `webhookSecret` over a secure channel.
3. We send a real OVH `.webm` URL once storage is provisioned.
4. You run the T4 version-exact check if possible (§8).
5. Joint end-to-end test with a real child recitation.

Steps 1 and 2 can happen in parallel today. Nice work on the turnaround — the contract matched on the first pass, which is rare.
