# Quran-ARS — Full Technical Documentation

> A study reference for the Quran Automatic Recitation Scoring (ARS) system: what it does,
> every model we evaluated, the architectures involved, how speech recognition works, the
> grader, the tashkeel problem, and the two-layer product design (memorization + tajweed).
> Written to be read top-to-bottom as a learning document.

---

## 0. FINAL ARCHITECTURE (the outcome) — read this first

The long exploration below (Cohere for words, a wav2vec2 phoneme recognizer for harakat, a
CATT+canonical tashkeel hybrid) was **superseded by a single model**:

**`NAMAA-Space/Cohere-Speech-Tashkeel-2B`** — a fine-tune of Cohere's Arabic ASR that outputs the
learner's **actual diacritized recitation**, with diacritics that are **acoustically derived**
(it writes the wrong vowel/consonant the reciter actually produced — verified on QuranMB against
the ground truth: case-endings *and* consonants like kh-vs-gh). So **one model, one pass** gives:

- **Words** (memorization) → `grader.grade()` on its output (tashkeel stripped for matching).
- **Actual diacritized recitation** → shown to the learner (their real recitation, not the
  canonical — this resolves the "cheating" concern; no CATT/canonical hybrid needed).
- **Harakat/tajweed errors** → `delivery/core/harakat_grader.py` compares its acoustic diacritics
  to the canonical, per correctly-recited word, with **tajweed tolerances** (waqf: word-final
  vowel→sukun is correct; shadda ignored for idghaam; sukun==unmarked for Uthmani).

**Measured:** word accuracy 0.98 clean / 0.93 phone-noise / 0.88 kids; harakat **false-rejection
0–1.4%** (stays near-zero even under phone noise, because noise-corrupted words fail the word
match and drop out of harakat-checking rather than becoming false flags); diacritic DER ~6.6%.
**Deploy verdict: GO.** Only unknown left is real kid-phone audio (the beta/flywheel).

Wired in `delivery/`: `core/namaa_model.py` + `core/harakat_grader.py`; `main.py` `/api/evaluate`
returns word grade + `user_recitation_diacritized` + `harakat_errors`. No longer used: base
Cohere-transcribe-03-2026, the phoneme recognizer (`checkpoints_phoneme`, PER 17%, kept only as a
fallback), and CATT. Sections 4–9 below document the reasoning and dead ends that led here.

---

## Table of Contents
1. [Product vision](#1-product-vision)
2. [System architecture: the two layers](#2-system-architecture-the-two-layers)
3. [How speech recognition (ASR) works — fundamentals](#3-how-speech-recognition-asr-works--fundamentals)
4. [The models we evaluated](#4-the-models-we-evaluated)
5. [The chosen model: Cohere Transcribe (deep dive)](#5-the-chosen-model-cohere-transcribe-deep-dive)
6. [Fine-tuning with LoRA — how it works](#6-fine-tuning-with-lora--how-it-works)
7. [The grader: how "how much did he get" is computed](#7-the-grader-how-much-did-he-get-is-computed)
8. [Tashkeel (diacritics): the problem and the solution](#8-tashkeel-diacritics-the-problem-and-the-solution)
9. [Layer 2: pronunciation / tajweed error detection](#9-layer-2-pronunciation--tajweed-error-detection)
10. [Data and datasets](#10-data-and-datasets)
11. [Experiments and results — the full journey](#11-experiments-and-results--the-full-journey)
12. [Deployment](#12-deployment)
13. [Glossary](#13-glossary)
14. [References](#14-references)

---

## 1. Product vision

Quran-ARS is an **automated sheikh** for testing Quran recitation. A learner (child or
adult) records themselves reciting a known target passage and sends the audio. The system
returns:

- **(Layer 1 — memorization / hifz):** which *words* they recited correctly, and which were
  wrong, skipped, or added → an overall "how much did you get" score.
- **(Layer 2 — tajweed / harakat):** whether they pronounced the *sounds* correctly (e.g. said
  a fatha where a damma is required), and *which* sound is wrong.

A human sheikh checks both: did you remember the words, and did you pronounce them correctly.
The two jobs need **two different models**, because they are fundamentally different tasks
(see §2).

---

## 2. System architecture: the two layers

```
                     ┌─────────────────────────────────────────────────────────────┐
   child's audio ───►│                                                             │
   (phone recording) │   LAYER 1 — WORDS                LAYER 2 — SOUNDS            │
                     │   ┌───────────────┐              ┌───────────────────┐      │
                     │   │ Cohere ASR    │              │ wav2vec2 phonetic │      │
                     │   │ (words → text)│              │ (audio → phonemes)│      │
                     │   └──────┬────────┘              └─────────┬─────────┘      │
                     │          │                                 │                │
   target ayah ─────►│   align to target ayah          expected phonemes (QPS)    │
   (known, from DB)  │          │                                 │                │
                     │   per-word correct/wrong/skip   align + diff → harakat errs │
                     │          │                                 │                │
                     │          └──────────────┬──────────────────┘                │
                     │                   combined report                           │
                     └─────────────────────────────────────────────────────────────┘
```

**Why two models?**
- **Cohere** transcribes *words* extremely well but outputs **no diacritics** and does not
  "hear" which vowel was produced. Ideal for Layer 1, useless for Layer 2.
- The **phonetic wav2vec2** transcribes *actual sounds* (phonemes incl. short vowels), so it
  can tell fatha from damma. Ideal for Layer 2, but not a clean word-grader.

Both models consume the **same recording**; their outputs are merged into one report.

**Key fact that shapes everything:** ordinary text-ASR models (Cohere, Whisper) *predict* the
expected diacritics from a language model — they do **not** acoustically verify the learner's
vowel. So harakat-error detection cannot come from the text-ASR; it needs the phonetic model
(Layer 2). This is the single most important architectural insight in the project.

---

## 3. How speech recognition (ASR) works — fundamentals

All the ASR models here share the same high-level pipeline:

```
raw waveform → feature extraction → acoustic encoder → decoder/head → text or phonemes
   (16 kHz)      (log-mel spectrogram)  (neural net)     (neural net)
```

### 3.1 Feature extraction (log-mel spectrogram)
Audio is a 1-D signal sampled at 16,000 samples/second. We don't feed raw samples; we compute
a **log-mel spectrogram**: slide a short window (~25 ms) across the audio, take the Fourier
transform to get frequency content, map it onto the **mel scale** (which mimics how human
hearing spaces pitches), and take the log of the energy. Result: a 2-D image-like array of
shape `(mel_bins, time_frames)` — e.g. 128 mel bins × ~1500 frames for 30 s. This is the
standard input to Whisper and Cohere.

### 3.2 The acoustic encoder
A neural network reads the spectrogram and produces a sequence of **hidden vectors**, one per
time step, that encode "what sound is happening here." Two dominant designs:
- **Transformer encoder** (Whisper): stacks of self-attention + feed-forward layers; every
  frame can attend to every other frame.
- **Conformer encoder** (Cohere): a transformer *augmented with convolutions*, so it captures
  both local acoustic detail (convolution) and long-range context (attention). Generally
  stronger and more noise-robust for speech (see §5).

### 3.3 The decoder / output head — two families
- **Seq2seq (encoder-decoder), autoregressive** (Whisper, Cohere): a decoder generates the
  transcription one token at a time, each token conditioned on the audio (via cross-attention)
  and on previously generated tokens (via self-attention). Good at producing fluent text; can
  "hallucinate" plausible words on unclear audio.
- **CTC (Connectionist Temporal Classification)** (wav2vec2 phonetic): the model emits one
  label per audio frame plus a "blank"; a collapsing rule merges repeats and removes blanks to
  get the final sequence. No autoregression, so it is **faithful frame-by-frame** — it does not
  invent words — which is exactly what pronunciation scoring needs.

### 3.4 Tokens
Text models output **sub-word tokens** (pieces of words) that are stitched into text. The
phonetic model outputs **phoneme/character symbols** (sounds). The vocabulary is the fixed set
of tokens a model can emit (e.g. Cohere: 16,384 tokens; the phonetic model: a small phonetic
alphabet).

---

## 4. The models we evaluated

We tested many models on **our own data** (real kid phone recordings + adult everyayah +
noisy RetaSy). The decision was always driven by measured performance on *our* task, never by
a model's general reputation.

| Model | Type / size | Tashkeel | Kids t4/t5/t6 | Adults | Noisy | Speed | Verdict |
|---|---|---|---|---|---|---|---|
| whisper-base-ar-quran (**V5**, old deploy) | Whisper-base 74M | ❌ | 0.88/0.07/– | – | – | fast | replaced |
| whisper-large-v3-turbo (raw) | Whisper 809M | ✗/weak | poor on kids | – | – | 0.7 s | no |
| **turbo-Quran** (MaddoggProduction) + our LoRA | Whisper 809M | ✅ | 0.88/0.40/0.69 | 0.90 | – | 0.7 s | good, beaten |
| **IJyad full-v3-Quran** | Whisper 1.5B | ✅ native | 0.88/0.53/0.96 | 0.98 | 0.73 | 1.4 s | strong runner-up |
| **Cohere Transcribe** ← **CHOSEN** | Conformer 2B | ❌ | 0.85/0.60/1.00 | 1.00 | 0.85 | **0.2 s** | **winner** |
| Cohere + tashkeel LoRA (3 attempts) | – | partial | degraded | – | – | – | ruled out |
| wav2vec2-quran-phonetics | wav2vec2-CTC | phonemes | — | — | — | fast | Layer 2 |

Accuracy numbers are grader accuracy (0–1) on the held-out clips; "noisy" is mean accuracy on
30 noisy RetaSy phone clips. See §11 for the full story behind each row.

### 4.1 The Whisper family (V5, turbo, turbo-Quran, IJyad)
**Whisper** (OpenAI) is a transformer **encoder-decoder** ASR trained on 680k hours. Fixed
**30-second** window, log-mel input, autoregressive text decoder. Variants:
- **base** (74M) — tiny, fast, weak; our old V5 deploy used `tarteel-ai/whisper-base-ar-quran`.
- **large-v3** (1.5B) — full-size, strongest Whisper; **IJyad/whisper-large-v3-Tarteel** is this,
  fine-tuned on Quran → **native Uthmani tashkeel**.
- **large-v3-turbo** (809M) — a *distilled* large-v3 with only **4 decoder layers** (vs 32) for
  speed; `MaddoggProduction/...-turbo-quran...` fine-tuned it on Quran+tashkeel.

Insight we discovered: **turbo is the shrunk v3**, so the **full v3 (IJyad) hears kids much
better** (test5 0.40→0.53, test6 0.69→0.96) — the lead's "v3 turbo isn't great" intuition was
right, but the fix was the *bigger* v3, not a different family.

### 4.2 wav2vec2 (Layer 2)
**wav2vec2** (Meta) is a **self-supervised** model: a CNN feature encoder + transformer,
pre-trained on unlabeled audio by a contrastive task (predict masked latent speech units), then
fine-tuned with **CTC** for recognition. For Quran, models like
`TBOGamer22/wav2vec2-quran-phonetics` fine-tune it to output **phonemes**, making it the
Layer-2 workhorse.

---

## 5. The chosen model: Cohere Transcribe (deep dive)

`CohereLabs/cohere-transcribe-03-2026` — a **2.07-billion-parameter Conformer encoder-decoder**
ASR, 14 languages incl. Arabic, Apache-2.0, self-hostable.

### 5.1 Why it won
On our data it was **best or tied on every axis** *and* far faster:
- Kids: test5 **0.60**, test6 **1.00** (best of all models)
- Adults: **1.00** (15/15)
- Noisy phone: **0.85** (most robust; IJyad 0.73)
- Speed: **0.20 s/ayah** — ~7× faster than IJyad
- **Faithful**: it transcribes what was actually said and does **not** hallucinate toward the
  target (we verified on mis-recited clips — it caught the errors instead of "correcting" them).

Its only gap is **no tashkeel** (§8).

### 5.2 Architecture (from its config)

```
audio 16 kHz
  │  log-mel: 128 mel bins  (feat_in = 128)
  ▼
CONFORMER ENCODER
  • 48 layers, d_model = 1280, 8 attention heads
  • relative-position self-attention  (self_attention_model = rel_pos)
  • depthwise-striding subsampling, factor 8  (dw_striding)  → 8× time compression
  • convolution kernel size 9 (local context) + FFN expansion ×4
  ▼
encoder hidden states  ──►  (linear projection to decoder width)
  ▼
TRANSFORMER DECODER  (lightweight, autoregressive)
  • self-attention (q/k/v/o_proj) + cross-attention to encoder + FFN (fc1/fc2)
  • decoder prompt controls language/task:
      <|startofcontext|><|startoftranscript|><|emo:undefined|><|ar|><|ar|>
      <|pnc|><|noitn|><|notimestamp|><|nodiarize|>
  ▼
OUTPUT HEAD  (proj_out): Linear → 16,384-token vocab, weights TIED to decoder embeddings
  ▼
tokens → text
```

**Why Conformer is fast *and* robust:** the heavy lifting is in the **encoder** (48 Conformer
layers with convolutions capture local acoustic cues → good in noise); the **decoder is
lightweight**, so generation is cheap → the 0.2 s speed. Whisper puts more weight in the
decoder, which is slower and more prone to hallucination.

### 5.3 How inference runs (transformers ≥ 5.4)
```python
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
proc  = AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026")
model = CohereAsrForConditionalGeneration.from_pretrained(..., torch_dtype=torch.bfloat16, device_map="auto")
inp = proc(audio, sampling_rate=16000, return_tensors="pt", language="ar")
inp.to(model.device, dtype=model.dtype)
out = model.generate(**inp, max_new_tokens=256)
text = proc.decode(out, skip_special_tokens=True)   # bare Arabic words
```
**Must use bf16**, not fp16 — fp16 overflows the attention mask (`-1e9`) and produces garbage.
This was a real bug we hit and fixed.

### 5.4 Environment
Runs in conda env `mlaudio`: Python 3.11, `transformers 5.13.1`, `torch 2.11.0+cu128`
(cu128 is required for this Blackwell GPU), plus `accelerate`, `sentencepiece`, `protobuf`.

---

## 6. Fine-tuning with LoRA — how it works

We fine-tune with **LoRA (Low-Rank Adaptation)**: instead of updating all billions of weights
(expensive, and risks destroying the model), we **freeze the base model** and inject tiny
trainable matrices next to chosen weight matrices.

For a frozen weight `W` (e.g. an attention `q_proj`), LoRA adds `ΔW = B·A` where `A` is
`r×d` and `B` is `d×r` with a small **rank r** (e.g. 16–32). Only `A` and `B` train. At r=32
this is ~0.1–0.3% of parameters. `alpha` scales the update (`ΔW·alpha/r`).

- **Advantages:** tiny (a few MB adapter), fast, and *reversible* — the base model is untouched,
  so we can keep its strengths.
- **Where we target it:** attention projections `q_proj`/`v_proj` (and optionally
  `k_proj`/`o_proj` + FFN `fc1`/`fc2` for more capacity).
- **Freeze the encoder, adapt the decoder:** for teaching *output* behavior (like tashkeel) we
  LoRA only the **decoder** and keep the **encoder frozen** to preserve acoustic robustness.

The Whisper fine-tune (`finetuning/finetune.py`) and the Cohere fine-tune
(`finetuning/finetune_cohere.py`) both use LoRA via the `peft` library.

---

## 7. The grader: how "how much did he get" is computed

File: `delivery/core/grader.py` (`QuranGrader`). Given the ASR transcription and the target
ayah text, it scores word-level correctness — **tolerantly**, because ASR and recitation vary.

1. **Normalization** — strip tashkeel and unify letter forms (`أإآٱ→ا`, `ة→ه`, `ى→ي`, remove
   hamza carriers, drop tatweel). This makes matching about *words*, not diacritics or spelling
   variants. (This is also *why* tashkeel doesn't affect the score.)
2. **Alignment** — **Needleman-Wunsch** global sequence alignment between the recited words and
   the canonical words: finds the best word-to-word correspondence allowing insertions,
   deletions, and substitutions (the classic dynamic-programming alignment from bioinformatics).
3. **Fuzzy match** — near-miss words are compared by **CER (Character Error Rate)** so a small
   mispronunciation still counts as the intended word rather than a hard miss.
4. **Score** — accuracy = fraction of canonical words correctly matched; the per-word
   correct/wrong/skipped labels drive the highlighted feedback.

**Threshold caveat:** we found RetaSy's own labels are noisy (many "incorrect" clips are
actually near-correct), so a clean pass/fail threshold can't be calibrated from RetaSy alone —
a cleaner test set is needed. This is a known open item, independent of model choice.

---

## 8. Tashkeel (diacritics): the problem and the solution

### 8.1 The problem
The client wants the learner's recitation shown **with tashkeel** ("it's Quran"). But:
- **Cohere outputs no tashkeel.**
- We tried to **fine-tune tashkeel into Cohere** — *3 attempts* (Uthmani targets, simplified
  standard-harakat targets, wide LoRA). All failed the same way: **more tashkeel came only at
  the cost of word accuracy** (test6 collapsed 1.0→0.15 in the aggressive run). Cause: Cohere's
  pretrained *undiacritized-Arabic prior* is so strong that forcing diacritics out of it breaks
  transcription; also Uthmani marks tokenize as awkward 3-token byte sequences. **Conclusion:
  Cohere cannot be cleanly fine-tuned to emit tashkeel.**

### 8.2 The critical realization
**No text-ASR "hears" harakat** — Cohere, IJyad, or a fine-tuned model all *predict* the most
likely marks from their language model. They never verify the learner's actual vowel. So
diacritics on the learner's output are, at best, a rendering of the *words* — not evidence of
correct pronunciation. (This is what makes Layer 2 a separate system.)

### 8.3 The v1 solution: canonical-hybrid diacritization (Quran-safe)
Because we know the **target ayah**, we diacritize by alignment, not by guessing:
- **Correctly-recited words** → copy the **exact canonical mushaf** text from `quran.db`
  (100% correct Uthmani script — we never "play with" scripture).
- **Mis-recited words** → fall back to **CATT** (SOTA Arabic diacritizer, Apache-2.0) on the
  actual spoken word, and **highlight it as an error**.

Result on our data: adults came back as **verbatim mushaf** (exact match), and a correctly
recited kids' surah was **25/27 words verbatim mushaf**. We verified that CATT *alone* is
**not** Quran-safe (it mis-diacritizes ~5–7% of scripture words, e.g. `الصَّلِحَتِ`→`الصُّلْحَتَ`),
which is exactly why we only use it for genuine errors.

Prototype: `finetuning/hybrid_tashkeel.py`. CATT via `pip install catt-tashkeel`.

### 8.4 The honest scope of v1 tashkeel
v1 tashkeel is a faithful **word-level** display (correct words shown correctly, errors marked).
It does **not** reveal a wrong *vowel* on an otherwise-correct word — that is Layer 2.

---

## 9. Layer 2: pronunciation / tajweed error detection

Layer 2 answers "did he say it with the right harakat, and which is wrong." It is a **separate
model** alongside Cohere, **not a replacement**.

### 9.1 The proven approach (from the literature)
```
child audio → Quran PHONETIC recognizer (wav2vec2-CTC) → phonemes the learner ACTUALLY said
target ayah → Quran G2P / QPS → expected phoneme sequence (correct harakat + tajweed)
align + diff → each mismatch = a pronunciation error → diagnose ("said fatha, expected damma")
```
- **Phonetic recognizer:** CTC is *faithful frame-by-frame* (no hallucination) — it transcribes
  the produced sounds, so a fatha vs damma difference shows up as different phoneme symbols.
- **QPS (Quran Phonetic Script):** encodes Arabic letters + short/long vowels at the phoneme
  level and articulation (Sifa) — the SOTA multi-level CTC system reached **0.16% PER** on
  expert data.
- **Alternative/complement — GOP (Goodness of Pronunciation):** force-align expected phonemes to
  the audio and score each one's acoustic likelihood; flag those below a threshold.

### 9.2 Building blocks that already exist (don't start from zero)
- **Models:** `TBOGamer22/wav2vec2-quran-phonetics` (phonetic output, 99.8% on clean words),
  `IbrahimSalah/Wav2vecLarge_quran_syllables_recognition`.
- **Benchmark / shared task:** **Iqra'Eval (ArabicNLP 2025)** — standardized Quran
  pronunciation-assessment task with data + baselines + published systems.
- **Datasets:** **QDAT** (1500 correct/incorrect recitations across 3 tajweed rules: Al-Mad,
  Ghunnah, Ikhfaa), the 870-hr QPS corpus, and our own **RetaSy** (ayah-level labels).

### 9.3 Feasibility findings (measured)

We tested `TBOGamer22/wav2vec2-quran-phonetics` directly:

- **Phoneme vocabulary distinguishes harakat** — 40-symbol romanized set with distinct short
  vowels `a/i/u` (fatha/kasra/damma), long vowels `ā/ī/ū`, and emphatics/pharyngeals
  (`ḍ ḥ ṣ ṭ ẓ ʿ`). The representation *can* express a fatha-vs-damma difference. ✅
- **On isolated words (its training domain, `Buraaq/quran-md-words`) it is near-perfect** —
  outputs match the reference transliteration almost exactly, vowels included
  (`رَبِّ`→`rabbi`, `الْعَالَمِينَ`→`l-ʿālamīna`). And it reported a genuine vowel difference
  (`الرَّحِيمِ` heard as `l-raḥīmu` vs reference `l-raḥīmi`) — i.e. the harakat-error signal works. ✅
- **On connected full ayahs, free recognition degrades badly** (drops/garbles phonemes) — it was
  trained on single words, so continuous recitation is out of domain. ❌
- **Silence-splitting does not segment continuous recitation into words** (few pauses). ❌

**Conclusion:** the core capability is proven; the model must be fed **word-level audio**. So the
build hinges on **word segmentation** of continuous recitation, then per-word phonemization, then
comparison to the expected `word_tr` (available for the whole Quran from `Buraaq/quran-md-words`).

**Prototype result (`finetuning/layer2_pronunciation.py`):** a full pipeline was built —
CTC **forced alignment** of the recitation to the expected phonemes → per-word time spans →
error detection (both slice re-decode and **GOP**) → harakat-vs-letter diagnosis. It runs
end-to-end, **but the false-positive rate is unacceptable**: on a *correctly*-recited clip it
flagged ~25/26 words. Causes: (1) this phonetic model is trained on **isolated words**, so its
emissions on **connected** recitation are unreliable (top prediction is often blank/`[PAD]` at
aligned frames); (2) **CTC emissions are peaky** (mostly blank + sharp spikes), which breaks
naive frame-averaged GOP; (3) alignment on out-of-domain audio is imprecise. **Takeaway:** the
*architecture* is validated, but a deployable checker needs a phonetic model trained on
**connected recitation** (Iqra'Eval / QPS 850-hr systems), a **proper CTC-GOP** formulation, and
**validation on QDAT/Iqra'Eval** to drive false positives near zero. This is multi-week R&D —
**not co-deployable with Layer 1 on a short timeline.**

### 9.4 The official Iqra'Eval foundation (what we build on)

Rather than reinvent, Layer 2 builds on the **Iqra'Eval** ecosystem (ArabicNLP 2025 / IQRA 2026
shared task) — the first open benchmark for Qur'anic Mispronunciation Detection & Diagnosis (MDD):

- **Training data** — `IqraEval/Iqra_train` (79 hrs; columns `audio`, `phoneme_ref` [canonical],
  `phoneme_aug` [for injecting mispronunciations], `sentence`, `tashkeel_sentence`; MSA connected
  reading from Common Voice) + `IqraEval/Iqra_TTS`.
- **Benchmark** — `IqraEval/QuranMB.v2` (1642 real recitations, audio) + `IqraEval/test_references`
  (expected phoneme string per clip) + `IqraEval/IqraEval_Test_GT` (`labels_test.csv`, the human
  ground-truth of what was *actually* pronounced) → enables measuring detection accuracy.
- **Baseline models** — `IqraEval/Iqra_{wav2vec2,hubert,wavlm,mhubert}_base` (speech-to-phoneme,
  raw checkpoints loaded via their Colab), plus `fryad-yaseen/quran-phoneme-ctc-*-v2`.
- **Phoneme scheme** — space-separated Buckwalter-style (`< a l < aa n a q A d i ...`:
  `aa`=long-ā, `E`=ʿayn, `$`=shīn, `nn`=shadda-noon, `<`=hamza) with short/long vowels ⇒ **encodes
  harakat**. Official evaluation API + metrics provided for reproducibility.

### 9.5 Plan and risks

1. **Phoneme recognizer for connected speech** — either fine-tune a standard `wav2vec2-CTC` on
   `Iqra_train` (clean, HF-loadable, matches the scheme) or adapt the **Whisper-large-v3
   speech-to-phoneme** approach (ANPLers, Iqra'Eval 2025 — Whisper handles connected recitation
   well). Replaces the word-trained model that failed the prototype.
2. **MDD + validation harness** — run the recognizer on `QuranMB.v2` → predicted phonemes →
   align to `test_references` → detected errors → score against `IqraEval_Test_GT` with the
   official metric. **Measure the false-rejection rate first** (wrongly flagging correct
   recitation is the cardinal sin on scripture).
3. **Harden for kid/phone audio** — the biggest risk: `Iqra_train`/QuranMB are adult MSA reading,
   not kids on phones. Expect a domain gap; likely need fine-tuning and/or collecting labeled kid
   mispronunciation data.
4. **Diagnosis + feedback + actual-harakat display** — report which phoneme/harakat is wrong,
   and use the recognizer's output to show the learner's *actual* diacritized recitation.

**Reality:** Layer 2 is a **multi-week R&D project**, not a wire-up — but the foundation
(benchmark, ground truth, training data, phoneme scheme, baseline models, eval API) is fully in
hand, so the path is concrete.

### 9.6 Training progress (our recognizer)

`finetuning/train_phoneme_recognizer.py` fine-tunes **wav2vec2-xls-r-300m + CTC** on the full
79-hr `Iqra_train` in the 68-phoneme IqraEval scheme. Full 8000-step run (loss 26 → 0.66): **PER improved 24.7% (ckpt-1000) → 17.2% (ckpt-8000)** on
QuranMB — a working *connected-speech* recognizer whose output tracks the reference (remaining
errors are mostly short-vowel a/u/i case-endings, i.e. the harakat distinctions). This validates
the approach vs. the failed word-trained prototype (§9.3). Training is **crash-resilient**
(`finetuning/run_phoneme_training.sh` auto-resumes from the latest checkpoint after transient CUDA
faults; saves every 500 steps).

**Honest status:** 17.2% PER is a solid foundation but **not yet deployment-safe** — ~1/6 phonemes
misrecognized would raise too many false rejections on scripture (SOTA is 0.16% PER with far more
data + a multi-level scheme). Levers to lower it: train longer/more epochs, beam search + phoneme
LM decoding, the QPS multi-level (Sifa) scheme or Whisper-S2P, then measure the official
**false-rejection rate** via the gated `IqraEval/IqraEval_Test_GT`, and harden for kid/phone audio.

---

## 10. Data and datasets

- **`data/quran_dataset_v6_allages`** — 4,348 (audio → diacritized text) pairs used for
  fine-tuning: 2,635 **Minshawi children's chorus** + 571 each of adult **Husary / AbdulBasit /
  Minshawy** (everyayah). All targets carry tashkeel; the Basmala was normalized. `audio` is a
  file path; `text` is the diacritized ayah; `reciter` labels the source.
- **`data/quran.db`** — the full diacritized Quran (Uthmani) used for canonical lookup and as
  grading targets; `get_ayah_range()` strips presentation-form markers (U+FC00–U+FDFF).
- **RetaSy** — external benchmark of real (often noisy, non-native, phone) recitations with
  ayah-level labels (correct / in_correct / not_match_aya / in_complete / not_related_quran).
  Great for realism, but its labels are **noisy** — used for *relative* comparison, not absolute
  thresholds.
- **Test clips** — `finetuning/test_samples/test 4/5/6.mp4`: real kid recitations of Surah
  95 / 112 / 109 used as the human-audible kid benchmark throughout.

**Known data limitation:** our kid training data is **chorus** (many children together), not
**solo phone** kids — which is why fine-tuning on it tends to be a *wash* for the solo-phone
deployment domain. The real fix is **real user audio** from the beta (the data flywheel).

---

## 11. Experiments and results — the full journey

A record of what we tried and what we learned (including dead ends — they are results too).

1. **V5 baseline (deployed):** whisper-base + old adapter. No tashkeel, kids test5 ≈ 0.07.
   Motivated the whole model search.
2. **turbo-Quran + all-ages LoRA:** first real win — tashkeel, kids test5 0.07→0.40, adults
   0.90. Checkpoint-800 beat the final checkpoint-1600 (overfitting) — confirming the
   "save several checkpoints, keep the best" instinct.
3. **Denoising (noisereduce, DeepFilterNet):** **ruled out** — both *hurt*, even on noisy
   RetaSy (0.53→0.39). Front-end denoising removes signal the model needs.
4. **Phone augmentation (reverb + pink noise + phone-mic lowpass, anti-silence guarded):**
   **a wash** — RetaSy 0.53→0.54, kids mixed. Synthetic noise ≠ real phone noise, and turbo was
   already noise-tolerant. (A real silent-augmentation bug was found and fixed first: peak-match
   only reduced, never boosted; guard added; verified 0/200 silent.)
5. **IJyad full-v3-Quran:** raw (no fine-tune) **beat our fine-tuned turbo** on kids AND adults
   (0.88/0.53/0.96, adults 0.98) with native tashkeel — the "use the bigger v3" lesson.
6. **Cohere Transcribe:** after fixing an fp16→bf16 inference bug, it **won every axis** and was
   7× faster and faithful (§5). Chosen as the ASR.
7. **Cohere tashkeel fine-tune ×3:** **ruled out** — fundamental accuracy↔tashkeel tradeoff
   (§8.1).
8. **CATT + canonical hybrid:** **the tashkeel solution** — Quran-safe diacritization for
   display (§8.3).
9. **Layer 2 research:** established the phonetic-ASR + QPS approach, found downloadable models,
   the Iqra'Eval benchmark, and QDAT (§9).

---

## 12. Deployment

- **Artifact:** the `delivery/` service is the deploy target (FastAPI). It exposes an async
  `POST /api/evaluate` for the backend, runs the grading pipeline, and calls back a webhook.
- **v1 (Layer 1) target pipeline:** Cohere ASR (bf16, GPU) → grader (align to target ayah) →
  per-word result + canonical-hybrid tashkeel display. Deps to add: `transformers>=5.4`,
  `catt-tashkeel`; runs on a GPU (T4/A10G fine; 2B model ≈ 4–5 GB VRAM).
- **Env note:** local testing is on a Blackwell GPU (RTX 5060 Ti, 16 GB) with torch cu128 /
  py3.11; the historical delivery stack targeted cu118/py3.10 — mind the mismatch when
  reproducing.
- **Status:** Layer 1 is functionally ready; **deploy is on hold until the owner gives the go.**

---

## 13. Glossary

- **ASR** — Automatic Speech Recognition (audio → text).
- **Tashkeel / harakat** — Arabic diacritics (short vowels: fatha=a, damma=u, kasra=i, plus
  shadda, sukun, tanwin). **Uthmani script** is the specific mushaf orthography (wasla `ٱ`,
  dagger-alef `ٰ`, etc.).
- **Hifz** — memorization of the Quran. **Tajweed** — the rules of correct pronunciation.
- **Conformer** — convolution-augmented transformer encoder (Cohere's encoder).
- **CTC** — Connectionist Temporal Classification: per-frame labeling + collapse; faithful, no
  hallucination (wav2vec2).
- **Seq2seq / autoregressive** — encoder-decoder that generates tokens one at a time (Whisper,
  Cohere).
- **LoRA** — Low-Rank Adaptation: freeze the base, train tiny injected matrices.
- **CER / PER / WER** — Character / Phoneme / Word Error Rate.
- **GOP** — Goodness of Pronunciation: acoustic-likelihood pronunciation score.
- **QPS** — Quran Phonetic Script: phoneme-level encoding of Quran incl. tajweed.
- **Needleman-Wunsch** — dynamic-programming global sequence alignment (used by the grader).
- **mel spectrogram** — perceptually-scaled time-frequency representation of audio.

---

## 14. References

**Chosen / evaluated models**
- Cohere Transcribe — https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
- IJyad full-v3-Quran — https://huggingface.co/IJyad/whisper-large-v3-Tarteel
- turbo-Quran — https://huggingface.co/MaddoggProduction/whisper-l-v3-turbo-quran-lora-dataset-mix
- Quran phonetic wav2vec2 — https://huggingface.co/TBOGamer22/wav2vec2-quran-phonetics

**Tashkeel**
- CATT: Character-based Arabic Tashkeel Transformer — https://arxiv.org/abs/2407.03236 · code https://github.com/abjadai/catt
- CATT-Whisper (multimodal, uses audio) — https://github.com/abjadai/catt-whisper

**Layer 2 — pronunciation / tajweed error detection**
- Mispronunciation Detection of Basic Quranic Recitation Rules (QDAT) — https://arxiv.org/abs/2305.06429
- Automatic Pronunciation Error Detection & Correction for Quran learners — https://arxiv.org/pdf/2509.00094
- A Critical Review of Knowledge-Centric Evaluation of Quranic Recitation (QPS / 850-hr) — https://arxiv.org/pdf/2510.12858
- Iqra'Eval shared task (ArabicNLP 2025) — https://aclanthology.org/2025.arabicnlp-sharedtasks.66.pdf
- Automatic Pronunciation Assessment — A Review (GOP) — https://arxiv.org/html/2310.13974
- Diacritic Recognition Performance in Arabic ASR — https://arxiv.org/abs/2302.14022

**Foundational architectures**
- Whisper (Robust Speech Recognition via Large-Scale Weak Supervision) — https://arxiv.org/abs/2212.04356
- Conformer (Convolution-augmented Transformer for Speech) — https://arxiv.org/abs/2005.08100
- wav2vec 2.0 (Self-Supervised Learning of Speech Representations) — https://arxiv.org/abs/2006.11477
- LoRA (Low-Rank Adaptation of Large Language Models) — https://arxiv.org/abs/2106.09685

---

*This document reflects the state of the project as of the model-selection and Layer-2
feasibility phase. Layer 1 (Cohere + grader + canonical-hybrid tashkeel) is ready pending the
owner's deploy signal; Layer 2 (phonetic pronunciation scoring) is in feasibility prototyping.*
