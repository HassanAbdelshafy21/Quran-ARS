# Quran-ARS — A Learning Guide

> A teach-yourself guide to how this project works, from the ground up: how a computer
> "hears" speech, how the data was built, how the models work, how the final system grades
> a child's Quran recitation (words **and** tajweed), and the lessons learned along the way.
> Read it top to bottom. No prior ML knowledge assumed; concepts are explained as they appear.
>
> Companion: `Quran-ARS-Technical-Documentation.md` is the terse reference + full experiment log.
> This document is the *why* and the *intuition*.

---

## Table of contents
1. [The problem in one page](#1-the-problem-in-one-page)
2. [Foundations: how a computer hears speech](#2-foundations-how-a-computer-hears-speech)
3. [How speech recognition (ASR) works](#3-how-speech-recognition-asr-works)
4. [The Arabic/Quran domain: tashkeel, tajweed, waqf](#4-the-arabicquran-domain-tashkeel-tajweed-waqf)
5. [The data — where it comes from and how pairs are built](#5-the-data--where-it-comes-from-and-how-pairs-are-built)
6. [The models, and how each one works](#6-the-models-and-how-each-one-works)
7. [The key insight: acoustic vs. language-model diacritics](#7-the-key-insight-acoustic-vs-language-model-diacritics)
8. [The final system, end to end](#8-the-final-system-end-to-end)
9. [The grading algorithms, explained](#9-the-grading-algorithms-explained)
10. [How we measure quality (the metrics)](#10-how-we-measure-quality-the-metrics)
11. [Lessons learned (what failed, and why)](#11-lessons-learned-what-failed-and-why)
12. [How to run and reproduce it](#12-how-to-run-and-reproduce-it)
13. [Glossary](#13-glossary)
14. [Where to learn more](#14-where-to-learn-more)

---

## 1. The problem in one page

We are building an **automatic sheikh**: a child records themselves reciting a known Quran
passage, and the app must judge two things a human teacher judges:

1. **Memorization (hifz):** did they recite the **right words**, in order, without skipping or
   adding? — a *word-level* question.
2. **Pronunciation (tajweed):** did they say each sound correctly, including the short vowels
   (**harakat**) and case-endings? — a *sound-level* question.

Why is this hard?
- **Children's voices** are acoustically different from adults (higher pitch, less stable).
- **Phone recordings** are noisy and compressed.
- **Short vowels in Arabic are not written** in normal text and are acoustically brief — even
  humans need clear audio to judge them.
- Most important: a mistake on the Quran must **never** be reported when the recitation was
  actually correct (a "false rejection" on scripture is unacceptable).

The whole project is a search for the right **model** (to turn audio into text) and the right
**grading logic** (to compare that text to the correct Quran) that satisfy both questions safely.

---

## 2. Foundations: how a computer hears speech

### 2.1 Sound is just numbers
A microphone measures air-pressure many thousands of times per second. We store **16,000
samples per second** (16 kHz) — each sample a number. So 10 seconds of audio = 160,000 numbers.
That raw list is the **waveform**.

### 2.2 We don't feed the raw waveform — we make a "picture" of it
Raw samples are hard for a model to use directly. Instead we compute a **spectrogram**: slide a
small window (~25 ms) across the audio and, for each window, use a **Fourier transform** to ask
"how much of each frequency (pitch) is present here?" Stack those windows and you get a 2-D
image: time on one axis, frequency on the other, brightness = energy.

We then squeeze the frequency axis onto the **mel scale** (which spaces pitches the way human
hearing does — we distinguish low pitches finely and high pitches coarsely) and take a log of
the energy. The result is a **log-mel spectrogram** — the standard input to modern speech models.
Think of it as a heat-map of "which sounds happen when." A 30-second clip becomes roughly a
`128 × 1500` array (128 mel bands × 1500 time frames).

**Intuition:** we turned "hearing" into "looking at a picture of the sound," which neural
networks are very good at.

---

## 3. How speech recognition (ASR) works

Every ASR model here follows the same shape:

```
waveform → log-mel spectrogram → ENCODER → DECODER/HEAD → text (or phonemes)
```

### 3.1 The encoder — "what sound is happening"
A neural network reads the spectrogram and outputs a sequence of **hidden vectors**, one per
time frame, each summarizing the sound around that moment. Two designs matter here:
- **Transformer encoder** (used by Whisper): every frame can "attend to" every other frame
  (self-attention), capturing long-range context.
- **Conformer encoder** (used by Cohere/NAMAA): a Transformer **plus convolutions**.
  Convolutions capture *local* detail (the exact shape of a consonant); attention captures
  *global* context. This combination is especially strong and **noise-robust** for speech.

### 3.2 The decoder/head — "turn sound into symbols"
Two families:

- **Autoregressive seq2seq** (Whisper, Cohere, NAMAA): a decoder generates the transcription one
  token at a time. Each new token looks at (a) the audio, via *cross-attention*, and (b) the
  words generated so far, via *self-attention*. This produces fluent text — but on unclear audio
  it can **hallucinate** a plausible word that wasn't said.

- **CTC** (Connectionist Temporal Classification, used by wav2vec2): the model outputs one label
  per frame plus a "blank," then a simple rule collapses repeats and removes blanks. Example:
  frames `b b _ a a _ t → "bat"`. CTC is **faithful frame-by-frame** — it doesn't invent words —
  which is exactly what pronunciation checking needs. Its weakness: outputs are "peaky" (mostly
  blank with sharp spikes), which complicates some downstream tricks.

### 3.3 Tokens
Text models emit **sub-word tokens** (word pieces) stitched into text. Phoneme models emit
**sound symbols**. The set of possible outputs is the **vocabulary** (e.g., NAMAA: 16,384 text
tokens; our phoneme experiment: 68 phoneme symbols).

---

## 4. The Arabic/Quran domain: tashkeel, tajweed, waqf

You must understand the domain to understand the grading.

- **Consonants and long vowels** are written in normal Arabic (`كتب` = k-t-b).
- **Tashkeel / harakat** = the small marks for **short vowels** and more:
  - fatha ( َ ) = short "a", damma ( ُ ) = short "u", kasra ( ِ ) = short "i"
  - sukun ( ْ ) = no vowel, shadda ( ّ ) = doubled/geminated consonant
  - tanwin ( ً ٌ ٍ ) = "-an / -un / -in" endings
- **Iʿrab (case-endings):** the **final** vowel of a word often changes with grammar
  (subject vs. object). Getting these right is a core recitation skill — and a common mistake.
- **Uthmani script:** the specific mushaf orthography (e.g. wasla-alef `ٱ`, dagger-alef `ٰ`,
  small silent marks). It looks slightly different from "standard harakat" but says the same
  sounds. Our canonical text (`quran.db`) is Uthmani; the model outputs standard harakat — so we
  **normalize** between them when comparing (see §9).
- **Tajweed** = the rules of correct pronunciation. Two that dominate the grading:
  - **Waqf (pausing):** when a reciter **stops** at the end of a phrase, the final vowel is
    dropped and becomes **sukun**. `تَعْبُدُونَ` (…-na) becomes `تَعْبُدُونْ` (…-n) at a pause.
    **This is correct** — not a mistake. Any grader that flags it is wrong.
  - **Idghaam (merging):** some adjacent letters merge, producing a **shadda**. Also not a
    mistake by itself.

These two rules are why naive harakat grading over-flags, and why our grader has explicit
**tolerances** for them (§9.3).

---

## 5. The data — where it comes from and how pairs are built

Every ASR/diacritization model learns from **(audio, text) pairs**. Here is where ours came from
and how such pairs are constructed — this is the part people underestimate.

### 5.1 The datasets we used
- **everyayah** — professional adult reciters (Husary, AbdulBasit, Minshawy), one clean audio
  file per ayah, with the exact ayah text. High quality, adult, studio. Our `quran_dataset_v6`
  used 571 ayahs each from three reciters + 2,635 clips of a **children's Minshawi chorus**
  → 4,348 diacritized pairs total.
- **RetaSy** — real, often noisy, non-native, phone recordings with ayah-level labels
  (correct / incorrect / …). Great for *realism testing*, but its labels are **noisy** (many
  "incorrect" clips are actually fine), so we used it only for *relative* comparison.
- **Buraaq/quran-md-words** — Quran audio segmented to the **single word**, each with a
  transliteration (`word_tr`). Used to test a phoneme recognizer and as a per-word reference.
- **IqraEval** ecosystem (the pronunciation-assessment benchmark): `Iqra_train` (79 hrs of MSA
  reading with **phoneme** transcriptions), `QuranMB.v2` (1,642 test recitations), and
  `IqraEval_Test_GT` (human ground-truth of what was *actually* pronounced). This is what let us
  **measure** pronunciation-error detection objectively.
- **Our own test clips** — real kid recitations of Surah 95/112/109 (`test 4/5/6.mp4`), the
  human-audible benchmark we kept returning to.

### 5.2 How an (audio → text) pair is actually built
Professional per-ayah files already come paired. But when you have a long recording (a whole
surah, or a chorus), you must **segment and align** it into per-ayah or per-word pairs. The
pipeline (see `finetuning/extract_child_segments.py`, `align_child_segments.py`):

1. **Silence-split** the long audio into candidate chunks (librosa detects gaps).
2. **Merge/trim** chunks to sensible durations (1–15 s).
3. **Transcribe** each chunk with a strong model (e.g. Whisper-large-v3-turbo).
4. **Align to the known text** by comparing the transcription to each candidate ayah using
   **CER** (character error rate) within a small position window, and keep the match only if the
   CER is below a threshold (e.g. 0.5). This throws away bad segments automatically.
5. For **children specifically**, we also used **pitch (F0)** to separate a child's voice from a
   teacher's when both are present.

**Why the care?** Garbage pairs poison training. A famous failure earlier in the project: an
audio-augmentation step accidentally produced **silent** clips; if you don't catch that, you
"train" on nothing and the model gets worse. Rule learned: *always listen to / verify the data
before training.*

### 5.3 The phoneme "alphabet"
For pronunciation work, text isn't enough — you need **phonemes** (sounds). The IqraEval scheme
writes each sound explicitly, e.g. `< i nn a m aa y a x $ …`:
- short vowels `a i u`, long vowels `aa ii uu`
- doubled (shadda) forms `nn bb $$`
- special consonants: `E` = ʿayn, `x` = kh, `g` = gh, `$` = sh, `<` = hamza
This alphabet **can express a fatha-vs-damma difference**, which is the whole point of tajweed
grading.

---

## 6. The models, and how each one works

We evaluated many models on **our own data** — decisions were always driven by measured results,
never a model's reputation. Here's each, with how it works.

### 6.1 Whisper (OpenAI) — the baseline family
Transformer **encoder-decoder**, trained on 680k hours, fixed **30-second** window. Variants:
- **base** (74M): tiny/fast/weak — our original deployed V5 (no tashkeel, kids ≈ 0.07).
- **large-v3** (1.5B): full-size, strongest. `IJyad/whisper-large-v3-Tarteel` is this,
  fine-tuned on Quran → native Uthmani tashkeel.
- **large-v3-turbo** (809M): a **distilled** large-v3 (only 4 decoder layers) for speed.
- Lesson: turbo is the *shrunk* v3; the **full v3 hears kids much better** (test5 0.40→0.53).

### 6.2 Cohere Transcribe — a Conformer ASR
`cohere-transcribe-03-2026`, ~2B params: **48-layer Conformer encoder** + a lightweight
Transformer decoder. Because the heavy lifting is in the encoder (convolutions → good in noise)
and the decoder is small, it is both **fast** (0.2 s/ayah, ~7× Whisper-turbo) and **robust**
(best on noisy phone clips) and **faithful** (doesn't hallucinate toward the target). Its gap:
it outputs **no tashkeel**.

### 6.3 wav2vec2 + CTC — the phoneme recognizer we trained
wav2vec2 is **self-supervised**: pre-trained on unlabeled audio (learn speech structure), then
fine-tuned with **CTC** for recognition. We fine-tuned `wav2vec2-xls-r-300m` on IqraEval's 79 hrs
to output **phonemes** (`finetuning/train_phoneme_recognizer.py`). It worked (PER 24.7%→17.2%),
proving connected-speech phoneme recognition is viable — but 17% error was too high to flag
harakat safely, and it needed a *second* model on top of the word model. Superseded by NAMAA.

### 6.4 Fine-tuning with LoRA — how we adapt a big model cheaply
Retraining billions of weights is expensive and risks destroying the model. **LoRA (Low-Rank
Adaptation)** freezes the original weights and injects tiny trainable matrices next to chosen
ones: for a frozen weight `W`, learn a small `ΔW = B·A` where `A,B` have a small **rank** `r`
(16–32). Only `A,B` train — a few MB, reversible, fast. We used LoRA to try to teach Cohere
tashkeel (it failed — §11) and to fine-tune Whisper variants.

### 6.5 NAMAA Cohere-Speech-Tashkeel — the winner (the whole system)
`NAMAA-Space/Cohere-Speech-Tashkeel-2B` is a fine-tune of Cohere's Arabic ASR that outputs
**fully diacritized** Arabic (words + harakat + case-endings) **acoustically** — i.e. it writes
the vowel it actually heard (§7). One model gives us words *and* pronunciation *and* the
diacritized display. Metrics: word acc 0.98 clean / 0.93 noisy / 0.88 kids; diacritic DER ~6.6%;
harakat false-rejection ~0–1.4%.

---

## 7. The key insight: acoustic vs. language-model diacritics

This single idea reorganized the whole project.

Ordinary ASR (Cohere, Whisper) predicts the **most likely** diacritics from a language model —
it "knows" what the correct vowel *should* be and writes that. So it will render `عِبَادِهِ`
(correct) even if the reciter mispronounced it. Useful for display, **useless for catching
vowel mistakes**, because it hides them.

NAMAA is different: its diacritics are **acoustically derived** — it writes the vowel the reciter
**actually produced**. We proved this on the IqraEval benchmark, where the ground truth tells us
what was really said:

| Reciter actually said | Canonical (correct) | NAMAA wrote |
|---|---|---|
| `عِبَادُهُ` (damma) | `عِبَادِهِ` (kasra) | `عِبَادُهُ` ✅ caught the case-ending error |
| `إِبْرَاهِيمُ` (damma) | `إِبْرَاهِيمَ` (fatha) | `إِبْرَاهِيمُ` ✅ |
| `يُخَشِّي` (**kh**) | `يُغَشِّي` (**gh**) | `يُخَشِّي` ✅ caught a **consonant** error |

Because it writes the actual sound, we can **compare its output to the correct Quran** to detect
real pronunciation errors — a single model doing both memorization and tajweed.

---

## 8. The final system, end to end

```
child's phone recording (audio)
        │
        ▼
   NAMAA model  ──►  diacritized transcription of what they ACTUALLY recited
        │                e.g.  "قُلْ يَا أَيُّهَا الْكَافِرُونْ لَا أَعْبُدُوا مَا تَعْبُدُونْ …"
        │
        ├─────────────► WORD grader ────► did they recite the right words?  (memorization score)
        │                 (compare to canonical, diacritics stripped)
        │
        └─────────────► HARAKAT grader ─► did they pronounce the vowels right?  (tajweed feedback)
                          (compare NAMAA's actual diacritics to canonical, tajweed-aware)

   target ayah is KNOWN (the app asks for a specific passage) → canonical text from quran.db
```

The service (`delivery/`) exposes `POST /api/evaluate`. For each recording it returns:
- `accuracy`, `passed`, `raw_score` — the memorization result
- `words[]` — per word: correct / wrong / skipped / extra
- `user_recitation_diacritized` — the child's **actual** diacritized recitation (their real
  words with their real vowels — not the "correct answer", so it doesn't cheat)
- `harakat_errors[]` — e.g. `عَبُدْتُمْ: letter ب said damma, expected fatha`
- Arabic feedback + (on failure) a reference reciter audio URL

**Worked example (Surah 109, a child who recited well):**
`passed: True, accuracy 0.88; harakat checked 11 words, 2 flagged` — e.g. `دِينَكُمْ` (said
`dīna-kum` where the case-ending should be `dīnu-kum`).

---

## 9. The grading algorithms, explained

### 9.1 Word grading (`delivery/core/grader.py`)
Goal: "how many of the correct words did they say, in order?" — tolerantly.

1. **Normalize** both texts: strip tashkeel, unify letter shapes (`أإآٱ→ا`, `ة→ه`, `ى→ي`, drop
   tatweel). Now comparison is about *words*, not spelling/diacritic style. (This is *why* the
   memorization score ignores tashkeel.)
2. **Align** the recited word list to the canonical word list with **Needleman-Wunsch** — a
   classic dynamic-programming algorithm (from DNA alignment) that finds the best word-to-word
   correspondence allowing **substitutions, insertions, deletions**. Intuitively it fills a grid
   of "best score to align the first *i* recited words with the first *j* canonical words" and
   backtracks the cheapest path.
3. **Fuzzy match** near-misses with **CER** so a tiny slip still counts as the intended word.
4. **Score** = fraction of canonical words matched. Threshold 0.85 → pass/fail. Per-word labels
   drive the highlighting.

### 9.2 Harakat grading (`delivery/core/harakat_grader.py`)
Goal: for each **correctly-recited word**, did the vowels match?

1. Only look at words the learner got **right at the letter level** (word errors are already
   reported by §9.1). Align recited↔canonical words by their bare (diacritic-free) form.
2. For a matched pair, split each into `(letter, harakat)` units and compare the **short-vowel**
   on each letter (fatha/damma/kasra/tanwin).
3. Normalize Uthmani→standard first so `ٱلـ`/dagger-alef differences don't count.

### 9.3 The tajweed tolerances (why the grader is trustworthy)
Naive letter-by-letter comparison flagged ~40–60% of *correct* recitation. Three domain-aware
tolerances fixed it:
- **Waqf:** never flag the **word-final** vowel (pausing correctly turns it to sukun).
- **Shadda:** ignore shadda presence/absence (idghaam/gemination are tajweed details).
- **Implicit sukun:** treat "no mark" and sukun as identical (Uthmani leaves sukun unwritten).
After these, the **false-rejection rate dropped to ~0.5%** on correct recitation, and stays
near-zero even under phone noise (because a noise-corrupted word usually also breaks the letters,
so it fails the word match and drops out of harakat-checking instead of becoming a false alarm).

---

## 10. How we measure quality (the metrics)

- **WER (Word Error Rate):** fraction of words wrong (substitutions+insertions+deletions ÷ words).
  Lower is better. Our "accuracy" ≈ 1 − a word-level error.
- **CER (Character Error Rate):** same, per character — used for fuzzy word matching.
- **PER (Phoneme Error Rate):** same, per phoneme — how we scored the phoneme recognizer (17.2%).
- **DER (Diacritic Error Rate):** fraction of diacritics wrong — how we score tashkeel (NAMAA
  ~6.6%).
- **The two that matter most for safety (pronunciation detection):**
  - **False-Rejection Rate (FRR):** how often we flag a **correct** sound as wrong. On scripture
    this must be tiny. Ours ≈ 0–1.4%.
  - **False-Acceptance Rate (FAR):** how often we **miss** a real error. NAMAA's acoustic
    diacritics keep this low (it writes the actual error).

Rule of thumb: raw accuracy on clean clips is easy; the **error rates on realistic, noisy,
correct recitation** are what decide whether it's safe to deploy.

---

## 11. Lessons learned (what failed, and why)

Dead ends are results too — each taught something:

1. **Bigger isn't automatically better, but the *right* bigger is.** Whisper-turbo (small) lost
   to full Whisper-v3 and to Cohere on hearing kids.
2. **Front-end denoising hurt.** noisereduce and DeepFilterNet both lowered accuracy — they strip
   signal the model needs. Don't "clean" audio before a robust model.
3. **Synthetic noise augmentation was a wash.** Fake phone noise ≠ real phone noise; it didn't
   transfer. (But it exposed a silent-clip bug — verify your data!)
4. **You cannot bolt tashkeel onto a model that resists it.** Three LoRA attempts to make base
   Cohere emit tashkeel all showed the same wall: more tashkeel ⇒ broken words. Its pretrained
   "undiacritized Arabic" prior was too strong. (NAMAA solved this by being *trained* for it.)
5. **A text diacritizer is not Quran-safe on its own.** CATT (a strong Arabic diacritizer) mis-marks
   ~5–7% of scripture words — fine as a fallback, not as the source of truth.
6. **No word-ASR "hears" harakat — but a purpose-built model can.** The turning point was proving
   NAMAA's diacritics are acoustic (§7).
7. **The grading logic is as important as the model.** The same NAMAA output went from "flags
   half of correct recitation" to "0.5% false rejection" purely by adding tajweed tolerances.
8. **Validate on the hard, realistic case.** Clean-audio numbers were never the question; noisy,
   correct, kid-domain error rates were.

---

## 12. How to run and reproduce it

**Environment:** conda env `mlaudio` — Python 3.11, `transformers>=5.4`, `torch 2.x+cu12x`,
`accelerate`, `sentencepiece`, `protobuf`, `librosa`, `soundfile`. A GPU with ≥8 GB
(the model needs ~5 GB in bf16). **bf16 is required** — fp16 overflows an attention mask and
produces garbage.

**Try the model directly:**
```python
import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
proc  = AutoProcessor.from_pretrained("NAMAA-Space/Cohere-Speech-Tashkeel-2B")
model = CohereAsrForConditionalGeneration.from_pretrained(
    "NAMAA-Space/Cohere-Speech-Tashkeel-2B", torch_dtype=torch.bfloat16, device_map="auto")
inp = proc(audio_16k_mono, sampling_rate=16000, return_tensors="pt", language="ar")
inp.to(model.device, dtype=model.dtype)
ids  = model.generate(**inp, max_new_tokens=256)
print(proc.decode(ids, skip_special_tokens=True))   # diacritized recitation
```

**The full grading pipeline** lives in `delivery/`:
- `core/namaa_model.py` — the model wrapper.
- `core/grader.py` — word/memorization grading.
- `core/harakat_grader.py` — tajweed grading with the tolerances of §9.3.
- `main.py` — the FastAPI service (`/api/evaluate`).
Build/run with `delivery/Dockerfile` + `docker-compose.yml` on a GPU host.

**Useful scripts to learn from** (`finetuning/`):
- `test_namaa.py` — the acoustic-vs-LM proof of §7.
- `layer1_report.py` — generates the human-readable grading report.
- `train_phoneme_recognizer.py` + `validate_mdd.py` — the phoneme-recognizer path (superseded,
  but a good CTC + pronunciation-detection example).

---

## 13. Glossary

- **ASR** — Automatic Speech Recognition (audio → text).
- **Waveform / sample rate** — the raw audio numbers; 16 kHz = 16,000 numbers/second.
- **Spectrogram / mel** — time-frequency "picture" of sound; mel = perceptual frequency scale.
- **Encoder / decoder** — the two halves of a seq2seq model (understand sound / produce symbols).
- **Conformer** — convolution-augmented Transformer encoder (Cohere/NAMAA).
- **CTC** — per-frame labeling + collapse; faithful, no hallucination (wav2vec2).
- **Autoregressive** — generates one token at a time, each conditioned on the previous.
- **Token / vocabulary** — the units a model emits, and the full set of them.
- **LoRA** — cheap fine-tuning by training tiny injected matrices; base weights frozen.
- **Tashkeel / harakat** — Arabic diacritics (short vowels + shadda/sukun/tanwin).
- **Iʿrab** — grammatical case-endings (the final vowel).
- **Tajweed** — rules of correct Quran pronunciation. **Waqf** — pausing (final vowel → sukun).
  **Idghaam** — merging (produces shadda).
- **Uthmani script** — the mushaf orthography (`ٱ`, `ٰ`, …).
- **Needleman-Wunsch** — dynamic-programming sequence alignment (used by the word grader).
- **WER / CER / PER / DER** — Word / Character / Phoneme / Diacritic Error Rate.
- **FRR / FAR** — False-Rejection / False-Acceptance Rate (flag a correct sound / miss a wrong one).
- **Acoustic vs. LM diacritics** — writing the vowel actually *heard* vs. the vowel *expected*.

---

## 14. Where to learn more

- Whisper (weakly-supervised ASR) — https://arxiv.org/abs/2212.04356
- Conformer (convolution-augmented Transformer) — https://arxiv.org/abs/2005.08100
- wav2vec 2.0 (self-supervised speech) — https://arxiv.org/abs/2006.11477
- CTC (the original paper) — https://www.cs.toronto.edu/~graves/icml_2006.pdf
- LoRA (low-rank fine-tuning) — https://arxiv.org/abs/2106.09685
- CATT (Arabic diacritization) — https://arxiv.org/abs/2407.03236
- Iqra'Eval (Quran pronunciation assessment benchmark) — https://aclanthology.org/2025.arabicnlp-sharedtasks.61/
- The models: NAMAA https://huggingface.co/NAMAA-Space/Cohere-Speech-Tashkeel-2B ·
  Cohere https://huggingface.co/CohereLabs/cohere-transcribe-03-2026

---

*Companion reference with the exhaustive experiment log and exact numbers:
`docs/Quran-ARS-Technical-Documentation.md`.*
