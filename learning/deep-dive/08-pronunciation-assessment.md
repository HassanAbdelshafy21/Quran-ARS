# Chapter 8 — Pronunciation Assessment (MDD)

**Goal:** the capstone — how to judge *how* something was pronounced, not just *what* was said.
The Mispronunciation Detection & Diagnosis (MDD) task, the two classic approaches (GOP via forced
alignment; phoneme recognition + comparison), the crucial **acoustic vs. language-model
diacritization** distinction, and how this project solved tajweed grading with a single model.

---

## 8.1 The task

**MDD** takes a learner's audio plus the **canonical** text (what they *should* say) and outputs,
per unit (phoneme/harakat), whether it was pronounced correctly, and if not, **what** was said
(diagnosis). Two subtasks:
- **Detection:** correct vs. mispronounced (the binary problem of Ch. 6.4).
- **Diagnosis:** which sound was actually produced (e.g. "said damma, expected kasra").

The defining constraint of *our* setting: it's on the **Quran**, so **false rejections must be
near zero** — flagging a correct recitation as wrong is unacceptable (Ch. 6.4).

---

## 8.2 Approach A — Goodness of Pronunciation (GOP) via forced alignment

The classical Computer-Aided Pronunciation Training method. Steps:

1. **Force-align** the audio to the *expected* phoneme sequence. Forced alignment = decoding
   constrained to a known transcript: find, for each expected phoneme, the frames it occupies.
   With a CTC model you can use `torchaudio.functional.forced_align`; with HMM/DNN models, Viterbi.
2. **Score each phoneme** by how well the audio at its frames matches it. The **GOP** of phoneme
   $p$ over its aligned frames $\mathcal{F}_p$ is the log-posterior ratio of the *expected* phoneme
   vs. the *best competitor*:

$$ \text{GOP}(p) = \log p(p\mid \mathbf{x}_{\mathcal{F}_p}) - \max_{q}\log p(q\mid \mathbf{x}_{\mathcal{F}_p})
   \;\approx\; \frac{1}{|\mathcal{F}_p|}\sum_{t\in\mathcal{F}_p}\Big(\log y_t[p] - \max_q \log y_t[q]\Big) $$

3. **Threshold:** if $\text{GOP}(p)$ is far below 0 (expected phoneme much less likely than the
   best competitor), flag a mispronunciation.

**Why GOP is elegant:** it directly answers "did the expected sound actually occur here?"
**Why it's fragile (and failed for us):**
- **CTC peakiness** (Ch. 3.6): the expected phoneme has high probability only on a spike frame and
  ~0 (blank) elsewhere; averaging over $\mathcal{F}_p$ underscores everything → massive **false
  rejections**. (Our prototype flagged ~25/26 words of a *correct* recitation.)
- **Alignment errors** propagate: bad boundaries → wrong frames → wrong scores.
- Needs a **strong, in-domain** acoustic model; our word-trained phoneme model on connected speech
  wasn't reliable enough.

---

## 8.3 Approach B — Phoneme recognition + comparison

1. **Freely recognize** the phonemes actually spoken (a CTC phoneme recognizer, Ch. 3/5).
2. **Align** the recognized phonemes to the canonical phonemes (Needleman–Wunsch, Ch. 6).
3. **Diff:** substitutions/deletions vs. the canonical = mispronunciations; the recognized symbol
   is the **diagnosis**.

This is faithful (no LM correcting errors) and gives diagnosis for free. Its ceiling is the
recognizer's **PER**: at 17% PER, ~1 in 6 phonemes is misrecognized → too many false rejections
for scripture. The SOTA (IqraEval systems, ~0.16% PER with big data + a multi-level phoneme scheme)
shows it's *possible*, but it's a heavy build and — critically — needs a **second model** on top of
the word model.

> **Where we got to:** Approach B with our own `wav2vec2-CTC` reached PER 17.2% on QuranMB — a
> genuine connected-speech recognizer, but not yet deploy-safe, and it required maintaining a
> separate model. Both A and B pushed us to ask: *can one model just output the actual diacritics?*

---

## 8.4 The pivotal distinction — acoustic vs. language-model diacritics

An ASR that outputs diacritized text can get them two ways:
- **Language-model diacritization:** predict the *expected* mark from context (the model "knows"
  the correct vowel and writes it). Great for readable output; **useless for MDD** — it *hides*
  the learner's error by writing the correct vowel regardless.
- **Acoustic diacritization:** write the vowel *actually heard*. This exposes errors — a
  mispronounced damma-for-kasra comes out as a damma.

**How to tell them apart empirically** (the test we ran, `finetuning/test_namaa.py`): feed clips
where the ground truth says the reciter made an error (IqraEval `Annotation ≠ Reference`). If the
model's output matches the **Annotation** (the actual), it's acoustic; if it matches the
**Reference** (canonical), it's LM. NAMAA matched the **Annotation** — writing `عِبَادُهُ` (the
wrong damma), `إِبْرَاهِيمُ` (wrong damma), `يُخَشِّي` (wrong *consonant* kh) — proving it's
acoustic. That single fact collapsed two models into one.

---

## 8.5 How this project does MDD (the final design)

One model, `NAMAA-Space/Cohere-Speech-Tashkeel-2B`, outputs the learner's **actual diacritized
recitation**. Then pure logic — no second model:

```
audio ── NAMAA ──► diacritized recitation (actual words + actual harakat)
                        │
   word grader (Ch.6): align to canonical (tashkeel stripped) → memorization score
   harakat grader:     for each CORRECTLY-recited word, compare NAMAA's harakat to canonical
                       with TAJWEED TOLERANCES → tajweed errors
```

**The harakat comparison (why it's safe):**
- Only check **correctly-recited words** (word errors are handled by the word grader) — this also
  means a noise-garbled word fails the word match and *drops out* of harakat checking instead of
  becoming a false alarm (why FRR stays ~0 under noise).
- Compare only the **short-vowel content** (fatha/damma/kasra), with tolerances grounded in Ch. 7:
  - **Waqf**: never flag the word-final vowel.
  - **Shadda**: ignore (idghaam/gemination).
  - **Implicit sukun**: treat unmarked = sukun (Uthmani).
- Normalize the canonical Uthmani → standard harakat so orthography differences don't count.

**Measured** (Ch. 6.4 method): FRR 0.5% clean / ~0% phone-noise, word acc 0.98/0.93/0.88, DER
~6.6%, and it *catches* real case-ending and consonant errors (FAR kept low by acoustic output).

---

## 8.6 The general lesson for pronunciation assessment

1. **Faithfulness beats fluency.** For MDD you want a model that reports what was *said*, not what
   was *expected* — the opposite of what makes a good dictation system. Acoustic diacritization (or
   a faithful phoneme recognizer) is the right tool; LM fusion is the wrong one.
2. **The model is half; the rules are half.** The same NAMAA output went from "flags half of
   correct scripture" to "0.5% FRR" purely through **domain-aware tolerances** (waqf/shadda/sukun).
   Pronunciation assessment is a *linguistics + ML* problem.
3. **Measure the false-rejection rate on correct data.** It's the gate that matters, and you can
   measure it without phoneme labels (run the grader on known-correct recitation).
4. **Prefer one faithful model + smart logic** over a tower of models, when you can get it. The
   whole project simplified from {word ASR + phoneme recognizer + text diacritizer} to {one
   acoustic-diacritizing ASR + a rule-based grader}.

---

## 8.7 What's still open (research directions for you)

- **Madd (elongation)** and **ghunnah duration** grading — needs duration modeling, not just
  vowel identity (Ch. 7.4).
- **Confidence-gated flagging**: only surface harakat errors above a confidence to push FRR even
  lower; calibrate against the IqraEval GT.
- **Real kid-phone data**: everything was validated on adult + synthetic noise + a few kid clips.
  A labeled kid-recitation set (with per-phoneme annotations) is the highest-value data to collect.
- **The official MDD metric**: with `IqraEval_Test_GT` now accessible, compute FRR/FAR/F1 in
  phoneme space by converting NAMAA's Arabic output to the IqraEval phoneme scheme (a G2P) — the
  last quantitative validation.

---

## Exercises
1. Explain, using CTC peakiness (Ch. 3.6), why frame-averaged GOP produces false rejections, and
   propose two fixes (spike scoring; forward–backward posteriors).
2. Design the empirical test that distinguishes acoustic from LM diacritization. What ground truth
   do you need, and what result proves "acoustic"?
3. Why does "only check correctly-recited words" make the harakat grader robust to noise? Trace a
   noise-garbled word through the pipeline.
4. Give the phonetic justification (Ch. 7) for each of the three tolerances (waqf, shadda, implicit
   sukun).
5. You must lower FRR from 1.4% to 0.3% without raising FAR much. Propose two concrete changes and
   how you'd validate them against the IqraEval GT.

---

*End of the deep-dive series. Re-read Chapter 1 now — with everything else in mind, the front end
will look different. Then reread the [Learning Guide](../Quran-ARS-Learning-Guide.md) and the
[Technical Documentation](../Quran-ARS-Technical-Documentation.md) — they'll read as summaries of
things you now understand from first principles.*
