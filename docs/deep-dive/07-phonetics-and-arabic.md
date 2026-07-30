# Chapter 7 — Phonetics & the Arabic/Quran Domain

**Goal:** understand what speech *is* physically (source–filter, formants), the phonetic
categories a model must distinguish, and the specific acoustics of Arabic and tajweed — so you
know exactly what the harakat grader is trying to hear.

---

## 7.1 How speech is produced: the source–filter model

Speech = a **source** of sound shaped by a **filter**:
- **Source.** For **voiced** sounds (all vowels, `m`, `n`, `l`, `b`…), the vocal folds vibrate,
  producing a buzz at the **fundamental frequency** $f_0$ (pitch: ~120 Hz adult male, ~210 female,
  **250–400 Hz children**). The buzz is rich in **harmonics** at $f_0, 2f_0, 3f_0,\dots$
  For **unvoiced** sounds (`s`, `sh`, `f`, `kh`…), the source is turbulent **noise** (air forced
  through a constriction), with no $f_0$.
- **Filter.** The **vocal tract** (throat, mouth, tongue, lips) acts as an acoustic tube whose
  shape emphasizes certain frequencies — **resonances** called **formants** ($F_1, F_2, F_3,\dots$).

So a vowel's spectrum = harmonics of $f_0$ (fine comb) under an envelope of formant peaks. The
**formants identify the vowel**; $f_0$ is the pitch (speaker/intonation). This is exactly why
**MFCCs** (Ch. 1) separate the two: the low cepstral coefficients capture the formant envelope
(the *what*), discarding pitch (the *who*).

---

## 7.2 Formants and the vowel triangle — the key to harakat

The three Arabic short vowels (harakat) map to classic vowel positions, defined by $F_1$ (inversely
related to tongue **height**) and $F_2$ (tongue **frontness**):

| Harakat | Vowel | Tongue | $F_1$ | $F_2$ |
|---|---|---|---|---|
| **kasra** ِ | /i/ ("ee") | high, front | low | high |
| **damma** ُ | /u/ ("oo") | high, back | low | low |
| **fatha** َ | /a/ ("ah") | low | high | mid |

So **fatha vs. damma vs. kasra is literally an $F_1/F_2$ pattern in the spectrogram.** A model that
"hears harakat" (NAMAA) is, in effect, reading the formant pattern of each short vowel. Two reasons
this is *hard*, tying back to earlier chapters:
- Short vowels are **brief** (few frames, Ch. 1) — little evidence.
- **High child $f_0$** spaces the harmonics widely, **under-sampling** the formant envelope (fewer
  harmonics fall under each formant peak), so formant estimation is noisier for kids. This is a
  real, physical reason kids are harder — not just "less data."

**Diphthongs / madd (elongation):** holding a vowel (madd) is a long steady formant pattern;
the length itself is phonemically meaningful in Quran recitation (2, 4, 6 counts).

---

## 7.3 Consonants and Arabic's distinctive inventory

Consonants are described by **place** (where the constriction is), **manner** (how air flows), and
**voicing**. Arabic has sounds rare in English that a general ASR may fumble — and that the
IqraEval phoneme scheme marks explicitly (Ch. 5 alphabet):

- **Emphatic (pharyngealized) consonants**: ص `S`, ض `D`, ط `T`, ظ `Z` — produced with the tongue
  root retracted, which **lowers the $F_2$ of neighboring vowels**. So emphasis is audible not
  just in the consonant but in how it *colors the adjacent vowel*. Distinguishing س `s` from ص `S`
  is a common tajweed target.
- **Pharyngeals / uvulars**: ع `E` (ʿayn), ح `H` (ḥ), ق `q`, غ `g` (gh), خ `x` (kh). The
  `يُخَشِّي` vs `يُغَشِّي` (kh vs gh) error NAMAA caught (Ch. 8 / §7 of the Learning Guide) is
  exactly a `x`↔`g` place/voicing confusion.
- **Hamza** ء `<` (glottal stop): a brief closure of the vocal folds — acoustically a gap + sharp
  onset.

**Coarticulation.** Sounds blend into neighbors (a consonant's formant transitions *point to* the
next vowel). This is why **isolated-word** phoneme models fail on **connected** recitation (Ch. 3):
in continuous speech the boundaries and transitions differ from clean single words.

---

## 7.4 The acoustics of tajweed rules the grader must respect

Recall (Learning Guide §4) that tajweed rules are **correct** pronunciation — the grader must not
flag them as errors. Their acoustics:

- **Waqf (pausing).** At a stop, the final short vowel is **not pronounced**; the word ends on a
  bare consonant (sukun). Acoustically the voiced vowel simply doesn't occur. A grader comparing
  the final letter's vowel to the "expected" mid-sentence vowel would see a mismatch — which is why
  our grader **never flags the word-final position** (Learning Guide §9.3).
- **Ghunnah (nasalization)** on ن/م with shadda: a ~2-count nasal hum (energy through the nasal
  cavity → a low-frequency resonance). It's a duration/nasality feature, not a vowel error.
- **Idghaam (merging):** a letter assimilates into the next, often realized as a **shadda**
  (gemination = a longer closure). Our grader **ignores shadda differences** for this reason.
- **Qalqalah**: a slight "bounce"/echo on ق ط ب ج د when they carry sukun — a brief burst. This is
  an inherently **local** cue — exactly the kind a **convolution** captures well (Ch. 2 §2.4).
- **Madd (elongation):** vowel length as meaning — a *duration* judgment, which frame-based models
  can measure but which our current word/harakat grader does **not** score (a natural future
  feature).

**Takeaway:** tajweed grading is not just "is the vowel right" — it's "is the vowel right *given*
the tajweed context." Our tolerances (waqf, shadda, implicit sukun) encode the most important of
these so the false-rejection rate stays near zero.

---

## 7.5 Why this domain knowledge changed the engineering

- Knowing that **harakat = formant patterns of brief vowels** explains why (a) a *word* ASR can't
  hear them (it predicts the expected vowel from the LM), and (b) a model *trained* to output
  acoustic diacritics (NAMAA) can — it learned to read those formant patterns.
- Knowing **waqf/idghaam/ghunnah** are correct-by-rule is why the grader has explicit tolerances;
  without phonetic knowledge you'd "fix" the model forever and never fix the real problem (the
  grading logic).
- Knowing **child $f_0$ under-samples formants** and **phone channels lose high frequencies**
  (fricatives) tells you *where* errors will concentrate and what real-world data to collect.

---

## Exercises
1. On a spectrogram, how do you tell /i/ (kasra) from /u/ (damma) at a glance? State the $F_1,F_2$
   pattern for each.
2. Why does a child's high $f_0$ make formant (and thus harakat) estimation harder? Relate
   harmonic spacing to formant sampling.
3. An emphatic ص lowers a neighboring vowel's $F_2$. If a learner says a plain س instead, what
   changes in the following vowel's spectrum — and why might a model detect the error on the
   *vowel* even if the consonant is ambiguous?
4. Classify each as source or filter phenomenon: pitch, vowel identity, whisper, ghunnah.
5. Explain why an *isolated-word* phoneme recognizer degrades on connected recitation, using
   coarticulation.

**Next:** [Chapter 8 — Pronunciation Assessment (MDD)](08-pronunciation-assessment.md): tying it
all together into how we grade tajweed.
