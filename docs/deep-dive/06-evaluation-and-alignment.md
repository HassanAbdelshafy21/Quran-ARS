# Chapter 6 — Evaluation & Alignment

**Goal:** compute the metrics that decide everything — WER/CER/PER/DER via edit distance, the
Needleman–Wunsch alignment behind the word grader, and the **detection** metrics (FRR/FAR,
precision/recall/F1, ROC) that gate pronunciation assessment. You'll be able to implement each.

---

## 6.1 Edit distance (Levenshtein) — the foundation of every error rate

The **edit distance** between a reference $\mathbf{r}$ (length $n$) and a hypothesis $\mathbf{h}$
(length $m$) is the minimum number of **substitutions (S)**, **insertions (I)**, and
**deletions (D)** to turn one into the other. Dynamic program on a grid $d[i][j]$ = edit distance
between $\mathbf{r}_{1:i}$ and $\mathbf{h}_{1:j}$:

$$
d[i][j] = \min\begin{cases}
d[i-1][j-1] + \mathbb{1}[r_i\neq h_j] & \text{(match/substitute)}\\
d[i-1][j] + 1 & \text{(delete } r_i)\\
d[i][j-1] + 1 & \text{(insert } h_j)
\end{cases}
$$

with $d[i][0]=i$, $d[0][j]=j$. Answer: $d[n][m]$. **Backtracking** the choices recovers the actual
S/I/D operations (the alignment). Complexity $O(nm)$.

**Worked example.** $\mathbf{r}=$ `k i t t e n`, $\mathbf{h}=$ `s i t t i n g`. Edit distance = 3
(k→s substitute, e→i substitute, insert g). Fill the grid to confirm.

Units:
- **WER** (Word Error Rate): tokens = **words**. $\text{WER} = (S+D+I)/N$, $N=$ #reference words.
- **CER**: tokens = **characters**. Robust when word boundaries are fuzzy; our grader uses CER to
  decide if a near-miss counts as the intended word.
- **PER**: tokens = **phonemes** (Ch. 3/8). Our phoneme recognizer: PER 17.2%.
- **DER** (Diacritic Error Rate): fraction of **diacritics** wrong (on correctly recognized
  letters). NAMAA: ~6.6%.

> Note WER can exceed 1.0 (many insertions). "Accuracy" $=1-\text{WER}$ is a convenience, not a
> probability. `jiwer` computes these; we used `jiwer.wer` on phoneme strings for PER.

---

## 6.2 Global alignment: Needleman–Wunsch (the word grader)

Edit distance is a special case of **global sequence alignment**. **Needleman–Wunsch (NW)** is the
same DP framed as *maximizing a score* (from bioinformatics, aligning DNA), with a match reward and
gap penalty. Our `grader.align(ref_words, hyp_words)` is exactly this on word tokens, producing a
list of operations:

- `equal` (word matched) → counts toward the memorization score.
- `sub` (substitution) → "you said X instead of Y."
- `delete` (reference word missing) → "you skipped Y."
- `insert` (extra hypothesis word) → "extra word" (ignored when the child passed — likely
  continued reciting).

**Why alignment and not just set-overlap?** Order and position matter in recitation; NW respects
sequence, so "skipped word 3" is distinguishable from "wrong word 3," and repeated words align
correctly. The grader then computes `accuracy = matched / total_reference_words`, threshold 0.85.

**Pseudocode (scoring form):**
```
score[i][j] = max(
   score[i-1][j-1] + (MATCH if r[i]==h[j] else MISMATCH),  # diagonal
   score[i-1][j]   + GAP,                                   # up   (delete r[i])
   score[i][j-1]   + GAP)                                   # left (insert h[j])
# backtrack from score[n][m] to read off equal/sub/delete/insert
```

---

## 6.3 Fuzzy matching (why our grader is *tolerant*)

Raw token equality is too strict — ASR spells a mispronounced word slightly differently but it's
clearly the intended word. So after normalization (strip tashkeel, unify alef/ya/ta-marbuta), the
grader treats two words as "equal" if their **CER is below a threshold** (e.g. 0.35). This turns a
brittle exact-match into a robust word grader. It's the difference between "0.85 pass" and
"unfairly failing a child for a spelling nuance."

---

## 6.4 Detection metrics — the real deploy gate

Word/diacritic error rates measure *transcription* quality. **Pronunciation assessment is a
detection problem**: for each phoneme/harakat, decide "correct" vs "mispronounced." Evaluate it as
a binary classifier against ground truth (which the IqraEval GT gives via `Reference_phn` vs
`Annotation_phn`). Build the confusion:

| | GT: correct | GT: mispronounced |
|---|---|---|
| **We say correct** | True Accept (TA) | **False Accept (FA)** — missed a real error |
| **We say wrong** | **False Reject (FR)** — flagged correct scripture ❌ | True Reject (TR) |

- **False-Rejection Rate** $\text{FRR} = \dfrac{FR}{FR+TA}$ — of all *correct* sounds, how many we
  wrongly flagged. **On the Quran this must be tiny.** Ours ≈ 0–1.4%.
- **False-Acceptance Rate** $\text{FAR} = \dfrac{FA}{FA+TR}$ — of all *real errors*, how many we
  missed.
- Treating "mispronunciation" as the positive class:
  $\text{Precision}=\dfrac{TR}{TR+FR}$, $\text{Recall}=\dfrac{TR}{TR+FA}=1-\text{FAR}$,
  $\text{F1}=\dfrac{2PR}{P+R}$.

**The trade-off.** A detector has a threshold (how confident before you flag). Lowering it catches
more errors (↑recall, ↓FAR) but flags more correct sounds (↑FRR). Sweeping the threshold traces the
**ROC curve** (recall vs. false-positive rate) or a **DET curve**; the **area under ROC (AUC)**
summarizes it. For scripture we deliberately pick an operating point with **very low FRR**, even at
some FAR cost — better to miss a subtle error than to wrongly correct a child on the Quran.

> **How we measured it here.** We had no phoneme-level Arabic labels for our own audio, so we
> measured FRR *directly in the product's logic*: run the harakat grader on **correct** adult
> recitation (known text) — every flag is by definition a false rejection → FRR = 0.5% clean,
> ~0% under phone noise. And `finetuning/validate_mdd.py` computes TA/TR/FR/FA against the IqraEval
> GT by triple-aligning Reference, Annotation, and Prediction (§6.2 alignment applied three ways).

---

## 6.5 Pitfalls in evaluation (learned the hard way)

- **Noisy labels.** RetaSy's "incorrect" clips were often actually correct → any model plateaued
  at ~66% "accuracy" on it. **A ceiling can be the test set, not the model.** Always inspect labels.
- **Domain match.** Metrics on clean adult studio audio don't predict kid-phone behavior. Report
  the number on the *realistic* condition (we augmented with phone noise, and kept a real kid set).
- **The right denominator.** FRR must be computed over *correct* items only; mixing in real errors
  hides the scripture-safety risk.
- **Aggregate vs. per-item.** A 17% PER can still mean *most* words are perfect and a few are
  garbage — look at distributions, not just means.

---

## Exercises
1. Compute WER for reference "قل هو الله أحد" (4 words) and hypothesis "قل هو الواحد أحد"
   (one substitution). *(WER = 1/4 = 0.25.)*
2. Fill the Levenshtein grid for `kitten`/`sitting` and backtrack the 3 operations.
3. A detector on 1000 correct phonemes and 100 mispronounced ones yields FR=5, FA=30. Compute
   FRR, FAR, precision, recall, F1. *(FRR=0.5%, FAR=30%, P=70/75=0.93, R=70/100=0.70, F1≈0.80.)*
4. Why can lowering the flag threshold *simultaneously* improve recall and worsen FRR? Draw the
   ROC intuition.
5. Explain why measuring FRR on *correct* recitation (as we did) is a valid and conservative
   deploy gate even without phoneme-level labels.

**Next:** [Chapter 7 — Phonetics & the Arabic/Quran Domain](07-phonetics-and-arabic.md): what the
model must actually hear.
