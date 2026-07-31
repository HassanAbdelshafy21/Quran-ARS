# Chapter 3 — CTC (Connectionist Temporal Classification)

**Goal:** understand how a model maps a length-$T$ sequence of encoder frames to a *shorter*,
unaligned label sequence (e.g. phonemes) **without frame-level labels** — the CTC loss, its
forward–backward algorithm (with a full numeric example), and how to decode. This is the engine
of the wav2vec2 phoneme recognizer in this repo.

---

## 3.1 The alignment problem

The encoder outputs $T$ frames (say 375). The label is a short sequence
$\mathbf{l} = (l_1,\dots,l_U)$, e.g. the phonemes `q u l` ($U=3$). We do **not** know which frames
correspond to which label — the speaker might hold "u" for 20 frames and "q" for 3. Classic
systems used HMMs to force an alignment. **CTC** lets the network learn without any alignment by
summing over **all** alignments that could produce the label.

Two tools make this work:
1. An extra **blank** symbol $\varnothing$ (means "no output here / repeat boundary").
2. A **collapse function** $\mathcal{B}$: given a length-$T$ frame-label path, (a) merge
   consecutive duplicates, then (b) remove blanks.

**Examples of $\mathcal{B}$** (target `q u l`, $T=6$):
- $\text{q q u u l l} \to \text{q u l}$ ✓
- $\text{q}\varnothing\text{q u l}\varnothing \to \text{q q u l}$ ✗ (the blank *separates* the two q's, giving `qq`)
- $\text{q}\varnothing\varnothing\text{u l l} \to \text{q u l}$ ✓
- $\varnothing\text{q u u}\varnothing\text{l} \to \text{q u l}$ ✓

Note the subtlety: to emit a **repeated** label (like `l l` in "hello"), you *need* a blank
between them, else they collapse to one. Blanks encode both "silence/none" and "repeat boundary."

---

## 3.2 The CTC objective

The network outputs, for each frame $t$, a probability distribution $y_t$ over the vocabulary
$\{\text{symbols}\}\cup\{\varnothing\}$ (a softmax). A **path** $\pi=(\pi_1,\dots,\pi_T)$ is one
label per frame; its probability (assuming frame-independence given the audio) is:

$$ p(\pi\mid \mathbf{x}) = \prod_{t=1}^{T} y_t[\pi_t] $$

The probability of the *label* $\mathbf{l}$ is the sum over **all** paths that collapse to it:

$$ p(\mathbf{l}\mid\mathbf{x}) = \sum_{\pi\,:\,\mathcal{B}(\pi)=\mathbf{l}} p(\pi\mid\mathbf{x}) $$

and the **CTC loss** is $-\log p(\mathbf{l}\mid\mathbf{x})$. The number of paths is exponential in
$T$, so we need a dynamic program — the **forward–backward algorithm**.

---

## 3.3 The forward algorithm (dynamic programming)

Build the **extended label** $\mathbf{l}'$ by inserting a blank between every label and at the
ends. For `q u l`:

$$ \mathbf{l}' = (\varnothing,\ \text{q},\ \varnothing,\ \text{u},\ \varnothing,\ \text{l},\ \varnothing), \quad |\mathbf{l}'| = 2U+1 = 7 $$

Define the **forward variable** $\alpha_t(s)$ = total probability of all paths that, by frame $t$,
have produced the prefix of $\mathbf{l}'$ up to position $s$:

$$ \alpha_t(s) = \sum_{\pi_{1:t}\,:\,\mathcal{B}(\pi_{1:t}) \,\hat=\, \mathbf{l}'_{1:s}} \prod_{t'=1}^{t} y_{t'}[\pi_{t'}] $$

**Initialization** (frame 1 can only be the first blank or the first real symbol):
$\alpha_1(1)=y_1[\varnothing]$, $\alpha_1(2)=y_1[\mathbf{l}'_2]$, all other $\alpha_1(s)=0$.

**Recursion.** A path reaching position $s$ at frame $t$ came from $s$ (stay), $s-1$ (advance
over a blank/symbol), or — only if the current symbol differs from the one two back and isn't a
blank — from $s-2$ (skip the blank between two distinct symbols):

$$
\alpha_t(s) = y_t[\mathbf{l}'_s]\cdot
\begin{cases}
\alpha_{t-1}(s) + \alpha_{t-1}(s-1) & \text{if } \mathbf{l}'_s=\varnothing \text{ or } \mathbf{l}'_s=\mathbf{l}'_{s-2}\\[4pt]
\alpha_{t-1}(s) + \alpha_{t-1}(s-1) + \alpha_{t-1}(s-2) & \text{otherwise}
\end{cases}
$$

**Termination:** $p(\mathbf{l}\mid\mathbf{x}) = \alpha_T(2U+1) + \alpha_T(2U)$ (path ends on the
final blank or the final symbol).

This is $O(T\cdot U)$ — linear, not exponential. The **backward** variable $\beta_t(s)$ is defined
symmetrically (probability of the *suffix* from $s$ onward). Together they give the
**gradient**: the derivative of the loss w.r.t. the logit for symbol $k$ at frame $t$ is

$$ \frac{\partial\,(-\log p)}{\partial\,a_t^k} = y_t[k] - \frac{1}{p(\mathbf{l}\mid\mathbf{x})}\!\!\sum_{s:\,\mathbf{l}'_s=k}\!\! \alpha_t(s)\beta_t(s) $$

i.e. "predicted probability minus the total responsibility of symbol $k$ at frame $t$." Standard
frameworks (`torch.nn.CTCLoss`, HuggingFace `Wav2Vec2ForCTC`) implement all of this; you pass
logits + label lengths and get the loss.

---

## 3.4 A full numeric example (do this by hand once)

Vocab $\{\text{a}, \varnothing\}$, target $\mathbf{l}=\text{a}$ ($U=1$), $T=3$. Extended label
$\mathbf{l}'=(\varnothing,\text{a},\varnothing)$, positions $s=1,2,3$. Suppose the network gives,
for each frame, $y_t[\text{a}]=0.7,\ y_t[\varnothing]=0.3$ (same every frame).

**Forward table** $\alpha_t(s)$:

| | $s{=}1\,(\varnothing)$ | $s{=}2\,(\text{a})$ | $s{=}3\,(\varnothing)$ |
|---|---|---|---|
| $t{=}1$ | $0.3$ | $0.7$ | $0$ |
| $t{=}2$ | $0.3\cdot0.3=0.09$ | $0.7\,(0.7+0.3)=0.70$ | $0.3\,(0.7)=0.21$ |
| $t{=}3$ | $0.3\cdot0.09=0.027$ | $0.7\,(0.70+0.09)=0.553$ | $0.3\,(0.21+0.70)=0.273$ |

(Row $t$ uses the recursion; e.g. $\alpha_2(2)=y[\text{a}]\,[\alpha_1(2)+\alpha_1(1)]$ since
$\mathbf{l}'_2=\text{a}$ and there's no $s-2$.)

**Termination:** $p(\text{a}) = \alpha_3(3)+\alpha_3(2) = 0.273+0.553 = 0.826$. So the loss is
$-\log 0.826 = 0.191$.

**Sanity check by brute force:** the paths over $\{a,\varnothing\}^3$ that collapse to `a` are all
$2^3=8$ paths *except* $\varnothing\varnothing\varnothing$. $p(\varnothing^3)=0.3^3=0.027$, so
$p(\text{a}) = 1 - 0.027 = 0.973$? — **No**: paths like $a\varnothing a$ collapse to `a a`, not
`a`. Enumerate: paths collapsing to `a` are exactly those with **at least one $a$ and no two
$a$'s separated by a blank**… the DP already handled this correctly ($0.826$). This is precisely
why you don't enumerate — the DP encodes the collapse rules. (Working out which of the 8 paths are
valid, and summing, reproduces $0.826$ — try it.)

---

## 3.5 Decoding: from probabilities to a transcript

At inference we have $y_t$ but no label. Options:

- **Greedy (best-path):** take $\arg\max_k y_t[k]$ at each frame, then apply $\mathcal{B}$
  (collapse repeats, drop blanks). Fast, what we used:
  ```python
  ids = logits.argmax(-1)          # per-frame best symbol
  out, prev = [], None
  for i in ids:
      if i != prev and i != BLANK: out.append(vocab_inv[i])
      prev = i
  ```
  Greedy is *not* optimal — the most probable *path* ≠ the most probable *label* (many paths sum
  to a label). Usually close enough.

- **Beam search (prefix beam):** keep the top-$B$ *label prefixes* (not paths), merging paths that
  collapse to the same prefix at each step. Recovers some of the probability mass greedy misses.

- **LM fusion / shallow fusion:** add a language-model score,
  $\log p_{\text{CTC}}(\mathbf{l}) + \lambda\log p_{\text{LM}}(\mathbf{l})$, during beam search, to
  prefer plausible sequences. **Caution for this project:** a strong LM would *correct*
  mispronunciations — good for WER, **bad for mispronunciation detection** (it hides the errors we
  must catch). This is why, for tajweed grading, we avoided LM fusion. See Ch. 8.

---

## 3.6 Properties that matter (and bit us)

- **Frame-independence assumption.** CTC assumes outputs are conditionally independent given the
  audio (the $\prod_t y_t$). It has no built-in output language model — hence it's **faithful**
  (won't hallucinate fluent-but-wrong text), which is *desirable* for pronunciation scoring.
- **Peaky posteriors.** In practice CTC learns to emit a symbol on **one** spike frame and blank
  everywhere else. So $y_t[\varnothing]\approx 1$ for most $t$. This is why naive **frame-averaged
  GOP** (Ch. 8) failed on our phoneme model: averaging a symbol's probability over its aligned
  frames includes many near-blank frames and underscores everything. You must score at the spike,
  or use the forward–backward posteriors, not raw frame means.
- **Length constraint.** $T$ must be $\ge$ the number of labels (plus room for blanks between
  repeats). Not an issue for speech (many frames per phoneme).

> **Project link.** `finetuning/train_phoneme_recognizer.py` fine-tunes `wav2vec2-xls-r-300m`
> with `ctc_loss_reduction="mean"` and `pad_token_id = blank`. It reached **PER 17.2%** on the
> IqraEval benchmark — a real connected-speech phoneme recognizer — but 17% frame errors + peaky
> posteriors made reliable *harakat-error* detection hard, which pushed us to NAMAA (Ch. 8).

---

## Exercises
1. Redo §3.4 with $y_t[\text{a}]=0.5$ every frame. Compute $p(\text{a})$ and the loss. Does more
   uncertainty raise or lower the loss?
2. For target `a a` ($U=2$) with $T=3$, write $\mathbf{l}'$ and explain why you *cannot* produce
   `a a` without a blank between the two a's. What's the minimum $T$?
3. Why is greedy decoding suboptimal? Construct a tiny case where the top path collapses to a
   different label than the highest-probability *label*.
4. Explain, using peakiness, why "average the expected phoneme's probability over its frames" is a
   bad Goodness-of-Pronunciation score, and propose a fix.
5. Give one reason CTC is *preferable* to an autoregressive decoder specifically for
   **mispronunciation detection**.

**Next:** [Chapter 4 — Sequence-to-Sequence & Decoding](04-seq2seq-decoding.md): the
autoregressive alternative (Whisper/Cohere/NAMAA), cross-attention, beam search, and hallucination.
