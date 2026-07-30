# Appendix A — Worked Examples: Viterbi/Forced Alignment & Attention Backprop

Two things people usually only half-understand: (1) how a decoder finds the single best path
(Viterbi) and how that becomes **forced alignment**, and (2) how gradients flow **backward**
through attention. We do both fully, by hand.

---

## A.1 Viterbi: the single most-likely path

CTC's forward algorithm (Ch. 3) *sums* over paths (probability of a label). **Viterbi** instead
finds the **single most-likely path** — a *max* where CTC had a *sum*. It's the backbone of HMM
decoding and of forced alignment.

### Setup
States $s \in \{1,\dots,S\}$, time $t \in \{1,\dots,T\}$. Emission $b_s(t)$ = probability that
state $s$ produced observation $t$ (from the acoustic model). Transition $a_{s'\to s}$ = probability
of moving from $s'$ to $s$. Define

$$ \delta_t(s) = \max_{\text{paths ending in } s \text{ at } t}\; P(\text{path}, \text{obs}_{1:t}) $$

**Recursion (max-product):**

$$ \delta_t(s) = b_s(t)\cdot \max_{s'}\big[\delta_{t-1}(s')\,a_{s'\to s}\big], \qquad
   \psi_t(s) = \arg\max_{s'}\big[\delta_{t-1}(s')\,a_{s'\to s}\big] $$

$\psi_t(s)$ is a **backpointer**: which previous state was best. At the end, take
$s^\* = \arg\max_s \delta_T(s)$ and **backtrack** $\psi$ from $t=T$ down to $1$ to read off the best
state sequence. Work in **log space** (sum of log-probs) to avoid underflow — max is unchanged.

### Worked example (3 states, 3 frames)
States $\{1,2,3\}$ (say phonemes q, u, l). Left-to-right transitions only: you may **stay** ($s\to
s$) or **advance** ($s\to s{+}1$), each with prob 0.5; start in state 1. Emissions (rows = state,
cols = frame), already the values $b_s(t)$:

| $b_s(t)$ | $t{=}1$ | $t{=}2$ | $t{=}3$ |
|---|---|---|---|
| s=1 (q) | 0.6 | 0.1 | 0.1 |
| s=2 (u) | 0.2 | 0.7 | 0.3 |
| s=3 (l) | 0.1 | 0.2 | 0.6 |

**t=1** (must start in state 1): $\delta_1(1)=0.6$, others unreachable ($\delta_1(2)=\delta_1(3)=0$).

**t=2:**
- $\delta_2(1)=b_1(2)\cdot\delta_1(1)\,a_{1\to1}=0.1\cdot0.6\cdot0.5=0.030$, $\psi=1$.
- $\delta_2(2)=b_2(2)\cdot\max[\delta_1(2)a_{2\to2},\,\delta_1(1)a_{1\to2}] =0.7\cdot\max[0,\,0.6\cdot0.5]=0.7\cdot0.30=0.210$, $\psi=1$.
- $\delta_2(3)=b_3(2)\cdot\max[\delta_1(3)a_{3\to3},\,\delta_1(2)a_{2\to3}]=0.2\cdot\max[0,0]=0$.

**t=3:**
- $\delta_3(2)=b_2(3)\cdot\max[\delta_2(2)a_{2\to2},\,\delta_2(1)a_{1\to2}]=0.3\cdot\max[0.210\cdot0.5,\,0.030\cdot0.5]=0.3\cdot0.105=0.0315$, $\psi=2$.
- $\delta_3(3)=b_3(3)\cdot\max[\delta_2(3)a_{3\to3},\,\delta_2(2)a_{2\to3}]=0.6\cdot\max[0,\,0.210\cdot0.5]=0.6\cdot0.105=0.063$, $\psi=2$.
- $\delta_3(1)=0.1\cdot(0.030\cdot0.5)=0.0015$.

**Best final** $s^\*=\arg\max_s\delta_3(s)=3$ (0.063). Backtrack: $\psi_3(3)=2$, $\psi_2(2)=1$.
Best path = **state 1 → 2 → 3** = frame1 is q, frame2 is u, frame3 is l. That's the alignment
`q u l` with one frame each. Probability of that path $=0.063$ (times start). ✔

### Viterbi vs. CTC-forward
- **Viterbi** = max over paths → the *single best* alignment (a hard segmentation).
- **CTC forward** = sum over paths → the *total* label probability (used for the loss).
Same DP skeleton, `max` vs. `logsumexp`.

---

## A.2 Forced alignment (constrain to a known transcript)

**Forced alignment** = "I already know *what* was said; tell me *when* each unit occurred." You run
the same max DP but the state sequence is **fixed to the known transcript** (you cannot output any
other symbols) — you only choose the *durations* (how many frames each unit holds).

With a **CTC** model this is: build the extended sequence $\mathbf{l}'$ (blanks between symbols,
Ch. 3), then run **Viterbi over $\mathbf{l}'$** with the CTC transition rules (stay / advance /
skip-a-blank-between-distinct-symbols). The backtracked path gives each symbol's frame span. That is
exactly what `torchaudio.functional.forced_align` computes, and what our Layer-2 prototype used to
get per-word time spans before scoring.

**Why it can go wrong** (Ch. 8): if the acoustic model is weak/out-of-domain, the forced path puts
boundaries in the wrong places, and any per-phoneme score computed on those frames is unreliable.
Forced alignment is only as good as the acoustic model underneath it.

**Micro-exercise.** Using §A.1's emissions but *forcing* the transcript `q l` (skip u), redo the DP
with only states {q, l} and left-to-right transitions. Where does the u-heavy frame 2 get assigned,
and what does the low probability tell you? (It reveals the transcript doesn't fit the audio — the
seed of mispronunciation detection.)

---

## A.3 The attention backward pass (gradients through softmax attention)

Forward (single head, Ch. 2), for one query $i$:

$$ e_{ij} = \frac{\mathbf{q}_i\!\cdot\!\mathbf{k}_j}{\sqrt{d}},\qquad
   \alpha_{ij} = \frac{e^{e_{ij}}}{\sum_{j'} e^{e_{ij'}}},\qquad
   \mathbf{z}_i = \sum_j \alpha_{ij}\,\mathbf{v}_j $$

Given the upstream gradient $\mathbf{g}_i = \partial \mathcal{L}/\partial \mathbf{z}_i$, we want the
gradients w.r.t. $\mathbf{v}, \mathbf{q}, \mathbf{k}$.

**(1) Through the value sum.** $\mathbf{z}_i=\sum_j\alpha_{ij}\mathbf{v}_j$, so

$$ \frac{\partial\mathcal{L}}{\partial \mathbf{v}_j} = \sum_i \alpha_{ij}\,\mathbf{g}_i,
   \qquad
   \frac{\partial\mathcal{L}}{\partial \alpha_{ij}} = \mathbf{g}_i\!\cdot\!\mathbf{v}_j =: d\alpha_{ij}. $$

**(2) Through the softmax.** The Jacobian of softmax is
$\partial\alpha_{ij}/\partial e_{ik} = \alpha_{ij}(\delta_{jk}-\alpha_{ik})$, so

$$ \frac{\partial\mathcal{L}}{\partial e_{ij}}
   = \sum_k d\alpha_{ik}\,\alpha_{ik}(\delta_{jk}-\alpha_{ij})
   = \alpha_{ij}\Big(d\alpha_{ij} - \underbrace{\textstyle\sum_k \alpha_{ik}\,d\alpha_{ik}}_{\bar d_i}\Big). $$

This neat "**subtract the weighted average**" form ($de_{ij}=\alpha_{ij}(d\alpha_{ij}-\bar d_i)$) is
the softmax backward you'll implement everywhere.

**(3) Through the scaled dot product.** $e_{ij}=\mathbf{q}_i\!\cdot\!\mathbf{k}_j/\sqrt{d}$, so

$$ \frac{\partial\mathcal{L}}{\partial \mathbf{q}_i} = \frac{1}{\sqrt d}\sum_j de_{ij}\,\mathbf{k}_j,
   \qquad
   \frac{\partial\mathcal{L}}{\partial \mathbf{k}_j} = \frac{1}{\sqrt d}\sum_i de_{ij}\,\mathbf{q}_i. $$

Then $\mathbf{q},\mathbf{k},\mathbf{v}$ each backprop into their projection matrices
$\mathbf{W}_Q,\mathbf{W}_K,\mathbf{W}_V$ by the usual linear-layer rule
$\partial\mathcal{L}/\partial\mathbf{W} = \mathbf{X}^\top(\partial\mathcal{L}/\partial(\mathbf{X}\mathbf{W}))$.

**Sanity intuition:** a value vector's gradient is the attention-weighted sum of downstream
gradients (positions that attended to it more get more credit); the score gradient is "how much
raising this score would help, minus the average over the row" (because softmax is a competition —
raising one logit lowers the others).

You can verify all of this numerically with finite differences; the from-scratch code
(`from_scratch.py`) includes a gradient check for the softmax step.

---

## A.4 Where each shows up in this project
- **Viterbi/forced alignment** → `torchaudio.forced_align` in the Layer-2 phoneme prototype
  (per-word spans), and the conceptual basis of GOP (Ch. 8).
- **Attention backward** → runs inside every training step of every model here; understanding it
  demystifies why attention is stable (softmax normalization) and where compute goes ($T^2$).

*See `from_scratch.py` for runnable implementations of the log-mel front end, the CTC forward/loss
(verified against Chapter 3's hand example and PyTorch), CTC greedy decode, and an attention
forward+backward with a finite-difference gradient check.*
