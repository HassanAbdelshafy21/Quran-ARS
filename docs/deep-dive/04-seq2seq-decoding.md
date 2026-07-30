# Chapter 4 — Sequence-to-Sequence & Decoding

**Goal:** understand the autoregressive encoder–decoder used by Whisper, Cohere, and **NAMAA** —
cross-attention, teacher forcing, how generation actually runs (greedy/beam/sampling), and the
mechanism behind **hallucination**. This is the model family we deployed.

---

## 4.1 The setup: condition on audio, generate tokens

A seq2seq ASR factorizes the probability of a transcript $\mathbf{y}=(y_1,\dots,y_U)$ given audio
$\mathbf{x}$ **left to right**:

$$ p(\mathbf{y}\mid\mathbf{x}) = \prod_{u=1}^{U} p(y_u \mid y_{<u}, \mathbf{x}) $$

- The **encoder** (Ch. 2) turns audio into memory $\mathbf{H}=\text{Enc}(\mathbf{x})\in\mathbb{R}^{T'\times d}$.
- The **decoder** produces one token at a time, each conditioned on **all previously generated
  tokens** ($y_{<u}$, via masked self-attention) **and the audio** ($\mathbf{H}$, via
  cross-attention).

Contrast with CTC (Ch. 3): CTC assumes frame independence and has *no* output history; seq2seq
explicitly models $p(y_u\mid y_{<u})$ — a built-in **language model**. That makes seq2seq fluent
but prone to hallucinate (§4.5).

---

## 4.2 The decoder block: three sub-layers

Each decoder layer has, in order (each residual + LayerNorm wrapped):

1. **Masked self-attention** over the tokens generated so far. "Masked" = a token at position $u$
   may only attend to positions $\le u$ (causal mask: set attention logits for future positions
   to $-\infty$ before softmax). This enforces the left-to-right factorization.
2. **Cross-attention**: queries come from the decoder state; **keys and values come from the
   encoder memory $\mathbf{H}$**. This is where the decoder "looks at the audio" —
   $ \text{softmax}(\mathbf{Q}_{\text{dec}}\mathbf{K}_{\text{enc}}^\top/\sqrt{d})\,\mathbf{V}_{\text{enc}} $.
   The attention weights form a soft **alignment** between output tokens and audio frames.
3. **FFN** (per-position MLP), as in the encoder.

A final linear **projection head** $\mathbf{W}_{\text{out}}$ maps the decoder state to
vocabulary-size **logits**, and softmax gives $p(y_u\mid y_{<u},\mathbf{x})$.

> **Cohere/NAMAA specifics:** the decoder is deliberately **light** (few layers) — most compute is
> in the 48-layer Conformer encoder. Its projections are `q_proj/k_proj/v_proj/o_proj` (attention)
> and `fc1/fc2` (FFN); the output head `proj_out` maps to a 16,384-token vocab and its weights are
> **tied** to the token embedding (a common trick: reuse the embedding matrix as the output
> classifier, saving parameters and improving quality).

---

## 4.3 Prompt tokens / task conditioning

These decoders are steered by special **prompt tokens** prepended to the output sequence. NAMAA's
processor seeds the decoder with, e.g.,
`<|startofcontext|><|startoftranscript|><|ar|><|ar|><|pnc|>…` — telling it "language = Arabic,
with punctuation, transcribe." Whisper does the same (`<|startoftranscript|><|ar|><|transcribe|>`).
The model learned during training to condition its output on these — which is how one model does
multiple languages/tasks. (In training, you also mask these prompt tokens out of the loss so the
model isn't graded on reproducing the fixed prompt.)

---

## 4.4 Training: teacher forcing and cross-entropy

You don't generate during training — that would be slow and unstable. Instead, **teacher
forcing**: feed the *ground-truth* previous tokens $y_{<u}$ and, in parallel for all $u$, predict
the next token. The loss is token-level **cross-entropy**:

$$ \mathcal{L} = -\sum_{u=1}^{U} \log p_\theta(y_u \mid y_{<u}, \mathbf{x}) $$

The causal mask lets this run in **one parallel pass** (each position sees only earlier
positions). Two practical notes you'll meet in the code:
- **Shift**: the decoder input is the target shifted right by one (prepend a start token); the
  labels are the target. HuggingFace models do this internally when you pass `labels=`.
- **Exposure bias**: at train time the decoder always sees *correct* history; at test time it sees
  its *own* (possibly wrong) outputs. This mismatch is one root of hallucination/derailing.

> **A real bug we hit:** the raw remote-code Cohere applied `log_softmax` in the head *and* then a
> `CrossEntropyLoss` (which applies log-softmax again) — a **double softmax** producing a wrong
> loss. The native `transformers` implementation was correct. Lesson: always verify what the loss
> actually computes before a long training run.

---

## 4.5 Inference: how generation actually runs

We generate token by token, each step: run the decoder on the tokens-so-far, get logits, pick the
next token, append, repeat until an end-of-sequence token or a max length.

**Decoding strategies:**
- **Greedy:** $y_u=\arg\max p(\cdot\mid y_{<u},\mathbf{x})$. Fast; can commit early to a bad token.
- **Beam search:** keep the top-$B$ *partial sequences* (beams) by cumulative log-prob; expand
  each, keep the best $B$. Approximately maximizes $\log p(\mathbf{y}\mid\mathbf{x})$. Better
  transcripts, $B\times$ slower. (Whisper defaults to beams; we used greedy/num_beams=1 for speed
  in tests.)
- **Sampling (temperature / top-k / top-p):** draw from the distribution instead of taking the max
  — used for diversity, generally *not* for ASR (we want the single most likely transcript). A
  **temperature** $\tau$ rescales logits $a/\tau$ before softmax: $\tau<1$ sharpens (more greedy),
  $\tau>1$ flattens.
- **KV-cache:** the decoder recomputes attention over all previous tokens each step; caching the
  per-layer keys/values makes generation $O(U)$ instead of $O(U^2)$. This is why `past_key_values`
  appears in the code.

**Cost:** autoregressive generation is inherently sequential ($U$ steps, can't parallelize over
output positions), which is why the **light decoder** in Cohere/NAMAA matters so much for latency.

---

## 4.6 Why seq2seq models *hallucinate* (and why it mattered here)

Because the decoder has a strong internal language model $p(y_u\mid y_{<u})$, on **unclear or
silent audio** it can generate fluent, plausible text that the cross-attention only weakly
supports — repeating phrases, inventing words, or "completing" a verse. Mechanistically:
- cross-attention gives little signal (audio unclear) → the token distribution is dominated by the
  language-model prior → it emits whatever is linguistically likely.
- exposure bias then compounds: one hallucinated token conditions the next.

**Consequences we measured:**
- We explicitly **tested faithfulness**: on mis-recited RetaSy clips, Cohere transcribed the
  *actual wrong words* rather than "correcting" them toward the target — good (it's a relatively
  faithful seq2seq). But general seq2seq models on very short/unclear kid clips did drift.
- This is the deep reason a **CTC** model (Ch. 3) is safer for pure faithfulness, and why, for
  the final tajweed grading, we rely on NAMAA writing the **acoustic** diacritic rather than any
  LM-fused decoding (Ch. 8).

**Mitigations** (general knowledge): no-repeat n-gram blocking, suppressing blank/again tokens,
a voice-activity detector to avoid decoding silence (NAMAA's card recommends a VAD), and
confidence thresholds on the average token log-prob.

---

## 4.7 Whisper's 30-second window (a concrete design point)

Whisper always consumes exactly 30 s of log-mel (pad or trim). Longer audio is **chunked** and
transcriptions stitched — which can cause boundary errors and repetition across chunks. Cohere's
processor similarly splits long audio and reassembles via a returned `audio_chunk_index`. For our
per-ayah/short-recitation use, clips fit in one window, so this isn't an issue — but it's a classic
gotcha you'll meet with long-form ASR.

---

## Exercises
1. Write out $p(\mathbf{y}\mid\mathbf{x})$ for $\mathbf{y}=(y_1,y_2,y_3)$ and identify which
   attention (self vs cross) supplies $y_{<u}$ and which supplies $\mathbf{x}$.
2. In beam search with $B=2$, hand-simulate two steps given a tiny vocabulary and made-up
   probabilities; show how a sequence that was 2nd-best at step 1 can win by step 2.
3. Why does temperature $\tau\to 0$ make sampling equivalent to greedy? Show it from the softmax.
4. Explain, in terms of cross-attention strength and the LM prior, why hallucination is worst on
   *silence* — and why a VAD helps.
5. Compare CTC and seq2seq on three axes: parallel training, output language model, faithfulness.
   Which would you choose for (a) a dictation app, (b) mispronunciation detection? Justify.

**Next:** [Chapter 5 — Self-Supervised Learning & Fine-Tuning](05-ssl-and-finetuning.md): where
wav2vec2's features come from, and the LoRA math.
