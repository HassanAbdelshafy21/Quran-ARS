# Chapter 5 — Self-Supervised Learning & Fine-Tuning

**Goal:** understand where wav2vec2's powerful features come from *without labels* (contrastive
pretraining + a learned quantizer), the pretrain→finetune recipe, and the **LoRA** math you used
to adapt big models cheaply.

---

## 5.1 Why self-supervision

Labeled speech (audio + transcript) is scarce and expensive; **unlabeled** audio is unlimited.
**Self-supervised learning (SSL)** invents a training signal from the data itself, learning
general speech representations that a small labeled set can then specialize. This is why
`wav2vec2-xls-r-300m` (pretrained on 436k hours across 128 languages) transfers so well to Arabic
phonemes with only 79 hours of labels (our phoneme recognizer, Ch. 3/8).

---

## 5.2 wav2vec2, precisely

Three components:

1. **CNN feature encoder** $f:\text{waveform}\to\mathbf{Z}$. A stack of strided 1-D convolutions
   turns raw 16 kHz audio directly into latent vectors $\mathbf{z}_t$ every ~20 ms (≈ a 50 Hz
   frame rate). **No hand-crafted log-mel** — the features are learned from the waveform.
2. **Transformer context network** $g:\mathbf{Z}\to\mathbf{C}$. Contextualized representations
   $\mathbf{c}_t$ (each $\mathbf{c}_t$ sees the whole utterance), as in Ch. 2.
3. **Quantizer** $q:\mathbf{z}_t\to\mathbf{q}_t$. Maps each latent to one of a **finite codebook**
   of discrete units via **product quantization** with the **Gumbel-softmax** trick (a
   differentiable approximation to picking a codebook entry). These discrete units are the
   "pseudo-labels" the model predicts.

### The pretraining task (masked contrastive prediction)
Randomly **mask** spans of the latent sequence $\mathbf{Z}$ (like BERT masks words). At each
masked time step $t$, the context vector $\mathbf{c}_t$ must **identify the true quantized latent
$\mathbf{q}_t$** among distractors sampled from other masked steps. The **contrastive loss**:

$$ \mathcal{L}_t = -\log \frac{\exp\!\big(\text{sim}(\mathbf{c}_t,\mathbf{q}_t)/\kappa\big)}
   {\sum_{\tilde{\mathbf{q}}\in Q_t}\exp\!\big(\text{sim}(\mathbf{c}_t,\tilde{\mathbf{q}})/\kappa\big)} $$

where $\text{sim}$ is cosine similarity, $\kappa$ a temperature, and $Q_t$ = the true unit + $K$
distractors. Intuition: "from context, predict which discrete sound unit was here" — the model
must learn phonetic structure to win. A **diversity** loss encourages using the whole codebook.
No transcripts were used anywhere.

### HuBERT (the cousin you'll see)
Same spirit, different target: cluster MFCC/features with k-means to get frame labels, then train
the Transformer to **predict the cluster id** of masked frames (a classification loss), iterating
the clustering on better features. IqraEval's baselines include `hubert_base` and `wavlm_base`
(WavLM adds denoising + speaker tasks). All are "pretrained speech encoders you fine-tune."

---

## 5.3 The pretrain → finetune recipe

1. **Pretrain** (someone already did this on huge unlabeled audio) → a general encoder.
2. **Add a task head** and **fine-tune on labels**:
   - For phoneme/character recognition: a linear layer + **CTC** (Ch. 3). This is exactly
     `Wav2Vec2ForCTC`.
   - Often **freeze the CNN feature encoder** early (it's already good, and it stabilizes
     training) — `model.freeze_feature_encoder()` in our trainer.
3. Train with a small **learning rate** and a warmup+decay schedule; the pretrained weights move
   gently while the new head learns fast.

> **Project link.** `finetuning/train_phoneme_recognizer.py`: base `wav2vec2-xls-r-300m`,
> `vocab_size=68` (IqraEval phoneme set), `ctc_loss_reduction="mean"`, feature encoder frozen,
> AdamW, cosine LR, ~30k steps. It went from loss 26 → ~0.5 and **PER 24.7% → 17.2%**. The loss
> curve flattening told us more data/epochs would help only slowly.

---

## 5.4 Full fine-tuning vs. parameter-efficient fine-tuning (PEFT)

Updating **all** weights of a 2B model per task is expensive (optimizer state ~2–3× the model in
memory), risks **catastrophic forgetting** (destroying pretrained skills), and yields a full-size
checkpoint per task. **PEFT** methods change only a tiny fraction of parameters.

### LoRA (Low-Rank Adaptation) — the math
Key empirical fact: the *update* a task needs to a big weight matrix is approximately **low-rank**.
So instead of learning a full $\Delta\mathbf{W}\in\mathbb{R}^{d\times d}$, factor it:

$$ \mathbf{W}' = \mathbf{W} + \Delta\mathbf{W}, \qquad \Delta\mathbf{W} = \frac{\alpha}{r}\,\mathbf{B}\mathbf{A},
   \quad \mathbf{A}\in\mathbb{R}^{r\times d},\ \mathbf{B}\in\mathbb{R}^{d\times r} $$

- **$\mathbf{W}$ is frozen**; only $\mathbf{A},\mathbf{B}$ train. With rank $r\ll d$ (e.g. 16–32),
  that's $2rd$ parameters instead of $d^2$ — often **<0.3%** of the model.
- **Init:** $\mathbf{A}$ random (small), $\mathbf{B}=\mathbf{0}$, so $\Delta\mathbf{W}=0$ at start —
  training begins exactly at the pretrained model (safe).
- **$\alpha$ (alpha)** is a scaling; the effective update is $\frac{\alpha}{r}\mathbf{B}\mathbf{A}$.
  A common heuristic is $\alpha=2r$.
- **Where to inject:** usually the attention projections `q_proj`,`v_proj` (and sometimes
  `k_proj`,`o_proj`,`fc1`,`fc2`). More targets = more capacity.
- **Merge for inference:** $\mathbf{W}'=\mathbf{W}+\frac{\alpha}{r}\mathbf{B}\mathbf{A}$ can be
  folded back into one matrix → **zero extra latency** at deploy. Adapters are a few MB and
  **reversible** (keep the base pristine).

**Why it's perfect for adapting a frozen giant:** cheap, fast, non-destructive, swappable.

> **Project links.** LoRA on Whisper's `q_proj/v_proj` for the everyayah fine-tunes; LoRA on
> Cohere's **decoder** (encoder frozen, to preserve acoustic robustness) in the tashkeel
> experiments. Crucial negative result: even wide LoRA (q/k/v/o + FFN) could **not** make base
> Cohere emit tashkeel without wrecking words — the pretrained "no-diacritics" prior was too
> strong. That told us the fix had to be a model *trained* for acoustic diacritics (NAMAA), not an
> adapter. **PEFT can adapt a skill the base already has latent; it can't easily install a skill
> the base actively resists.**

---

## 5.5 Optimization essentials (for completeness)

- **Loss → gradients → update.** Backpropagation computes $\partial\mathcal{L}/\partial\theta$;
  **AdamW** updates each parameter using running estimates of the gradient mean and variance
  (adaptive step sizes) plus decoupled weight decay (L2 regularization).
- **Learning-rate schedule.** **Warmup** (ramp up over the first few hundred steps) avoids early
  instability; **cosine decay** anneals to ~0. We used warmup+cosine everywhere.
- **Precision.** **bf16** (bfloat16) has the same exponent range as fp32 but fewer mantissa bits —
  it trains stably where **fp16 overflows** (fp16's small range overflowed an attention mask value
  and produced garbage — a real bug we fixed by using bf16).
- **Gradient accumulation.** Simulate a big batch on small VRAM by summing gradients over several
  micro-batches before an optimizer step (effective batch = `batch_size × grad_accum`).
- **Gradient clipping.** Cap the gradient norm (e.g. 1.0) to prevent rare exploding updates.
- **Checkpoint often; resume on crash.** Our phoneme run died overnight on a transient CUDA
  fault; a resilient wrapper resumed from the latest checkpoint. Always save frequently.

---

## Exercises
1. A weight is $1024\times1024$. Full fine-tune = how many params? LoRA with $r=16$ = how many?
   What fraction? *(1.05M vs 32,768 → ~3.1%.)*
2. Why is $\mathbf{B}$ initialized to zero? What would happen to early training if both $\mathbf{A}$
   and $\mathbf{B}$ were random?
3. In the wav2vec2 contrastive loss, what happens to $\mathcal{L}_t$ if the true unit and a
   distractor are equally similar to $\mathbf{c}_t$? What does that push the encoder to do?
4. Explain "catastrophic forgetting" and two ways LoRA mitigates it.
5. Why does freezing the CNN feature extractor stabilize CTC fine-tuning early on?

**Next:** [Chapter 6 — Evaluation & Alignment](06-evaluation-and-alignment.md): edit distance,
WER/CER/PER/DER, Needleman–Wunsch, and detection metrics.
