# Chapter 2 — Neural Architectures & Attention

**Goal:** understand, at the level of the actual equations, how the `128 × T` log-mel becomes a
sequence of hidden vectors — through MLPs, CNNs, RNN/LSTMs, and above all the **Transformer** and
the **Conformer** block that Cohere/NAMAA use. By the end you can derive self-attention and
explain every sub-layer of an encoder.

---

## 2.0 The one primitive: a learned linear layer + nonlinearity

Everything is built from the **affine map** $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$
(a matrix multiply) followed by a **nonlinearity** $\sigma$. A stack of these is a **multilayer
perceptron (MLP)**:

$$ \mathbf{h}^{(\ell)} = \sigma\!\left(\mathbf{W}^{(\ell)}\mathbf{h}^{(\ell-1)} + \mathbf{b}^{(\ell)}\right) $$

Nonlinearities you'll meet: **ReLU** $\max(0,x)$, **GELU** $x\,\Phi(x)$ (smooth ReLU, used in
Transformers), **SiLU/Swish** $x\,\sigma(x)$. Without $\sigma$, stacked linear layers collapse to
one linear layer — the nonlinearity is what gives depth its power. Training adjusts all
$\mathbf{W},\mathbf{b}$ by gradient descent on a loss (Ch. 3–5).

Two ubiquitous helpers:
- **Residual connection**: $\mathbf{y} = \mathbf{x} + f(\mathbf{x})$. Lets gradients flow through
  deep stacks (the "+ $\mathbf{x}$" is a gradient highway) and lets each block learn a *change*.
- **Layer normalization**: normalize a vector to zero mean / unit variance across its features,
  then scale+shift by learned $\gamma,\beta$. Stabilizes training. (BatchNorm normalizes across
  the batch instead — used in conv stacks; LayerNorm dominates Transformers.)

---

## 2.1 CNNs — local patterns and downsampling

A **1-D convolution** slides a small learned filter $\mathbf{w}$ (length $K$) across time:

$$ y[t] = \sum_{k=0}^{K-1} w[k]\, x[t + k] + b $$

with many filters producing many output channels. Properties that matter for speech:
- **Locality**: a filter sees a small window — perfect for the *local* shape of a consonant or a
  formant transition.
- **Weight sharing / translation invariance**: the same filter applies everywhere, so a pattern
  is detected regardless of *when* it occurs (fixing the shift-sensitivity from Ch. 1).
- **Stride > 1 downsamples**: a stride-2 conv halves the time length. Speech encoders use a few
  strided convs up front to shrink ~3000 frames to a manageable length (the **subsampling** in a
  Conformer; Cohere uses `dw_striding` with factor 8 → 8× shorter).

CNNs alone (with enough depth) were competitive ASR encoders, but they see only a bounded context
(the **receptive field**). To model long-range dependencies (a word's meaning depends on far-away
words) we want something that connects any two positions directly → attention.

---

## 2.2 RNNs and LSTMs — sequential memory (the pre-Transformer workhorse)

A **recurrent neural network** processes a sequence left-to-right, carrying a hidden state:

$$ \mathbf{h}_t = \sigma\!\left(\mathbf{W}_x \mathbf{x}_t + \mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{b}\right) $$

$\mathbf{h}_t$ is a summary of everything up to $t$. Problem: gradients through many steps
**vanish or explode** (multiplying by $\mathbf{W}_h$ repeatedly), so plain RNNs forget long
context.

The **LSTM (Long Short-Term Memory)** fixes this with a **cell state** $\mathbf{c}_t$ and three
**gates** (each a sigmoid $\in(0,1)$ that multiplies a vector, i.e. "how much to let through"):

$$
\begin{aligned}
\mathbf{f}_t &= \sigma(\mathbf{W}_f[\mathbf{h}_{t-1},\mathbf{x}_t]) &&\text{forget gate} \\
\mathbf{i}_t &= \sigma(\mathbf{W}_i[\mathbf{h}_{t-1},\mathbf{x}_t]) &&\text{input gate} \\
\mathbf{o}_t &= \sigma(\mathbf{W}_o[\mathbf{h}_{t-1},\mathbf{x}_t]) &&\text{output gate} \\
\tilde{\mathbf{c}}_t &= \tanh(\mathbf{W}_c[\mathbf{h}_{t-1},\mathbf{x}_t]) &&\text{candidate} \\
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t &&\text{new cell} \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t) &&\text{output}
\end{aligned}
$$

The additive cell update $\mathbf{c}_t = \mathbf{f}_t\odot\mathbf{c}_{t-1} + \dots$ is the key: if
$\mathbf{f}_t\approx 1$, information is carried unchanged across many steps (a gradient highway in
time). **BiLSTMs** run one LSTM forward and one backward and concatenate, so each frame sees both
past and future. BiLSTM+CTC was *the* ASR encoder for years, and the QDAT tajweed work in this
repo uses an LSTM. But RNNs are **inherently sequential** (can't parallelize over time) and still
struggle with very long range — which is why Transformers took over.

---

## 2.3 The Transformer: attention is the whole idea

**Motivation.** Let every position look directly at every other position and pull in what's
relevant, in one parallel operation. That is **self-attention**.

### 2.3.1 Scaled dot-product attention (derive it)
Given a sequence of vectors stacked as rows of $\mathbf{X}\in\mathbb{R}^{T\times d}$, project into
three roles with learned matrices:

$$ \mathbf{Q}=\mathbf{X}\mathbf{W}_Q,\quad \mathbf{K}=\mathbf{X}\mathbf{W}_K,\quad \mathbf{V}=\mathbf{X}\mathbf{W}_V $$

- **Query** $\mathbf{q}_i$: "what is position $i$ looking for?"
- **Key** $\mathbf{k}_j$: "what does position $j$ offer?"
- **Value** $\mathbf{v}_j$: "the content position $j$ will contribute if attended to."

The **attention weight** from $i$ to $j$ is a normalized similarity of query and key:

$$ \alpha_{ij} = \frac{\exp(\mathbf{q}_i\!\cdot\!\mathbf{k}_j / \sqrt{d_k})}{\sum_{j'} \exp(\mathbf{q}_i\!\cdot\!\mathbf{k}_{j'} / \sqrt{d_k})} $$

and the output at $i$ is the weighted sum of values:

$$ \mathbf{z}_i = \sum_j \alpha_{ij}\,\mathbf{v}_j, \qquad
   \text{i.e. } \mathbf{Z} = \underbrace{\text{softmax}\!\left(\tfrac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)}_{T\times T\text{ attention matrix}}\mathbf{V} $$

**Why $\sqrt{d_k}$?** The dot product $\mathbf{q}\cdot\mathbf{k}$ of two random $d_k$-dim vectors
has variance $\propto d_k$; dividing by $\sqrt{d_k}$ keeps the softmax inputs at a sane scale so
gradients don't vanish for large $d_k$. **Why softmax?** It turns similarities into a probability
distribution over positions (nonnegative, sums to 1) — "how much of my new value comes from each
position."

**Complexity:** the $\mathbf{Q}\mathbf{K}^\top$ matrix is $T\times T$ → $O(T^2 d)$ time and
$O(T^2)$ memory. This quadratic cost in sequence length is *the* practical limit of Transformers
(and why speech encoders subsample with convs first — §2.1 — and why long audio is chunked).

### 2.3.2 Multi-head attention
One attention can only average one way. **Multi-head** runs $h$ attentions in parallel on
different learned projections (each of dim $d_k=d/h$), then concatenates and projects:

$$ \text{MHA}(\mathbf{X}) = \big[\text{head}_1,\dots,\text{head}_h\big]\mathbf{W}_O,\quad
   \text{head}_m = \text{Attn}(\mathbf{X}\mathbf{W}_Q^m,\mathbf{X}\mathbf{W}_K^m,\mathbf{X}\mathbf{W}_V^m) $$

Different heads specialize (one tracks the previous phoneme, one the far context, etc.). Cohere's
decoder attention uses `q_proj/k_proj/v_proj/o_proj` — exactly these four matrices.

### 2.3.3 Positional encoding (attention has no order!)
Self-attention is **permutation-equivariant** — shuffle the inputs and outputs shuffle the same
way; it has *no notion of position*. We must inject order. Two schemes:
- **Absolute (sinusoidal, original Transformer/Whisper):** add to each position a fixed vector
  $ \text{PE}[t,2i]=\sin(t/10000^{2i/d}),\ \text{PE}[t,2i+1]=\cos(t/10000^{2i/d})$. Different
  frequencies encode position; the network can attend "relative" by trig identities.
- **Relative (Conformer):** bias the attention score by a function of $i-j$ (how far apart), so
  the model reasons about *distance* directly. Cohere's config says `self_attention_model: rel_pos`
  — relative positional attention, better for variable-length speech.

### 2.3.4 The full Transformer encoder block
Two sub-layers, each wrapped in residual + LayerNorm:

$$
\begin{aligned}
\mathbf{X}' &= \mathbf{X} + \text{MHA}(\text{LN}(\mathbf{X})) &&\text{(mix across time)}\\
\mathbf{X}'' &= \mathbf{X}' + \text{FFN}(\text{LN}(\mathbf{X}')) &&\text{(process each position)}
\end{aligned}
$$

The **feed-forward network (FFN)** is a per-position MLP that expands then contracts:
$\text{FFN}(\mathbf{x}) = \mathbf{W}_2\,\sigma(\mathbf{W}_1\mathbf{x})$, typically $4\times$ wider
inside (the `fc1`/`fc2` in Cohere). Stack $L$ such blocks → the encoder. Whisper-large-v3 has 32
such layers; a *decoder* block adds a **cross-attention** sub-layer (Ch. 4).

**Intuition of one block:** attention *moves information between time steps* (context mixing); the
FFN *transforms each step's content* (feature computation). Alternate them $L$ times and you get
deep, globally-aware representations.

---

## 2.4 The Conformer block — why Cohere/NAMAA are strong on speech

Pure Transformers model global context well but are weaker at the *local* fine structure that
distinguishes phonemes. **Conformer** = Transformer **+ a convolution module**, giving both. Each
Conformer block is a **"macaron"** sandwich (half-step FFNs on both ends):

$$
\begin{aligned}
\mathbf{x} &\leftarrow \mathbf{x} + \tfrac12\,\text{FFN}(\mathbf{x}) \\
\mathbf{x} &\leftarrow \mathbf{x} + \text{MHSA}_{\text{rel}}(\mathbf{x}) &&\text{global context}\\
\mathbf{x} &\leftarrow \mathbf{x} + \text{Conv}(\mathbf{x}) &&\text{local patterns}\\
\mathbf{x} &\leftarrow \mathbf{x} + \tfrac12\,\text{FFN}(\mathbf{x}) \\
\mathbf{x} &\leftarrow \text{LN}(\mathbf{x})
\end{aligned}
$$

The **Conv module** is: pointwise conv → GLU gating → **depthwise** conv (kernel ~9, per-channel,
cheap, captures local time structure) → BatchNorm → SiLU → pointwise conv. Depthwise-separable
convs are the efficient conv you'll see everywhere.

**Cohere's encoder (from its config), decoded:**
- `n_layers: 48`, `d_model: 1280`, `n_heads: 8` — 48 Conformer blocks, 1280-dim, 8 heads.
- `self_attention_model: rel_pos` — relative positional MHSA.
- `subsampling: dw_striding`, `subsampling_factor: 8` — depthwise-strided front end shrinks time
  8× (so the $O(T^2)$ attention is affordable).
- `conv_kernel_size: 9`, `ff_expansion_factor: 4`, `feat_in: 128` — 128-band log-mel input.

This is exactly why Cohere/NAMAA are **fast and noise-robust**: convolutions nail local acoustic
cues (good under noise), relative attention handles context, and the heavy encoder + *light*
decoder means cheap generation.

---

## 2.5 Encoder vs. decoder, and where this project's models sit

| Model | Encoder | Decoder/head | Notes |
|---|---|---|---|
| Whisper | Transformer (self-attn) | Transformer (self + cross attn), autoregressive | 30 s window |
| Cohere / **NAMAA** | **Conformer** (48 layers) | light Transformer, autoregressive | fast, robust |
| wav2vec2 | CNN feature extractor + Transformer | linear + **CTC** head (no autoregression) | Ch. 3 |

The **encoder** always turns audio → hidden states $\mathbf{H}\in\mathbb{R}^{T'\times d}$. What
differs is how those become symbols: a **CTC head** (Ch. 3) or an **autoregressive decoder**
(Ch. 4). Everything above is shared machinery.

---

## Exercises
1. Self-attention is $O(T^2)$. If subsampling reduces $T=3000$ frames to $T'=375$ (factor 8), by
   what factor does the attention matrix shrink? *(64×.)* Why does this make long-audio ASR
   practical?
2. Write the attention matrix for $T=3$ with queries/keys such that position 2 attends almost
   entirely to position 1. What must $\mathbf{q}_2\cdot\mathbf{k}_1$ be relative to the others?
3. In an LSTM, what value of the forget gate $\mathbf{f}_t$ makes the cell perfectly remember?
   Relate this to the residual connection in a Transformer — what do they have in common?
4. Why does the Conformer put a convolution *between* the attention and the second FFN? What kind
   of tajweed cue (Ch. 7) is inherently *local* and thus conv-friendly (e.g. **qalqalah**'s
   consonant "bounce")?
5. Remove the $\sqrt{d_k}$ scaling and argue what happens to the softmax and gradients as
   $d_k$ grows.

**Next:** [Chapter 3 — CTC](03-ctc.md): turning encoder states into text *without* knowing the
alignment.
