# Speech AI — Deep Dive Series

A rigorous, self-study path into speech processing and ASR, grounded in the Quran-ARS project.
This goes **deep**: the DSP math, the architectures at the layer level, CTC's forward–backward
algorithm, attention, decoding, evaluation, and pronunciation assessment. It assumes you can read
basic linear algebra and probability; everything else is built up.

If the [Learning Guide](../Quran-ARS-Learning-Guide.md) is the *tour*, this is the *textbook*.

## How to read this
Read in order — each chapter uses the previous. Every chapter has: **intuition → the math →
a worked example → how this project uses it → exercises**. Work the exercises; speech is learned
by doing.

## Chapters
1. **[Signals & Features](01-signals-and-features.md)** — sampling, quantization, framing,
   the DFT/FFT, the power spectrum, mel filterbanks, log-mel, MFCC/DCT, deltas. How raw audio
   becomes model input.
2. **[Neural Architectures & Attention](02-neural-architectures-attention.md)** — MLPs, CNNs,
   RNN/LSTM, the Transformer (self-attention math step by step), positional encoding, and the
   Conformer block that Cohere/NAMAA use.
3. **[CTC](03-ctc.md)** — the alignment problem, the CTC collapse, the forward–backward
   algorithm and loss (with a full numeric example), greedy/beam decoding, LM fusion, peakiness.
4. **[Sequence-to-Sequence & Decoding](04-seq2seq-decoding.md)** — encoder–decoder, cross-
   attention, teacher forcing, autoregressive generation, beam search, temperature/sampling,
   and *why* seq2seq models hallucinate.
5. **[Self-Supervised Learning & Fine-Tuning](05-ssl-and-finetuning.md)** — wav2vec2's
   contrastive objective and quantizer, HuBERT, the pretrain→finetune recipe, transfer, and the
   LoRA math we used.
6. **[Evaluation & Alignment](06-evaluation-and-alignment.md)** — Levenshtein edit distance,
   WER/CER/PER/DER computed exactly, Needleman–Wunsch, and detection metrics (FRR/FAR, precision/
   recall/F1, ROC) for pronunciation assessment.
7. **[Phonetics & the Arabic/Quran Domain](07-phonetics-and-arabic.md)** — the source–filter
   model, formants, the IPA, Arabic phonology, and the acoustics of tajweed (madd, ghunnah,
   idghaam, qalqalah) — what the model must actually hear.
8. **[Pronunciation Assessment (MDD)](08-pronunciation-assessment.md)** — the mispronunciation
   detection & diagnosis task, Goodness-of-Pronunciation (GOP) math, forced alignment,
   text vs. audio diacritization, and how this project ended up doing it with one model.

## Notation
- Scalars $x$, vectors $\mathbf{x}$, matrices $\mathbf{X}$. Time index $t$ or $n$; frequency
  index $k$; layer index $\ell$.
- $\mathbb{R}$ reals, $\mathbb{E}[\cdot]$ expectation, $\odot$ elementwise product.
- Audio sample rate $f_s$ (Hz); a signal $x[n]$ is a sequence of samples.
