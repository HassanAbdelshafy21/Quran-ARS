# Chapter 1 — Signals & Features

**Goal:** understand exactly how a sound in the air becomes the `128 × T` array a neural network
sees, and *why* each step is there. By the end you can implement a log-mel front end from scratch
and explain every parameter (window length, hop, `n_fft`, mel bands).

---

## 1.1 What a digital audio signal *is*

A microphone converts air pressure $p(t)$ (a continuous function of time) into a voltage, and an
**analog-to-digital converter (ADC)** does two things:

1. **Sampling** — measure the voltage every $T_s$ seconds. The **sample rate** is
   $f_s = 1/T_s$. Speech uses $f_s = 16{,}000$ Hz (16 kHz). The signal becomes a sequence
   $x[n] = p(nT_s)$, $n = 0,1,2,\dots$
2. **Quantization** — round each measurement to a finite number of levels. 16-bit audio has
   $2^{16} = 65{,}536$ levels. This adds a tiny **quantization noise**; with 16 bits the
   signal-to-noise ratio is about $6.02 \times 16 \approx 96$ dB — inaudible. (Derivation: each
   extra bit halves the step size, adding $20\log_{10}2 \approx 6.02$ dB of SNR.)

So "audio" in code is just a 1-D array of floats (we normalize the integers to $[-1, 1]$).
10 s at 16 kHz $= 160{,}000$ numbers.

### The Sampling Theorem (why 16 kHz?)
**Nyquist–Shannon:** a signal containing no frequencies above $B$ Hz is *perfectly* reconstructed
from samples taken at rate $f_s > 2B$. The highest representable frequency is the **Nyquist
frequency** $f_s/2$.

At $f_s = 16$ kHz we can represent up to **8 kHz**. Human speech energy that carries phonetic
information lives mostly below 8 kHz (vowel formants < 4 kHz; fricatives like *s*, *sh* up to
~8 kHz). That's why 16 kHz is the ASR standard: it captures the speech-relevant band while
halving the data vs. 32 kHz. Frequencies above Nyquist, if present, **alias** (fold back) into
the band as fake low tones — so ADCs apply an **anti-alias low-pass filter** before sampling.

> **Project link:** every model here resamples input to 16 kHz first (`librosa.load(..., sr=16000)`).
> A phone recording at 44.1 kHz is downsampled; a low-passed/compressed phone signal loses the
> 4–8 kHz band, which is one reason phone audio is harder.

---

## 1.2 Why not feed raw samples to the network?

You *can* (some models do), but raw waveforms are:
- **High-rate and long**: 160k numbers for 10 s. Expensive.
- **Phase-sensitive and shift-sensitive**: the same word starting 5 ms later is a very different
  array of numbers, though it sounds identical. The network would waste capacity learning
  invariances we can hand it for free.

The **short-time Fourier transform (STFT)** turns the waveform into a time–frequency
representation that is compact and where "which sounds are present" is explicit. That is the
front end used by Whisper, Cohere, and NAMAA.

---

## 1.3 Frequency: the Discrete Fourier Transform (DFT)

**Idea:** any finite signal can be written as a sum of sinusoids of different frequencies. The
DFT tells you *how much* of each frequency is present.

For a frame of $N$ samples $x[0..N-1]$, the DFT produces $N$ complex numbers:

$$ X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-j 2\pi k n / N}, \qquad k = 0,1,\dots,N-1 $$

- $e^{-j2\pi kn/N} = \cos(2\pi kn/N) - j\sin(2\pi kn/N)$ is a probe sinusoid at frequency
  bin $k$. The sum correlates the signal with that sinusoid.
- Bin $k$ corresponds to physical frequency $f_k = k\, f_s / N$ Hz. Bins $0..N/2$ are the real
  frequencies $0..f_s/2$; the upper half is a mirror image (for real signals), so we keep
  $N/2+1$ bins.
- $|X[k]|$ is the **magnitude** (how much of that frequency); $\angle X[k]$ is the **phase**.
  For speech features we usually keep magnitude and discard phase.

**The FFT** (Fast Fourier Transform) is just an $O(N\log N)$ algorithm to compute the DFT instead
of the naive $O(N^2)$. Same output. This is why `n_fft` is a power of two (e.g. 400 → padded to
512): the FFT is fastest and cleanest there.

**Worked micro-example.** Take $N=4$, $x=[1,0,-1,0]$ (a half-cycle). Then
$X[1]=\sum_n x[n]e^{-j\pi n/2} = 1\cdot1 + 0 + (-1)(-1) + 0 = 2$, and $X[0]=0$ (no DC/mean),
so the DFT correctly reports energy at bin 1 (a sinusoid completing one cycle over 4 samples =
$f_s/4$) and none at DC. Compute $X[2]$ yourself and check it's 0.

---

## 1.4 Short-time analysis: framing and windowing

Speech is **non-stationary** — its spectrum changes as you move from one phoneme to the next. A
single DFT over a whole utterance would blur everything together. Solution: chop the signal into
short **frames** that are approximately stationary, and DFT each.

- **Frame length** ≈ 25 ms. At 16 kHz that's 400 samples. Why 25 ms? Long enough to resolve
  the low pitches / formants (frequency resolution $\approx f_s/N$; 400 samples → 40 Hz bins),
  short enough that the phoneme doesn't change much within it.
- **Hop (stride)** ≈ 10 ms = 160 samples. Frames **overlap** (25 ms frames every 10 ms), so we
  get a smooth sequence of ~100 frames per second. This overlap is what makes the spectrogram
  a continuous "movie."

**Windowing.** Cutting a hard rectangular frame creates discontinuities at the edges, and the DFT
of a hard edge smears energy across all bins (**spectral leakage**). We multiply each frame by a
tapered **window** $w[n]$ that goes to zero at the edges. The **Hann window**:

$$ w[n] = 0.5\left(1 - \cos\!\frac{2\pi n}{N-1}\right) $$

reduces leakage. So each frame is $x_w[n] = w[n]\,x[n]$ before the FFT.

**Putting it together (the STFT):**

$$ \text{STFT}[t,k] = \sum_{n=0}^{N-1} w[n]\, x[t\cdot H + n]\, e^{-j2\pi kn/N} $$

where $H$ is the hop and $t$ indexes frames. The **power spectrum** of frame $t$ is
$P[t,k] = |\text{STFT}[t,k]|^2$. A **spectrogram** is $P$ (usually shown in dB).

> **Project parameters (Whisper-style, shared by Cohere/NAMAA front ends):** 25 ms window,
> 10 ms hop, `n_fft = 400` (→ 201 usable bins), 30 s clips → ~3000 frames before mel pooling.

---

## 1.5 Perceptual frequency: the mel scale

Human hearing does not perceive frequency linearly. We discriminate low pitches finely and high
pitches coarsely — an octave from 100→200 Hz sounds like the same "distance" as 1000→2000 Hz. The
**mel scale** models this. A common formula:

$$ m(f) = 2595 \,\log_{10}\!\left(1 + \frac{f}{700}\right), \qquad
   f(m) = 700\left(10^{m/2595} - 1\right) $$

We build a **mel filterbank**: $M$ triangular filters (e.g. $M=80$ or $128$) equally spaced *on
the mel axis*, each summing the power spectrum over a band. Filter $i$ has response
$H_i[k]$ (a triangle rising from its left edge to its center, falling to its right edge), and:

$$ S[t,i] = \sum_{k} H_i[k]\, P[t,k] $$

This compresses the $201$ linear FFT bins into $M$ perceptually spaced bands. Low-frequency
bands are narrow (fine resolution where formants live); high-frequency bands are wide.

> **Project link:** Whisper/turbo use 80 mel bins; large-v3 and the Cohere/NAMAA front end use
> **128**. More bands = finer spectral detail, useful for fine phonetic distinctions.

---

## 1.6 The log: matching loudness and dynamic range

Perceived **loudness** is roughly logarithmic (decibels). Also, speech power spans a huge dynamic
range (a vowel is orders of magnitude louder than silence). We take the log:

$$ \tilde{S}[t,i] = \log\!\big(S[t,i] + \epsilon\big) $$

($\epsilon$ avoids $\log 0$.) This is the **log-mel spectrogram** — the standard neural ASR input.
It compresses dynamic range and makes multiplicative effects (like a channel gain) **additive**,
which the network handles more easily. Typically we also **normalize** (subtract mean, divide by
std) per utterance or with dataset statistics.

**That's the whole front end:** `waveform → frame+window → FFT → |·|² → mel filterbank → log →
normalize → log-mel [M × T]`. That array is what the encoder in Chapter 2 consumes.

---

## 1.7 MFCCs and the DCT (classical features you should know)

Before deep learning, systems used **MFCCs (Mel-Frequency Cepstral Coefficients)**. One more step
after log-mel: apply a **Discrete Cosine Transform (DCT)** across the mel axis:

$$ c[t,j] = \sum_{i=1}^{M} \tilde{S}[t,i]\, \cos\!\left[\frac{\pi j (i - \tfrac12)}{M}\right],
   \quad j = 0,\dots,J-1 $$

and keep the first $J \approx 13$ coefficients. Why the DCT?
- It **decorrelates** the mel bands (neighboring mel bands are correlated; GMM-based models
  needed roughly independent features).
- The **cepstrum** separates the slowly-varying **spectral envelope** (the vocal-tract shape →
  the phoneme identity, low DCT indices) from the fast **pitch harmonics** (high indices). Low
  MFCCs capture *what* was said; they discard the speaker's pitch — useful for speaker-independent
  recognition.

**Deltas.** Classical systems appended $\Delta$ (velocity) and $\Delta\Delta$ (acceleration) —
finite differences over time — to inject dynamics:
$\Delta c[t] \approx \frac{\sum_{\tau} \tau\,(c[t+\tau]-c[t-\tau])}{2\sum_\tau \tau^2}$.

**Do modern models use MFCCs?** Mostly **no** — deep networks prefer **log-mel** (the DCT throws
away information a neural net can exploit, and CNNs like the correlated structure). But you must
know MFCCs: they dominate the literature, small/edge models still use them, and the QDAT tajweed
work (Ch. 8) used MFCC+LSTM. The Wav2Vec2 line skips hand features entirely and learns from the
waveform (Ch. 5) — the modern extreme.

---

## 1.8 The whole pipeline in code (annotated)

```python
import numpy as np, librosa

y, sr = librosa.load("clip.wav", sr=16000)          # 1. resample to 16 kHz, floats in [-1,1]
S = librosa.feature.melspectrogram(                  # 2..5 in one call:
        y=y, sr=sr,
        n_fft=400,          # DFT size (25 ms)  -> frequency resolution
        hop_length=160,     # 10 ms stride      -> ~100 frames/sec
        win_length=400,     # window length
        window="hann",      # taper (reduce leakage)
        n_mels=128,         # mel filterbank size
        power=2.0)          # |STFT|^2 power spectrum
log_mel = np.log(S + 1e-6)                            # 6. log dynamic-range compression
log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)   # 7. normalize
# log_mel.shape == (128, T),  T ≈ len(y)/hop_length
```

Every ASR model in this repo does exactly this internally (the `AutoProcessor` /
`Wav2Vec2FeatureExtractor` wraps it).

---

## 1.9 Why this matters for *this* project

- **Kid voices** have higher pitch (higher $f_0$, ~250–400 Hz vs. ~120 for men). Pitch shows up
  as harmonic spacing; the **mel + log** front end and the deep encoder learn to be pitch-robust,
  but very high pitch with sparse harmonics can under-sample the formants — part of why kids are
  harder.
- **Phone channel**: band-limiting (losing 4–8 kHz) and codec compression distort the upper mel
  bands (fricatives like *s/sh/kh*). This is visible in the log-mel and is a real source of error.
- **Short vowels (harakat)** are brief — they occupy few frames. With 10 ms hop a 60 ms vowel is
  ~6 frames. That's why harakat detection is fundamentally an *acoustic-resolution* problem, and
  why a model that "hears" them (NAMAA) is remarkable.

---

## Exercises
1. At $f_s=16$ kHz, `n_fft=400`: what physical frequency does FFT bin $k=50$ correspond to? What
   is the frequency spacing between bins? *(Answer: $f_k = k f_s/N = 50\cdot16000/400 = 2000$ Hz;
   spacing $= 40$ Hz.)*
2. A 25 ms window gives 40 Hz frequency resolution. If you wanted 20 Hz resolution, what window
   length (ms) do you need, and what do you lose? *(50 ms; worse time resolution — the
   time–frequency uncertainty trade-off.)*
3. Implement the mel formula: how many Hz is 1000 mel? Is the mel-per-Hz slope larger at 200 Hz
   or 5000 Hz? Explain what that means for resolving vowel formants.
4. Take a 1 s clip, compute its log-mel, then reconstruct audio from magnitude only (Griffin–Lim).
   Why does it sound "phasey"? What information did we throw away, and does ASR need it?
5. Why do modern neural ASR models prefer log-mel over MFCC? Give two reasons tied to §1.7.

**Next:** [Chapter 2 — Neural Architectures & Attention](02-neural-architectures-attention.md),
where the `128 × T` log-mel becomes hidden states.
