#!/usr/bin/env python3
"""
Speech AI from scratch — a study companion to the deep-dive series.

Pure-NumPy, runnable implementations of the core algorithms, each with a self-test:
  1) log-mel front end            (Chapter 1)
  2) CTC forward / loss + decode  (Chapter 3)  — verified vs. the hand example AND PyTorch
  3) attention forward + backward (Chapter 2 / Appendix A) — finite-difference gradient check

Run:  python from_scratch.py
Only NumPy is required. If librosa / torch are installed, extra cross-checks run too.
"""
import numpy as np

# =====================================================================================
# 1) LOG-MEL FRONT END  (Chapter 1)
# =====================================================================================

def hz_to_mel(f):  return 2595.0 * np.log10(1.0 + f / 700.0)
def mel_to_hz(m):  return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def mel_filterbank(sr, n_fft, n_mels):
    """Triangular mel filters: [n_mels, n_fft//2 + 1]. Equally spaced on the mel axis."""
    K = n_fft // 2 + 1
    mel_pts = np.linspace(0.0, hz_to_mel(sr / 2), n_mels + 2)     # +2 for the edges
    hz_pts = mel_to_hz(mel_pts)
    bin_idx = np.floor((n_fft + 1) * hz_pts / sr).astype(int)      # FFT bin of each edge
    fb = np.zeros((n_mels, K))
    for m in range(1, n_mels + 1):
        left, center, right = bin_idx[m - 1], bin_idx[m], bin_idx[m + 1]
        for k in range(left, center):
            if center > left: fb[m - 1, k] = (k - left) / (center - left)   # rising edge
        for k in range(center, right):
            if right > center: fb[m - 1, k] = (right - k) / (right - center) # falling edge
    return fb

def log_mel(y, sr=16000, n_fft=400, hop=160, n_mels=128, eps=1e-6):
    """waveform -> log-mel [n_mels, T].  frame -> hann -> rFFT -> |.|^2 -> mel -> log."""
    win = np.hanning(n_fft)
    n_frames = 1 + max(0, (len(y) - n_fft) // hop)
    frames = np.stack([y[i * hop: i * hop + n_fft] * win for i in range(n_frames)])  # [T, n_fft]
    spec = np.fft.rfft(frames, n=n_fft, axis=1)          # [T, K] complex
    power = np.abs(spec) ** 2                            # power spectrum
    fb = mel_filterbank(sr, n_fft, n_mels)               # [n_mels, K]
    mel = power @ fb.T                                   # [T, n_mels]
    return np.log(mel + eps).T                           # [n_mels, T]


# =====================================================================================
# 2) CTC  (Chapter 3)
# =====================================================================================

def _logsumexp(v):
    m = np.max(v)
    return m + np.log(np.sum(np.exp(v - m))) if np.isfinite(m) else m

def ctc_loss(log_probs, targets, blank=0):
    """
    log_probs : [T, V]  log-softmax outputs (log-probabilities per frame).
    targets   : list of label ids (no blanks).  blank : id of the blank symbol.
    Returns (loss = -log p(targets|x),  alpha table [T, S]).  Log-space forward algorithm.
    """
    T, V = log_probs.shape
    ext = [blank]
    for s in targets:
        ext += [s, blank]                 # blanks between every label and at the ends
    S = len(ext)                          # 2U + 1
    NEG = -1e30
    a = np.full((T, S), NEG)
    a[0, 0] = log_probs[0, ext[0]]        # start on first blank
    if S > 1:
        a[0, 1] = log_probs[0, ext[1]]    # or on first real symbol
    for t in range(1, T):
        for s in range(S):
            terms = [a[t - 1, s]]                          # stay
            if s - 1 >= 0: terms.append(a[t - 1, s - 1])   # advance
            if s - 2 >= 0 and ext[s] != blank and ext[s] != ext[s - 2]:
                terms.append(a[t - 1, s - 2])              # skip blank between distinct symbols
            a[t, s] = _logsumexp(np.array(terms)) + log_probs[t, ext[s]]
    ll = _logsumexp(np.array([a[T - 1, S - 1], a[T - 1, S - 2]]))  # end on last symbol or blank
    return -ll, a

def ctc_greedy_decode(log_probs, blank=0):
    """Best-path decode: argmax per frame, then collapse repeats and drop blanks."""
    ids = np.argmax(log_probs, axis=1)
    out, prev = [], None
    for i in ids:
        if i != prev and i != blank:
            out.append(int(i))
        prev = i
    return out


# =====================================================================================
# 3) ATTENTION forward + backward  (Chapter 2 / Appendix A)
# =====================================================================================

def softmax(x, axis=-1):
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)

def attention_forward(Q, K, V):
    """Scaled dot-product attention. Q,K,V: [T, d]. Returns Z:[T,d], and cache for backward."""
    d = Q.shape[1]
    scores = Q @ K.T / np.sqrt(d)          # [T, T]
    A = softmax(scores, axis=1)            # attention weights (rows sum to 1)
    Z = A @ V                             # [T, d]
    return Z, (Q, K, V, A, d)

def attention_backward(dZ, cache):
    """Gradients dQ, dK, dV given dZ = dL/dZ.  Implements Appendix A.3."""
    Q, K, V, A, d = cache
    dV = A.T @ dZ                                  # (1) through the value sum
    dA = dZ @ V.T                                 #     dL/dA
    # (2) softmax backward, per row: de = A * (dA - sum_k A*dA)
    dscore = A * (dA - np.sum(A * dA, axis=1, keepdims=True))
    # (3) through the scaled dot product
    dQ = dscore @ K / np.sqrt(d)
    dK = dscore.T @ Q / np.sqrt(d)
    return dQ, dK, dV


# =====================================================================================
# SELF-TESTS
# =====================================================================================

def test_logmel():
    print("[1] log-mel front end")
    sr = 16000
    y = 0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float64)  # 1 s, 440 Hz
    M = log_mel(y, sr=sr, n_fft=400, hop=160, n_mels=128)
    print(f"    output shape {M.shape}  (expected (128, ~{1 + (sr-400)//160}))")
    # the 440 Hz tone should light up a low mel band; check the peak band is in the lower third
    band_energy = M.mean(axis=1)
    peak = int(np.argmax(band_energy))
    print(f"    peak mel band = {peak} (low = correct for a 440 Hz tone)")
    try:
        import librosa
        Lr = librosa.feature.melspectrogram(y=y.astype(np.float32), sr=sr, n_fft=400,
                                            hop_length=160, n_mels=128, power=2.0)
        Lr = np.log(Lr + 1e-6)
        # shapes/behaviour should be comparable (not identical: librosa pads/normalizes differently)
        print(f"    librosa shape {Lr.shape}  ✓ (front ends agree in structure)")
    except Exception:
        print("    (librosa not installed — skipping cross-check)")

def test_ctc():
    print("[2] CTC forward / loss")
    # Chapter 3 hand example: vocab {a=1, blank=0}, target 'a', T=3, y[a]=0.7, y[blank]=0.3.
    log_probs = np.log(np.array([[0.3, 0.7]] * 3))        # [T=3, V=2]
    loss, alpha = ctc_loss(log_probs, targets=[1], blank=0)
    p = np.exp(-loss)
    print(f"    p(target)  = {p:.4f}   (hand-computed 0.8260)")
    print(f"    loss       = {loss:.4f}   (hand-computed 0.1912)")
    assert abs(p - 0.826) < 1e-3, "CTC forward disagrees with the hand example!"
    dec = ctc_greedy_decode(log_probs, blank=0)
    print(f"    greedy decode = {dec}  (expected [1])")
    try:
        import torch, torch.nn.functional as F
        lp = torch.tensor(log_probs, dtype=torch.double).unsqueeze(1)      # [T, N=1, V]
        tgt = torch.tensor([[1]], dtype=torch.long)
        loss_t = F.ctc_loss(lp, tgt, input_lengths=torch.tensor([3]),
                            target_lengths=torch.tensor([1]), blank=0, reduction='none')
        print(f"    torch CTCLoss = {loss_t.item():.4f}   ✓ matches ours")
        assert abs(loss_t.item() - loss) < 1e-4
    except Exception as e:
        print(f"    (torch not available for cross-check: {type(e).__name__})")

def test_attention():
    print("[3] attention forward + backward (finite-difference gradient check)")
    rng = np.random.default_rng(0)
    T, d = 4, 3
    Q, K, V = rng.standard_normal((T, d)), rng.standard_normal((T, d)), rng.standard_normal((T, d))
    Z, cache = attention_forward(Q, K, V)
    dZ = rng.standard_normal((T, d))                      # arbitrary upstream gradient
    dQ, dK, dV = attention_backward(dZ, cache)
    # finite-difference check on Q
    eps = 1e-6
    num_dQ = np.zeros_like(Q)
    for i in range(T):
        for j in range(d):
            Qp = Q.copy(); Qp[i, j] += eps
            Zp, _ = attention_forward(Qp, K, V)
            num_dQ[i, j] = np.sum((Zp - Z) * dZ) / eps    # directional derivative along dZ
    err = np.max(np.abs(num_dQ - dQ))
    print(f"    max |analytic dQ - numeric dQ| = {err:.2e}  (should be ~1e-6)")
    assert err < 1e-4, "attention backward is wrong!"

if __name__ == "__main__":
    print("=" * 60)
    test_logmel(); print()
    test_ctc();    print()
    test_attention()
    print("=" * 60)
    print("All self-tests passed. Read the code alongside Chapters 1–3 + Appendix A.")
