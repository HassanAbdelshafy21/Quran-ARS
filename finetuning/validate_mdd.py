#!/usr/bin/env python3
"""Official MDD evaluation on QuranMB using the gated GT (IqraEval_Test_GT).

For each clip we have: Reference_phn (expected), Annotation_phn (human GT of what was
actually said), and our model's Prediction. Anchoring on the reference, a phoneme is a
TRUE error where Annotation != Reference; the model DETECTS an error where Prediction !=
Reference. From that:
  FRR (false rejection) = FR / (correctly-pronounced ref phonemes)   [scripture-safety gate]
  FAR (false acceptance)= FA / (truly-mispronounced ref phonemes)
  detection P/R/F1 on mispronunciations.
Run in `mlaudio` env:  python finetuning/validate_mdd.py <checkpoint_dir> [n]
"""
import io, sys, json, torch, numpy as np, soundfile as sf, librosa
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor

CK = sys.argv[1] if len(sys.argv) > 1 else "finetuning/checkpoints_phoneme/checkpoint-8000"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 400


def align_to_ref(ref, hyp):
    """Needleman-Wunsch align hyp tokens to ref tokens; return list over ref positions:
    the aligned hyp token (or None if deleted). Substitutions/matches map 1:1."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i][j] = min(dp[i-1][j-1] + c, dp[i-1][j] + 1, dp[i][j-1] + 1)
    # backtrack
    i, j = n, m; out = [None] * n
    while i > 0 and j > 0:
        c = 0 if ref[i-1] == hyp[j-1] else 1
        if dp[i][j] == dp[i-1][j-1] + c: out[i-1] = hyp[j-1]; i -= 1; j -= 1
        elif dp[i][j] == dp[i-1][j] + 1: out[i-1] = None; i -= 1      # ref deleted in hyp
        else: j -= 1                                                  # hyp insertion (ignore)
    return out


def main():
    vocab = json.load(open(f"{CK}/vocab.json")); id2ph = {v: k for k, v in vocab.items()}; BLANK = vocab["[PAD]"]
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    model = Wav2Vec2ForCTC.from_pretrained(CK).to("cuda").eval()

    def predict(y):
        iv = fe(y, sampling_rate=16000, return_tensors="pt").input_values.to("cuda")
        with torch.no_grad(): ids = torch.argmax(model(iv).logits, dim=-1)[0].cpu().tolist()
        out, prev = [], None
        for i in ids:
            if i != prev and i != BLANK: out.append(id2ph.get(i, "?"))
            prev = i
        return out

    p = hf_hub_download("IqraEval/QuranMB.v2", "data/test-00000-of-00001.parquet", repo_type="dataset")
    t = pq.ParquetFile(p).read().to_pydict()
    gt = hf_hub_download("IqraEval/IqraEval_Test_GT", "labels_test.csv", repo_type="dataset")
    import csv
    GT = {r["ID"]: (r["Reference_phn"].split(), r["Annotation_phn"].split())
          for r in csv.DictReader(open(gt, encoding="utf-8"))}

    TA = TR = FR = FA = 0
    for i in range(min(N, len(t["ID"]))):
        ID = t["ID"][i]
        if ID not in GT: continue
        ref, ann = GT[ID]
        y, sr = sf.read(io.BytesIO(t["audio"][i]["bytes"]), dtype="float32")
        if y.ndim > 1: y = y.mean(1)
        if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        pred = predict(y)
        ann_a = align_to_ref(ref, ann)     # what was truly said, per ref position
        pred_a = align_to_ref(ref, pred)   # what model heard, per ref position
        for k in range(len(ref)):
            true_err = (ann_a[k] != ref[k])          # GT: mispronounced here?
            det_err = (pred_a[k] != ref[k])          # model: flagged here?
            if not true_err and not det_err: TA += 1
            elif true_err and det_err: TR += 1
            elif not true_err and det_err: FR += 1   # false rejection
            elif true_err and not det_err: FA += 1   # false acceptance

    corr = TA + FR; mis = TR + FA
    frr = FR / corr if corr else 0
    far = FA / mis if mis else 0
    prec = TR / (TR + FR) if (TR + FR) else 0
    rec = TR / (TR + FA) if (TR + FA) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    print(f"=== MDD on QuranMB ({CK.split('/')[-1]}, n clips scored) ===")
    print(f"phonemes: correct={corr}  mispronounced={mis}")
    print(f"FALSE-REJECTION RATE (flag correct as wrong) = {frr:.3f}   <- scripture-safety gate")
    print(f"FALSE-ACCEPTANCE RATE (miss real errors)     = {far:.3f}")
    print(f"detection  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}")


if __name__ == "__main__":
    main()
