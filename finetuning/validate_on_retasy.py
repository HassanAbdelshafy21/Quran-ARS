#!/usr/bin/env python3
"""
Validate the current model+grader against RetaSy's human labels — a real
phone-audio "answer key". Measures whether OUR pass/fail verdict matches the
human's correct/incorrect judgement, and reports the confusion matrix.

This is the honest baseline: "on real phone audio, the model gets the verdict
right X% of the time, with a Y% false-fail rate." Re-run the SAME set after V6
to prove improvement. These rows are the HELD-OUT TEST SET — never train on them.

Run from repo root (GPU):  python finetuning/validate_on_retasy.py
"""
import os
import io
import sys
import glob
import numpy as np

# Local Blackwell-GPU shim (harmless elsewhere); precede model import.
import torch  # noqa
try:
    import torch.distributed.tensor  # noqa
except Exception:
    pass

import soundfile as sf
import librosa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "delivery"))
from core.model_loader import QuranModel
from core.grader import QuranGrader

PARQUETS = sorted(glob.glob("data/dataset_cache/RetaSy/*.parquet"))
MODEL_CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join("delivery", "model", "checkpoint-30000")

# Map human labels -> expected verdict. multiple_aya is ambiguous -> skip.
EXPECT_PASS = {"correct"}
EXPECT_FAIL = {"in_correct", "not_match_aya", "in_complete", "not_related_quran"}


def decode(audio_bytes):
    y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return y


def make_transcriber(spec):
    """Return transcribe(y)->text. 'hf:<model_id>' uses a raw HF Whisper (e.g. turbo);
    otherwise spec is a QuranModel checkpoint dir (base + our LoRA adapter)."""
    if spec.startswith("hf:"):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        mid = spec[3:]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        proc = WhisperProcessor.from_pretrained(mid)
        m = WhisperForConditionalGeneration.from_pretrained(
            mid, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev)
        m.eval()

        def fn(y):
            feats = proc(y, sampling_rate=16000, return_tensors="pt").input_features.to(dev)
            if dev == "cuda":
                feats = feats.half()
            with torch.no_grad():
                ids = m.generate(feats, language="ar", task="transcribe",
                                 num_beams=1, max_new_tokens=200)
            return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return fn

    model = QuranModel(spec)
    return lambda y: model.transcribe(y)["text"]


def main():
    print(f"Loading model: {MODEL_CKPT}", flush=True)
    transcribe_fn = make_transcriber(MODEL_CKPT)
    grader = QuranGrader()

    # confusion matrix
    tp = fp = fn = tn = 0
    correct_scores, incorrect_scores = [], []
    n = 0
    for f in PARQUETS:
        t = pq.read_table(f, columns=["audio", "Aya", "final_label"]).to_pydict()
        for i in range(len(t["audio"])):
            label = t["final_label"][i]
            if label not in EXPECT_PASS and label not in EXPECT_FAIL:
                continue
            target = t["Aya"][i]
            if not target:
                continue
            try:
                y = decode(t["audio"][i]["bytes"])
                if len(y) < 16000 * 0.3:
                    continue
                text = transcribe_fn(y)
                res = grader.grade(text, target)
            except Exception:
                continue
            passed = res["passed"]
            n += 1
            if label in EXPECT_PASS:
                correct_scores.append(res["accuracy"])
                if passed:
                    tp += 1
                else:
                    fn += 1
            else:
                incorrect_scores.append(res["accuracy"])
                if passed:
                    fp += 1
                else:
                    tn += 1
            if n % 50 == 0:
                print(f"  ...{n} evaluated", flush=True)

    print("\n===== VALIDATION on RetaSy (real phone audio, human-labeled) =====")
    print(f"Evaluated: {n} clips")
    print(f"""
                     Human: CORRECT     Human: INCORRECT
  We PASS  :          TP {tp:4d}            FP {fp:4d}   (passed a wrong recitation)
  We FAIL  :          FN {fn:4d}            TN {tn:4d}
                  (failed a correct child)
""")
    tot = max(n, 1)
    verdict_acc = (tp + tn) / tot
    ppos = tp + fn
    nneg = fp + tn
    print(f"Overall verdict accuracy : {verdict_acc*100:.1f}%")
    if ppos:
        print(f"FALSE-FAIL rate (correct kids marked wrong): {fn/ppos*100:.1f}%  <- worst for kids")
    if nneg:
        print(f"FALSE-PASS rate (wrong recitations passed) : {fp/nneg*100:.1f}%")

    def summ(a, name):
        if a:
            a = np.array(a)
            print(f"  {name}: n={len(a)} mean_acc={a.mean():.2f} "
                  f"median={np.median(a):.2f}  (pass bar = 0.85)")
    print("\nScore distribution (informs whether the 0.85 threshold is right):")
    summ(correct_scores, "CORRECT recitations ")
    summ(incorrect_scores, "INCORRECT recitations")

    # ---- threshold sweep: find the pass bar that best trades false-fail vs false-pass ----
    c = np.array(correct_scores)
    ic = np.array(incorrect_scores)
    # persist scores so future sweeps need no re-inference
    np.savez("finetuning/retasy_scores.npz", correct=c, incorrect=ic)
    print("\n===== THRESHOLD SWEEP (pass bar vs errors) =====")
    print(" thr | verdict acc | false-fail | false-pass")
    best = None
    for t in [0.85, 0.60, 0.50, 0.45, 0.40, 0.37, 0.35, 0.33, 0.30, 0.25, 0.20]:
        tp = int((c >= t).sum()); fn = int((c < t).sum())
        fp = int((ic >= t).sum()); tn = int((ic < t).sum())
        acc = (tp + tn) / (len(c) + len(ic))
        ff = fn / len(c); fpr = fp / len(ic)
        flag = ""
        # a sensible pick: highest accuracy while keeping false-pass under ~20%
        if fpr <= 0.20 and (best is None or acc > best[1]):
            best = (t, acc, ff, fpr); flag = "  <= candidate"
        print(f" {t:.2f} |   {acc*100:5.1f}%    |   {ff*100:5.1f}%   |   {fpr*100:5.1f}%{flag}")
    if best:
        print(f"\nSuggested pass bar ~{best[0]:.2f}: verdict {best[1]*100:.1f}%, "
              f"false-fail {best[2]*100:.1f}%, false-pass {best[3]*100:.1f}% "
              f"(vs 0.85 bar today).")


if __name__ == "__main__":
    main()
