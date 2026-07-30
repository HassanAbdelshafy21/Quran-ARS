#!/usr/bin/env python3
"""Validate fine-tuned Cohere checkpoints: tashkeel presence + accuracy on kids,
adults, and noisy RetaSy vs raw-Cohere baselines. Run in `mlaudio` env.
  python finetuning/validate_cohere.py finetuning/checkpoints_cohere_tashkeel/checkpoint-*"""
import os, sys, re, io, glob, random
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch, numpy as np, librosa, soundfile as sf, sqlite3, pyarrow.parquet as pq
try: import torch.distributed.tensor  # noqa
except Exception: pass
sys.path.insert(0, "delivery")
from core.segmenter import AudioSegmenter
from core.grader import QuranGrader
from datasets import load_from_disk
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from peft import PeftModel

MODEL = "CohereLabs/cohere-transcribe-03-2026"
MARK = re.compile(r"[ﰀ-﷿]"); DIAC = re.compile(r"[ً-ْٰ]")
g = QuranGrader(); seg = AudioSegmenter(); conn = sqlite3.connect("data/quran.db")
proc = AutoProcessor.from_pretrained(MODEL)

def gt(s):
    c = conn.cursor(); c.execute("SELECT aya_text FROM quran WHERE sura_no=? ORDER BY aya_no", (s,))
    return MARK.sub("", " ".join(r[0] for r in c.fetchall())).strip()

# preload eval sets
kids = {"test 4.mp4": 95, "test 5.mp4": 112, "test 6.mp4": 109}
ksegs = {fn: seg.segment_file("finetuning/test_samples/" + fn) for fn in kids}
ds = load_from_disk("data/quran_dataset_v6_allages")
random.seed(7); adults = random.sample([r for r in ds if r["reciter"] != "Minshawi_Child"], 15)
reta = []
for f in sorted(glob.glob("data/dataset_cache/RetaSy/*.parquet")):
    t = pq.read_table(f, columns=["audio", "Aya", "final_label"]).to_pydict()
    for i in range(len(t["audio"])):
        if t["final_label"][i] != "correct" or not t["Aya"][i]: continue
        try:
            y, sr = sf.read(io.BytesIO(t["audio"][i]["bytes"]), dtype="float32")
            if y.ndim > 1: y = y.mean(1)
            if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            if len(y) >= 4800: reta.append((y, MARK.sub("", t["Aya"][i]).strip()))
        except Exception: pass
        if len(reta) >= 25: break
    if len(reta) >= 25: break

def make_tr(model):
    def tr(y):
        y = np.asarray(y, dtype=np.float32)
        inp = proc(y, sampling_rate=16000, return_tensors="pt", language="ar")
        inp.to(model.device, dtype=model.dtype)
        out = model.generate(**inp, max_new_tokens=256)
        d = proc.decode(out, skip_special_tokens=True)
        return (d[0] if isinstance(d, list) else d).strip()
    return tr

def evaluate(model, label):
    tr = make_tr(model); tk = 0; n = 0
    kacc = []
    for fn, s in kids.items():
        txt = " ".join(tr(x["audio_data"]) for x in ksegs[fn])
        kacc.append(g.grade(txt, gt(s))["accuracy"]); tk += bool(DIAC.search(txt)); n += 1
    aacc = []
    for r in adults:
        y, _ = librosa.load(r["audio"], sr=16000); txt = tr(y)
        aacc.append(g.grade(txt, r["text"])["accuracy"]); tk += bool(DIAC.search(txt)); n += 1
    racc = []
    for y, ref in reta:
        txt = tr(y); racc.append(g.grade(txt, ref)["accuracy"]); tk += bool(DIAC.search(txt)); n += 1
    print(f"{label:26s} | kids {kacc[0]:.2f}/{kacc[1]:.2f}/{kacc[2]:.2f} | "
          f"adults {np.mean(aacc):.2f} | noisy {np.mean(racc):.2f} | tashkeel {tk}/{n}", flush=True)

print("baselines (raw Cohere): kids .85/.60/1.0 | adults 1.0 | noisy .85 | tashkeel 0/N\n")
base = CohereAsrForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda").eval()
for ck in sys.argv[1:]:
    m = PeftModel.from_pretrained(base, ck).to("cuda").eval()
    evaluate(m, os.path.basename(ck.rstrip("/")))
    m = m.unload()  # remove adapter, restore base for next
