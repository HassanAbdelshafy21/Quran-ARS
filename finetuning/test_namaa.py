#!/usr/bin/env python3
"""Test NAMAA-Space/Cohere-Speech-Tashkeel-2B as a potential single model.
(1) word accuracy + tashkeel on our kids/adults; (2) the decisive ACOUSTIC test:
on QuranMB clips where GT Annotation != Reference (a real pronunciation error),
does NAMAA's diacritized output reflect the ACTUAL error or "correct" to canonical?
Run in `mlaudio` env."""
import os, sys, re, io, sqlite3, csv
os.environ.setdefault("HF_HUB_OFFLINE", "0")
import torch, numpy as np, librosa, soundfile as sf
sys.path.insert(0, "delivery")
from core.segmenter import AudioSegmenter
from core.grader import QuranGrader
from datasets import load_from_disk
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

M = "NAMAA-Space/Cohere-Speech-Tashkeel-2B"
MARK = re.compile("[ﰀ-﷿ﹰ-﻿]"); DIAC = re.compile("[ً-ٰ]")
seg = AudioSegmenter(); g = QuranGrader(); conn = sqlite3.connect("data/quran.db")
proc = AutoProcessor.from_pretrained(M)
model = CohereAsrForConditionalGeneration.from_pretrained(M, torch_dtype=torch.bfloat16).to("cuda").eval()

def tr(y):
    y = np.asarray(y, dtype=np.float32)
    inp = proc(y, sampling_rate=16000, return_tensors="pt", language="ar"); inp.to(model.device, dtype=model.dtype)
    out = model.generate(**inp, max_new_tokens=256); d = proc.decode(out, skip_special_tokens=True)
    return (d[0] if isinstance(d, list) else d).strip()

def surah(s):
    c = conn.cursor(); c.execute("SELECT aya_text FROM quran WHERE sura_no=? ORDER BY aya_no", (s,))
    return MARK.sub("", " ".join(r[0] for r in c.fetchall())).strip()

print("=" * 70, "\n(1) KIDS + ADULTS: word accuracy + tashkeel\n")
for fn, s in {"test 4.mp4": 95, "test 5.mp4": 112, "test 6.mp4": 109}.items():
    segs = seg.segment_file("finetuning/test_samples/" + fn)
    txt = " ".join(tr(x["audio_data"]) for x in segs)
    print(f"{fn} (surah {s}): acc={g.grade(txt, surah(s))['accuracy']:.2f} tashkeel={bool(DIAC.search(txt))}")
    print(f"   OUT: {txt[:95]}\n")
ds = load_from_disk("data/quran_dataset_v6_allages"); import random; random.seed(3)
for r in random.sample([x for x in ds if x["reciter"] != "Minshawi_Child"], 3):
    y, _ = librosa.load(r["audio"], sr=16000); txt = tr(y)
    print(f"adult: acc={g.grade(txt, r['text'])['accuracy']:.2f} | OUT: {txt}  | TGT: {r['text']}")

print("\n" + "=" * 70, "\n(2) DECISIVE ACOUSTIC TEST on QuranMB (GT Annotation != Reference)\n")
p = hf_hub_download("IqraEval/QuranMB.v2", "data/test-00000-of-00001.parquet", repo_type="dataset")
t = pq.ParquetFile(p).read().to_pydict()
gt = hf_hub_download("IqraEval/IqraEval_Test_GT", "labels_test.csv", repo_type="dataset")
GT = {r["ID"]: (r["Reference_phn"].strip(), r["Annotation_phn"].strip())
      for r in csv.DictReader(open(gt, encoding="utf-8"))}
shown = 0
for i in range(len(t["ID"])):
    ID = t["ID"][i]
    if ID not in GT: continue
    ref, ann = GT[ID]
    if ref == ann: continue           # only clips with a real pronunciation deviation
    y, sr = sf.read(io.BytesIO(t["audio"][i]["bytes"]), dtype="float32")
    if y.ndim > 1: y = y.mean(1)
    if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    out = tr(y)
    print(f"ID {ID}")
    print(f"  REFERENCE (expected): {ref[:80]}")
    print(f"  ANNOTATION (actual) : {ann[:80]}")
    print(f"  NAMAA output        : {out[:80]}\n")
    shown += 1
    if shown >= 8: break
