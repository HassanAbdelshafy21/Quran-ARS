#!/usr/bin/env python3
"""Quran-safe diacritization: given Cohere's bare transcription + the known
target ayah(s) (canonical diacritized), restore EXACT mushaf tashkeel for
correctly-recited words by alignment; fall back to CATT only for genuinely
mis-recited words (no canonical answer). Demo on adults (expect exact) + kids."""
import os, sys, re, sqlite3, random
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, librosa, torch, difflib
try: import torch.distributed.tensor  # noqa
except Exception: pass
sys.path.insert(0, "delivery")
from core.segmenter import AudioSegmenter
from datasets import load_from_disk
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from catt_tashkeel import CATTEncoderDecoder

MARK = re.compile("[ﰀ-﷿ﹰ-﻿]")
# full Arabic diacritic/annotation ranges (standard harakat + Uthmani marks) + tatweel
DIAC = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭ࣓-ࣿـ]")
def bare(w):
    w = w.replace("ٰ", "ا")  # dagger-alef (long-a) -> explicit alef (matches Cohere)
    w = DIAC.sub("", w)
    w = re.sub("[أإآٱ]", "ا", w)  # hamza-alef variants -> alef
    w = w.replace("ة", "ه").replace("ى", "ي")  # ta-marbuta->ha, alef-maqsura->ya
    w = w.replace("ؤ", "و").replace("ئ", "ي").replace("ء", "")  # hamza carriers
    return w.strip()

seg = AudioSegmenter(); conn = sqlite3.connect("data/quran.db")
proc = AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026")
cm = CohereAsrForConditionalGeneration.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026", torch_dtype=torch.bfloat16).to("cuda").eval()
catt = CATTEncoderDecoder()

def coh(y):
    y = np.asarray(y, dtype=np.float32)
    inp = proc(y, sampling_rate=16000, return_tensors="pt", language="ar"); inp.to(cm.device, dtype=cm.dtype)
    o = cm.generate(**inp, max_new_tokens=256); d = proc.decode(o, skip_special_tokens=True)
    return (d[0] if isinstance(d, list) else d).strip()

def canon_surah(s):
    c = conn.cursor(); c.execute("SELECT aya_text FROM quran WHERE sura_no=? ORDER BY aya_no", (s,))
    return MARK.sub("", " ".join(r[0] for r in c.fetchall())).strip()

def hybrid(spoken_text, canon_text):
    """exact canonical mushaf for matched words; CATT for mis-recited words."""
    sp = spoken_text.split(); cn = canon_text.split()
    sm = difflib.SequenceMatcher(None, [bare(w) for w in sp], [bare(w) for w in cn], autojunk=False)
    out, errs, n_can = [], [], 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.extend(cn[j1:j2]); n_can += (j2 - j1)      # EXACT mushaf tashkeel
        elif tag in ("replace", "insert"):
            for w in sp[i1:i2]:
                d = catt.do_tashkeel(w, verbose=False).strip(); out.append(d); errs.append((w, d))
    return " ".join(out), errs, n_can

print("=" * 72, "\nADULTS  (correct recitation -> expect EXACT canonical mushaf)\n")
ds = load_from_disk("data/quran_dataset_v6_allages"); random.seed(3)
for r in random.sample([x for x in ds if x["reciter"] != "Minshawi_Child"], 4):
    y, _ = librosa.load(r["audio"], sr=16000)
    sp = coh(y); rec, errs, ncan = hybrid(sp, r["text"])
    print(f"COHERE(bare): {sp}")
    print(f"HYBRID out  : {rec}")
    print(f"CANONICAL   : {r['text']}")
    print(f"  -> EXACT MATCH: {rec.strip() == r['text'].strip()} | from-canonical: {ncan} words | CATT: {len(errs)}\n")

print("=" * 72, "\nKIDS  (correct words -> canonical; real errors -> CATT)\n")
for fn, s in {"test 4.mp4": 95, "test 5.mp4": 112, "test 6.mp4": 109}.items():
    segs = seg.segment_file("finetuning/test_samples/" + fn)
    sp = " ".join(coh(x["audio_data"]) for x in segs)
    rec, errs, ncan = hybrid(sp, canon_surah(s))
    print(f"### {fn}")
    print(f"COHERE(bare): {sp}")
    print(f"HYBRID out  : {rec}")
    print(f"  from-canonical(exact mushaf): {ncan} words | CATT-fallback: {[e[1] for e in errs]}\n")
