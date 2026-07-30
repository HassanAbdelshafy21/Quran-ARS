#!/usr/bin/env python3
"""Layer 1 output demo: for each test recitation, run Cohere -> grader -> produce the
product report: score, per-word correct/wrong/missed, Arabic feedback, and the
diacritized display (canonical mushaf for correct words, CATT for mis-recited). Run in `mlaudio`."""
import os, sys, re, sqlite3
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch
sys.path.insert(0, "delivery")
from core.segmenter import AudioSegmenter
from core.grader import QuranGrader
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from catt_tashkeel import CATTEncoderDecoder

MARK = re.compile("[ﰀ-﷿ﹰ-﻿]"); DIAC = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭ࣓-ࣿـ]")
def bare(w):
    w = w.replace("ٰ", "ا"); w = DIAC.sub("", w)
    return re.sub("[أإآٱ]", "ا", w).replace("ة", "ه").replace("ى", "ي").replace("ؤ","و").replace("ئ","ي").replace("ء","").strip()

seg = AudioSegmenter(); grader = QuranGrader(); conn = sqlite3.connect("data/quran.db")
proc = AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026")
cm = CohereAsrForConditionalGeneration.from_pretrained("CohereLabs/cohere-transcribe-03-2026", torch_dtype=torch.bfloat16).to("cuda").eval()
catt = CATTEncoderDecoder()

def coh(y):
    y = np.asarray(y, dtype=np.float32)
    inp = proc(y, sampling_rate=16000, return_tensors="pt", language="ar"); inp.to(cm.device, dtype=cm.dtype)
    o = cm.generate(**inp, max_new_tokens=256); d = proc.decode(o, skip_special_tokens=True)
    return (d[0] if isinstance(d, list) else d).strip()

def surah_words_diac(s):
    c = conn.cursor(); c.execute("SELECT aya_text FROM quran WHERE sura_no=? ORDER BY aya_no", (s,))
    return MARK.sub("", " ".join(r[0] for r in c.fetchall())).strip().split()

SURAH = {95: "التِّين", 112: "الإخلاص", 109: "الكافرون"}

for fn, s in {"test 6.mp4": 109, "test 5.mp4": 112, "test 4.mp4": 95}.items():
    segs = seg.segment_file(f"finetuning/test_samples/{fn}")
    spoken = " ".join(coh(x["audio_data"]) for x in segs)
    canon_diac = surah_words_diac(s)
    target = " ".join(canon_diac)
    res = grader.grade(spoken, target)
    # walk alignment -> status per canonical word (+ what child said)
    status = []; ri = 0; extras = []
    for op, r_w, h_w in res["debug_ops"]:
        if op == "equal": status.append(("correct", h_w)); ri += 1
        elif op == "sub": status.append(("wrong", h_w)); ri += 1
        elif op == "delete": status.append(("missed", None)); ri += 1
        elif op == "insert": extras.append(h_w)
    # diacritized display: canonical for correct, CATT(child word) for wrong, mark missed
    disp = []
    for i, (st, said) in enumerate(status):
        cw = canon_diac[i] if i < len(canon_diac) else ""
        if st == "correct": disp.append(cw)
        elif st == "wrong":
            d = catt.do_tashkeel(said, verbose=False).strip() if said else ""
            disp.append(f"«{d}»")            # « » marks a mis-recited word
        elif st == "missed": disp.append(f"⟦{cw}⟧")   # ⟦ ⟧ marks a skipped word

    print("\n" + "═" * 64)
    print(f"  سورة {SURAH[s]}  (Surah {s})   —   {fn}")
    print("═" * 64)
    pct = int(round(res["accuracy"] * 100))
    verdict = "✅ PASSED" if res["passed"] else "❌ NEEDS REVIEW"
    print(f"  SCORE: {pct}%   ({res['raw_score']} words)   {verdict}")
    print(f"\n  WHAT WE HEARD (raw): {spoken}")
    print(f"\n  YOUR RECITATION (diacritized; «wrong»  ⟦skipped⟧):")
    print(f"    {' '.join(disp)}")
    print(f"\n  WORD-BY-WORD:")
    icons = {"correct": "✓", "wrong": "✗", "missed": "∅"}
    line = "   "
    for i, (st, said) in enumerate(status):
        cw = canon_diac[i] if i < len(canon_diac) else ""
        tag = icons[st]
        cell = f"{tag}{cw}" + (f"(said:{said})" if st == "wrong" and said else "")
        line += cell + "   "
        if len(line) > 90: print(line); line = "   "
    if line.strip(): print(line)
    if res["mistakes"]:
        print(f"\n  الملاحظات (feedback):")
        for m in res["mistakes"][:12]: print(f"    • {m}")
    if extras: print(f"\n  extra words spoken (ignored when passed): {extras}")
