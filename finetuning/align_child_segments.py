#!/usr/bin/env python3
"""
Align extracted child segments to their correct ayah text -> trainable pairs.

Approach (validated on Surah 80): transcribe each child segment, clean Whisper's
repetition loops (word + phrase level), then best-match it to an ayah of that
surah (or the Basmala). Emit only confident matches (phonetic CER <= MATCH_CER);
drop the rest. Precision over recall -> clean training pairs.

Output: aligned_metadata.jsonl of {audio, text, surah, ayah, cer}. Resumable:
re-running skips surahs already in the output. Optional surah filter via argv.

Run from repo root (GPU):  python finetuning/align_child_segments.py [surah ...]
"""
import os
import re
import sys
import json
import sqlite3
from collections import defaultdict

import torch  # noqa
try:
    import torch.distributed.tensor  # noqa (local Blackwell shim; harmless elsewhere)
except Exception:
    pass

import jiwer
import librosa

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "delivery"))
from core.grader import QuranGrader
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# whisper-large-v3-turbo is used ONLY here, to LABEL the data (read the clear chorus
# far better than our small fine-tuned model; turbo ~= large-v3 quality on clear
# audio but ~2x faster). It never ships; V6 is still our model.
BIG_MODEL = "openai/whisper-large-v3-turbo"

SEG_DIR = "data/dataset_cache/audio/Minshawy_Child_Segments"
MANIFEST = os.path.join(SEG_DIR, "manifest.jsonl")
DB = "data/quran.db"
OUT = os.path.join(SEG_DIR, "aligned_metadata.jsonl")
MARK = re.compile(r'[ﰀ-﷿]')
# Position-anchored matching: the mushaf is in ayah order, so we track a running
# position and only match each segment against a small WINDOW around it. That
# constrains the choice, which lets us safely LOOSEN the CER gate (vs searching
# all 286 ayahs of a long surah, which forced a tight gate and dropped ~90%).
WIN_BACK, WIN_FWD = 1, 3       # search ayahs [pos-1 .. pos+3]
MATCH_CER = 0.50               # looser than 0.40 — position already constrains it
BASMALA = "بسم الله الرحمن الرحيم"


def collapse(t):
    """Remove Whisper repetition loops at word and phrase (2-4 gram) level."""
    w = t.split()
    out = []
    for x in w:                                   # collapse immediate word repeats
        if not out or out[-1] != x:
            out.append(x)
    w = out
    for n in (4, 3, 2):                           # collapse immediate phrase repeats
        res, i = [], 0
        while i < len(w):
            if i + 2 * n <= len(w) and w[i:i + n] == w[i + n:i + 2 * n]:
                res.extend(w[i:i + n]); i += n
                while i + n <= len(w) and res[-n:] == w[i:i + n]:
                    i += n
            else:
                res.append(w[i]); i += 1
        w = res
    return " ".join(w)


def targets_for(conn, surah, norm):
    cur = conn.cursor()
    cur.execute("SELECT aya_no, aya_text FROM quran WHERE sura_no=? ORDER BY aya_no", (surah,))
    tgts = [(0, norm(BASMALA), BASMALA)]          # Basmala = "ayah 0"
    for n, t in cur.fetchall():
        clean = MARK.sub("", t).strip()
        tgts.append((n, norm(clean), clean))
    return tgts


def main():
    grader = QuranGrader()
    norm = grader.normalize
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {BIG_MODEL} on {device} (downloads ~3GB on first run)...", flush=True)
    processor = WhisperProcessor.from_pretrained(BIG_MODEL)
    big = WhisperForConditionalGeneration.from_pretrained(
        BIG_MODEL, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
    big.eval()

    def transcribe(y):
        feats = processor(y, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        if device == "cuda":
            feats = feats.half()
        with torch.no_grad():
            ids = big.generate(feats, language="ar", task="transcribe",
                               num_beams=1, max_new_tokens=200)
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    conn = sqlite3.connect(DB)

    rows = [json.loads(l) for l in open(MANIFEST, encoding="utf-8")]
    by_surah = defaultdict(list)
    for r in rows:
        by_surah[int(r["surah"])].append(r)
    for s in by_surah:
        by_surah[s].sort(key=lambda r: r["start"])
    if len(sys.argv) > 1:
        want = set(int(x) for x in sys.argv[1:])
        by_surah = {s: v for s, v in by_surah.items() if s in want}

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            try:
                done.add(int(json.loads(l)["surah"]))
            except Exception:
                pass

    emitted = dropped = 0
    with open(OUT, "a", encoding="utf-8") as out:
        for surah in sorted(by_surah):
            if surah in done:
                continue
            tgts = targets_for(conn, surah, norm)   # ordered: [(0,Basmala),(1,a1),...]
            pos = 0                                  # running position pointer
            for seg in by_surah[surah]:
                p = os.path.join(SEG_DIR, seg["file"])
                if not os.path.exists(p):
                    continue
                y, _ = librosa.load(p, sr=16000)
                txt = collapse(norm(transcribe(y)))
                if not txt:
                    dropped += 1
                    continue
                # 1) prefer a match within the position window
                lo = max(0, pos - WIN_BACK)
                hi = min(len(tgts), pos + WIN_FWD + 1)
                window = tgts[lo:hi]
                bi, best = min(enumerate(window),
                               key=lambda kv: jiwer.cer(txt, kv[1][1]) if kv[1][1] else 1.0)
                cer = jiwer.cer(txt, best[1]) if best[1] else 1.0
                matched_idx = lo + bi
                # 2) fallback: if the window fails, re-anchor with a global best-match
                #    (recovers from pointer drift / stuck position)
                if cer > MATCH_CER:
                    gi, gbest = min(enumerate(tgts),
                                    key=lambda kv: jiwer.cer(txt, kv[1][1]) if kv[1][1] else 1.0)
                    gcer = jiwer.cer(txt, gbest[1]) if gbest[1] else 1.0
                    if gcer <= MATCH_CER:
                        best, cer, matched_idx = gbest, gcer, gi
                if cer <= MATCH_CER:
                    out.write(json.dumps({
                        "audio": p, "text": best[2], "surah": surah,
                        "ayah": best[0], "cer": round(cer, 3),
                    }, ensure_ascii=False) + "\n")
                    emitted += 1
                    pos = matched_idx + 1           # advance past the matched ayah
                else:
                    dropped += 1
            print(f"  surah {surah:3d}: emitted {emitted}, dropped {dropped}", flush=True)

    print(f"\nDone. Aligned pairs: {emitted} | dropped: {dropped}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
