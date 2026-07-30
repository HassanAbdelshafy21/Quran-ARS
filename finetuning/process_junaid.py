#!/usr/bin/env python3
"""
Segment + align Muhammad Taha Al-Junaid (single child) Juz-Amma surahs into
per-ayah (audio -> tashkeel text) pairs. Single reciter, so: silence-split each
surah into ayah chunks, transcribe with turbo, position-align to the surah's
ayahs (window + CER gate). Also verifies the track->surah mapping via content.

Run from repo root (GPU):  python finetuning/process_junaid.py
"""
import os, re, glob, json, sqlite3
import numpy as np
import torch
try:
    import torch.distributed.tensor  # noqa
except Exception:
    pass
import librosa, soundfile as sf, jiwer
from transformers import WhisperForConditionalGeneration, WhisperProcessor

SRC = "data/dataset_cache/audio/junaid_juzamma"
OUT = "data/dataset_cache/audio/junaid_segments"
DB = "data/quran.db"
MARK = re.compile(r'[ﰀ-﷿]')
BIG = "openai/whisper-large-v3-turbo"
SR = 16000
TOP_DB, MIN_DUR, MAX_DUR, MERGE_GAP = 30, 1.0, 20.0, 0.4
WIN_BACK, WIN_FWD, MATCH_CER = 1, 3, 0.50


def collapse(t):
    w = t.split(); out = []
    for x in w:
        if not out or out[-1] != x: out.append(x)
    w = out
    for n in (4, 3, 2):
        res, i = [], 0
        while i < len(w):
            if i + 2 * n <= len(w) and w[i:i+n] == w[i+n:i+2*n]:
                res.extend(w[i:i+n]); i += n
                while i + n <= len(w) and res[-n:] == w[i:i+n]: i += n
            else:
                res.append(w[i]); i += 1
        w = res
    return " ".join(w)


def main():
    os.makedirs(OUT, exist_ok=True)
    import sys; sys.path.insert(0, "delivery")
    from core.grader import QuranGrader
    norm = QuranGrader().normalize
    conn = sqlite3.connect(DB)
    proc = WhisperProcessor.from_pretrained(BIG)
    model = WhisperForConditionalGeneration.from_pretrained(BIG, torch_dtype=torch.float16).to("cuda").eval()

    def tr(y):
        f = proc(y, sampling_rate=16000, return_tensors="pt").input_features.to("cuda").half()
        with torch.no_grad():
            ids = model.generate(f, language="ar", task="transcribe", num_beams=1, max_new_tokens=200)
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def ayahs(s):
        c = conn.cursor(); c.execute("SELECT aya_no,aya_text FROM quran WHERE sura_no=? ORDER BY aya_no", (s,))
        return [(n, norm(MARK.sub("", t)), MARK.sub("", t).strip()) for n, t in c.fetchall()]

    emitted = dropped = 0
    out = open(os.path.join(OUT, "aligned_metadata.jsonl"), "w", encoding="utf-8")
    for f in sorted(glob.glob(os.path.join(SRC, "*.mp3"))):
        surah = int(os.path.basename(f)[:3])
        y, _ = librosa.load(f, sr=SR)
        iv = librosa.effects.split(y, top_db=TOP_DB, frame_length=2048, hop_length=512)
        # merge close chunks
        merged, i = [], 0
        while i < len(iv):
            s, e = iv[i]; j = i + 1
            while j < len(iv) and (iv[j][0] - e) / SR < MERGE_GAP: e = iv[j][1]; j += 1
            if MIN_DUR <= (e - s) / SR <= MAX_DUR: merged.append((s, e))
            i = j
        tgts = ayahs(surah); pos = 0; se = 0
        for k, (s, e) in enumerate(merged):
            seg = y[s:e]
            txt = collapse(norm(tr(seg)))
            if not txt or not tgts: dropped += 1; continue
            lo = max(0, pos - WIN_BACK); hi = min(len(tgts), pos + WIN_FWD + 1)
            bi, best = min(enumerate(tgts[lo:hi]), key=lambda kv: jiwer.cer(txt, kv[1][1]) if kv[1][1] else 1.0)
            cer = jiwer.cer(txt, best[1]) if best[1] else 1.0; idx = lo + bi
            if cer <= MATCH_CER:
                fn = f"j{surah:03d}_{k:03d}.wav"; sf.write(os.path.join(OUT, fn), seg, SR)
                out.write(json.dumps({"audio": os.path.join(OUT, fn), "text": best[2],
                                      "surah": surah, "ayah": best[0], "cer": round(cer, 3)}, ensure_ascii=False) + "\n")
                emitted += 1; pos = idx + 1; se += 1
            else:
                dropped += 1
        print(f"  surah {surah}: {se}/{len(tgts)} ayahs aligned (chunks {len(merged)})", flush=True)
    out.close()
    print(f"\nJunaid: aligned {emitted} pairs, dropped {dropped} -> {OUT}")


if __name__ == "__main__":
    main()
