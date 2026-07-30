#!/usr/bin/env python3
"""
Extract child-voice segments from the Minshawi "Mushaf Al-Muallim" full-surah
recordings (teacher recites -> children repeat -> pause).

Method (see separation_plan.md):
  1. Split each surah by silence into speech chunks.
  2. Estimate each chunk's median pitch (F0). An adult-male teacher sits
     ~120-230 Hz; a children's chorus sits ~300-380 Hz — a clean, verifiable split.
  3. Keep chunks with F0 >= CHILD_F0_HZ (child), drop teacher/silence, write them
     out with a manifest.

Careful by design:
  - Precision over recall: the cutoff sits ABOVE the teacher band, so borderline
    chunks are dropped rather than mislabeled. Better to lose some kid audio than
    poison training with teacher audio.
  - Every surah drops one CHILD sample into _samples_to_verify/ so you can spot-
    check separation across the whole Quran, not just one surah.
  - manifest.jsonl logs every kept segment (surah, idx, start, dur, f0).

Runs surahs in parallel across CPU cores (the F0 step is CPU signal-processing;
a GPU does not accelerate it). Run from repo root:
    python finetuning/extract_child_segments.py [surah ...]
"""
import os
import sys
import glob
import json
import numpy as np
import librosa
import soundfile as sf
from multiprocessing import Pool

SRC_DIR = "data/dataset_cache/audio/Minshawy_Muallim_FullSurah"
OUT_DIR = "data/dataset_cache/audio/Minshawy_Child_Segments"
SAMPLE_DIR = os.path.join(OUT_DIR, "_samples_to_verify")
MANIFEST = os.path.join(OUT_DIR, "manifest.jsonl")

SR = 16000
TOP_DB = 32            # silence threshold for splitting
RAW_MIN = 0.35         # min raw chunk length to classify by pitch (s)
MERGE_GAP = 0.6        # merge consecutive CHILD chunks separated by < this gap (s) —
                       # rejoins one ayah that the silence-split broke into fragments
MIN_DUR = 1.0          # drop merged segments shorter than this (s)
MAX_DUR = 15.0         # drop segments longer than this (teacher passages / failed splits)
CHILD_F0_HZ = 300.0    # >= this median F0 = child. Above the 260-300 teacher-leak band;
                       # the clean child cluster is 330+ Hz.
F0_MIN, F0_MAX = 80, 500
N_WORKERS = 4


def median_f0(seg):
    """Median voiced F0 over the central ~2s of a chunk (fast + robust)."""
    if len(seg) > SR * 2:
        mid = len(seg) // 2
        seg = seg[mid - SR: mid + SR]
    f0, _, _ = librosa.pyin(seg, fmin=F0_MIN, fmax=F0_MAX, sr=SR)
    f0 = f0[~np.isnan(f0)]
    return float(np.median(f0)) if f0.size else 0.0


def process_surah_file(path):
    """Worker: extract child segments from one surah. Writes wavs, returns stats."""
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        y, _ = librosa.load(path, sr=SR)
    except Exception as e:
        return {"surah": name, "error": str(e), "n_chunks": 0, "n_child": 0,
                "child_secs": 0.0, "teacher_secs": 0.0, "rows": [], "f0s": []}

    intervals = librosa.effects.split(y, top_db=TOP_DB, frame_length=2048, hop_length=512)

    # 1) classify each usable raw chunk by pitch
    labeled, f0s = [], []
    for s, e in intervals:
        if (e - s) / SR < RAW_MIN:
            continue
        f0 = median_f0(y[s:e])
        f0s.append(f0)
        labeled.append((s, e, f0 >= CHILD_F0_HZ))

    # 2) merge consecutive CHILD chunks (short gaps) into whole-ayah utterances
    rows = []
    child_secs = teacher_secs = 0.0
    sample_saved = False
    i = 0
    while i < len(labeled):
        s, e, is_child = labeled[i]
        if not is_child:
            teacher_secs += (e - s) / SR
            i += 1
            continue
        j, end = i + 1, e
        while j < len(labeled) and labeled[j][2] and (labeled[j][0] - end) / SR < MERGE_GAP:
            end = labeled[j][1]
            j += 1
        seg = y[s:end]
        dur = len(seg) / SR
        if MIN_DUR <= dur <= MAX_DUR:
            fn = f"{name}_{i:04d}.wav"
            sf.write(os.path.join(OUT_DIR, fn), seg, SR)
            child_secs += dur
            rows.append({"surah": name, "idx": i, "file": fn,
                         "start": round(s / SR, 2), "dur": round(dur, 2)})
            if not sample_saved:
                sf.write(os.path.join(SAMPLE_DIR, f"CHILD_{fn}"), seg, SR)
                sample_saved = True
        i = j

    return {"surah": name, "n_chunks": len(intervals), "n_child": len(rows),
            "child_secs": child_secs, "teacher_secs": teacher_secs, "rows": rows, "f0s": f0s}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    wanted = sys.argv[1:]
    files = sorted(glob.glob(os.path.join(SRC_DIR, "*.mp3")))
    if wanted:
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in wanted]
    if not files:
        print(f"No surah files in {SRC_DIR}")
        return
    # Big surahs first for better load balancing across workers.
    files.sort(key=lambda p: os.path.getsize(p), reverse=True)
    print(f"Extracting child segments from {len(files)} surahs "
          f"({N_WORKERS} workers)...", flush=True)

    all_f0, child_secs, teacher_secs, n_saved = [], 0.0, 0.0, 0
    errors = []
    with open(MANIFEST, "w", encoding="utf-8") as mf:
        with Pool(N_WORKERS) as pool:
            for i, res in enumerate(pool.imap_unordered(process_surah_file, files), 1):
                if res.get("error"):
                    errors.append((res["surah"], res["error"]))
                    print(f"  [{i}/{len(files)}] surah {res['surah']}: ERROR {res['error']}", flush=True)
                    continue
                child_secs += res["child_secs"]
                teacher_secs += res["teacher_secs"]
                n_saved += res["n_child"]
                all_f0.extend(res["f0s"])
                for r in res["rows"]:
                    mf.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"  [{i}/{len(files)}] surah {res['surah']}: {res['n_chunks']:4d} chunks | "
                      f"child {res['n_child']:3d} ({res['child_secs']/60:5.1f} min)", flush=True)

    # ---- verification report ----
    print("\n===== VERIFICATION =====")
    print(f"Child segments saved : {n_saved}  -> {OUT_DIR}")
    print(f"Child audio total    : {child_secs/3600:.2f} h")
    print(f"Teacher audio total  : {teacher_secs/3600:.2f} h")
    print(f"Manifest             : {MANIFEST}")
    print(f"Audit samples        : {SAMPLE_DIR}  (CHILD_* — one per surah)")
    if errors:
        print(f"Surahs with errors   : {len(errors)} -> {[e[0] for e in errors]}")
    a = np.array([f for f in all_f0 if f > 0])
    if a.size:
        print("\nF0 histogram (bimodal = clean teacher/child split; cutoff at "
              f"{CHILD_F0_HZ:.0f} Hz):")
        edges = list(range(80, 410, 30))
        hist, _ = np.histogram(a, bins=edges)
        mx = max(hist.max(), 1)
        for i, h in enumerate(hist):
            bar = "#" * int(50 * h / mx)
            mark = " <- cutoff" if edges[i] <= CHILD_F0_HZ < edges[i + 1] else ""
            print(f"  {edges[i]:3d}-{edges[i+1]:3d}Hz | {bar}{mark}")


if __name__ == "__main__":
    main()
