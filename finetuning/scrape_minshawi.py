#!/usr/bin/env python3
"""
Download the Minshawi "Mushaf Al-Muallim" (teacher recites, children repeat)
from archive.org — the source of the child-voice training data.

The original archive.org identifier used by the old version of this script died;
this uses the live re-upload. Downloads full-surah MP3s (teacher + children +
pauses). Splitting out the child segments is a downstream processing step.

Default is the WHOLE QURAN (~1.1 GB mp3, ~53 h total audio, ~18-24 h of child
voice) — ~25x more kid data than the original Fatiha+Juz-Amma set (~45 min kids).
Set TARGET_SURAHS to [1] + list(range(78, 115)) to reproduce the original subset.
"""
import os
import json
import time
import urllib.request

IDENT = "Al-MushafAl-MualimForChildrenRecitedByMohamedSiddiqEl-Minshawi"
OUTPUT_DIR = "data/dataset_cache/audio/Minshawy_Muallim_FullSurah"

# Whole Quran = maximum child data. For the original subset use:
#   TARGET_SURAHS = [1] + list(range(78, 115))
TARGET_SURAHS = list(range(1, 115))

UA = {"User-Agent": "Mozilla/5.0"}


def surah_id(name):
    try:
        return int(name.split(".")[0])
    except ValueError:
        return -1


def list_mp3s():
    with urllib.request.urlopen(f"https://archive.org/metadata/{IDENT}", timeout=30) as r:
        meta = json.load(r)
    return sorted(f["name"] for f in meta["files"]
                  if f.get("name", "").lower().endswith(".mp3"))


def download(name, dest):
    url = f"https://archive.org/download/{IDENT}/{name}"
    req = urllib.request.Request(url, headers=UA)
    tmp = dest + ".part"
    with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)  # 1 MB
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wanted = set(TARGET_SURAHS)
    files = [n for n in list_mp3s() if surah_id(n) in wanted]
    print(f"{len(files)} surah files -> {OUTPUT_DIR}")

    for i, name in enumerate(files, 1):
        dest = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"[{i}/{len(files)}] {name} already present, skipping", flush=True)
            continue
        print(f"[{i}/{len(files)}] downloading {name} ...", flush=True)
        for attempt in range(3):
            try:
                download(name, dest)
                break
            except Exception as e:
                print(f"    attempt {attempt + 1} failed: {e}", flush=True)
                time.sleep(2 ** attempt)
        else:
            print(f"    GAVE UP on {name}", flush=True)

    done = sum(1 for n in files
               if os.path.exists(os.path.join(OUTPUT_DIR, n)))
    print(f"Done: {done}/{len(files)} surah files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
