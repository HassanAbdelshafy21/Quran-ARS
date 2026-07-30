#!/usr/bin/env python3
"""Layer 2 — Mispronunciation Detection prototype.
Pipeline: known target ayah -> expected phonemes (Buraaq word_tr) ; child audio ->
phonetic wav2vec2 emissions ; CTC forced-alignment gives per-word time spans ;
free-decode each word and compare to expected -> flag harakat (vowel) vs letter errors.
Run in `mlaudio` env."""
import os, sys, re, difflib
os.environ.setdefault("HF_HUB_OFFLINE", "0")
import torch, numpy as np, librosa, torchaudio.functional as AF
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

PH = "TBOGamer22/wav2vec2-quran-phonetics"
VOWELS = set("aiuāīū")
proc = Wav2Vec2Processor.from_pretrained(PH)
model = Wav2Vec2ForCTC.from_pretrained(PH).to("cuda").eval()
tok = proc.tokenizer
BLANK = tok.pad_token_id
vocab = tok.get_vocab()

def emissions(y):
    iv = proc(y, sampling_rate=16000, return_tensors="pt").input_values.to("cuda")
    with torch.no_grad():
        lp = torch.log_softmax(model(iv).logits, dim=-1)
    return lp[0].cpu()  # [T, V]

def free_decode(y):
    iv = proc(y, sampling_rate=16000, return_tensors="pt").input_values.to("cuda")
    with torch.no_grad():
        ids = torch.argmax(model(iv).logits, dim=-1)
    return proc.batch_decode(ids)[0].strip()

def to_ids(s):
    out = []
    for ch in s:
        if ch in vocab: out.append(vocab[ch])
        elif ch.lower() in vocab: out.append(vocab[ch.lower()])
        else: out.append(vocab.get("[UNK]"))
    return out

def buraaq_surah_words(surahs):
    """{surah: [(word_ar, word_tr), ...]} in ayah/word order."""
    from datasets import load_dataset, Audio
    dd = load_dataset("Buraaq/quran-md-words", split="train", streaming=True).cast_column("audio", Audio(decode=False))
    want = set(surahs); got = {s: [] for s in surahs}; seen = set()
    for s in dd:
        sid = s["surah_id"]
        if sid in want:
            key = (sid, s["ayah_id"], s["word_index"])
            if key in seen: continue
            seen.add(key)
            got[sid].append((s["ayah_id"], s["word_index"], s["word_ar"], s["word_tr"]))
        if sid > max(want) and all(got[s] for s in want):
            break
    for s in got:
        got[s].sort(key=lambda r: (r[0], r[1]))
        got[s] = [(w[2], w[3]) for w in got[s]]
    return got

def diff_word(expected_tr, heard):
    """char-level diff -> list of (kind, exp, got)."""
    sm = difflib.SequenceMatcher(None, expected_tr, heard, autojunk=False)
    issues = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal": continue
        e, g = expected_tr[i1:i2], heard[j1:j2]
        kind = "harakat" if (set(e) | set(g)) & VOWELS and not (set(e) | set(g)) - VOWELS else "letter"
        issues.append((kind, e, g))
    return issues

GOP_THRESH = -5.0  # per-phoneme: flag if expected sound is this much less likely than the best competitor

def assess(y, expected_words):
    """GOP-based: forced-align expected phonemes, then score each phoneme's acoustic
    likelihood vs the best competitor at its aligned frames (no slice re-decoding)."""
    id2tok = {v: k for k, v in vocab.items()}
    targets, word_of, char_of = [], [], []
    for wi, w in enumerate(expected_words):
        for ch in w:
            tid = to_ids(ch)[0]; targets.append(tid); word_of.append(wi); char_of.append(ch)
    emis = emissions(y)  # [T,V] log-probs
    aligned, scores = AF.forced_align(emis.unsqueeze(0), torch.tensor([targets]), blank=BLANK)
    spans = AF.merge_tokens(aligned[0], scores[0])
    # per target-token (char) frame span, in order
    tok_span, ti = {}, 0
    for sp in spans:
        if sp.token == BLANK or ti >= len(targets): continue
        tok_span[ti] = (sp.start, sp.end); ti += 1
    report = {}
    for i, tid in enumerate(targets):
        wi = word_of[i]; report.setdefault(wi, [])
        if i not in tok_span:
            report[wi].append((char_of[i], None, None)); continue
        s, e = tok_span[i]; fr = emis[s:e+1]              # [n, V]
        exp_lp = fr[:, tid].mean().item()
        best = fr.max(dim=-1)
        best_lp = best.values.mean().item()
        gop = exp_lp - best_lp                             # <=0 ; near 0 = good
        actual = id2tok.get(int(best.indices.mode().values), "?")
        report[wi].append((char_of[i], gop, actual))
    out = []
    for wi, w in enumerate(expected_words):
        chars = report.get(wi, [])
        issues = []
        for ch, gop, act in chars:
            if gop is None or gop < GOP_THRESH:
                kind = "harakat" if ch in VOWELS else "letter"
                issues.append((kind, ch, "" if gop is None else act, gop))
        out.append((w, chars, issues))
    return out

if __name__ == "__main__":
    print("loading Buraaq word_tr for surahs 95/112/109 ...", flush=True)
    words = buraaq_surah_words([95, 112, 109])
    tests = {"test 5.mp4": 112, "test 6.mp4": 109, "test 4.mp4": 95}
    for fn, surah in tests.items():
        y, _ = librosa.load(f"finetuning/test_samples/{fn}", sr=16000)
        exp = [tr for (_, tr) in words[surah]]
        print(f"\n{'='*70}\n{fn}  (surah {surah}, {len(exp)} expected words)\n{'='*70}")
        nflag = 0
        for w, chars, issues in assess(y, exp):
            flag = "OK " if not issues else "‼ "
            nflag += bool(issues)
            note = "" if not issues else "  " + "; ".join(f"[{k}] '{e}'→'{g}'(gop{p:.1f})" for k, e, g, p in issues)
            print(f"  {flag} {w:14s}{note}")
        print(f"  -> {nflag}/{len(exp)} words flagged")
