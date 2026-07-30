#!/usr/bin/env python3
"""Layer 2 — train a connected-speech phoneme recognizer (wav2vec2-CTC) on
IqraEval/Iqra_train (MSA connected reading, phoneme_ref targets). Produces a
standard HF Wav2Vec2ForCTC in the IqraEval phoneme scheme, for MDD against QuranMB.

Run in `mlaudio` env (GPU). Streams the train split to avoid a huge upfront download.
  python finetuning/train_phoneme_recognizer.py --max_steps 8000 --batch_size 8
"""
import os, io, re, tempfile, json, argparse, time, math, random
os.environ.setdefault("HF_HUB_OFFLINE", "0")
import torch, torch.nn as nn, numpy as np, librosa
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor

BASE = "facebook/wav2vec2-xls-r-300m"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="finetuning/checkpoints_phoneme")
    p.add_argument("--max_steps", type=int, default=8000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--max_audio_s", type=float, default=15.0)
    p.add_argument("--resume_from", default=None, help="checkpoint dir to resume model+vocab from")
    return p.parse_args()


def build_vocab():
    """Phoneme set from the dev split (small, covers the inventory)."""
    dev = load_dataset("IqraEval/Iqra_train", split="dev").cast_column("audio", Audio(decode=False))
    phones = set()
    for r in dev:
        phones.update(r["phoneme_ref"].split())
    toks = ["[PAD]", "[UNK]"] + sorted(phones)
    return {t: i for i, t in enumerate(toks)}


def decode_audio(a, max_s):
    b = a.get("bytes")
    if b is None:
        y, _ = librosa.load(a["path"], sr=16000)
    else:
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(a.get("path", ".mp3"))[1] or ".mp3", delete=False) as f:
            f.write(b); tmp = f.name
        try: y, _ = librosa.load(tmp, sr=16000)
        finally: os.unlink(tmp)
    return y[: int(max_s * 16000)].astype(np.float32)


class MapData(Dataset):
    """Map-style over the cached (memory-mapped) train split — reliable, workers OK."""
    def __init__(self, vocab, fe, max_s):
        self.vocab, self.fe, self.max_s = vocab, fe, max_s
        self.ds = load_dataset("IqraEval/Iqra_train", split="train").cast_column("audio", Audio(decode=False))
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        r = self.ds[i]; unk = self.vocab["[UNK]"]
        try:
            y = decode_audio(r["audio"], self.max_s)
            if len(y) < 1600: y = np.zeros(1600, dtype=np.float32)
            iv = self.fe(y, sampling_rate=16000).input_values[0]
            labels = [self.vocab.get(p, unk) for p in r["phoneme_ref"].split()] or [unk]
            return {"input_values": np.asarray(iv, dtype=np.float32), "labels": labels}
        except Exception:
            return {"input_values": np.zeros(1600, dtype=np.float32), "labels": [self.vocab["[UNK]"]]}


def collate(batch, pad_val=0.0):
    maxT = max(len(b["input_values"]) for b in batch)
    maxL = max(len(b["labels"]) for b in batch)
    iv = np.full((len(batch), maxT), pad_val, dtype=np.float32)
    am = np.zeros((len(batch), maxT), dtype=np.int64)
    lab = np.full((len(batch), maxL), -100, dtype=np.int64)
    for i, b in enumerate(batch):
        iv[i, :len(b["input_values"])] = b["input_values"]
        am[i, :len(b["input_values"])] = 1
        lab[i, :len(b["labels"])] = b["labels"]
    return (torch.from_numpy(iv), torch.from_numpy(am), torch.from_numpy(lab))


def main():
    a = parse(); os.makedirs(a.output_dir, exist_ok=True)
    step0 = 0
    if a.resume_from:
        vocab = json.load(open(os.path.join(a.resume_from, "vocab.json")))
        m = re.search(r"checkpoint-(\d+)", a.resume_from); step0 = int(m.group(1)) if m else 0
        print(f"resuming from {a.resume_from} at step {step0}; vocab {len(vocab)}", flush=True)
    else:
        print("building phoneme vocab from dev split ...", flush=True)
        vocab = build_vocab()
        json.dump(vocab, open(os.path.join(a.output_dir, "vocab.json"), "w"), ensure_ascii=False)
        print(f"vocab size {len(vocab)}: {list(vocab)[:20]} ...", flush=True)

    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                  do_normalize=True, return_attention_mask=True)
    model = Wav2Vec2ForCTC.from_pretrained(
        a.resume_from or BASE, vocab_size=len(vocab), ctc_loss_reduction="mean",
        pad_token_id=vocab["[PAD]"], ignore_mismatched_sizes=True).to("cuda")
    model.freeze_feature_encoder()
    model.train()

    dl = DataLoader(MapData(vocab, fe, a.max_audio_s), batch_size=a.batch_size,
                    collate_fn=collate, num_workers=4, shuffle=True, drop_last=True, persistent_workers=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=a.lr)
    warmup = 500
    def lr_at(s): return (s / warmup) if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, a.max_steps - warmup)))

    step = step0; micro = 0; window = 0.0; t0 = time.time(); opt.zero_grad()
    while step < a.max_steps:
        for iv, am, lab in dl:
            iv, am, lab = iv.to("cuda"), am.to("cuda"), lab.to("cuda")
            loss = model(input_values=iv, attention_mask=am, labels=lab).loss / a.grad_accum
            loss.backward(); window += loss.item(); micro += 1
            if micro % a.grad_accum: continue
            for g in opt.param_groups: g["lr"] = a.lr * lr_at(step)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); opt.zero_grad(); step += 1
            if step % 25 == 0:
                print(f"step {step:5d}/{a.max_steps} loss {window/25:.3f} lr {opt.param_groups[0]['lr']:.2e} {(time.time()-t0)/step:.2f}s/it", flush=True)
                window = 0.0
            if step % a.save_steps == 0:
                d = os.path.join(a.output_dir, f"checkpoint-{step}"); model.save_pretrained(d)
                json.dump(vocab, open(os.path.join(d, "vocab.json"), "w"), ensure_ascii=False)
                print("saved", d, flush=True)
            if step >= a.max_steps: break
    d = os.path.join(a.output_dir, f"checkpoint-{step}"); model.save_pretrained(d)
    json.dump(vocab, open(os.path.join(d, "vocab.json"), "w"), ensure_ascii=False)
    print("saved final", d)


if __name__ == "__main__":
    main()
