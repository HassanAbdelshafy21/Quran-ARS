#!/usr/bin/env python3
"""Probe: lock down the correct training construction for Cohere Transcribe.
Manual teacher-forcing: decoder_input_ids = labels = [lang-prompt + text + eos],
prompt masked to -100. loss_function=ForCausalLMLoss shifts internally.
LoRA on decoder q/v (encoder frozen). Overfit 2 samples -> generate must emit
the tashkeel target. Run in `mlaudio` env."""
import os, re
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch, torch.nn as nn, numpy as np, librosa
try: import torch.distributed.tensor  # noqa
except Exception: pass
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk

MODEL = "CohereLabs/cohere-transcribe-03-2026"
DIAC = re.compile(r"[ً-ْٰ]")
PAD, EOS = 2, 3

proc = AutoProcessor.from_pretrained(MODEL)
tok = proc.tokenizer
model = CohereAsrForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")

prompt_ids = proc.get_decoder_prompt_ids(language="ar", punctuation=True)
print("prompt_ids:", prompt_ids, "len", len(prompt_ids))

ds = load_from_disk("data/quran_dataset_v6_allages")
samples = [r for r in ds if r["reciter"] != "Minshawi_Child"][:2]
audios = [librosa.load(r["audio"], sr=16000)[0].astype(np.float32) for r in samples]
texts = [r["text"] for r in samples]

# audio features
feats = proc(audio=audios, sampling_rate=16000, language="ar", return_tensors="pt", padding=True)
input_features = feats["input_features"].to("cuda", dtype=torch.bfloat16)
length = feats["length"].to("cuda") if "length" in feats else None

# build decoder_input_ids + labels manually
seqs = []
for t in texts:
    tids = tok.encode(t, add_special_tokens=False)
    seqs.append(prompt_ids + tids + [EOS])
maxlen = max(len(s) for s in seqs)
dii, lab = [], []
for s in seqs:
    pad = maxlen - len(s)
    ids = s + [PAD] * pad
    l = list(s) + [PAD] * pad
    for i in range(len(l)):
        if i < len(prompt_ids) or i >= len(s): l[i] = -100  # mask prompt + pad
    dii.append(ids); lab.append(l)
decoder_input_ids = torch.tensor(dii, device="cuda")
labels = torch.tensor(lab, device="cuda")
print("decoder_input_ids", tuple(decoder_input_ids.shape), "labels", tuple(labels.shape))

# LoRA on decoder q/v only
dec_qv = [n for n, m in model.named_modules()
          if isinstance(m, nn.Linear) and "decoder" in n.lower() and (n.endswith("q_proj") or n.endswith("v_proj"))]
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=dec_qv, lora_dropout=0.0, bias="none"))
model.print_trainable_parameters()

kw = dict(input_features=input_features, decoder_input_ids=decoder_input_ids, labels=labels)
if length is not None: kw["length"] = length
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
model.train()
for step in range(40):
    out = model(**kw)
    out.loss.backward(); opt.step(); opt.zero_grad()
    if step % 5 == 0 or step == 39: print(f"step {step:2d} loss {out.loss.item():.4f}")

model.eval()
with torch.no_grad():
    for a, tgt in zip(audios, texts):
        inp = proc(a, sampling_rate=16000, return_tensors="pt", language="ar")
        inp.to(model.device, dtype=model.dtype)
        ids = model.generate(**inp, max_new_tokens=64)
        got = proc.decode(ids, skip_special_tokens=True)
        got = got[0] if isinstance(got, list) else got
        print("\nTARGET:", tgt[:55], "| tashkeel:", bool(DIAC.search(tgt)))
        print("GOT   :", got[:55], "| tashkeel:", bool(DIAC.search(got)))
