#!/usr/bin/env python3
"""Fine-tune Cohere Transcribe to emit Quranic tashkeel, without harming its
acoustic robustness: LoRA on the DECODER q/v only (encoder frozen). Teacher
forcing with decoder_input_ids = labels = [lang-prompt + text + eos], prompt
masked to -100; loss_function=ForCausalLMLoss shifts internally. Saves the LoRA
adapter every --save_steps. Run in `mlaudio` env (transformers 5.13, torch cu128).

  python finetuning/finetune_cohere.py --max_steps 2400 --batch_size 4 \
      --grad_accum 4 --lr 1e-4 --save_steps 300
"""
import os, re, math, random, argparse, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch, torch.nn as nn, numpy as np, librosa
try: import torch.distributed.tensor  # noqa
except Exception: pass
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk

MODEL = "CohereLabs/cohere-transcribe-03-2026"
PAD, EOS = 2, 3

_ANNOT = re.compile(r"[ٰۖ-ۭۥۦۧۨ۩۪-ࣰۭ-ࣿ]")
def simplify_tashkeel(s):
    """Uthmani script -> standard harakat: keep fatha/damma/kasra/shadda/sukun/
    tanwin (U+064B-0652), drop wasla/dagger-alef/Quranic annotation marks."""
    s = s.replace("ٱ", "ا")   # alef-wasla -> alef
    s = s.replace("ۡ", "ْ")   # small-high-sukun -> sukun
    s = _ANNOT.sub("", s)               # dagger-alef, small waw/ya, annotation marks
    return re.sub(r"\s+", " ", s).strip()


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", default="data/quran_dataset_v6_allages")
    p.add_argument("--output_dir", default="finetuning/checkpoints_cohere_tashkeel")
    p.add_argument("--max_steps", type=int, default=2400)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save_steps", type=int, default=300)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--targets", default="q_proj,v_proj")
    p.add_argument("--max_audio_s", type=float, default=25.0)
    return p.parse_args()


class Pairs(Dataset):
    def __init__(self, rows, prompt_ids, tok, max_audio_s, simple=True):
        self.rows = rows; self.prompt_ids = prompt_ids; self.tok = tok
        self.max = max_audio_s; self.simple = simple
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        y, _ = librosa.load(r["audio"], sr=16000)
        y = y[: int(self.max * 16000)].astype(np.float32)
        text = simplify_tashkeel(r["text"]) if self.simple else r["text"]
        tids = self.tok.encode(text, add_special_tokens=False)
        seq = self.prompt_ids + tids + [EOS]
        return y, seq


def make_collate(proc, n_prompt):
    def collate(batch):
        audios = [b[0] for b in batch]; seqs = [b[1] for b in batch]
        feats = proc(audio=audios, sampling_rate=16000, language="ar",
                     return_tensors="pt", padding=True)
        maxlen = max(len(s) for s in seqs)
        dii, lab = [], []
        for s in seqs:
            ids = s + [PAD] * (maxlen - len(s))
            l = [(-100 if (i < n_prompt or i >= len(s)) else ids[i]) for i in range(maxlen)]
            dii.append(ids); lab.append(l)
        out = {"input_features": feats["input_features"],
               "decoder_input_ids": torch.tensor(dii),
               "labels": torch.tensor(lab)}
        if "length" in feats: out["length"] = feats["length"]
        return out
    return collate


def main():
    a = parse(); os.makedirs(a.output_dir, exist_ok=True)
    proc = AutoProcessor.from_pretrained(MODEL)
    prompt_ids = proc.get_decoder_prompt_ids(language="ar", punctuation=True)
    model = CohereAsrForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")

    suffixes = tuple(a.targets.split(","))
    dec_qv = [n for n, m in model.named_modules()
              if isinstance(m, nn.Linear) and "decoder" in n.lower()
              and n.endswith(suffixes)]
    print(f"LoRA targets ({a.targets}): {len(dec_qv)} modules")
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_r * 2,
                                             target_modules=dec_qv, lora_dropout=0.05, bias="none"))
    model.print_trainable_parameters()

    rows = list(load_from_disk(a.dataset_path)); random.seed(42); random.shuffle(rows)
    dl = DataLoader(Pairs(rows, prompt_ids, proc.tokenizer, a.max_audio_s),
                    batch_size=a.batch_size, shuffle=True, num_workers=4,
                    collate_fn=make_collate(proc, len(prompt_ids)), drop_last=True)
    print(f"{len(rows)} pairs | {len(dl)} batches/epoch | target {a.max_steps} steps")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=0.01)
    warmup = 100
    def lr_at(s):
        if s < warmup: return s / warmup
        prog = (s - warmup) / max(1, a.max_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    trainable = [p for p in model.parameters() if p.requires_grad]
    model.train(); step = 0; micro = 0; t0 = time.time(); window = 0.0
    opt.zero_grad()
    while step < a.max_steps:
        for batch in dl:
            b = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()}
            b["input_features"] = b["input_features"].to(torch.bfloat16)
            loss = model(**b).loss / a.grad_accum
            loss.backward(); window += loss.item(); micro += 1
            if micro % a.grad_accum != 0:
                continue
            for g in opt.param_groups: g["lr"] = a.lr * lr_at(step)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); opt.zero_grad(); step += 1
            if step % 20 == 0:
                print(f"step {step:4d}/{a.max_steps} loss {window/20:.4f} "
                      f"lr {opt.param_groups[0]['lr']:.2e} {(time.time()-t0)/step:.2f}s/it", flush=True)
                window = 0.0
            if step % a.save_steps == 0:
                d = os.path.join(a.output_dir, f"checkpoint-{step}")
                model.save_pretrained(d); print("saved", d, flush=True)
            if step >= a.max_steps: break
        if step >= a.max_steps: break
    d = os.path.join(a.output_dir, f"checkpoint-{step}")
    model.save_pretrained(d); print("saved final", d)


if __name__ == "__main__":
    main()
