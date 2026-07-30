
import os
import torch
try:
    import torch.distributed.tensor  # noqa: needed for PEFT load on some torch builds (local Blackwell GPU)
except Exception:
    pass
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_from_disk, Value
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback
)
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Constants
if not torch.cuda.is_available():
    raise RuntimeError("No GPU found. Training requires a GPU.")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

BASE_MODEL = "tarteel-ai/whisper-base-ar-quran"
AUGMENT = False
ADAPTER_MODEL = "KheemP/whisper-base-quran-lora"
DATASET_PATH = "data/quran_dataset_v5"
OUTPUT_DIR = "finetuning/checkpoints_v5"

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        # Actually usually we map dataset first to input_features.
        # But if we do dynamic padding:
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous step
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        
        # HOTFIX: Ensure 'input_ids' is NOT passed to Whisper
        if "input_ids" in batch:
            del batch["input_ids"]
            
        return batch

def phone_augment(y, sr):
    """Simulate phone-recorded conditions on clean studio audio: room echo,
    background noise (SNR 14-28 dB), and phone-mic low-pass. Peak-matched to the
    input loudness, and GUARDED so it can never return a silent/NaN clip -- a
    silent (audio, text) pair would poison training."""
    import numpy as np, random, scipy.signal
    y = y.astype(np.float32)
    peak_in = float(np.max(np.abs(y)))
    if peak_in < 1e-4:
        return y                                              # input already silent; leave it
    z = y.copy()
    # room echo (short exponential impulse)
    ir = np.exp(-np.arange(int(0.04 * sr)) / (random.uniform(0.15, 0.30) * sr)).astype(np.float32)
    ir[0] = 1.0
    z = np.convolve(z, ir / ir.sum(), mode="full")[:len(y)].astype(np.float32)
    # colored (pink) background noise at a chosen SNR
    x = np.random.randn(len(z)); X = np.fft.rfft(x); f = np.fft.rfftfreq(len(z)); f[0] = f[1]
    n = np.fft.irfft(X / np.sqrt(f), len(z)).astype(np.float32)
    n /= (np.max(np.abs(n)) + 1e-8)
    rms = np.sqrt(np.mean(z ** 2)) + 1e-8
    snr = random.uniform(14, 28)
    z = z + n * ((rms / (10 ** (snr / 20))) / (np.sqrt(np.mean(n ** 2)) + 1e-8))
    # phone-mic low-pass
    sos = scipy.signal.butter(4, random.uniform(3500, 7000), btype="low", fs=sr, output="sos")
    z = scipy.signal.sosfilt(sos, z).astype(np.float32)
    # peak-match to the original loudness
    z = z / (np.max(np.abs(z)) + 1e-8) * peak_in
    # SAFETY: never return silence/NaN -> fall back to the clean original
    if (not np.isfinite(z).all() or np.max(np.abs(z)) < 0.5 * peak_in
            or np.sqrt(np.mean(z ** 2)) < 1e-3):
        return y
    return z.astype(np.float32)


def prepare_dataset(batch, processor):
    # Prepare output dictionary
    new_batch = {
        "audio": [],
        "text": [],
        "surah": [],
        "ayah": [],
        "id": [],
        "input_features": [],
        "labels": [],
        "reciter": []
    }
    
    import librosa
    
    # Iterate over the batch
    for i in range(len(batch["audio"])):
        audio_path = batch["audio"][i]
        text = batch["text"][i]
        try:
            surah = batch.get("surah", [None]*len(batch["audio"]))[i]
            ayah = batch.get("ayah", [None]*len(batch["audio"]))[i]
        except:
            surah, ayah = None, None

        id_val = batch["id"][i] if "id" in batch else f"id_{i}"
        reciter = batch["reciter"][i] if "reciter" in batch else "Unknown"
        
        try:
            # OPTIMIZATION: Check Duration from Path first (Fast)
            # This avoids loading/resampling audio for the 17k files we will skip
            duration = librosa.get_duration(path=audio_path)
            
            # LOGIC: Train on ALL files < 30s
            if duration > 30.0:
                # Skip > 30s
                continue
            
            # Load audio -> Resample to 16kHz (Only for valid files)
            audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
            
            # Keep the clean original, plus a phone/noise-augmented copy so the model
            # learns to hear kid audio recorded on real phones (the deployment domain).
            items_to_add = [(audio_array, "original")]
            if AUGMENT:
                try:
                    items_to_add.append((phone_augment(audio_array, sampling_rate), "phone_aug"))
                except Exception as aug_err:
                    print(f"Augmentation failed for {audio_path}: {aug_err}")

            # Add to batch
            for audio_data, kind in items_to_add:
                features = processor.feature_extractor(audio_data, sampling_rate=sampling_rate).input_features[0]
                lab = processor.tokenizer(text).input_ids

                if len(lab) > 448:
                    continue
                
                new_batch["audio"].append(audio_path)
                new_batch["text"].append(text)
                new_batch["surah"].append(surah)
                new_batch["ayah"].append(ayah)
                new_batch["id"].append(f"{id_val}_{kind}")
                new_batch["input_features"].append(features)
                new_batch["labels"].append(lab)
                new_batch["reciter"].append(reciter)

        except Exception as e:
            print(f"Skipping bad file {audio_path}: {e}")
            continue
            
    return new_batch






def train():
    global BASE_MODEL, AUGMENT
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment", action="store_true", help="add a phone_aug copy per sample (default off; was a wash on turbo)")
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--base_model", type=str, default=BASE_MODEL,
                        help="Base Whisper to fine-tune (e.g. openai/whisper-large-v3-turbo)")
    parser.add_argument("--num_proc", type=int, default=8, help="Preprocessing workers (lower for less RAM)")
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH)
    parser.add_argument("--resume_adapter", type=str, default=None,
                        help="Path to a LoRA adapter to continue training from (V6 = V5 + new data)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset size for debugging")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="LoRA lr (research: ~1e-4)")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout (regularize small data)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=None, help="default = 10% of max_steps")
    parser.add_argument("--save_steps", type=int, default=None, help="checkpoint interval (save several to pick best)")
    args, _ = parser.parse_known_args()
    BASE_MODEL = args.base_model
    AUGMENT = args.augment

    print(f"Loading processor for {BASE_MODEL}...")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)

    print(f"Loading dataset from {args.dataset_path}...")
    try:
        dataset = load_from_disk(args.dataset_path)
    except Exception as e:
        print(f"Could not load dataset: {e}")
        return

    print(f"Initial Dataset Size: {len(dataset)}")
    
    # Optional slicing for debug
    if args.max_samples is not None and args.max_samples < len(dataset):
        print(f"Slicing dataset to {args.max_samples} samples for debugging...")
        dataset = dataset.select(range(args.max_samples))

    # Simple check for full dataset
    if len(dataset) < 100 and args.max_samples is None:
        print("WARNING: Dataset seems very small. Are you sure this is correct?")

    print("Preprocessing dataset...")
    # Map dataset first
    dataset = dataset.map(
        lambda x: prepare_dataset(x, processor), 
        batched=True, 
        batch_size=32,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        num_proc=args.num_proc
    )
    
    # Filter out any None/Empty entries if prepare_dataset failed (it currently just skips internally but returns lists, 
    # but HF map expects consistent lengths. prepare_dataset returns dict of lists. 
    # If we skipped items, lists might be shorter than batch? 
    # Actually, prepare_dataset in this code iterates batch and appends valid ones. 
    # HF `map` with `batched=True` expects the function to return a dict of lists with the *same* length as input batch 
    # OR a new length if it's a 1-to-many or filter operation. 
    # Since we are constructing a NEW batch from scratch in `prepare_dataset`, 
    # and we pass `remove_columns`, this effectively replaces the dataset.
    # So the length change is handled by HF.
    
    print(f"Processed Dataset Size: {len(dataset)}")

    # Create Train/Test Split
    test_size = 0.05
    if len(dataset) < 20: 
        test_size = 0.2 # larger split for tiny debug sets
        
    print(f"Splitting dataset into Train (?) and Test ({test_size})...")
    split_dataset = dataset.train_test_split(test_size=test_size, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"Train Size: {len(train_dataset)}")
    print(f"Eval Size: {len(eval_dataset)}")

    print(f"Loading model {BASE_MODEL}...")
    # Load in 8bit or 4bit if bitsandbytes is available, else fp32 or fp16
    # For now assume we want to setup for LoRA
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Prepare for k-bit training - SKIPPED for FP16 compatibility
    # model = prepare_model_for_kbit_training(model)
    # However, for gradient checkpointing to work, we need this:
    model.enable_input_require_grads()
    model.config.use_cache = False

    # Apply LoRA
    # We can either load existing adapter or start new
    # If starting new:
    # config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_2_SEQ_LM)
    # model = get_peft_model(model, config)
    
    # But plan says: "Load existing adapter to continue training"
    # To continue training a LoRA adapter, we load it.
    # However, PeftModel.from_pretrained puts it in inference mode usually. 
    # We need to set is_trainable=True.
    
    # Apply LoRA
    # V4: Fresh Start from Base Model
    if args.resume_adapter:
        print(f"Continuing from existing adapter: {args.resume_adapter}")
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
    else:
        print("Initializing new LoRA config...")
        config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, config)

    model.print_trainable_parameters()

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps if args.warmup_steps is not None else max(int(0.1 * args.max_steps), 0),
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="no",            # real quality gate is the RetaSy validation, not internal eval
        predict_with_generate=False,
        save_steps=args.save_steps if args.save_steps else (1000 if args.max_steps > 1000 else 10),
        logging_steps=50 if args.max_steps > 50 else 1,
        report_to="none",
        load_best_model_at_end=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
    )

    
    
    # Check for existing checkpoints
    last_checkpoint = None
    if os.path.exists(args.output_dir): 
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
        if checkpoints:
            last_checkpoint = True 
            print("Found existing checkpoints. Resuming training...")

    print("Starting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
if __name__ == "__main__":
    train()
