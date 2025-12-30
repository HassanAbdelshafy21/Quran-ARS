
import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_from_disk
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
ADAPTER_MODEL = "KheemP/whisper-base-quran-lora"
DATASET_PATH = "data/quran_dataset"
OUTPUT_DIR = "finetuning/checkpoints_v4"

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

        return batch

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
        surah = batch["surah"][i]
        ayah = batch["ayah"][i]
        id_val = batch["id"][i]
        reciter = batch["reciter"][i]
        
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
            
            # Identify if it's the Child reciter
            is_child = "Minshawy_Teacher" in str(reciter)
            
            # We will add Original to ALL files < 30s
            items_to_add = [(audio_array, "original")]

            import random
            import numpy as np
            
            # Augment ONLY Adult reciters (to avoid chipmunking the child further)
            if not is_child:
                try:
                    # Force Augmentation (Pitch Shift)
                    n_steps = random.uniform(2.0, 4.0) 
                    augmented_audio = librosa.effects.pitch_shift(audio_array, sr=sampling_rate, n_steps=n_steps)
                    items_to_add.append((augmented_audio, "augmented"))
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset size for debugging")
    args, _ = parser.parse_known_args()

    print(f"Loading processor for {BASE_MODEL}...")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)

    print(f"Loading dataset from {DATASET_PATH}...")
    try:
        dataset = load_from_disk(DATASET_PATH)
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
        num_proc=8
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
    print("Initializing new LoRA config...")
    config = LoraConfig(
        r=32, 
        lora_alpha=64, 
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05, 
        bias="none", 
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, config)

    model.print_trainable_parameters()

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=1e-5,
        warmup_steps=500 if args.max_steps > 500 else 0,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000 if args.max_steps > 1000 else 10,
        eval_steps=1000 if args.max_steps > 1000 else 10,
        logging_steps=50 if args.max_steps > 50 else 1,
        report_to="none",
        load_best_model_at_end=True if args.max_steps > 100 else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, 
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
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
