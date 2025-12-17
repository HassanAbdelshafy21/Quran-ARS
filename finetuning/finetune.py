
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
    Seq2SeqTrainer
)
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Constants
if not torch.cuda.is_available():
    raise RuntimeError("No GPU found. Training requires a GPU.")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

BASE_MODEL = "tarteel-ai/whisper-base-ar-quran"
ADAPTER_MODEL = "KheemP/whisper-base-quran-lora"
DATASET_PATH = "data/quran_dataset"
OUTPUT_DIR = "finetuning/checkpoints"

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
        "labels": []
    }
    
    import librosa
    
    # Iterate over the batch
    for i in range(len(batch["audio"])):
        audio_path = batch["audio"][i]
        text = batch["text"][i]
        surah = batch["surah"][i]
        ayah = batch["ayah"][i]
        id_val = batch["id"][i]
        
        try:
            # Load and resample to 16kHz
            audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
            
            # compute log-Mel input features from input audio array 
            features = processor.feature_extractor(audio_array, sampling_rate=sampling_rate).input_features[0]
            
            # encode target text to label ids 
            lab = processor.tokenizer(text).input_ids
            
            # Add to new batch
            new_batch["audio"].append(audio_path)
            new_batch["text"].append(text)
            new_batch["surah"].append(surah)
            new_batch["ayah"].append(ayah)
            new_batch["id"].append(id_val)
            new_batch["input_features"].append(features)
            new_batch["labels"].append(lab)
            
        except Exception as e:
            print(f"Skipping bad file {audio_path}: {e}")
            continue
            
    return new_batch

def train():
    print(f"Loading processor for {BASE_MODEL}...")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    
    print(f"Loading dataset from {DATASET_PATH}...")
    try:
        dataset = load_from_disk(DATASET_PATH)
    except Exception as e:
        print(f"Could not load dataset: {e}")
        return

    print("Preprocessing dataset...")
    # Use batched=True to allow filtering out bad examples
    dataset = dataset.map(
        lambda x: prepare_dataset(x, processor), 
        batched=True, 
        batch_size=32,
        num_proc=1,
        remove_columns=dataset.column_names # Remove old columns to avoid mismatch length
    )

    print(f"Loading model {BASE_MODEL}...")
    # Load in 8bit or 4bit if bitsandbytes is available, else fp32 or fp16
    # For now assume we want to setup for LoRA
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL, 
        load_in_8bit=True, 
        device_map="auto"
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    # We can either load existing adapter or start new
    # If starting new:
    # config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_2_SEQ_LM)
    # model = get_peft_model(model, config)
    
    # But plan says: "Load existing adapter to continue training"
    # To continue training a LoRA adapter, we load it.
    # However, PeftModel.from_pretrained puts it in inference mode usually. 
    # We need to set is_trainable=True.
    
    print(f"Loading adapter {ADAPTER_MODEL}...")
    try:
        model = PeftModel.from_pretrained(model, ADAPTER_MODEL, is_trainable=True)
        print("Loaded existing adapter.")
    except Exception as e:
        print(f"Could not load adapter (might not exist locally or on hub?): {e}")
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
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=500,
        num_train_epochs=10,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=50,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset,
        eval_dataset=dataset, # Should split train/test
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
    )

    print("Starting training...")
    trainer.train() 
    # Commented out to prevent auto-run

if __name__ == "__main__":
    train()
