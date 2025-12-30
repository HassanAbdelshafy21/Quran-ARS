import torch
from transformers import WhisperProcessor
from datasets import load_from_disk
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import numpy as np

# Copied from finetune.py
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
        
        # HOTFIX CHECK
        if "input_ids" in batch:
            print("!!! WARNING: input_ids found in batch (inside collator) !!!")
            del batch["input_ids"]
        
        return batch

def prepare_dataset(batch, processor):
    # Dummy prepare just to get structure if needed, but we load processed dataset mostly?
    # Actually finetune maps it.
    pass

def test_on_real_data():
    print("Loading Processor...")
    processor = WhisperProcessor.from_pretrained("tarteel-ai/whisper-base-ar-quran")
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    print("Loading Dataset from disk...")
    # This is the RAW dataset before mapping in finetune.py
    # So we need to simulate the mapping "prepare_dataset" step manually for a few items
    dataset = load_from_disk("data/quran_dataset")
    print(f"Dataset Size: {len(dataset)}")
    
    # Take 2 samples
    samples = dataset.select(range(2))
    
    # We need to run the PREPROCESSING that finetune.py does
    # copying code from finetune.py prepare_dataset
    print("Preprocessing 2 samples...")
    
    import librosa
    
    processed_features = []
    
    for i in range(len(samples)):
        audio_path = samples[i]["audio"]
        text = samples[i]["text"]
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Feature extraction
        input_features = processor.feature_extractor(y, sampling_rate=sr).input_features[0]
        
        # Labels
        labels = processor.tokenizer(text).input_ids
        
        processed_features.append({
            "input_features": input_features,
            "labels": labels
        })

    print("Calling Collator...")
    batch = collator(processed_features)
    
    print("\n--- Final Batch Keys ---")
    keys = list(batch.keys())
    print(keys)
    
    if "input_ids" in keys:
        print("FAIL: input_ids is PERSISTENT in the batch.")
    else:
        print("SUCCESS: input_ids is NOT in the batch.")

if __name__ == "__main__":
    test_on_real_data()
