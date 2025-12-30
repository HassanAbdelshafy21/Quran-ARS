import torch
from transformers import WhisperProcessor
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
        
        # DEBUG: Print keys
        print(f"DEBUG: Batch keys before fix: {batch.keys()}")
        
        # HOTFIX: Ensure 'input_ids' is NOT passed to Whisper
        if "input_ids" in batch:
            print("DEBUG: Deleting input_ids")
            del batch["input_ids"]
            
        print(f"DEBUG: Batch keys after fix: {batch.keys()}")
        return batch

def test_collator():
    print("Testing DataCollator...")
    MODEL_ID = "tarteel-ai/whisper-base-ar-quran"
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Mock Data
    dummy_input = np.random.randn(80, 3000) # Mel spec features approx
    dummy_labels = [1, 2, 3, 4, 50257]
    
    features = [
        {"input_features": dummy_input, "labels": dummy_labels},
        {"input_features": dummy_input, "labels": dummy_labels}
    ]

    print("Calling collator...")
    batch = collator(features)
    
    print("--- Final Batch Keys ---")
    for k in batch.keys():
        print(f"Key: {k}, Type: {type(batch[k])}")
        
    if "input_ids" in batch:
        print("FAIL: input_ids found in batch!")
    else:
        print("SUCCESS: input_ids not found in batch.")

if __name__ == "__main__":
    test_collator()
