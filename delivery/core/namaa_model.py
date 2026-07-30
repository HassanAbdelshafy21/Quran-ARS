"""NAMAA Cohere-Speech-Tashkeel ASR wrapper — the single model for Quran-ARS.

Outputs the learner's ACTUAL diacritized recitation (words + harakat, acoustically
derived), enabling both word-level (memorization) and harakat-level (tajweed) grading
from one pass. Same CohereAsr class; needs transformers>=5.4 + a GPU (bf16, ~5 GB).
"""
import torch
import numpy as np
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

MODEL_ID = "NAMAA-Space/Cohere-Speech-Tashkeel-2B"


class NamaaModel:
    def __init__(self, model_id=MODEL_ID, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading NAMAA Cohere-Speech-Tashkeel on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = CohereAsrForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=self.dtype).to(self.device).eval()
        print("NAMAA loaded.")

    def transcribe(self, audio_array, sampling_rate=16000):
        """Returns {"text": <diacritized Arabic>, "words":[{word,start,end}]}."""
        y = np.asarray(audio_array, dtype=np.float32)
        inp = self.processor(y, sampling_rate=sampling_rate, return_tensors="pt", language="ar")
        inp.to(self.device, dtype=self.dtype)
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=256)
        d = self.processor.decode(out, skip_special_tokens=True)
        text = (d[0] if isinstance(d, list) else d).strip()
        words = [{"word": w, "start": None, "end": None} for w in text.split()]
        return {"text": text, "words": words}

    def transcribe_simple(self, audio_array, sampling_rate=16000):
        return self.transcribe(audio_array, sampling_rate)["text"]
