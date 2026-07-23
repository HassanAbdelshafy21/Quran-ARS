import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import os
import sys
import numpy as np

# Ensure UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

class QuranModel:
    """
    Wrapper for the V5 Fine-Tuned Whisper Model.
    Supports word-level timestamps extraction.
    """
    def __init__(self, checkpoint_path, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Model from {checkpoint_path} on {self.device}...")
        
        BASE_MODEL = "tarteel-ai/whisper-base-ar-quran"
        self.processor = WhisperProcessor.from_pretrained(BASE_MODEL)

        # The shipped checkpoint is a LoRA adapter (adapter_config.json), not a full
        # model. Load the base model, then apply + merge the adapter so the rest of
        # this class runs against a plain WhisperForConditionalGeneration.
        adapter_config = (
            os.path.join(checkpoint_path, "adapter_config.json")
            if os.path.isdir(checkpoint_path) else None
        )
        if adapter_config and os.path.exists(adapter_config):
            from peft import PeftModel
            base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
            self.model = PeftModel.from_pretrained(base, checkpoint_path)
            self.model = self.model.merge_and_unload()  # fold LoRA into base weights
            self.model = self.model.to(self.device)
        else:
            # Plain full-model directory or HuggingFace hub id
            self.model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path).to(self.device)
        
        # Optimization
        if self.device == "cuda":
            self.model.half() # FP16
            
        # Force Arabic Config
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="ar", task="transcribe")
        self.model.config.suppress_tokens = []

        # The fine-tuned base ships a generation config without no_timestamps_token_id,
        # which makes generate(return_timestamps=True) raise on modern transformers.
        # Derive it from the tokenizer (no external download).
        gen = self.model.generation_config
        if getattr(gen, "no_timestamps_token_id", None) is None:
            gen.no_timestamps_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")

        print("Model Loaded Successfully.")

    def transcribe(self, audio_array, sampling_rate=16000):
        """
        Transcribes a raw audio numpy array.
        Returns dict with 'text' and 'words' (with timestamps).
        """
        # Prepare inputs
        inputs = self.processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(self.device)
        
        if self.device == "cuda":
            inputs = inputs.half()
            
        # Generate with timestamps enabled
        with torch.no_grad():
            # First: Generate with beam search for best transcription
            generated_ids = self.model.generate(
                inputs, 
                max_new_tokens=400,
                num_beams=5, 
                early_stopping=True,
                return_timestamps=True
            )
            
        # Decode full text
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # Extract word-level timestamps
        audio_duration = len(audio_array) / sampling_rate
        words_with_timestamps = []
        try:
            # Decode with timestamps to get segment-level timing
            decoded_with_ts = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=False,
                decode_with_timestamps=True
            )[0]

            # Whisper emits a timestamp token before each spoken chunk, e.g.
            # "<|0.00|>text<|1.20|>more text". Depending on version/audio it may emit
            # only an opening timestamp with no closing one. Split on every timestamp
            # token and treat each chunk as spanning from its timestamp to the next one
            # (or to the end of the audio for the final chunk).
            import re
            parts = re.split(r'<\|(\d+\.\d+)\|>', decoded_with_ts)
            # parts = [leading_special_tokens, t0, text0, t1, text1, ...]
            chunks = []  # (start_time, text)
            i = 1
            while i < len(parts):
                start_time = float(parts[i])
                text = parts[i + 1] if i + 1 < len(parts) else ""
                chunks.append((start_time, text.strip()))
                i += 2

            for idx, (start_time, chunk_text) in enumerate(chunks):
                if not chunk_text:
                    continue
                end_time = chunks[idx + 1][0] if idx + 1 < len(chunks) else audio_duration
                if end_time < start_time:
                    end_time = audio_duration

                chunk_words = chunk_text.split()
                if not chunk_words:
                    continue

                time_per_word = (end_time - start_time) / len(chunk_words)
                for w_idx, word in enumerate(chunk_words):
                    words_with_timestamps.append({
                        "word": word,
                        "start": round(start_time + w_idx * time_per_word, 2),
                        "end": round(start_time + (w_idx + 1) * time_per_word, 2)
                    })

            if not words_with_timestamps:
                raise ValueError("no timestamp tokens found in decoded output")

        except Exception as e:
            print(f"Timestamp extraction failed (non-critical): {e}")
            # Fallback: return words without timestamps
            for word in transcription.split():
                words_with_timestamps.append({
                    "word": word,
                    "start": None,
                    "end": None
                })
        
        return {
            "text": transcription,
            "words": words_with_timestamps
        }
    
    def transcribe_simple(self, audio_array, sampling_rate=16000):
        """
        Simple transcription without timestamps (backward compatible).
        Returns plain text string.
        """
        # Prepare inputs
        inputs = self.processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(self.device)
        
        if self.device == "cuda":
            inputs = inputs.half()
            
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs, 
                max_new_tokens=400,
                num_beams=5, 
                early_stopping=True
            )
            
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription.strip()
