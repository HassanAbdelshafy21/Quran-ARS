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
        
        self.processor = WhisperProcessor.from_pretrained("tarteel-ai/whisper-base-ar-quran")
        
        # Handle local paths correctly for newer transformers versions
        if os.path.isdir(checkpoint_path):
            self.model = WhisperForConditionalGeneration.from_pretrained(
                checkpoint_path, 
                local_files_only=True,
                trust_remote_code=True
            ).to(self.device)
        else:
            self.model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path).to(self.device)
        
        # Optimization
        if self.device == "cuda":
            self.model.half() # FP16
            
        # Force Arabic Config
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="ar", task="transcribe")
        self.model.config.suppress_tokens = []
        
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
        words_with_timestamps = []
        try:
            # Decode with timestamps to get segment-level timing
            decoded_with_ts = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=False, 
                decode_with_timestamps=True
            )[0]
            
            # Parse timestamp tokens: <|0.00|>text<|0.50|>
            import re
            # Pattern: <|start_time|>text<|end_time|>
            segments = re.findall(r'<\|(\d+\.\d+)\|>(.*?)<\|(\d+\.\d+)\|>', decoded_with_ts)
            
            for start_str, segment_text, end_str in segments:
                start_time = float(start_str)
                end_time = float(end_str)
                segment_text = segment_text.strip()
                
                if not segment_text:
                    continue
                
                # Split segment into words and distribute timestamps
                segment_words = segment_text.split()
                if len(segment_words) == 0:
                    continue
                    
                time_per_word = (end_time - start_time) / len(segment_words)
                
                for idx, word in enumerate(segment_words):
                    word_start = round(start_time + idx * time_per_word, 2)
                    word_end = round(start_time + (idx + 1) * time_per_word, 2)
                    words_with_timestamps.append({
                        "word": word,
                        "start": word_start,
                        "end": word_end
                    })
                    
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
