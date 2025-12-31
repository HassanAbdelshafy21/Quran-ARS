import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import os
import sys

# Ensure UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

class QuranModel:
    """
    Wrapper for the V5 Fine-Tuned Whisper Model.
    """
    def __init__(self, checkpoint_path, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Model from {checkpoint_path} on {self.device}...")
        
        self.processor = WhisperProcessor.from_pretrained("tarteel-ai/whisper-base-ar-quran")
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
        """
        # Prepare inputs
        inputs = self.processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(self.device)
        
        if self.device == "cuda":
            inputs = inputs.half()
            
        # Generate with Beam Search for better accuracy on children's speech
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs, 
                max_new_tokens=400,
                num_beams=5, 
                early_stopping=True
            )
            
        # Decode
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription.strip()
