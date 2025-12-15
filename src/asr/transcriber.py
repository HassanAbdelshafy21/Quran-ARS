
import os
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperConfig
from peft import PeftModel, PeftConfig
from src.utils.audio import preprocess_audio
import librosa

class ASRTranscriber:
    def __init__(self, base_model_name: str = "tarteel-ai/whisper-base-ar-quran", 
                 lora_model_name: str = "KheemP/whisper-base-quran-lora",
                 cache_dir: str = None):
        """
        Initializes the ASR model with optional LoRA adapter.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = WhisperProcessor.from_pretrained(base_model_name, cache_dir=cache_dir)
        
        # Load base model
        self.model = WhisperForConditionalGeneration.from_pretrained(base_model_name, cache_dir=cache_dir)
        
        # Load LoRA adapter if provided
        if lora_model_name:
            print(f"Loading LoRA adapter: {lora_model_name}")
            try:
                self.model = PeftModel.from_pretrained(self.model, lora_model_name, cache_dir=cache_dir)
                print("LoRA adapter loaded successfully.")
            except Exception as e:
                print(f"Failed to load LoRA adapter: {e}. proceed with base model.")
        
        self.model.to(self.device)
        self.model.config.forced_decoder_ids = None # Needed for flexibility in some whisper versions
        
    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribes the given audio file.
        
        Args:
            audio_path: Path to the audio file.
            
        Returns:
            dict: {"text": str, "chunks": list}
        """
        # Ensure 16kHz
        # For inference, we can load directly with librosa to get arrays, 
        # but preprocess_audio saves it to disk which might be redundant if we just want the array.
        # However, it's safer to ensure we have a clean file. 
        # Let's load array using librosa directly for speed if file exists.
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        # Resample to 16000 Hz
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Preprocess input
        input_features = self.processor(y, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(self.device)
        
        # Generate token ids
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)
            
        # Decode token ids to text
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # Basic chunking isn't natively supported by simple .generate() without return_timestamps=True 
        # and using appropriate model config. 
        # For MVP, we stick to text.
        
        return {
            "text": transcription,
            "chunks": [] # Implement word timestamps later if needed
        }

if __name__ == "__main__":
    # Test script
    transcriber = ASRTranscriber()
    # Dummy test to see if it loads
    print("Model loaded.")
