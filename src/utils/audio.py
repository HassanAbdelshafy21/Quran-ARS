
import librosa
import soundfile as sf
import os
import numpy as np

def preprocess_audio(input_path: str, output_path: str = None) -> str:
    """
    Loads an audio file, converts it to 16kHz mono, and saves it.
    
    Args:
        input_path: Path to the input audio file.
        output_path: Path to save the processed audio. If None, appends '_processed.wav' to input filename.
        
    Returns:
        Path to the processed audio file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_processed.wav"
        
    # Load audio: librosa automatically resamples to sr if provided, and converts to mono if mono=True
    # sr=16000 is Whisper's expected sample rate
    y, sr = librosa.load(input_path, sr=16000, mono=True)
    
    # Save as WAV (subtype 'PCM_16' is standard 16-bit)
    sf.write(output_path, y, 16000, subtype='PCM_16')
    
    return output_path
