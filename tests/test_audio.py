
import pytest
import os
import numpy as np
import soundfile as sf
import librosa
from src.utils.audio import preprocess_audio

@pytest.fixture
def temp_audio_file(tmp_path):
    # Create a dummy audio file (44.1kHz, Stereo)
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    # Stereo signal: two channels
    y = np.vstack((np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t))).T
    
    file_path = tmp_path / "test_stereo_44k.wav"
    sf.write(str(file_path), y, sr)
    return str(file_path)

def test_preprocess_audio_exists(temp_audio_file):
    output = preprocess_audio(temp_audio_file)
    assert os.path.exists(output)
    
    # Clean up (if output wasn't in tmp_path, but here it acts relative if not absolute)
    # The default behavior saves next to input, which is in tmp_path, so it's fine.

def test_preprocess_audio_properties(temp_audio_file):
    output_path = preprocess_audio(temp_audio_file)
    
    # Load back to check properties
    y, sr = librosa.load(output_path, sr=None, mono=False)
    
    assert sr == 16000 # Checked sample rate
    assert y.ndim == 1 or (y.ndim == 2 and y.shape[0] == 1) # Check mono (1D or 1 channel)
    
    # librosa.load with mono=False returns (channels, samples) for multi-channel, or (samples,) for mono
    # actually librosa.load(mono=False) might return (2, N) for stereo. 
    # If the file is mono, it returns (N,).
    
    # Double check with soundfile to be sure about file format on disk
    info = sf.info(output_path)
    assert info.samplerate == 16000
    assert info.channels == 1
