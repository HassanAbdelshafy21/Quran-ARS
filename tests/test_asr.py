
import pytest
from unittest.mock import MagicMock, patch
import os
import torch
from src.asr.transcriber import ASRTranscriber

# Mock transformers and peft to avoid downloading models during CI/CD or local test cycles
# unless we explicitly want integration tests.
# For this environment, real download might be slow and flaky.
# Let's do a mock test first to verify logic, then maybe a real test if requested.

@pytest.fixture
def mock_asr_deps():
    with patch("src.asr.transcriber.WhisperProcessor") as mock_processor, \
         patch("src.asr.transcriber.WhisperForConditionalGeneration") as mock_model_cls, \
         patch("src.asr.transcriber.PeftModel") as mock_peft:
         
        mock_model_instance = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model_instance
        
        mock_peft_instance = MagicMock()
        mock_peft.from_pretrained.return_value = mock_peft_instance
        
        mock_processor_instance = MagicMock()
        mock_processor.from_pretrained.return_value = mock_processor_instance
        
        # Setup generate return
        mock_peft_instance.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_processor_instance.batch_decode.return_value = ["تسۡت"]
        
        mock_peft_instance.config.forced_decoder_ids = None
        
        yield {
            "processor": mock_processor_instance,
            "model": mock_model_instance,
            "peft": mock_peft_instance
        }

def test_asr_initialization(mock_asr_deps):
    transcriber = ASRTranscriber()
    assert transcriber.model is not None
    assert transcriber.processor is not None

def test_asr_transcription_flow(mock_asr_deps, tmp_path):
    # Create dummy audio file
    import soundfile as sf
    import numpy as np
    
    file_path = tmp_path / "test.wav"
    sf.write(str(file_path), np.random.uniform(-1, 1, 16000), 16000)
    
    transcriber = ASRTranscriber()
    result = transcriber.transcribe(str(file_path))
    
    assert "text" in result
    assert result["text"] == "تسۡت"
