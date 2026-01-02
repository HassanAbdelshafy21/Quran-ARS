
# Fine-Tuning Plan: Whisper LoRA for Quran

This document outlines the strategy to fine-tune `tarteel-ai/whisper-base-ar-quran` with the adapter `KheemP/whisper-base-quran-lora` (or start a fresh one) using your local data.

## Goal
Improve the ASR accuracy by training on a dataset of <Audio, Text> pairs derived from Quranic recitations.

## 1. Dataset Preparation
To fine-tune, we need a dataset. We can construct one programmatically using `everyayah.com` audio and our `quran.db`.

### 1.1 Data Source
- **Audio**: Download full Quran recitations (e.g., Mishary Rashid or Abdul Basit) split by Ayah.
- **Text**: Use `quran.db` (Uthmani or Clean script).

### 1.2 Dataset Structure
We will create a HuggingFace-compatible dataset generator.
- **Input**: Audio path (resampled to 16kHz).
- **Target**: Text (normalized or with tashkeel).

## 2. Environment Setup
We need to ensure `peft`, `bitsandbytes` (optional for 8-bit), and `transformers` are set up.
> [!WARNING]
> Fine-tuning requires a GPU with at least 8GB VRAM (for Base model + LoRA).

## 3. Training Script
We will create `src/training/finetune.py`.

### Key Components:
1.  **Model Loading**:
    ```python
    model = WhisperForConditionalGeneration.from_pretrained("tarteel-ai/whisper-base-ar-quran")
    # Load existing adapter to continue training
    model = PeftModel.from_pretrained(model, "KheemP/whisper-base-quran-lora", is_trainable=True)
    ```
2.  **Data Processing**:
    - Use `WhisperProcessor` to convert audio to `input_features`.
    - Tokenize text labels.
3.  **Trainer**:
    - Use `Seq2SeqTrainer` from `transformers`.
    - Arguments: `per_device_train_batch_size=8`, `learning_rate=1e-5`, `fp16=True`.

## 4. Evaluation
Compare WER (Word Error Rate) on a validation set (e.g., Surahs not in training) before and after training.

## 5. Next Steps
1.  **Generate Dataset**: Script to download ~1000 ayahs.
2.  **Implement Training Code**: `finetune.py`.
3.  **Run Training**: Execute locally.
