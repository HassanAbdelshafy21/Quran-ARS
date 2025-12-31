# 🎙️ Quran ASR for Kids (V5 Production)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Whisper](https://img.shields.io/badge/OpenAI-Whisper-green)
![License](https://img.shields.io/badge/License-MIT-purple)

**An advanced AI system designed to listen, transcribe, and grade children's recitation of the Holy Quran.**

Unlike standard ASR models that fail on child speech ("mumbling", incorrect tajweed, stuttering), this system is fine-tuned specifically for children's voices and includes a **"Tolerant Grading"** layer that focuses on phonetic intent rather than perfect spelling.

---

## 🚀 Key Features

* **👶 Child-Centric ASR (V5-30k):** Fine-tuned on the "Minshawi with Child Repeat" dataset. It understands high-pitched, hesitant, and mumbled speech that standard Whisper models miss.
* **🧠 Tolerant Grader:** A custom grading engine (`grader.py`) that uses **Phonetic Normalization** and **Fuzzy Alignment**. It forgivingly grades "Wal-ti" as correct for "Wal-teen", mirroring a human teacher's leniency for beginners.
* **✂️ Smart Segmentation (VAD):** Handles long recordings (15+ minutes) by intelligently splitting audio based on breath pauses, ensuring no words are cut off.
* **🔌 Production Feedback API:** A FastAPI backend that returns detailed, actionable feedback in **Arabic JSON** (`كلمة ناقصة`, `خطأ`, etc.), ready for mobile app integration.
* **🔉 Corrective Loop:** pinpoint accuracy allows the frontend to play the *exact* word the child missed, followed by the Teacher's (Minshawi) correct pronunciation.

---

## �️ Installation

### Prerequisites

* Python 3.9+
* CUDA-enabled GPU (Recommended for real-time inference)
* FFmpeg (for audio processing)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/YourUsername/Quran-ARS.git
cd Quran-ARS

# 2. Install dependencies
pip install -r requirements.txt
```

---

## �️ Usage (Backend API)

The system runs as a **FastAPI** server.

### 1. Start the Server

```bash
python backend/main.py
```

* The server will load the **V5-30k Model** (Heavy) onto the GPU.
* It listens on `http://0.0.0.0:8000`.

### 2. Test the API

You can use the included test script to simulate a mobile app request.

```bash
python backend/test_client.py
```

### 3. API Endpoint: `/grade_recitation`

**Input (Multipart Form):**

* `file`: The Audio File (mp3/wav/m4a).
* `target_ayah`: The correct Quranic text (Surah).

**Output (JSON - Arabic):**

```json
{
  "status": "success",
  "user_recitation": "قُلْ يَا هِيُهَا الْكَافِقِّدُونَ ...",
  "expected_recitation": "قُلْ يَا أَيُّهَا الْكَافِرُونَ ...",
  "passed": false,
  "accuracy": 0.59,
  "mistakes": [
    "خطأ: قلت 'هِيُهَا' بدلاً من 'أَيُّهَا'",
    "كلمة ناقصة: 'لَا'"
  ]
}
```

---

## 📊 Model Performance

We benchmarked multiple versions against a challenging **Child Validation Set** (Real classroom recordings).

| Model Version | Standard WER | Child Accuracy | Notes |
| :--- | :---: | :---: | :--- |
| **Tarteel Base** | > 80% | Low | Fails on mumbling/repetitions. |
| **V3 (50k)** | ~40% | Moderate | Suffered from "Dictionary Overfitting". |
| **V5 (30k) 🏆** | **~25%** | **High** | **Golden Candidate.** Best balance of phonetic accuracy and stability. |

> **Note:** "Standard WER" is misleading for Arabic. Our **Tolerant Grader** boosts the effective user accuracy to **~85-95%** by ignoring spelling variations (Imla'i vs Uthmani).

---

## 📂 Project Structure

* `backend/` - The Production API code.
  * `core/model_loader.py` - Wraps the V5 Whisper Model (Beam Search enabled).
  * `core/grader.py` - The Logic Brain (Phonetic normalization & Grading).
  * `core/segmenter.py` - VAD logic for long files.
  * `main.py` - FastAPI Entry point.
* `finetuning/` - Training scripts and research.
  * `finetune.py` - The Hugging Face training script.
  * `quran_grader.py` - The original prototype of the grader.
* `data/` - Dataset management scripts.

---

## 🔮 Future Roadmap (The "Data Flywheel")

1. **Launch V1:** Deploy V5-30k to the first 50 users.
2. **Collect Data:** Save "Wrong" predictions that users flag as "Actually Correct".
3. **Train V6:** Fine-tune specifically on these edge cases (Accents, Lisp, Stuttering).
4. **Edge Deployment:** Port the model to ONNX/CoreML for offline use on iPads/Android.

---

## 📜 License

This project is licensed under the MIT License.
