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
* **🗣️ Speaking Teacher (TTS):** The system generates **Audio Feedback** using natural Arabic TTS. It praises correct recitation ("Ahsant!") and verbally corrects specific mistakes ("You forgot 'Wa-Huwa'").
* **🔌 Production Feedback API:** A FastAPI backend that returns detailed, actionable feedback in **Arabic JSON**, ready for mobile app integration.

---

## 🛠️ Project Structure

The repository is organized for production deployment:

* **`backend/`**: The Core API.
  * `main.py`: FastAPI entry point.
  * `core/`: Model logic, Grader, TTS, and VAD.
* **`data/logs/`**: All logs and JSON responses are saved here.
* **`reports/`**: Generated Markdown reports (Student vs Parent views).
* **`temp_storage/`**: Temporary audio files (recordings & TTS).
* **`finetuning/`**: Research and Training scripts.

---

## 💻 Installation

### Prerequisites

* Python 3.9+
* CUDA-enabled GPU (Recommended)
* FFmpeg

### Setup

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

The system runs as a **FastAPI** server.

### 1. Start the Server

```bash
python backend/main.py
```

* The server will load the **V5-30k Model** (Heavy) onto the GPU.
* It listens on `http://0.0.0.0:8000`.

### 2. Run the Demo Client

We have a test script to simulate a mobile app request.

```bash
python backend/test_client.py
```

* **Edit `backend/test_client.py`** to switch between test files (e.g., `test 3.ogg` for Al-Hadid or `test 6.mp4` for Al-Kafirun).
* The script sends the audio to the API and saves the response to `data/logs/response_log.json`.

---

## 📱 The "Dual View" Experience

The system generates two levels of feedback:

### 🅰️ Parent View (Monitoring)

* Full Transcription of what the child said.
* Detailed list of mistakes.
* Accuracy Score.

### 🅱️ Student View (Learning)

* **Interactive Feedback:** "The Teacher" speaks to them.
* **Sheikh Comparison:** If they make mistakes, they get a link to Sheikh Minshawi's correct recitation.
* **Celebration:** If they get 100%, the correction section disappears!

---

## 📊 Benchmarks

| Model | Child Accuracy | Notes |
| :--- | :---: | :--- |
| **Baseline** | Low | Fails on mumbling. |
| **V5 (30k) 🏆** | **High (~95%)** | **Golden Candidate.** Best balance of phonetic accuracy and stability. |

---

## 📜 License

This project is licensed under the MIT License.
