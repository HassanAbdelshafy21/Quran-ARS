# Quran-ARS (Quran Recitation AI System)

An AI-powered system designed to assist in learning Quran recitation. The system analyzes audio input, transcribes it, and compares it against the Quranic text to detect mistakes in words, pronunciation, and Tajweed.

## 📌 Project Overview

The goal of Quran-ARS is to build a robust feedback loop for Quran students. The system:
1.  **Transcribes** user recitation using state-of-the-art ASR models (currently Whisper + LoRA).
2.  **Aligns** the recitation with the correct Ayah from a verified Quran database.
3.  **Detects** errors (planned features):
    *   Word mistakes (substitution, omission, addition).
    *   Harakat/Vowel errors.
    *   Pronunciation accuracy.
    *   Tajweed rule violations (timing, pitch, nasalization).

## 📂 Project Structure

```
├── data/               # Contains Quran database and sample audio files
│   ├── quran/          # Raw SQL data for Quran text
│   └── quran.db        # SQLite database generated from SQL data
├── Planning/           # Project roadmap, plans, and research documents
├── scripts/            # Utility scripts (if any)
├── src/                # Source code
│   ├── asr/            # Automatic Speech Recognition (Whisper integration)
│   ├── quran_db/       # Database interface for querying Ayahs
│   └── utils/          # Helper functions (e.g., audio preprocessing)
├── tests/              # Unit tests
└── requirements.txt    # Python dependencies
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- `ffmpeg` (required by `librosa`/`soundfile` for audio processing)

### Steps

1.  **Clone the repository:**
    ```bash
    git clone <repo-url>
    cd Quran-ARS
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🛠️ Usage

### 1. Audio Transcription (ASR)
The system currently uses a Whisper-based model tuned for Quranic Arabic.

```python
from src.asr.transcriber import ASRTranscriber

# Initialize the model (downloads weights on first run)
transcriber = ASRTranscriber()

# Transcribe an audio file
result = transcriber.transcribe("data/sample_001001.mp3")
print(result["text"])
```

### 2. Querying the Quran Database
You can search for Ayahs or retrieve them by Surah and Ayah number.

```python
from src.quran_db.core import QuranDB

db = QuranDB()

# Get specific Ayah (Surah 1, Ayah 1)
ayah = db.get_ayah(1, 1)
print(ayah.aya_text)

# Search for text
results = db.search_text("الحمد")
for res in results:
    print(f"{res.sura_name_en} ({res.sura_no}:{res.aya_no})")
```

### 3. Regenerating the Database
If you need to rebuild `data/quran.db` from the source SQL file:
```bash
python src/quran_db/importer.py
```

## 🧪 Testing

Run the test suite using `pytest`. Ensure you are in the root directory.

```bash
# Run all tests
PYTHONPATH=. pytest

# Run specific test file
PYTHONPATH=. pytest tests/test_asr.py
```

## 🚧 Status & Roadmap

- [x] **Quran Database:** SQL importer and SQLite interface implemented.
- [x] **Basic ASR:** Whisper Base + LoRA adapter integration working.
- [x] **Audio Utils:** Basic preprocessing (resampling to 16kHz).
- [ ] **Forced Alignment:** Move to Wav2Vec2 or Whisper-timestamp-based alignment for phoneme-level accuracy.
- [ ] **Error Detection:** Implement logic to compare transcription/alignment vs. expected text.
- [ ] **API:** Build FastAPI backend to serve the model.

## 📄 License
MIT License

Copyright (c) 2025 Hassan Abdelshafy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
