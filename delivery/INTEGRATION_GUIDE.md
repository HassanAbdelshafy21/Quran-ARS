# 🔌 Quran ASR Kids — Backend Integration Guide

## 📁 Folder Structure

```
delivery/
├── main.py                         # FastAPI Server (Entry Point)
├── test_client.py                  # Demo Client (CLI)
├── requirements.txt                # Python Dependencies
├── INTEGRATION_GUIDE.md            # This File
│
├── core/                           # Core Processing Modules
│   ├── grader.py                   # Grading Engine (Alignment + Error Analysis)
│   ├── model_loader.py             # Whisper ASR Model (Transcription + Timestamps)
│   ├── segmenter.py                # Audio VAD Segmenter
│   └── tts.py                      # Text-to-Speech Feedback (Arabic)
│
├── model/                          # Fine-Tuned Model (LoRA Adapter)
│   └── checkpoint-30000/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── preprocessor_config.json
│
├── data/
│   └── quran.db                    # SQLite Quran Database (all 6236 Ayahs)
│
└── temp_storage/                   # Auto-created: temporary audio & feedback files
```

---

## 🚀 Setup (Step by Step)

### 1. Prerequisites

- **Python 3.9+**
- **CUDA GPU** (recommended for speed, CPU works but slower)
- **FFmpeg** installed and in PATH (required by librosa)
- **Internet connection** (first run downloads base model ~300MB from HuggingFace)

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install PyTorch with CUDA (if GPU available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install all other dependencies
pip install -r requirements.txt
```

### 3. Start the Server

```bash
cd delivery
python main.py
```

Output:
```
==================================================
Initializing Quran ASR Backend...
Model Path: .../delivery/model/checkpoint-30000
==================================================
Loading Model from .../checkpoint-30000 on cuda...
Model Loaded Successfully.
Backend Ready! 🚀
API Docs: http://localhost:8000/docs
==================================================
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 4. Test with Demo Client

```bash
python test_client.py --file test_audio.ogg --text "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ" --surah 1 --ayah 1
```

### 5. Open Swagger UI

Go to `http://localhost:8000/docs` in your browser for interactive API testing.

---

## 📡 API Endpoints

### `POST /grade_recitation` — Main Grading Endpoint

#### Request

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|:------|:-----|:--------:|:------------|
| `file` | File | ✅ | Audio file (MP3, WAV, OGG, M4A) |
| `target_ayah` | String | ✅ | Expected Quranic text (or from your DB) |
| `surah_num` | Integer | ❌ | Surah number (1-114) — enables Sheikh reference audio |
| `ayah_num` | Integer | ❌ | Ayah number — uses per-ayah reference audio |

#### Response (JSON)

```json
{
  "request_id": "5dd616b3-1a1a-4d7c-8536-d266054eb2e4",
  "status": "success",
  "user_recitation": "بسم الله الرحمن الرحيم",
  "expected_recitation": "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ",
  "passed": true,
  "accuracy": 1.0,
  "raw_score": "4/4",
  "mistakes": [],
  "words": [
    {
      "word": "بسم",
      "expected": "بسم",
      "is_correct": true,
      "score": 1.0,
      "error_type": null,
      "error_type_ar": null,
      "char_errors": [],
      "timestamp_start": 0.50,
      "timestamp_end": 0.92
    }
  ],
  "feedback_audio": "http://localhost:8000/audio/feedback_xxx.mp3",
  "reference_audio": null,
  "segments_processed": 1
}
```

#### Response Fields Reference

| Field | Type | Description |
|:------|:-----|:------------|
| `request_id` | String | Unique request ID (UUID) |
| `status` | String | `"success"` or error message |
| `user_recitation` | String | Full transcription of what the child said |
| `expected_recitation` | String | The correct Quranic text |
| `passed` | Boolean | `true` if accuracy > 85% |
| `accuracy` | Float | Overall accuracy score (0.0 to 1.0) |
| `raw_score` | String | Score as fraction (e.g. "3/4") |
| `mistakes` | Array[String] | Human-readable mistake descriptions (Arabic) |
| `words` | Array[Object] | **Per-word analysis** (see below) |
| `feedback_audio` | String | URL to download TTS feedback audio (MP3) |
| `reference_audio` | String/null | URL to Sheikh recitation (null if passed) |
| `segments_processed` | Integer | Number of audio segments processed |

#### `words[]` Object

| Field | Type | Description |
|:------|:-----|:------------|
| `word` | String/null | What the child said (null = missed word) |
| `expected` | String/null | Correct word (null = extra word) |
| `is_correct` | Boolean | Whether word matched (exact or fuzzy) |
| `score` | Float | Word similarity score (0.0 to 1.0) |
| `error_type` | String/null | `"substitution"` / `"deletion"` / `"insertion"` / null |
| `error_type_ar` | String/null | `"استبدال"` / `"حذف"` / `"زيادة"` / null |
| `char_errors` | Array[Object] | Character-level errors (see below) |
| `timestamp_start` | Float/null | Word start time in seconds |
| `timestamp_end` | Float/null | Word end time in seconds |

#### `char_errors[]` Object

| Field | Type | Description |
|:------|:-----|:------------|
| `type` | String | `"استبدال_حرف"` / `"حذف_حرف"` / `"زيادة_حرف"` / `"تغيير_حركة"` |
| `type_en` | String | `"char_substitution"` / `"char_deletion"` / `"char_insertion"` / `"diacritic_change"` |
| `position` | Integer | Position in the word |
| `got` | String | What was spoken (for substitution/insertion) |
| `expected` | String | What was expected (for substitution/deletion/diacritic) |
| `char` | String | (diacritic_change only) The base letter affected |

---

### `POST /report_issue` — Report Grading Error

Use this to flag incorrect grades. Audio is saved for future model retraining.

**Request (JSON Body):**
```json
{
  "request_id": "5dd616b3-...",
  "user_comment": "The grading was wrong"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Reported."
}
```

---

### `GET /health` — Health Check

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "2.0.0"
}
```

---

## 💻 Integration Examples

### Python

```python
import requests

response = requests.post(
    "http://YOUR_SERVER:8000/grade_recitation",
    files={"file": open("recording.ogg", "rb")},
    data={
        "target_ayah": "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ",
        "surah_num": 1,
        "ayah_num": 1
    }
)

result = response.json()
print(f"Passed: {result['passed']}")
print(f"Score: {int(result['accuracy'] * 100)}%")
for word in result['words']:
    status = "✅" if word['is_correct'] else "❌"
    print(f"  {status} {word['word']} (score: {word['score']})")
```

### cURL

```bash
curl -X POST http://localhost:8000/grade_recitation \
  -F "file=@recording.ogg" \
  -F "target_ayah=بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ" \
  -F "surah_num=1" \
  -F "ayah_num=1"
```

### JavaScript (Fetch)

```javascript
const formData = new FormData();
formData.append('file', audioBlob, 'recording.ogg');
formData.append('target_ayah', 'بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ');
formData.append('surah_num', '1');
formData.append('ayah_num', '1');

const response = await fetch('http://YOUR_SERVER:8000/grade_recitation', {
    method: 'POST',
    body: formData
});
const result = await response.json();

// Display results
console.log(`Score: ${(result.accuracy * 100).toFixed(0)}%`);
result.words.forEach(w => {
    console.log(`${w.is_correct ? '✅' : '❌'} ${w.word} → ${w.expected}`);
});
```

### Dart / Flutter

```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

var request = http.MultipartRequest('POST', 
    Uri.parse('http://YOUR_SERVER:8000/grade_recitation'));
request.files.add(await http.MultipartFile.fromPath('file', audioPath));
request.fields['target_ayah'] = 'بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ';
request.fields['surah_num'] = '1';
request.fields['ayah_num'] = '1';

var response = await request.send();
var body = await response.stream.bytesToString();
var result = jsonDecode(body);

print('Score: ${(result['accuracy'] * 100).toInt()}%');
```

---

## 🧠 How It Works

### Pipeline

```
Audio File → VAD Segmenter → Whisper ASR → Tolerant Grader → TTS Feedback
                                 ↓                ↓
                          Transcription     Per-Word Analysis
                          + Timestamps      + Char-Level Errors
```

### Technology

| Component | Technology | Details |
|:----------|:-----------|:--------|
| ASR Model | OpenAI Whisper | Base: `tarteel-ai/whisper-base-ar-quran` + LoRA fine-tuning on child speech |
| Grading | Needleman-Wunsch DP | Fuzzy alignment with CER threshold 0.45 |
| Char Analysis | Needleman-Wunsch DP | Character-level alignment + diacritics comparison |
| TTS | edge-tts | Arabic voice `ar-EG-SalmaNeural` (Egyptian, Female) |
| Audio | librosa + VAD | 16kHz, silence splitting, segment merging |
| API | FastAPI | Auto-docs at `/docs`, async TTS |

### Model Details

- **Base Model**: `tarteel-ai/whisper-base-ar-quran` (~300MB, auto-downloaded from HuggingFace on first run)
- **Fine-Tuning**: LoRA adapter (r=32, alpha=64, targets: q_proj + v_proj)
- **Checkpoint**: `checkpoint-30000` (~4.7MB adapter, included in this folder)
- **Inference**: Beam search (5 beams) with timestamps, FP16 on CUDA

### Grading Rules

- **Pass threshold**: accuracy > 85%
- **Fuzzy matching**: Words with CER < 0.45 are accepted (handles child pronunciation)
- **Smart filter**: Extra words are forgiven when passed (kids often continue reciting)
- **Reference audio**: Only sent when child fails (Sheikh Minshawi from everyayah.com)

---

## ⚠️ Notes

1. **First Run**: Downloads base model (~300MB) from HuggingFace. Needs internet.
2. **GPU**: Strongly recommended. CPU works but inference is ~10x slower.
3. **Port**: Default is 8000. Change in `main.py` → `uvicorn.run(..., port=XXXX)`.
4. **CORS**: If calling from a web browser, add CORS middleware:
   ```python
   from fastapi.middleware.cors import CORSMiddleware
   app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
   ```
5. **Production**: Use `gunicorn` with uvicorn workers:
   ```bash
   gunicorn main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```
