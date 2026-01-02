Got you — let’s fix this properly.

Here is the **entire plan in ONE single Markdown file**, no breaks, no cuts.
Just copy everything between the `markdown and` at the end into a `.md` file.

````markdown
# 📘 Quran Recitation AI System — Full Technical Plan (Engineering Document)
*Version 1.0 – Prepared for Development*

---

# 1. Introduction

This project aims to build an **AI-powered Quran Recitation Evaluation System** capable of:

- Processing **student audio**
- Generating **accurate Quranic transcription with tashkeel**
- Comparing the transcription with the **Quran database**
- Detecting:
  - Word errors  
  - Harakat (tashkeel) mistakes  
  - Pronunciation errors  
  - Simple tajweed issues  
- Generating **instant feedback** (text + TTS)
- Producing **daily performance reports** for parents

This document defines the **full engineering plan** from A→Z.

---

# 2. Core AI Components

## 2.1 ASR Engine (Primary Model)

We will use:

- **Base model:** `tarteel-ai/whisper-base-ar-quran`  
- **LoRA adapter:** `KheemP/whisper-base-quran-lora`

Benefits:

- Quran-specific ASR  
- Outputs **text with tashkeel**  
- Low WER (~6%)  
- Lightweight Whisper-base → fast inference  
- Perfect for MVP and later expansion

---

## 2.2 Comparison Engine

Responsible for:

- Tokenizing student text vs reference ayah  
- Word-level error detection:
  - Substitution  
  - Deletion  
  - Insertion  
- Harakat (tashkeel) mismatch detection  
- Optional: timestamps per word

---

## 2.3 Scoring Engine

Calculates:

- WER  
- Word accuracy  
- Harakat accuracy  
- Simple tajweed score  
- Overall score (0–100)

---

## 2.4 Feedback Engine

Generates:

- Arabic & English summaries  
- Detailed tips per mistake  
- Encouraging child-friendly messages  
- Optional TTS audio feedback

---

## 2.5 Daily Report Engine

Generates a daily report including:

- Attempts & scores  
- Common mistakes  
- Difficult letters  
- Weekly trend  
- Parent-friendly PDF/HTML report  

---

# 3. System Architecture

```text
Audio Input
    ↓
Audio Preprocessing (16kHz, mono)
    ↓
Quran ASR Model (Whisper LoRA)
    ↓
Transcribed Text with Tashkeel
    ↓
Quran DB Lookup (Correct Ayah)
    ↓
Text Alignment Engine
    ↓
Error Detection (words + harakat)
    ↓
Scoring Engine
    ↓
Feedback Engine (text + audio)
    ↓
JSON Response to Backend
    ↓
Attempt Logging (DB)
    ↓
Daily Report Generator → Parent Report
````

---

# 4. Development Plan (A → Z)

## Phase 1 — Project Setup (Day 1–2)

* Create project structure
* Create environment + `requirements.txt`
* Prepare folders:

```text
src/
  asr/
  evaluation/
  feedback/
  quran_db/
  utils/
  reports/
  api/
data/
tests/
```

---

## Phase 2 — Quran Database Module (Day 2–3)

* Load Quran JSON with tashkeel
* Implement:

  ```python
  get_ayah_text(ayah_key)
  get_ayah_words(ayah_key)
  ```

---

## Phase 3 — Audio Preprocessing (Day 3)

* Implement:

  ```python
  preprocess_audio(input_path) -> processed.wav
  ```

* Convert audio to 16kHz mono WAV.

---

## Phase 4 — ASR Module (Whisper LoRA) (Day 3–5)

Implement:

```python
load_model()
transcribe_audio(model, audio_path)
```

Outputs:

* Text with tashkeel
* Optional word timestamps

---

## Phase 5 — Text Alignment Engine (Day 5–7)

Implement:

```python
align_text(student_text, correct_text)
```

Detect:

* Wrong words
* Missing words
* Extra words
* Harakat mismatches

Output (concept):

* List of tokens
* Error types per word (correct / substitution / deletion / insertion)
* Harakat mismatch flags

---

## Phase 6 — Scoring Engine (Day 7–8)

Implement:

```python
compute_scores(alignment)
```

Calculates:

* WER
* Word accuracy
* Harakat accuracy
* Simple tajweed score (Phase 1)
* Overall score (0–100)

---

## Phase 7 — Feedback Engine (Day 8–9)

Implement:

```python
generate_feedback(alignment, scores, language="ar")
```

Includes:

* Short summary (Arabic + English)
* Specific tips (words / letters to practice)
* Motivational, child-friendly comments
* Optional TTS text payloads (for an external TTS service)

---

## Phase 8 — Attempt Logging (Day 9)

Each recitation attempt is stored in the backend DB, including:

* `student_id`
* `ayah_key`
* `student_text`
* `correct_text`
* `scores` (JSON)
* `errors` (JSON)
* `attempt_id`
* `created_at`

This is used later by the Daily Report Engine.

---

## Phase 9 — FastAPI Endpoint (Day 10–11)

**Endpoint:**

`POST /api/v1/evaluate_recitation`

**Flow:**

1. Receive audio + metadata (ayah_key, user_id, attempt_id, language_ui).
2. Save & preprocess audio.
3. Run ASR (Whisper LoRA) → `student_text` (+ optional word timings).
4. Fetch reference ayah from Quran DB → `correct_text`.
5. Run text alignment → word-level + harakat errors.
6. Compute scores (WER, accuracy, harakat, simple tajweed).
7. Generate feedback (text + optional TTS text).
8. Log attempt in DB.
9. Return JSON response to backend.

---

## Phase 10 — Daily Report Module (Day 12–14)

A daily cron job (e.g. 20:00 server time):

```python
build_daily_report(student_id, date)
```

For each student, it:

* Fetches all attempts from that day
* Aggregates:

  * Total ayahs
  * Total attempts
  * Time spent (if tracked)
  * Average accuracy, harakat, tajweed
  * Common words with errors
  * Common harakat mistakes
  * Difficult letters (e.g. ص ض ق ط…)
* Builds a **Daily Report JSON**
* Backend converts it to PDF/HTML and sends to parents (email / WhatsApp / dashboard)

---

## Phase 11 — Testing & Refinement (Day 14–16)

* Test with real Quran recitations (children + adults)
* Verify:

  * Tashkeel correctness
  * Word error detection
  * Harakat mismatch detection
  * Score ranges (not too harsh, not too easy)
* Tune thresholds & feedback messages
* Fix bugs and edge cases

---

# 5. Backend API Contract

## 5.1 Endpoint: `POST /api/v1/evaluate_recitation`

### Request (multipart/form-data)

| Field         | Type   | Description                          |
| ------------- | ------ | ------------------------------------ |
| `audio_file`  | file   | Student recitation audio (WAV/MP3)   |
| `ayah_key`    | string | e.g. `"1:1"`                         |
| `user_id`     | string | Student identifier                   |
| `attempt_id`  | string | Unique attempt ID (frontend/backend) |
| `language_ui` | string | `"ar"` or `"en"` (feedback language) |

---

### Response (JSON)

```json
{
  "ayah_key": "1:1",
  "student_text": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمَ",
  "correct_text": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
  "scores": {
    "overall": 86,
    "wer": 0.14,
    "harakat_score": 90,
    "tajweed_simple": 75
  },
  "word_details": [
    {
      "index": 1,
      "student": "بِسْمِ",
      "expected": "بِسْمِ",
      "start": 0.10,
      "end": 0.45,
      "correct": true,
      "errors": []
    },
    {
      "index": 4,
      "student": "الرَّحِيمَ",
      "expected": "الرَّحِيمِ",
      "start": 1.20,
      "end": 1.65,
      "correct": false,
      "errors": [
        {
          "type": "harakah_error",
          "description_ar": "يجب أن تكون كسرة في آخر الكلمة، وليست فتحة.",
          "description_en": "The ending should be kasrah, not fatha."
        }
      ]
    }
  ],
  "feedback": {
    "summary_ar": "تلاوة جيدة عمومًا، انتبه لحركة آخر كلمة في (الرحيم) ومد كلمة (الرحمن).",
    "summary_en": "Overall good recitation, pay attention to the final harakah in 'الرحيم' and the madd in 'الرحمن'.",
    "tips_ar": [
      "أعد نطق كلمة (الرَّحِيمِ) مع كسر الحرف الأخير.",
      "حاول إطالة المد في (الرَّحْمَٰنِ) بمقدار حركتين."
    ],
    "tips_en": [
      "Repeat the word 'الرَّحِيمِ' with a kasrah on the last letter.",
      "Try extending the madd in 'الرَّحْمَٰنِ' by about two counts."
    ]
  },
  "audio_feedback": {
    "tts_enabled": true,
    "url": "https://your-cdn.com/tts/attempt-001-feedback-ar.mp3"
  },
  "meta": {
    "processing_time_ms": 850,
    "attempt_id": "attempt-001",
    "timestamp": "2025-12-06T20:15:00Z"
  }
}
```

---

# 6. Daily Report System

## 6.1 Data Collected Daily

For each student per day, we collect:

* Total ayahs recited
* Total attempts
* Total recitation time (if tracked)
* Average accuracy score
* Average harakat score
* Average simple tajweed score
* Most-missed words
* Common harakat mistakes
* Difficult letters (e.g. ص، ض، ق، ط)
* Weekly accuracy trend (last 7 days)

---

## 6.2 Daily Report JSON Structure

```json
{
  "student_id": "STU_001",
  "student_name": "Ahmed Hassan",
  "date": "2025-12-06",
  "summary": {
    "total_ayahs": 12,
    "total_attempts": 18,
    "total_time_seconds": 780,
    "accuracy_score": 87,
    "harakat_score": 90,
    "tajweed_score": 76,
    "overall_rating": "Good",
    "improvement_from_yesterday": 8
  },
  "mistakes_analysis": {
    "most_missed_words": [
      "العالمين",
      "الصراط",
      "الرحيم"
    ],
    "common_harakat_errors": [
      "Dammah → Fatha",
      "Kasrah → Sukoon"
    ],
    "hard_letters": [
      "ص",
      "ض",
      "ق"
    ],
    "tajweed_issues": [
      "Short Madd in (الرحمن)",
      "Weak Ghunnah in (من)"
    ]
  },
  "progress": {
    "weekly_scores": [72, 75, 79, 81, 86, 87, 87]
  },
  "ai_feedback": [
    "Great improvement in accuracy today.",
    "Pay attention to Madd in سورة الفاتحة.",
    "Practice the letter (ص) for clearer pronunciation."
  ]
}
```

---

## 6.3 Example Parent Report (Human-Readable)

```text
🌟 Daily Quran Recitation Report – Ahmed Hassan
📅 Date: 6 December 2025

🎯 Performance Summary
- Ayahs recited: 12
- Attempts: 18
- Total time: 13 minutes
- Accuracy: 87%
- Harakat accuracy: 90%
- Tajweed score: 76%
- Rating: ⭐ Good
- Improvement since yesterday: +8%

❗ Common Mistakes
- Misread words: العالمين، الصراط، الرحيم
- Harakat issues: Kasrah → Sukoon, Dammah → Fatha
- Difficult letters: ص، ض، ق

📈 Weekly Progress
Scores: 72% → 75% → 79% → 81% → 86% → 87% → 87%

🧠 AI Feedback
- Excellent improvement in reading accuracy today.
- Focus on Madd in (الرحمن) and Ghunnah in (من).
- Practice pronouncing the letter (ص) clearly.

💚 Keep going! Your child is improving day by day.
```

---

# 7. Deployment Plan

## 7.1 Options & Approximate Costs (1 USD ≈ 47 EGP)

| Platform          | Cost/hr (USD) | Cost/hr (EGP) | Notes                                   |
| ----------------- | ------------- | ------------- | --------------------------------------- |
| RunPod Serverless | 0.40–2.17     | 19–102        | Best price, pay-per-second, scales to 0 |
| Modal             | 0.59–2.50     | 28–118        | Great Python developer workflow         |
| Hugging Face EP   | 0.50–2.50     | 23–118        | Very easy, costly if always running     |
| Replicate         | 0.81–5.04     | 38–237        | Good for demos, not main production     |

## 7.2 Recommended Strategy

* **MVP / Early Production:**

  * Use **RunPod Serverless** with a T4-class GPU.
  * Advantages:

    * Pay-per-second billing
    * Scales to zero (no cost when idle)
    * Good performance for Whisper-base.

* **Later / Higher Scale:**

  * Consider **Modal** for better dev workflow and more complex pipelines.
  * If the client wants enterprise-level infra, move to AWS/GCP with containers later.

---

# 8. Final Development Timeline

| Phase                 | Estimated Days |
| --------------------- | -------------- |
| Setup + Quran DB      | 1–3            |
| ASR Module            | 3–5            |
| Alignment + Scoring   | 5–8            |
| Feedback Engine       | 8–9            |
| API Endpoint          | 10–11          |
| Daily Reports Module  | 12–14          |
| Testing & Refinements | 14–16          |

Total estimated time for MVP: **~16 days**.

---

# 9. Next Step

Once this document is approved, the next action is:

> **Start coding Phase 1 (Project Setup + Quran DB Module).**

```

If you paste that into a `.md` file, you’ll have a single, clean document.

When you’re ready to move on, tell me:

**“Start coding now”** and we’ll begin implementing step by step.
```
