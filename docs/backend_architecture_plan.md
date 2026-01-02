# 🏗️ Backend Architecture: Handling Long Recitations & Feedback

**Objective:** Process 15-minute+ continuous recitations without "clipping" words, and provide specific, actionable feedback (with audio).

## 1. The Core Challenge: " The Bad Cut"

If a child says: "Bismi-llahi-rrahma..." and we cut the audio there:

* **Model hears:** "Bismi-llahi-rrahma" (incomplete).
* **Grader says:** "Wrong! You missed 'Al-Raheem'".
* **Reality:** The child was right, but our code cut them off.

## 2. Solution: Intelligent Segmentation Pipeline

Instead of arbitrary 30-second cuts, we use **Silence-Based Segmentation (VAD)**.

### Step 1: Voice Activity Detection (VAD)

We scan the 15-minute file for "Silence Gaps" (> 300ms).

* **Logic:** We *only* cut when the child takes a breath.
* **Tool:** `webrtcvad` or `silero-vad` (Fast & Accurate).
* **Safety Margin:** Add 100ms padding before/after each cut to capture the start/end of words.

### Step 2: The "Sliding Window" Transcription

Even with VAD, a child might recite 3 Ayahs in one breath.

* **Action:** We send the "Breath Chunk" to the Model.
* **Output:** The Model returns the text for those 3 Ayahs combined.
* **Alignment:** We align this long text against the expected Surah to find out which Ayahs were covered.

## 3. The Feedback Loop Logic

Once we have the transcription, we run the `QuranGrader`.

### Scenario: Child makes a mistake in Ayah 4

1. **Detection:** Grader flags: `Mistake: Said "Maliki" instead of "Maaliki"`.
2. **Response Payload:**

    ```json
    {
      "ayah_number": 4,
      "status": "wrong_pronunciation",
      "user_said": "Maliki yawm...",
      "correction_text": "Maaliki yawm...",
      "reference_audio_url": "https://api.quran.com/.../ayah/1/4/audio.mp3" 
    }
    ```

3. **App Behavior:**
    * Highlight the word in **Red**.
    * Show the correct word in **Green**.
    * **Play Button:** Plays the "Reference Audio" (e.g., Al-Afasy or Minshawi) for *just that Ayah* so the child hears the right rhythm.

## 4. Proposed Tech Stack

* **Server:** FastAPI (Python).
* **VAD:** `silero-vad` (Robust to noise).
* **Model:** V5-30k (Loaded via HuggingFace `pipeline`).
* **Alignment:** `quran_grader` (Our custom module).
* **Audio Source:** Use `EveryAyah.com` API for fetching the "Correct Audio" to send back to the user.

## 5. Development Steps

1. **Prototype VAD:** Create `test_segmentation.py` to chop a 15-min file without breaking words.
2. **Aligner Update:** Update `quran_grader` to handle "Multi-Ayah" inputs.
3. **API Endpoint:** Build `/submit_recitation` endpoint.
