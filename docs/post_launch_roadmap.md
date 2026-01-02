# 🔮 Phase 6: Post-Launch & Iteration Roadmap

**Objective:** Transform the V5 "Prototype" into a V6 "World Class Product" using real-world data.

## 1. The "Data Flywheel" Strategy (Crucial)

The hardest part of this project was finding child data. Now, you have 50+ users creating it for you.
**Action Plan:**

1. **Log Everything:** (With user consent) Save every audio recording and the corresponding "Grade" (Pass/Fail) to your server.
2. **The "Report Problem" Button:** Add a button in the UI: *"I said it right!"*.
    * If a user clicks this, flag that audio file as a **Golden Negative** (Model failed, Human succeeded).
    * These are the *most valuable* training samples you will ever get.

## 2. Dataset V6 (The "Real World" Set)

After 1 month of usage:

* Collect the top 1,000 "Reported" audios.
* Manually transcribe them (or have a teacher verify them).
* **Retrain:** Fine-tune V5 for just 1-2 epochs on this new "Hard Mode" dataset.
* **Result:** V6 will stop making the specific mistakes that annoy your actual users.

## 3. Advanced Features (The "Nice to Haves")

### A. Teacher-Child Separation (Diarization)

* **Problem:** Currently, if a teacher speaks over the kid, the model gets confused.
* **Solution:** Train a small classifier (or use Pyannote Audio) to detect "Adult Voice" vs "Child Voice" and **mute the Adult** before sending to Whisper.
* *Status:* We skipped this for V5 to meet the deadline, but it's the next logical tech upgrade.

### B. Edge Deployment (Cheaper Hosting)

* **Problem:** Running GPUs is expensive ($$$).
* **Solution:** Convert V5 to **ONNX Int8** or **CoreML** (for iOS).
* **Benefit:** Run the model *directly on the user's phone*. zero server cost, zero latency, works offline.

## 4. Timeline Recommendation

| Month | Activity | Goal |
|---|---|---|
| **Month 1** | Launch V5 (Beta). | Collect 1,000+ real child recordings. |
| **Month 2** | "Data Cleaning Sprint". | Filter the data. Identify common "False Fails". |
| **Month 3** | Train **V6**. | Fix the top 10 complaints. |
| **Month 4** | Research "On-Device" (CoreML). | Move away from cloud servers to save money. |

---

**Summary:**
The V5 Model is your "Base". The **User Data** is your "Fuel". The app isn't just a product; it's a data-gathering machine for V6.
