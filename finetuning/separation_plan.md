
# Separation Strategy Analysis

## Goal

Extract pure child audio from "Teacher-Child" repetition files.
Target Data:

- `Al_Husayni_Al_Azazi_Children_mp3s`
- `downloaded_mp3s` (Minshawi Muallim)

## Hypothesis: "The Echo Pattern"

These recitations typically follow a strict pattern:

1. Teacher recites Ayah N.
2. Short Silence.
3. Child(ren) repeat Ayah N.
4. Short Silence.
5. Teacher recites Ayah N+1.

## Proposed Algorithm

1. **Segment**: Split audio by silence.
2. **Transcribe**: Run Whisper on *every* chunk.
3. **Align**:
    - Match chunks to Ayah text.
    - If we see **Sequence A** followed by **Sequence A** (with high similarity), we identify pair (Teacher, Child).
    - Heuristics:
        - `Chunk[i] == Ayah N` AND `Chunk[i+1] == Ayah N` -> `i=Teacher`, `i+1=Child`.
        - Verify pitch? (Optional, maybe overkill if pattern is robust).
        - Verify duration? Child often slower or same speed.

## Edge Cases

- **Merged Ayahs**: Teacher reads 1+2. Child reads 1+2. (Script handles multi-ayah overlap).
- **Missed Split**: Teacher + Child in one chunk. (Detected by duration? Or transcript length vs Ayah length?)
- **Non-Repetition**: Some parts might strictly be teacher (e.g. intro).
- **Azazi Specifics**: Does he have distinct intro/outro music?

## Action Plan

1. Create `debug_separation.py`.
2. Pick `Al_Husayni_Al_Azazi_Children_mp3s/001.mp3` and `112.mp3` (short).
3. Run segmentation and transcription.
4. Print the sequence of detected texts.
5. Manually verify if the "Duplicate" pattern holds.
