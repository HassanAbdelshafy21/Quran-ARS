import jiwer
import re
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 1. The Real Data (From our V5-30k Benchmark)
# Kid Recitation (Surah At-Tin)
TRANSCRIPTION_RAW = "وَالتِّي مِنْ وَزَّيْتُونِي وَتُورِسِي مِينِينَ"
# The Real Surah (Ground Truth)
GROUND_TRUTH = "وَٱلتِّينِ وَٱلزَّيۡتُونِ وَطُورِ سِينِينَ"

def normalize_phonetic(text):
    # Aggressive Normalization for Grading (Ignoring Spelling Rules)
    text = re.sub(r'[إأٱآ]', 'ا', text)   # All Alefs -> Bare Alif
    text = re.sub(r'ة', 'ه', text)        # Ta Marbuta -> Ha
    text = re.sub(r'ى', 'ي', text)        # Alif Maqsura -> Ya
    # Remove ALL diacritics (Fatha, Kasra, etc)
    text = re.sub(r'[\u064B-\u0652\u0670]', '', text) 
    # Remove Othmani symbols
    text = re.sub(r'[\u06D6-\u06ED]', '', text)
    # Simplify common phonetic confusions
    # Zay (ز) vs Dhal (ذ) vs Za (ظ) - Kids mix these
    # Taa (ط) vs Ta (ت)
    # Saad (ص) vs Seen (س)
    # text = text.replace('ط', 'ت').replace('ص', 'س').replace('ظ', 'ز').replace('ذ', 'ز')
    return text.strip()

def grade_recitation(student_text, reference_text):
    print(f"\n--- Grading Attempt ---")
    print(f"Student 'Heard': {student_text}")
    print(f"Quran Actual:    {reference_text}")
    
    # 1. Normalize
    stud_norm = normalize_phonetic(student_text)
    ref_norm = normalize_phonetic(reference_text)
    
    print(f"\n[Normalized Comparison]")
    print(f"Student: {stud_norm}")
    print(f"Quran:   {ref_norm}")
    
    # 2. Align & Score
    output = jiwer.process_words(ref_norm, stud_norm)
    
    print(f"\n[Analysis]")
    # We Iterate over the alignment
    # Miss = User missed a word
    # Sub = User said wrong word (or spelled badly)
    # Hit = Correct
    
    alignment = output.alignments[0]
    
    score = 0
    total_words = len(alignment)
    
    for op in alignment:
        # op is a Chunk (hit, sub, del, ins)
        if op.type == 'equal':
            print(f"✅ {ref_norm[op.ref_start_idx:op.ref_end_idx]} matches {stud_norm[op.hyp_start_idx:op.hyp_end_idx]}")
            score += 1
        elif op.type == 'substitute':
            # Check Similarity (Is it just a spelling error?)
            ref_w = " ".join(ref_norm[op.ref_start_idx:op.ref_end_idx])
            hyp_w = " ".join(stud_norm[op.hyp_start_idx:op.hyp_end_idx])
            
            # Simple Character Overlap
            match_ratio = jiwer.cer(ref_w, hyp_w)
            
            if match_ratio < 0.6: # Relaxed Error Rate for Child Speech
                print(f"⚠️ Tolerated {hyp_w} (Expected: {ref_w}) - Close Enough")
                score += 1
            else:
                print(f"❌ MISTAKE: Said '{hyp_w}' instead of '{ref_w}'")
                
        elif op.type == 'delete':
            ref_w = " ".join(ref_norm[op.ref_start_idx:op.ref_end_idx])
            print(f"❌ MISSED: {ref_w}")
        elif op.type == 'insert':
            hyp_w = " ".join(stud_norm[op.hyp_start_idx:op.hyp_end_idx])
            print(f"❌ ADDED: {hyp_w}")

    print(f"\nFinal Score: {score}/{len(reference_text.split())} words correct")

# Run Demo
grade_recitation(TRANSCRIPTION_RAW, GROUND_TRUTH)
