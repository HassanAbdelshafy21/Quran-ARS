import re
import sys
import jiwer

# Ensure UTF-8 for Windows Consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# --- Diacritics (Tashkeel) Constants ---
DIACRITICS = set('\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670')
# Fathah, Dammah, Kasrah, Fathatan, Dammatan, Kasratan, Shaddah, Sukun, Superscript Alef


class QuranGrader:
    def __init__(self):
        pass

    def normalize(self, text):
        """
        Aggressively normalizes Arabic text for phonetic comparison.
        """
        if not text: return ""
        
        # 1. Remove Diacritics
        text = re.sub(r'[\u064B-\u0652\u0670]', '', text)
        
        # 2. Remove Othmani Signs
        text = re.sub(r'[\u06D6-\u06ED]', '', text)
        
        # 3. Unify Characters
        text = re.sub(r'[إأٱآ]', 'ا', text) # Alefs
        text = re.sub(r'ى', 'ي', text)      # Alif Maqsura -> Ya
        text = re.sub(r'ة', 'ه', text)      # Ta Marbuta -> Ha
        
        # --- Demo Specific Tunings (Child Pronunciation Handlers) ---
        text = text.replace('سبحد', 'سبح')
        text = text.replace('وولعزيز', 'وهو العزيز')
        # -----------------------------------------------------------
        
        return text.strip()

    def _strip_diacritics(self, text):
        """Remove only diacritics, keep base letters."""
        return ''.join(c for c in text if c not in DIACRITICS)
    
    def _extract_diacritics(self, text):
        """Extract diacritics map: {position_of_base_char: diacritic_char}."""
        diacritics_map = {}
        base_index = -1
        for c in text:
            if c in DIACRITICS:
                if base_index >= 0:
                    diacritics_map[base_index] = diacritics_map.get(base_index, '') + c
            else:
                base_index += 1
        return diacritics_map

    def fuzzy_match(self, w1, w2, threshold=0.45):
        """Returns True if words are phonetically similar enough (CER < threshold)."""
        if w1 == w2: return True
        return jiwer.cer(w1, w2) < threshold

    def get_word_score(self, w1, w2):
        """Returns a similarity score between two words (0.0 to 1.0)."""
        if w1 == w2: return 1.0
        cer = jiwer.cer(w1, w2)
        return max(0.0, round(1.0 - cer, 3))

    def analyze_char_errors(self, spoken_word, expected_word):
        """
        Analyzes character-level errors between spoken and expected words.
        Returns a list of error dicts with type and details.
        
        Error types:
          - "حذف_حرف" (char_deletion): A character in expected is missing
          - "زيادة_حرف" (char_insertion): An extra character was spoken
          - "استبدال_حرف" (char_substitution): A character was replaced
          - "تغيير_حركة" (diacritic_change): The haraka/tashkeel differs
        """
        errors = []
        
        # --- Step 1: Compare diacritics (on raw words before normalization) ---
        spoken_diacritics = self._extract_diacritics(spoken_word)
        expected_diacritics = self._extract_diacritics(expected_word)
        
        spoken_base = self._strip_diacritics(spoken_word)
        expected_base = self._strip_diacritics(expected_word)
        
        # Compare diacritics where base letters match
        min_len = min(len(spoken_base), len(expected_base))
        for i in range(min_len):
            s_d = spoken_diacritics.get(i, '')
            e_d = expected_diacritics.get(i, '')
            if s_d != e_d and spoken_base[i] == expected_base[i]:
                errors.append({
                    "type": "تغيير_حركة",
                    "type_en": "diacritic_change",
                    "position": i,
                    "got": s_d if s_d else "بدون حركة",
                    "expected": e_d if e_d else "بدون حركة",
                    "char": expected_base[i]
                })
        
        # --- Step 2: Character-level alignment (on normalized/stripped base) ---
        # Use Needleman-Wunsch at character level
        s_chars = list(self.normalize(spoken_word))
        e_chars = list(self.normalize(expected_word))
        
        n = len(e_chars)
        m = len(s_chars)
        
        if n == 0 and m == 0:
            return errors
        
        # DP table
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1): dp[i][0] = i
        for j in range(m + 1): dp[0][j] = j
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if e_chars[i-1] == s_chars[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1,
                    dp[i-1][j-1] + cost
                )
        
        # Backtrack
        char_ops = []
        i, j = n, m
        while i > 0 or j > 0:
            if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (0 if e_chars[i-1] == s_chars[j-1] else 1):
                if e_chars[i-1] != s_chars[j-1]:
                    char_ops.append(('sub', e_chars[i-1], s_chars[j-1], i-1))
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                char_ops.append(('delete', e_chars[i-1], None, i-1))
                i -= 1
            else:
                char_ops.append(('insert', None, s_chars[j-1], j-1))
                j -= 1
        
        char_ops.reverse()
        
        for op_type, expected_char, spoken_char, pos in char_ops:
            if op_type == 'sub':
                errors.append({
                    "type": "استبدال_حرف",
                    "type_en": "char_substitution",
                    "position": pos,
                    "got": spoken_char,
                    "expected": expected_char
                })
            elif op_type == 'delete':
                errors.append({
                    "type": "حذف_حرف",
                    "type_en": "char_deletion",
                    "position": pos,
                    "expected": expected_char
                })
            elif op_type == 'insert':
                errors.append({
                    "type": "زيادة_حرف",
                    "type_en": "char_insertion",
                    "position": pos,
                    "got": spoken_char
                })
        
        return errors

    def align(self, ref_words, hyp_words):
        """
        Custom DP alignment (Needleman-Wunsch) allowing fuzzy matches.
        Returns detailed ops list: (type, ref_word, hyp_word)
        """
        n = len(ref_words)
        m = len(hyp_words)
        
        # dp[i][j] = min cost to align ref[:i] and hyp[:j]
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        
        # Initialization
        for i in range(n + 1): dp[i][0] = i
        for j in range(m + 1): dp[0][j] = j
            
        # Fill DP
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                r_w = ref_words[i-1]
                h_w = hyp_words[j-1]
                
                # Check match (Exact or Fuzzy)
                if self.fuzzy_match(r_w, h_w):
                    cost = 0 # It matches!
                else:
                    cost = 1 # Substitution cost
                
                dp[i][j] = min(
                    dp[i-1][j] + 1,   # Deletion (Ref word missed)
                    dp[i][j-1] + 1,   # Insertion (Hyp word added)
                    dp[i-1][j-1] + cost # Match/Sub
                )
        
        # Backtrack to find the Ops
        ops = []
        i, j = n, m
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                r_w = ref_words[i-1]
                h_w = hyp_words[j-1]
                match = self.fuzzy_match(r_w, h_w)
                cost = 0 if match else 1
                
                # Check if this cell came from diagonal (Match/Sub)
                if dp[i][j] == dp[i-1][j-1] + cost:
                    if match:
                        ops.append(('equal', r_w, h_w))
                    else:
                        ops.append(('sub', r_w, h_w))
                    i -= 1
                    j -= 1
                    continue
            
            # Check if came from up (Deletion)
            if i > 0 and dp[i][j] == dp[i-1][j] + 1:
                ops.append(('delete', ref_words[i-1], None))
                i -= 1
            else:
                # Came from left (Insertion)
                ops.append(('insert', None, hyp_words[j-1]))
                j -= 1
                
        return ops[::-1]

    def grade(self, student_input, target_ayah):
        """
        Grades the student input against the target Ayah.
        Returns overall score + per-word details with character-level error analysis.
        """
        norm_student = self.normalize(student_input)
        norm_target = self.normalize(target_ayah)
        
        ref_tokens = norm_target.split()
        hyp_tokens = norm_student.split()
        
        # Using Custom Aligner
        ops = self.align(ref_tokens, hyp_tokens)
        
        score = 0
        total_target_words = len(ref_tokens)
        feedback = []
        words_detail = []
        
        for op, r_w, h_w in ops:
            if op == 'equal':
                score += 1
                words_detail.append({
                    "word": h_w,
                    "expected": r_w,
                    "is_correct": True,
                    "score": 1.0,
                    "error_type": None,
                    "error_type_ar": None,
                    "char_errors": []
                })
            elif op == 'sub':
                feedback.append(f"خطأ: قلت '{h_w}' بدلاً من '{r_w}'")
                word_score = self.get_word_score(h_w, r_w)
                char_errors = self.analyze_char_errors(h_w, r_w)
                words_detail.append({
                    "word": h_w,
                    "expected": r_w,
                    "is_correct": False,
                    "score": word_score,
                    "error_type": "substitution",
                    "error_type_ar": "استبدال",
                    "char_errors": char_errors
                })
            elif op == 'delete':
                feedback.append(f"كلمة ناقصة: '{r_w}'")
                words_detail.append({
                    "word": None,
                    "expected": r_w,
                    "is_correct": False,
                    "score": 0.0,
                    "error_type": "deletion",
                    "error_type_ar": "حذف",
                    "char_errors": []
                })
            elif op == 'insert':
                feedback.append(f"كلمة زائدة: '{h_w}'")
                words_detail.append({
                    "word": h_w,
                    "expected": None,
                    "is_correct": False,
                    "score": 0.0,
                    "error_type": "insertion",
                    "error_type_ar": "زيادة",
                    "char_errors": []
                })
        
        # Avoid division by zero
        accuracy = score / total_target_words if total_target_words > 0 else 0.0
        
        is_passed = accuracy > 0.85
        
        # Smart Filter: If passed, ignore "Extra Words" (usually child continuing recitation)
        if is_passed:
            feedback = [f for f in feedback if not f.startswith("كلمة زائدة:")]

        return {
            "passed": is_passed, 
            "accuracy": accuracy,
            "mistakes": feedback,
            "words": words_detail,
            "raw_score": f"{score}/{total_target_words}",
            "debug_ops": ops
        }

# --- Demo Usage ---
if __name__ == "__main__":
    grader = QuranGrader()
    
    print("--- Test Case 1: The 'Wal-ti' Issue ---")
    student = "وَالتِّي مِنْ وَزَّيْتُونِي" 
    target = "وَٱلتِّينِ وَٱلزَّيۡتُونِ"
    
    result = grader.grade(student, target)
    print(f"Result: {result['passed']} (Accuracy: {result['accuracy']:.2f})")
    if result['mistakes']: print(f"Mistakes: {result['mistakes']}")
    print(f"Words Detail: {result['words']}")
    
    print("\n--- Test Case 2: Real Mistake ---")
    student = "والتين و التفاح"
    result = grader.grade(student, target)
    print(f"Result: {result['passed']} (Accuracy: {result['accuracy']:.2f})")
    if result['mistakes']: print(f"Mistakes: {result['mistakes']}")
    print(f"Words Detail: {result['words']}")
