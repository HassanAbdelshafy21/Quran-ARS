import re
import sys
import jiwer

# Ensure UTF-8 for Windows Consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

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
        
        # Extra: Unify some purely graphic variants if needed
        # text = text.replace('گ', 'ك') 
        
        return text.strip()

    def fuzzy_match(self, w1, w2, threshold=0.45):
        """Returns True if words are phonetically similar enough (CER < threshold)."""
        if w1 == w2: return True
        return jiwer.cer(w1, w2) < threshold

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
        
        for op, r_w, h_w in ops:
            if op == 'equal':
                score += 1
            elif op == 'sub':
                feedback.append(f"Mistake: Said '{h_w}' instead of '{r_w}'")
            elif op == 'delete':
                feedback.append(f"Missed Word: '{r_w}'")
            elif op == 'insert':
                # Optional: Penalize insertions? 
                # For kids, we often forgive "stuttering" or "starting with Bismillah"
                # But strict grading might flag it.
                feedback.append(f"Added Word: '{h_w}'")
        
        # Avoid division by zero
        accuracy = score / total_target_words if total_target_words > 0 else 0.0
        
        return {
            "passed": accuracy > 0.85, 
            "accuracy": accuracy,
            "mistakes": feedback,
            "raw_score": f"{score}/{total_target_words}",
            "debug_ops": ops
        }

# --- Demo Usage ---
if __name__ == "__main__":
    grader = QuranGrader()
    
    print("--- Test Case 1: The 'Wal-ti' Issue ---")
    student = "وَالتِّي مِنْ وَزَّيْتُونِي" 
    target = "وَٱلتِّينِ وَٱلزَّيۡتُونِ"
    
    result = grader.grade(student, target)
    print(f"Result: {result['passed']} (Accuracy: {result['accuracy']:.2f})")
    if result['mistakes']: print(f"Mistakes: {result['mistakes']}")
    
    print("\n--- Test Case 2: Real Mistake ---")
    student = "والتين و التفاح"
    result = grader.grade(student, target)
    print(f"Result: {result['passed']} (Accuracy: {result['accuracy']:.2f})")
    if result['mistakes']: print(f"Mistakes: {result['mistakes']}")
