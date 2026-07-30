"""Harakat (tajweed) grading — enabled by NAMAA's acoustic diacritics.

For each CORRECTLY-recited word (same letters as the canonical), compare the learner's
actual harakat (from NAMAA) to the canonical harakat and flag differences (e.g. said
fatha where a damma is required). Words with letter/word errors are left to the
word-level grader. Canonical (Uthmani) is normalized to standard harakat to match
NAMAA's output.
"""
import re
import difflib

# standard harakat we compare on
HARAKAT = set("ًٌٍَُِّْ")  # tanwin, fatha, damma, kasra, shadda, sukun
NAMES = {"َ": "fatha", "ُ": "damma", "ِ": "kasra", "ْ": "sukun",
         "ّ": "shadda", "ً": "tanwin-fath", "ٌ": "tanwin-damm", "ٍ": "tanwin-kasr", "": "none"}
_STRIP = re.compile("[ٰۖ-ۜ۟-ۤۥۦ۪ۧۨ-ۭ]")
_DIAC = re.compile("[ً-ْٰ]")


def simplify(s):
    """Uthmani -> standard harakat (so canonical is comparable to NAMAA output)."""
    s = s.replace("ٱ", "ا").replace("ۡ", "ْ")  # wasla->alef, small-sukun->sukun
    return _STRIP.sub("", s)


def _bare(w):
    return re.sub("[أإآٱ]", "ا", _DIAC.sub("", w)).replace("ة", "ه").replace("ى", "ي").strip()


def _units(word):
    """List of (base_letter, harakat-string) for a diacritized word."""
    out = []
    for ch in word:
        if ch in HARAKAT:
            if out: out[-1][1].append(ch)
        elif ch.isalpha() or ch in "ءؤئآأإٱى":
            out.append([ch, []])
    return [(b, "".join(sorted(d))) for b, d in out]


def _hk_name(diacs):
    return "+".join(NAMES.get(d, d) for d in diacs) if diacs else "none"


def grade_harakat(spoken_diac, canonical_diac):
    """Returns {"words":[...], "harakat_errors":[...], "checked":int, "wrong":int}.
    Only words whose letters match the canonical are harakat-checked."""
    sp = spoken_diac.split()
    cn = [simplify(w) for w in canonical_diac.split()]
    sm = difflib.SequenceMatcher(None, [_bare(w) for w in sp], [_bare(w) for w in cn], autojunk=False)
    words, errors, checked = [], [], 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            continue  # letter/word errors -> handled by the word grader
        for k in range(i2 - i1):
            sw, cw = sp[i1 + k], cn[j1 + k]
            su, cu = _units(sw), _units(cw)
            if len(su) != len(cu):
                continue  # letter mismatch after all -> skip
            checked += 1
            word_errs = []
            last = len(cu) - 1
            for idx, ((sb, sh), (cb, ch)) in enumerate(zip(su, cu)):
                # WAQF: the word-final vowel becomes sukun when the reciter pauses — this is
                # correct tajweed, not an error, so never flag the last letter.
                if idx == last:
                    continue
                # Compare only the SHORT-VOWEL content (fatha/damma/kasra/tanwin). Ignore
                # shadda (idghaam/gemination) and treat sukun == unmarked, because Uthmani
                # leaves sukun implicit — so only genuine vowel swaps are flagged.
                def vowels(x):
                    return set(x) - {"ّ", "ْ"}
                sv, cv = vowels(sh), vowels(ch)
                if sv != cv:
                    word_errs.append({"letter": cb, "expected": _hk_name(ch), "got": _hk_name(sh)})
            status = "correct" if not word_errs else "harakat_error"
            words.append({"word": sw, "status": status, "errors": word_errs})
            if word_errs:
                errors.append({"word": sw, "expected_word": cw, "details": word_errs})
    return {"words": words, "harakat_errors": errors, "checked": checked, "wrong": len(errors)}
