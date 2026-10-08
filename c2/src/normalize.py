"""
Rule-based normalization of the code-mixed transcript before translation.

    meeting-eka   -> meeting eka     (hyphenated Sinhala clitic split off)
    bankඑකට       -> bank එකට        (Latin stem glued to Sinhala suffix)
    uh, umm, hmm  -> removed         (fillers)

Repeated words are NOT collapsed: Sinhala reduplication is meaningful
("podi podi"), and a rule cannot tell it from a stutter.
"""
import re

FILLERS = {"uh", "uhh", "um", "umm", "hmm", "hm", "mm", "er", "erm", "ah", "aah", "eh", "ehh", "අ"}

CLITICS = r"(eka|ekak|eke|ekata|eken|ekka|ekath|wala|walata|walin|wage)"

_HYPHEN_CLITIC = re.compile(rf"\b([A-Za-z]+)-{CLITICS}\b", re.IGNORECASE)
_LATIN_SINHALA = re.compile(r"([A-Za-z])([඀-෿])")
_SINHALA_LATIN = re.compile(r"([඀-෿‍])([A-Za-z])")


def split_mixed_words(text):
    text = _HYPHEN_CLITIC.sub(r"\1 \2", text)
    text = _LATIN_SINHALA.sub(r"\1 \2", text)
    return _SINHALA_LATIN.sub(r"\1 \2", text)


def remove_fillers(text):
    kept = [w for w in text.split() if w.strip(",.?!").lower() not in FILLERS]
    return " ".join(kept)


def tidy(text):
    text = re.sub(r"\s+([,.?!])", r"\1", text)
    text = re.sub(r"([,.?!]){2,}", r"\1", text)
    return re.sub(r"\s+", " ", text).strip(" ,")


def normalize(text):
    return tidy(remove_fillers(split_mixed_words(text)))


if __name__ == "__main__":
    import sys
    print(normalize(" ".join(sys.argv[1:]) or "uh meeting-eka Friday ekata reschedule karamu"))
