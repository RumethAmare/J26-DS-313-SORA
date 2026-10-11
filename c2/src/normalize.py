"""
Rule-based normalization of the code-mixed transcript before translation.

    meeting-eka   -> meeting eka     (hyphenated Sinhala clitic split off)
    bankඑකට       -> bank එකට        (Latin stem glued to Sinhala suffix)
    uh, umm, hmm  -> removed         (fillers)
    binduwai hatai -> 0 7            (spoken Sinhala digits, so numbers survive translation)
    onne          -> one             (spelling variants the model rarely saw)

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


# Spoken digits in a phone/NIC/account number. Only the "-ai" list forms are
# mapped: bare "eka"/"deka" also mean "the"/"two things" in ordinary speech.
SPOKEN_DIGITS = {
    "binduwai": "0", "binduwayi": "0", "ekai": "1", "ekayi": "1", "dekai": "2", "dekayi": "2",
    "thunai": "3", "thunayi": "3", "tunai": "3", "hatharai": "4", "hatarai": "4", "hathara": "4",
    "pahai": "5", "pahayi": "5", "hayai": "6", "hayayi": "6", "hathai": "7", "hatai": "7",
    "hathayi": "7", "atai": "8", "atayi": "8", "namayai": "9", "nawayai": "9", "namayayi": "9",
}

# Romanized spellings of one word, mapped to the form most common in training.
SPELLING = {"onne": "one", "onee": "one", "oney": "one", "onneda": "oneda", "mokadda": "mokada"}


def _swap_words(text, table):
    def swap(m):
        w = m.group(0)
        return table.get(w.lower(), w)
    return re.sub(r"[A-Za-z]+", swap, text)


def spoken_numbers(text):
    return _swap_words(text, SPOKEN_DIGITS)


def fix_spelling(text):
    return _swap_words(text, SPELLING)


def prepare(text):
    """Safe input fixes, usable even with a model trained without normalize()."""
    return fix_spelling(spoken_numbers(text))


def normalize(text):
    return tidy(remove_fillers(split_mixed_words(prepare(text))))


# NLLB's tokenizer drops the zero-width joiner, so a conjunct like ට්‍ර comes
# out split: "ට් රාන්ස්පෝට්", "අවශ් යතාවය", "වෛද් ය". Two safe re-joins:
#   1. a lone consonant+virama ("ට්") never stands as a word, so always join;
#   2. otherwise join only if some gold Sinhala word starts with the joined
#      conjunct ("අවශ්‍ය..."), so real word breaks ("මෝටර් රථ",
#      "නිවේදනයක් යවමුද") are left alone.
_BROKEN_CONJUNCT = re.compile(r"(^|\s)([\u0d9a-\u0dc6]\u0dca) ([\u0dbb\u0dba])")
_SPLIT_PAIR = re.compile(r"(\S*\u0dca) ([\u0dbb\u0dba]\S*)")
_PUNCT = ".,?!:;\"'()"
_SINHALA_VOCAB = None


def sinhala_vocab():
    """Conjunct word-starts in the gold clean_si text (data/train.jsonl), loaded
    once: "අවශ්‍යතාවය" contributes "අවශ්‍ය"."""
    global _SINHALA_VOCAB
    if _SINHALA_VOCAB is None:
        import json
        from pathlib import Path
        _SINHALA_VOCAB = set()
        path = Path(__file__).resolve().parents[1] / "data" / "train.jsonl"
        if path.exists():
            for line in open(path, encoding="utf-8"):
                for u in json.loads(line)["utterances"]:
                    for w in (u["clean_si"] or "").split():
                        w = w.strip(_PUNCT)
                        for m in re.finditer("්‍", w):
                            _SINHALA_VOCAB.add(w[:m.end() + 1])
    return _SINHALA_VOCAB


def fix_sinhala(text, vocab=None):
    text = _BROKEN_CONJUNCT.sub(r"\1\2" + "\u200d" + r"\3", text)
    vocab = sinhala_vocab() if vocab is None else vocab

    def join(m):
        left = m.group(1).lstrip(_PUNCT)
        known = (left + "\u200d" + m.group(2)[0]) in vocab
        return m.group(1) + "\u200d" + m.group(2) if known else m.group(0)
    return _SPLIT_PAIR.sub(join, text)


# Digits must survive translation exactly: C4 redacts NICs and phone numbers by
# their exact characters. The model sometimes adds, drops or reorders digits
# ("1995 123 089 34" -> "1995123809346"), so copy each number back from the
# source when the digits differ. Only done when source and output have the same
# number of numbers, so the pairing is unambiguous.
_NUMBER = re.compile(r"\d+(?:[ ,.:/-]+\d+)*")


def fix_numbers(src, hyp):
    src_nums, hyp_nums = _NUMBER.findall(src), list(_NUMBER.finditer(hyp))
    if not src_nums or len(src_nums) != len(hyp_nums):
        return hyp
    out, last = [], 0
    for s, h in zip(src_nums, hyp_nums):
        digits_s, digits_h = re.sub(r"\D", "", s), re.sub(r"\D", "", h.group())
        fixed = h.group()
        if digits_s != digits_h:
            # keep the output's style: one block stays one block
            fixed = digits_s if not re.search(r"\D", h.group()) else s.strip(" ,.")
        out += [hyp[last:h.start()], fixed]
        last = h.end()
    return "".join(out) + hyp[last:]


def postprocess(src, hyp):
    """Model output -> final text: exact numbers, re-joined Sinhala letters."""
    return fix_sinhala(fix_numbers(src, hyp))


# --- Sinhala script -> romanized Singlish (for synthetic training data) ------
_CONS = dict(zip("කඛගඝඞඟචඡජඣඤඥටඨඩඪණඬතථදධනඳපඵබභමඹයරලවශෂසහළෆ",
                 "k k g g ng ng ch ch j j gn gn t t d d n nd th th d d n nd p p b b m mb "
                 "y r l w sh sh s h l f".split()))
_SIGNS = dict(zip("ාැෑිීුූෘෙේෛොෝෞ", "a a a i i u u ru e e ai o o au".split()))
_INDEP = dict(zip("අආඇඈඉඊඋඌඍඑඒඓඔඕඖ", "a a a a i i u u ru e e ai o o au".split()))
# How people actually vary the spelling, picked at random per word.
_VARIANTS = {"ී": ["i", "ee"], "ූ": ["u", "oo"], "ැ": ["a", "e"], "ව": ["w", "v"],
             "ත": ["th", "t"], "ං": ["n", "ng"]}


def romanize(text, rng=None):
    """'මචං කොහොමද' -> 'machan kohomada'. Latin text passes through."""
    def pick(ch, default):
        return rng.choice(_VARIANTS[ch]) if rng and ch in _VARIANTS else default

    out, i = [], 0
    while i < len(text):
        ch = text[i]
        if ch in _CONS:
            out.append(pick(ch, _CONS[ch]))
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt in _SIGNS:
                out.append(pick(nxt, _SIGNS[nxt]))
                i += 1
            elif nxt == "\u0dca":   # virama: no vowel
                i += 1
            else:
                out.append("a")      # inherent vowel
        elif ch in _INDEP:
            out.append(_INDEP[ch])
        elif ch == "ං":
            out.append(pick(ch, "n"))
        elif ch == "ඃ":
            out.append("h")
        elif ch not in "\u200c\u200d":
            out.append(ch)
        i += 1
    return "".join(out)


if __name__ == "__main__":
    import sys
    print(normalize(" ".join(sys.argv[1:]) or "uh meeting-eka Friday ekata reschedule karamu"))
