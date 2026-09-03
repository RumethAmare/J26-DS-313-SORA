#!/usr/bin/env python3
"""
Sinhala <-> Latin folding normalizer.

WHY THIS EXISTS
---------------
The C1 corpus is script-split: 16 recordings write Sinhala in native Unicode,
9 write it romanized in Latin, 2 mix within the recording. A Whisper hypothesis
in native Sinhala scored against a romanized reference is 100% wrong by
construction, which is the leading suspect for the 0.937 WER in the prior
baseline (SORA_Dataset/results/c1_report.md).

So Phase 0 scores every decode mode TWICE: raw as-annotated, and folded through
this module. The gap between the two is the measurement of how much of that WER
is script mismatch rather than recognition error.

HOW IT WORKS
------------
Team romanization is informal ("kohomada", "karanne", "sthuthiyi"), not
ISO-15919, so folding is deliberately LOSSY and applied SYMMETRICALLY to both
sides. Distinctions Sinhala orthography makes but informal romanization does
not are collapsed on both sides:

  - aspirated / unaspirated   (kh->k, th->t, dh->d, bh->b, ...)
  - retroflex / dental        (ට and ත both -> t; ඩ and ද both -> d)
  - long / short vowels       (ආ and අ both -> a; aa -> a)
  - sibilants                 (ශ, ෂ, ස all -> s; sh -> s)
  - v / w                     (both -> v, since "wenne"/"venne" both occur)
  - doubled consonants        (karanne -> karane, ekka -> eka)

fold("කොහොමද") == fold("kohomada") == "kohomada"
fold("කරන්නේ") == fold("karanne")  == "karane"

This is a SCORING aid and a diagnostic, not a linguistic transliterator. It is
intentionally many-to-one: it will merge genuinely distinct words. That is
acceptable because both hypothesis and reference pass through it identically,
so the comparison stays fair — but folded WER is a floor, not the headline
number, and must always be reported alongside the raw figure.

No third-party dependency: the table is inlined so this keeps working under the
offline-mode requirement of C1_FINAL_IMPLEMENTATION.md §3.
"""
import re
import unicodedata

# ── Sinhala Unicode block ─────────────────────────────────────
SINHALA_RANGE = (0x0D80, 0x0DFF)
VIRAMA = "්"          # hal kirima — suppresses the inherent vowel
ZERO_WIDTH = "‌‍"  # ZWNJ / ZWJ — used for touching letters, yansaya, rakaransaya

# Consonants, already folded: aspirates onto plain, retroflex onto dental,
# prenasalized onto plain, all three sibilants onto s.
CONSONANTS = {
    "ක": "k", "ඛ": "k",                    # ක ඛ
    "ග": "g", "ඝ": "g", "ඟ": "g",     # ග ඝ ඟ
    "ඞ": "n",                                    # ඞ
    "ච": "c", "ඡ": "c",                    # ච ඡ
    "ජ": "j", "ඣ": "j", "ඦ": "j",     # ජ ඣ ඦ
    "ඤ": "n", "ඥ": "n",                    # ඤ ඥ
    "ට": "t", "ඨ": "t",                    # ට ඨ
    "ඩ": "d", "ඪ": "d", "ඬ": "d",     # ඩ ඪ ඬ
    "ණ": "n",                                    # ණ
    "ත": "t", "ථ": "t",                    # ත ථ
    "ද": "d", "ධ": "d", "ඳ": "d",     # ද ධ ඳ
    "න": "n",                                    # න
    "ප": "p", "ඵ": "p",                    # ප ඵ
    "බ": "b", "භ": "b", "ඹ": "b",     # බ භ ඹ
    "ම": "m",                                    # ම
    "ය": "y",                                    # ය
    "ර": "r",                                    # ර
    "ල": "l", "ළ": "l",                    # ල ළ
    "ව": "v",                                    # ව
    "ශ": "s", "ෂ": "s", "ස": "s",     # ශ ෂ ස
    "හ": "h",                                    # හ
    "ෆ": "f",                                    # ෆ
}

# Independent vowels, long folded onto short.
IND_VOWELS = {
    "අ": "a", "ආ": "a", "ඇ": "a", "ඈ": "a",   # අ ආ ඇ ඈ
    "ඉ": "i", "ඊ": "i",                                  # ඉ ඊ
    "උ": "u", "ඌ": "u",                                  # උ ඌ
    "ඍ": "ru", "ඎ": "ru", "ඏ": "lu", "ඐ": "lu",
    "එ": "e", "ඒ": "e", "ඓ": "ai",                  # එ ඒ ඓ
    "ඔ": "o", "ඕ": "o", "ඖ": "au",                  # ඔ ඕ ඖ
}

# Dependent vowel signs. NFC composes the two-part signs (ො ෝ ෞ) first.
DEP_VOWELS = {
    "ා": "a", "ැ": "a", "ෑ": "a",   # ා ැ ෑ
    "ි": "i", "ී": "i",                   # ි ී
    "ු": "u", "ූ": "u",                   # ු ූ
    "ෘ": "ru", "ෟ": "lu",                 # ෘ ෟ
    "ෙ": "e", "ේ": "e", "ෛ": "ai",   # ෙ ේ ෛ
    "ො": "o", "ෝ": "o", "ෞ": "au",   # ො ෝ ෞ
}

_SINHALA_RE = re.compile(r"[඀-෿]")
_WS_RE = re.compile(r"\s+")

# `\w` excludes Unicode category Mn, so a plain [^\w\s] strips Sinhala dependent
# vowel signs and the virama — "ඔයාට කොහොමද" shreds into "ඔය ට ක හ මද". The
# legacy pattern below has that bug; it is kept ONLY to reproduce the prior
# baseline. Real scoring uses _PUNCT_RE, which spares the Sinhala block.
_PUNCT_RE = re.compile(r"[^\w\s඀-෿]", re.UNICODE)
_PUNCT_RE_LEGACY = re.compile(r"[^\w\s]", re.UNICODE)

# Aspirate and other Latin digraphs. Order matters: these run BEFORE the
# doubled-consonant collapse, so "sthuthi" -> "stuti" not "sthuti".
_DIGRAPHS = [
    ("th", "t"), ("dh", "d"), ("kh", "k"), ("gh", "g"),
    ("ph", "p"), ("bh", "b"), ("jh", "j"), ("chh", "c"),
    ("ch", "c"), ("sh", "s"), ("zh", "s"),
]
_DOUBLE_RE = re.compile(r"([a-z])\1+")


def is_sinhala_script(text):
    """True if the text contains at least one Sinhala-block character."""
    return bool(_SINHALA_RE.search(text))


def sinhala_to_latin(text):
    """
    Transliterate Sinhala script into the folded Latin space.

    Walks the string tracking whether a consonant is still awaiting its vowel:
    a bare consonant takes the inherent 'a', a dependent sign replaces it, and
    the virama cancels it. Non-Sinhala characters pass through untouched, so
    mixed-script text (the R0018/R0022 case) is handled in one pass.
    """
    text = unicodedata.normalize("NFC", text)
    out = []
    pending = False  # a consonant is waiting to learn its vowel

    for ch in text:
        if ch in ZERO_WIDTH:
            continue
        if ch in CONSONANTS:
            if pending:
                out.append("a")
            out.append(CONSONANTS[ch])
            pending = True
        elif ch in DEP_VOWELS:
            out.append(DEP_VOWELS[ch])
            pending = False
        elif ch == VIRAMA:
            pending = False
        elif ch in IND_VOWELS:
            if pending:
                out.append("a")
            out.append(IND_VOWELS[ch])
            pending = False
        else:
            if pending:
                out.append("a")
                pending = False
            out.append(ch)

    if pending:
        out.append("a")
    return "".join(out)


def latin_fold(text):
    """Collapse the distinctions informal romanization does not reliably make."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _PUNCT_RE.sub(" ", text)

    for src, dst in _DIGRAPHS:
        text = text.replace(src, dst)

    text = text.replace("w", "v").replace("q", "k").replace("x", "ks").replace("z", "s")
    text = _DOUBLE_RE.sub(r"\1", text)          # karanne -> karane, aa -> a
    return _WS_RE.sub(" ", text).strip()


def fold(text):
    """
    Map text into the common comparison space, whatever script it arrived in.

    Safe to apply to already-Latin text: sinhala_to_latin is a no-op there.
    """
    return latin_fold(sinhala_to_latin(text))


def normalize_raw(text):
    """
    The as-annotated normalizer: lowercase, strip punctuation, squeeze spaces.

    Text keeps whatever script it was written in — this is the honest "did the
    model produce the reference as annotated" number, and the headline figure
    Phase 0 reports.
    """
    text = _PUNCT_RE.sub(" ", text.lower())
    return _WS_RE.sub(" ", text).strip()


def normalize_legacy(text):
    """
    Bug-compatible with SORA_Dataset/scripts/c1/evaluate.py.

    That normalizer strips Sinhala vowel signs and the virama along with the
    punctuation, so its native-script references were shredded before scoring.
    Kept solely so Phase 0 can reproduce the prior 0.937 WER and prove the new
    harness is wired to the same data. Never report a number from this.
    """
    text = _PUNCT_RE_LEGACY.sub(" ", text.lower())
    return _WS_RE.sub(" ", text).strip()


# ── self-test ─────────────────────────────────────────────────
# Native-script spellings paired with how the team actually romanizes them.
_PAIRS = [
    ("කොහොමද", "kohomada"),
    ("කරන්නේ", "karanne"),
    ("ඔයා", "oya"),
    ("මම", "mama"),
    ("ගොඩක්", "godak"),
    ("ස්තූතියි", "sthuthiyi"),
    ("පුළුවන්ද", "puluwanda"),
    ("නෑ", "naa"),
    ("හරි", "hari"),
    ("දෙන්න", "denna"),
    ("එකක්", "ekak"),
    ("තියෙනවා", "thiyenawa"),
]


def _selftest():
    failures = []
    for sinhala, roman in _PAIRS:
        a, b = fold(sinhala), fold(roman)
        mark = "ok " if a == b else "FAIL"
        if a != b:
            failures.append((sinhala, roman, a, b))
        print(f"  [{mark}] {sinhala:12} -> {a:12} | {roman:12} -> {b}")

    print("\n  mixed-script passthrough:")
    mixed = "මගේ card එක block කරන්න"
    print(f"    {mixed!r} -> {fold(mixed)!r}")

    print("\n  raw vs legacy vs folded on a native-script line:")
    line = "ඔයාට කොහොමද help කරන්නේ"
    print(f"    raw    : {normalize_raw(line)}")
    print(f"    legacy : {normalize_legacy(line)}   <- vowel signs destroyed")
    print(f"    folded : {fold(line)}")
    if normalize_raw(line) != line.lower():
        failures.append(("normalize_raw", "must preserve Sinhala marks",
                         normalize_raw(line), line.lower()))

    if failures:
        print(f"\n{len(failures)}/{len(_PAIRS)} pairs did not fold together:")
        for sinhala, roman, a, b in failures:
            print(f"  {sinhala} -> {a!r}  !=  {roman} -> {b!r}")
        return 1
    print(f"\nAll {len(_PAIRS)} Sinhala/Latin pairs fold together.")
    return 0


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    for arg in sys.argv[1:]:
        print(f"{arg!r} -> {fold(arg)!r}")
