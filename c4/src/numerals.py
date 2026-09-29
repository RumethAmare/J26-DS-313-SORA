"""
Spoken-numeral decoding for Sinhala-English code-mixed transcripts.

Identifiers in this corpus are often *spoken*, not written. A phone number can
appear in any of these forms, all meaning 0775554433:

    0775554433                                          compact digits
    0 7 7 5 5 5 4 4 3 3                                 spaced digits
    077 5554 433                                        grouped digits
    බිංදුවයි හතයි හතයි පන්සිය පනස් පහයි හතළිස් හතරයි තිස් තුන    Sinhala number words
    binduwai hatai hatai pansiya panas pahai ...        romanized number words

A digit-only regex finds the first three and misses the rest entirely. Since
C4's primary metric is recall and a missed identifier is a disclosure, the
number-word forms have to be decoded rather than ignored.

## How the spoken form works

Sinhala numerals are additive within a group, and groups are concatenated to
build the digit string. The conjunctive suffix -යි (romanized -ai / -yi) marks
the END of a group:

    බිංදුවයි   හතයි  හතයි  පන්සිය පනස් පහයි   හතළිස් හතරයි   තිස් තුන
    [0]       [7]   [7]   [500+50+5]      [40+4]        [30+3]
    "0"    +  "7" + "7" + "555"        + "44"        + "33"   = "0775554433"

So each group is summed, rendered as digits, and the groups are concatenated.
That is different from normal numeral parsing, where the whole utterance would
sum to one value.

Number words are also frequently written without spaces between constituents
(අනූඑකයි = අනූ + එක + යි = 91), so matching is longest-prefix within a token,
not whole-token lookup.

## Known ambiguity

Romanization is lossy: "hata" is both හත (7) and හැට (60). This module resolves
it to 7, and requires an explicit "haeta"/"heta" spelling for 60, because 7 is
overwhelmingly the more common reading in digit-sequence context. Ambiguous
decodes are reported so the caller can flag them for human verification rather
than silently trusting them.
"""
from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Lexicon. Values are the numeric contribution of each constituent.
# ---------------------------------------------------------------------------

_SINHALA = {
    # zero
    "බිංදුව": 0, "බින්දුව": 0, "ශුන්‍ය": 0, "බිංදු": 0,
    # units
    "එක": 1, "දෙක": 2, "තුන": 3, "හතර": 4, "පහ": 5,
    "හය": 6, "හත": 7, "අට": 8, "නවය": 9, "නමය": 9,
    # teens
    "දහය": 10, "එකොළහ": 11, "දොළහ": 12, "දහතුන": 13, "දාහතර": 14,
    "පහළොව": 15, "දහසය": 16, "දහහත": 17, "දහඅට": 18,
    "දහනවය": 19, "දහනමය": 19,
    # tens (standalone and combining forms)
    "විස්ස": 20, "විසි": 20,
    "තිහ": 30, "තිස්": 30,
    "හතළිහ": 40, "හතළිස්": 40,
    "පනහ": 50, "පනස්": 50,
    "හැට": 60,
    "හැත්තෑව": 70, "හැත්තෑ": 70,
    "අසූව": 80, "අසූ": 80,
    "අනූව": 90, "අනූ": 90,
    # hundreds. These are multiplicative, not additive: එකසිය is 1x100 = 100,
    # NOT එක + සිය = 101. Each multiple therefore needs its own entry, or
    # longest-prefix matching silently adds the multiplier to the hundred.
    "එකසිය": 100, "සියය": 100, "සිය": 100,
    "දෙසිය": 200, "තුන්සිය": 300, "හාරසිය": 400, "පන්සිය": 500,
    "හයසිය": 600, "හත්සිය": 700, "අටසිය": 800, "නවසිය": 900,
    # thousands
    "එක්දාස්": 1000, "දහස": 1000, "දාස්": 1000,
    "දෙදහස": 2000, "දෙදාස්": 2000,
}

_LATIN = {
    # zero -- "zero" is English but appears mid-Sinhala, which is the point
    "binduwa": 0, "bindu": 0, "zero": 0, "sunya": 0, "shunya": 0,
    # units
    "eka": 1, "ek": 1,
    "deka": 2, "de": 2,
    "thuna": 3, "tuna": 3, "thun": 3,
    "hathara": 4, "hatara": 4, "hathare": 4,
    "paha": 5, "pas": 5,
    "haya": 6, "hay": 6,
    "hatha": 7, "hata": 7, "hath": 7,
    "ata": 8, "atta": 8, "att": 8,
    "navaya": 9, "nawaya": 9, "namaya": 9, "nava": 9,
    # teens
    "dahaya": 10, "ekolaha": 11, "dolaha": 12, "dahathuna": 13,
    "dahahatara": 14, "pahalova": 15, "pahaloha": 15, "dahasaya": 16,
    "dahahatha": 17, "dahaata": 18, "dahanavaya": 19, "dahanamaya": 19,
    # tens
    "vissa": 20, "visi": 20, "wissa": 20, "wisi": 20,
    "thiha": 30, "this": 30, "tis": 30, "tiha": 30,
    "hathaliha": 40, "hathalis": 40, "hatalis": 40,
    "panaha": 50, "panas": 50,
    "haeta": 60, "heta": 60, "haetta": 60,
    "haeththaewa": 70, "haeththae": 70, "heththawa": 70, "heththa": 70,
    "asuwa": 80, "asu": 80,
    "anuwa": 90, "anu": 90,
    # hundreds -- multiplicative, see the Sinhala table above
    "ekasiya": 100, "siyaya": 100, "siya": 100,
    "desiya": 200, "thunsiya": 300, "harasiya": 400, "pansiya": 500,
    "hayasiya": 600, "hathsiya": 700, "atasiya": 800, "navasiya": 900,
    # thousands
    "ekdas": 1000, "dahasa": 1000, "das": 1000,
    "dedahasa": 2000, "dedas": 2000,
}

# Conjunctive suffixes marking the end of a group, longest first.
_SUFFIX_SI = ("යි",)
_SUFFIX_LA = ("yi", "ai", "i")

# Constituents that are ambiguous under romanization; decoding one lowers
# confidence rather than failing, so the caller can route it to human review.
_AMBIGUOUS_LATIN = {"hata", "hath", "asu", "anu", "de", "ek"}

# එක / eka is the most common word in Singlish that is ALSO a digit. As a
# classifier it means "the" and follows a noun -- "number එක", "account එක",
# "NIC එක", "mobile number eka". Reading it as 1 turns 0712345678 into
# 1712345678, which is a silent corruption of the identifier.
#
# The conjunctive form (එකයි / ekai) is never the classifier, so only the BARE
# form is trimmed, and only from the START of a run, where the classifier sits.
_CLASSIFIER_BARE = {"එක", "eka"}

_SI_KEYS = sorted(_SINHALA, key=len, reverse=True)
_LA_KEYS = sorted(_LATIN, key=len, reverse=True)

_SINHALA_BLOCK = re.compile(r"[඀-෿]")


def _is_sinhala(token: str) -> bool:
    return bool(_SINHALA_BLOCK.search(token))


def _suffix_candidates(token: str, sinhala: bool):
    """
    Yield (stem, had_suffix) readings of a token, most-likely first.

    A token can strip more than one way -- "binduwai" yields "binduw" under -ai
    and "binduwa" under -i, and only the second is a real word. So every reading
    is offered and the caller keeps the first that fully decomposes, rather than
    committing to the longest suffix and failing.
    """
    suffixes = _SUFFIX_SI if sinhala else _SUFFIX_LA
    for suf in suffixes:
        if len(token) > len(suf) and token.endswith(suf):
            yield token[: -len(suf)], True
    yield token, False


def _decompose(stem: str, sinhala: bool) -> list[int] | None:
    """
    Split a stem into constituent values by longest-prefix matching.
    'අනූඑක' -> [90, 1].  Returns None if the stem is not fully consumed.
    """
    lexicon, keys = (_SINHALA, _SI_KEYS) if sinhala else (_LATIN, _LA_KEYS)
    values: list[int] = []
    i = 0
    while i < len(stem):
        for key in keys:
            if stem.startswith(key, i):
                values.append(lexicon[key])
                i += len(key)
                break
        else:
            return None
    return values or None


def _token_value(token: str) -> tuple[list[int], bool, bool] | None:
    """
    Decode one whitespace-delimited token.
    Returns (constituent values, ends_group, is_ambiguous) or None.
    """
    cleaned = token.strip().strip(",.‘’'\"")
    if not cleaned:
        return None
    sinhala = _is_sinhala(cleaned)
    if not sinhala:
        cleaned = cleaned.lower()

    for stem, had_suffix in _suffix_candidates(cleaned, sinhala):
        values = _decompose(stem, sinhala)
        if values is None:
            continue
        ambiguous = (not sinhala) and any(
            stem == a or stem.startswith(a) for a in _AMBIGUOUS_LATIN
        )
        return values, had_suffix, ambiguous
    return None


def decode_number_words(phrase: str) -> tuple[str, bool] | None:
    """
    Decode a run of number words into its digit string.

    Returns (digits, ambiguous) or None if the phrase is not a numeral run.

        >>> decode_number_words("බිංදුවයි හතයි හතයි පන්සිය පනස් පහයි")[0]
        '077555'
    """
    tokens = phrase.split()
    if not tokens:
        return None

    digits: list[str] = []
    group: list[int] = []
    ambiguous = False
    decoded_any = False

    for token in tokens:
        decoded = _token_value(token)
        if decoded is None:
            return None
        values, ends_group, tok_ambiguous = decoded
        decoded_any = True
        ambiguous = ambiguous or tok_ambiguous

        for value in values:
            # An additive group runs in non-increasing magnitude: 500, 50, 5.
            # An increase means the previous group ended without a conjunctive
            # suffix to mark it -- "එක එකසිය විසිතුන" is 1 then 123, not 124.
            if group and value > group[-1]:
                digits.append(str(sum(group)))
                group = []
            group.append(value)

        if ends_group:
            digits.append(str(sum(group)))
            group = []

    if group:
        digits.append(str(sum(group)))
    if not decoded_any:
        return None
    return "".join(digits), ambiguous


# ---------------------------------------------------------------------------
# Encoding: digits -> spoken form. The inverse of decode_number_words, used to
# generate synthetic spoken identifiers. Every output is required to round-trip
# through the decoder, so the two are tested against each other.
# ---------------------------------------------------------------------------

_SI_UNITS = {1: "එක", 2: "දෙක", 3: "තුන", 4: "හතර", 5: "පහ",
             6: "හය", 7: "හත", 8: "අට", 9: "නවය"}
_SI_TEENS = {10: "දහය", 11: "එකොළහ", 12: "දොළහ", 13: "දහතුන", 14: "දාහතර",
             15: "පහළොව", 16: "දහසය", 17: "දහහත", 18: "දහඅට", 19: "දහනවය"}
# Tens have a standalone form (40 alone) and a combining form (40 + units).
_SI_TENS_ALONE = {2: "විස්ස", 3: "තිහ", 4: "හතළිහ", 5: "පනහ", 6: "හැට",
                  7: "හැත්තෑව", 8: "අසූව", 9: "අනූව"}
_SI_TENS_COMB = {2: "විසි", 3: "තිස්", 4: "හතළිස්", 5: "පනස්", 6: "හැට",
                 7: "හැත්තෑ", 8: "අසූ", 9: "අනූ"}
_SI_HUNDREDS = {1: "එකසිය", 2: "දෙසිය", 3: "තුන්සිය", 4: "හාරසිය", 5: "පන්සිය",
                6: "හයසිය", 7: "හත්සිය", 8: "අටසිය", 9: "නවසිය"}
_SI_ZERO = "බිංදුව"

# Romanized speech in the corpus is read digit by digit ("binduwai hatai
# hayai"), so only units are needed. Suffixed forms close a group.
_LA_DIGIT = {0: "binduwa", 1: "eka", 2: "deka", 3: "thuna", 4: "hathara",
             5: "paha", 6: "haya", 7: "hatha", 8: "ata", 9: "namaya"}


def _si_group_words(group: str, joined: bool) -> list[str]:
    """Words for one 1-3 digit group, e.g. '555' -> ['පන්සිය', 'පනස්', 'පහ']."""
    n = int(group)
    if n == 0:
        return [_SI_ZERO]
    words: list[str] = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        words.append(_SI_HUNDREDS[hundreds])
    if 10 <= rest <= 19:
        words.append(_SI_TEENS[rest])
    elif rest >= 20:
        tens, units = divmod(rest, 10)
        if units == 0:
            words.append(_SI_TENS_ALONE[tens])
        elif joined:
            # Written without a space, as in the corpus: හතළිස්පහ
            words.append(_SI_TENS_COMB[tens] + _SI_UNITS[units])
        else:
            words.extend([_SI_TENS_COMB[tens], _SI_UNITS[units]])
    elif rest:
        words.append(_SI_UNITS[rest])
    return words


def encode_number_words(digits: str, groups: list[int] | None = None,
                        romanized: bool = False, joined: bool = False) -> str:
    """
    Render a digit string the way it is spoken.

    `groups` gives the length of each spoken group and must sum to len(digits);
    None means one digit per group. A group longer than one digit may not start
    with 0 -- "05" is not a number a speaker says, it is "0" then "5". Every
    group except the last carries the conjunctive suffix, which is what lets
    the decoder recover the boundaries.

        >>> encode_number_words("0775554433", [1, 1, 1, 3, 2, 2])
        'බිංදුවයි හතයි හතයි පන්සිය පනස් පහයි හතළිස් හතරයි තිස් තුන'
    """
    if not digits.isdigit():
        raise ValueError(f"not a digit string: {digits!r}")
    if romanized:
        if groups not in (None, [1] * len(digits)):
            raise ValueError("romanized speech is digit-by-digit only")
        groups = [1] * len(digits)
    groups = groups or [1] * len(digits)
    if sum(groups) != len(digits) or any(not 1 <= g <= 3 for g in groups):
        raise ValueError(f"groups {groups} do not partition {len(digits)} digits")

    out: list[str] = []
    i = 0
    for k, size in enumerate(groups):
        group = digits[i : i + size]
        i += size
        if size > 1 and group[0] == "0":
            raise ValueError(f"group {group!r} has a leading zero")
        words = ([_LA_DIGIT[int(group)]] if romanized
                 else _si_group_words(group, joined))
        if k < len(groups) - 1:
            words[-1] += "i" if romanized else "යි"
        out.extend(words)
    return " ".join(out)


def find_number_word_runs(text: str, min_digits: int = 6):
    """
    Locate maximal runs of number words in free text.

    Yields (start_char, end_char, digits, ambiguous) with offsets into `text`,
    for runs decoding to at least `min_digits` digits -- short runs are ordinary
    counting words ("deka" = two things) rather than spoken identifiers.
    """
    # Token spans, so offsets map back to the original string exactly.
    spans = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]

    i = 0
    while i < len(spans):
        if _token_value(spans[i][2]) is None:
            i += 1
            continue
        j = i
        while j < len(spans) and _token_value(spans[j][2]) is not None:
            j += 1

        # Trim a leading classifier "එක"/"eka" -- see _CLASSIFIER_BARE.
        start_idx = i
        while start_idx < j - 1:
            tok = spans[start_idx][2].strip(",.‘’'\"").lower()
            if tok in _CLASSIFIER_BARE or tok.strip(",.") in _CLASSIFIER_BARE:
                start_idx += 1
            else:
                break

        run_text = text[spans[start_idx][0] : spans[j - 1][1]]
        decoded = decode_number_words(run_text)
        if decoded and len(decoded[0]) >= min_digits:
            yield spans[start_idx][0], spans[j - 1][1], decoded[0], decoded[1]
        i = j
