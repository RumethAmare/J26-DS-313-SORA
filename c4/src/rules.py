"""
Rule layer for Sri Lankan structured identifiers (SO3).

Detects NIC, PHONE, ACCOUNT, DOB and EMAIL in Sinhala-English code-mixed text.
These have fixed formats, so deterministic rules find them without training
data, and the rules can be changed without retraining the model (NFR7).

The rules are written against the forms the corpus actually contains, not
against clean written text:

    compact          0771234567 · 953201456V · 199512345678
    spaced/grouped   0 7 7 1 2 3 4 5 6 7 · 077 123 4567 · +94 77 123 4567
    number words     බිංදුවයි හතයි හතයි ... · zero seven six, eight nine zero ...
    prefixed refs    LN1234NG56789 · CEB/KW/1234/567890 · D L G 1 2 3 ... · N අනූ අටයි ...
    extensions       0112345678 extension 12 · extension number 123
    dates            12th of August 1995 · June 3rd 1990 · 1995 අගෝස්තු 12 වැනිදා

A digit sequence alone does not say what it is -- a 12-digit number can be a
new-format NIC or an account number. The nearest preceding keyword ("NIC",
"account", "ගිණුම්", "mobile" ...) decides; the number's own structure decides
only when there is no keyword. Recall is the primary metric, so an unlabelled
long digit sequence is still reported (as ACCOUNT) rather than dropped.

Returned offsets follow the corpus span convention: codepoint slice indices,
no edge whitespace, no sentence-final punctuation.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

from numerals import decode_number_words, find_number_word_runs

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    start: int
    end: int
    label: str
    surface: str
    value: str = ""                 # normalised identifier, e.g. 0771234567
    role: str = "PRIVATE_INDIVIDUAL"
    ambiguous: bool = False         # route to human verification
    source: str = "rule"
    entity_id: str = ""             # set by cross-script resolution (PERSON)
    score: float = 1.0              # model confidence; rules are certain

    @property
    def redact(self) -> bool:
        return self.role != "ORGANISATION"


# ---------------------------------------------------------------------------
# Context keywords. The nearest keyword before a candidate (by end position,
# longest on a tie) classifies it. "number" alone is deliberately absent: in
# "account number" it would outrank "account" and misclassify every account.
# ---------------------------------------------------------------------------

_CONTEXT = [
    # "identity card" must outrank the ACCOUNT keyword "card" it ends with.
    ("NIC", r"\bnic\b|\bid (?:number|no\.?|eka|එක)|\bidentity(?: card)?\b|හැඳුනුම්පත්"
            r"|හැඳුනුම්"),
    ("ACCOUNT", r"\baccount\b|\bacc\.? ?no\b|\bcard\b|\breference\b|\bref\b|\bpolicy\b"
                r"|\bcustomer (?:id|number)\b|\bbill\b|ගිණුම්|ගිණුම"),
    ("PHONE", r"\bphone\b|\bmobile\b|\bcontact\b|\bcall\b|\btel\b|\bwhatsapp\b"
              r"|\bhotline\b|දුරකථන|ඇමතුම්|කෝල්|අමතන්න"),
    ("DOB", r"\bbirth|\bborn\b|\bdob\b|\bbirthday\b|උපන්"),
    ("AMOUNT", r"\brs\b\.?|\brupees\b|\brupiyal\b|රුපියල්|\blkr\b|\bamount\b|\bpaid\b"
               r"|\bpayment\b|\bgewwa\b|ගෙව්වා|\bdebit\b|\bcredit\b|\btransaction\b"),
]
_CONTEXT_RE = [(cls, re.compile(p, re.IGNORECASE)) for cls, p in _CONTEXT]
_HOTLINE_RE = re.compile(r"\bhotline\b|ක්ෂණික", re.IGNORECASE)

CONTEXT_WINDOW = 50


def _nearest_keyword(left: str) -> str | None:
    best: tuple[int, int, str] | None = None
    for cls, rx in _CONTEXT_RE:
        for m in rx.finditer(left):
            key = (m.end(), m.end() - m.start(), cls)
            if best is None or key[:2] > best[:2]:
                best = key
    return best[2] if best else None


RIGHT_WINDOW = 25
# An utterance is an answer turn when little besides the identifier is said:
# "ow, 199512345678." / "Yes, December 15th 1984." / "2013 මාර්තු 12."
ANSWER_TURN_MAX_WORDS = 3


def _first_keyword(right: str) -> str | None:
    best: tuple[int, int, str] | None = None
    for cls, rx in _CONTEXT_RE:
        for m in rx.finditer(right):
            key = (m.start(), -(m.end() - m.start()), cls)
            if best is None or key[:2] < best[:2]:
                best = key
    return best[2] if best else None


def context_class(text: str, start: int, window: int = CONTEXT_WINDOW,
                  preceding: str = "", end: int | None = None) -> str | None:
    """
    Class of the keyword that describes the candidate at text[start:end].

    1. The nearest keyword before it in the same utterance.
    2. Otherwise the first keyword just after it: Sinhala is verb-final, so
       "0112345678 ta call karanna" / "0112345678 අමතන්න" put it last.
    3. Otherwise, only for an answer turn, the end of the previous utterance:
       "Date of birth?" / "2013 මාර්තු 12" split question and answer. An
       amount keyword never carries over -- it cannot veto a later turn.
    """
    left = text[max(0, start - window) : start]
    # A keyword describes the next number only: in "NIC eka 953201456V, number
    # eka 0771234567" the NIC keyword must not reach past the NIC to the phone.
    cut = max((i for i, ch in enumerate(left) if ch.isdigit()), default=-1)
    local = _nearest_keyword(left[cut + 1 :])
    if local is not None or end is None:
        return local

    right = text[end : end + RIGHT_WINDOW]
    right = right[: next((i for i, ch in enumerate(right) if ch.isdigit()), len(right))]
    after = _first_keyword(right)
    if after is not None:
        return after

    # Words in the candidate's own sentence: "17th of August. I need to be
    # home that day." is still an answer -- the second sentence is not.
    before = re.split(r"[.?!](?:\s|$)", text[:start])[-1]
    after = re.split(r"[.?!](?:\s|$)", text[end:])[0]
    other_words = len(before.split()) + len(after.split())
    if preceding and cut < 0 and other_words <= ANSWER_TURN_MAX_WORDS:
        carried = _nearest_keyword(preceding[-window:])
        return None if carried == "AMOUNT" else carried
    return None


def _is_hotline(text: str, start: int) -> bool:
    return bool(_HOTLINE_RE.search(text[max(0, start - CONTEXT_WINDOW) : start]))


# ---------------------------------------------------------------------------
# Identifier structure
# ---------------------------------------------------------------------------

_TODAY = dt.date.today()


def _valid_nic_day(day: int) -> bool:
    # 1-366, or 501-866 for women. Day 0 and 367-500 are never issued.
    return 1 <= day <= 366 or 501 <= day <= 866


def is_old_nic(digits: str) -> bool:
    return len(digits) == 9 and _valid_nic_day(int(digits[2:5]))


def is_new_nic(digits: str) -> bool:
    return (len(digits) == 12 and 1900 <= int(digits[:4]) <= _TODAY.year
            and _valid_nic_day(int(digits[4:7])))


def normalise_phone(digits: str) -> str | None:
    """Sri Lankan number in local form (0 + 9 digits), or None."""
    if len(digits) == 11 and digits.startswith("94") and digits[2] != "0":
        return "0" + digits[2:]
    if len(digits) == 10 and digits[0] == "0" and digits[1] != "0":
        return digits
    return None


def classify_digits(digits: str, ctx: str | None, letter: str = "") -> tuple[str, str, bool] | None:
    """
    Decide what a digit sequence is. Returns (label, value, ambiguous) or None.

    The keyword overrides structure because structure is ambiguous: a new NIC
    and a 12-digit account number are both twelve digits.
    """
    n = len(digits)
    # Speakers add or repeat digits (false starts decode to extra digits); with
    # a NIC keyword right before it, a near-NIC length is still the NIC,
    # flagged for review.
    # A NIC starts with its birth year (YY or 19/20YY), so never with 0.
    nic_like = ctx == "NIC" and 9 <= n <= 14 and digits[0] != "0"
    if letter:
        if n == 9:
            return "NIC", digits + letter.upper(), not is_old_nic(digits)
        if nic_like:
            return "NIC", digits + letter.upper(), True
        return None
    if ctx == "AMOUNT":
        return None

    phone = normalise_phone(digits)
    if nic_like:
        return "NIC", digits, n not in (9, 12)
    if ctx == "ACCOUNT" and n >= 6:
        return "ACCOUNT", digits, False
    if ctx == "PHONE" and phone:
        return "PHONE", phone, False
    if phone:
        return "PHONE", phone, ctx is None
    if n == 11 and digits[0] == "0" and digits[1] != "0":
        # A phone number with one digit said twice -- misspoken, still a phone.
        return "PHONE", digits, True
    if is_new_nic(digits):
        return "NIC", digits, ctx is not None
    if n == 9 and is_old_nic(digits):
        # A spoken NIC often loses its V. Without the letter it could be an
        # account number, so it is flagged rather than trusted.
        return "NIC", digits, True
    if 10 <= n <= 18:
        return "ACCOUNT", digits, ctx is None
    return None


# ---------------------------------------------------------------------------
# Candidate patterns
# ---------------------------------------------------------------------------

# Digits separated by single spaces or hyphens, an optional leading +, an
# optional trailing V/X (old NIC). Must not touch letters, slashes or a
# decimal point, so dates (1995.08.12) and references (CEB/KW/12) are left
# for their own patterns.
_DOTTED = re.compile(r"(?<![\w.])\d{1,4}(?:\.\d{1,4}){2,}(?![\w]|\.\d)")


def _looks_like_date(token: str) -> bool:
    groups = token.split(".")
    return len(groups) == 3 and any(len(g) == 4 for g in groups)


def _read_out(span: str) -> bool:
    """Spoken digit by digit: single digits or digit words, one per token."""
    tokens = span.replace(",", " ").split()
    return len(tokens) >= 7 and all(len(t) == 1 and t.isdigit() or not t.isdigit()
                                    for t in tokens)


_DIGIT_RUN = re.compile(
    r"(?<![\w/.\-+])(\+)?(\d+(?:[ \-]\d+)*)(?:( ?)([VvXx]))?(?![\w/\-]|[.,]\d)"
)

# Reference numbers with a letter prefix, written as one token.
_REF_TOKEN = re.compile(r"(?<![\w/\-])[A-Z][A-Z0-9]*(?:[-/][A-Z0-9]+)*(?![\w/\-])")

# A letter prefix directly before a digit or number-word run: "D L G ", "POL ", "N ".
_PREFIX_BEFORE = re.compile(r"(?<![\w])(?:[A-Z](?: [A-Z]){1,4}|[A-Z]{1,6}) $")
_PREFIX_STOP = {"NIC", "ID", "NO", "TP", "PIN", "OTP", "LKR", "RS", "I", "A",
                "OK", "SMS", "ATM", "DOB"}

_EXTENSION_AFTER = re.compile(
    r"(?:,)? (?:extension|ext\.?)(?: number| no\.?)? \d{1,5}(?![\d])"
    r"|(?:,)? අභ්‍යන්තර අංකය? \d{1,5}(?![\d])",
    re.IGNORECASE,
)
_EXTENSION_ALONE = re.compile(
    r"(?<![\w])(?:extension|ext\.?)(?: number| no\.?)? \d{1,5}(?![\d])"
    r"|අභ්‍යන්තර අංකය? \d{1,5}(?![\d])",
    re.IGNORECASE,
)

_EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")

# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

MONTHS_EN = ["january", "february", "march", "april", "may", "june", "july",
             "august", "september", "october", "november", "december"]
MONTHS_SI = ["ජනවාරි", "පෙබරවාරි", "මාර්තු", "අප්‍රේල්", "මැයි", "ජූනි", "ජූලි",
             "අගෝස්තු", "සැප්තැම්බර්", "ඔක්තෝබර්", "නොවැම්බර්", "දෙසැම්බර්"]
_MONTH_NUM = {m: i + 1 for i, m in enumerate(MONTHS_EN)}
_MONTH_NUM.update({m: i + 1 for i, m in enumerate(MONTHS_SI)})
_MONTH_NUM["sept"] = 9

_M_EN = r"(?P<mon>" + "|".join(MONTHS_EN + ["sept"]) + r")"
_M_SI = r"(?P<mon>" + "|".join(MONTHS_SI) + r")"
# Any ordinal suffix is accepted: ASR writes "9rd" and "21th".
_DAY = r"(?P<day>\d{1,2})(?:st|nd|rd|th)?"
_YEAR = r"(?P<year>(?:19|20)\d{2})"
_SI_DAY_SUFFIX = r"(?: ?(?:වැනිදා|වෙනිදා))?"

_DATE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    rf"\b{_DAY}(?: of)? {_M_EN}\b(?:,? {_YEAR}\b)?",              # 12th of August 1995
    rf"\b{_M_EN} {_DAY}\b(?:,? {_YEAR}\b)?",                      # August 12, 1995
    rf"\b{_YEAR} {_M_EN} {_DAY}\b",                               # 1995 August 12th
    rf"\b(?P<day>\d{{1,2}})[/.\-](?P<mnum>\d{{1,2}})[/.\-]{_YEAR}\b",  # 12/08/1995
    rf"\b{_YEAR}[/.\-](?P<mnum>\d{{1,2}})[/.\-](?P<day>\d{{1,2}})\b",  # 1995-08-12
    rf"{_YEAR} {_M_SI} {_DAY}{_SI_DAY_SUFFIX}",                   # 1995 අගෝස්තු 12 වැනිදා
    rf"{_M_SI} {_DAY}{_SI_DAY_SUFFIX}(?:,? {_YEAR}\b)?",          # අගෝස්තු 12, 1995
)]
_LONE_ORDINAL = re.compile(r"\b(?P<day>\d{1,2})(?:st|nd|rd|th)\b", re.IGNORECASE)

# Minimum age at which a full date with no context is taken to be a birth date.
_MIN_AGE_YEARS = 15


def _date_ok(day: int, month: int, year: int | None) -> bool:
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return False
    return year is None or 1900 <= year <= _TODAY.year


def _is_birth_date(text: str, start: int, end: int, year: int | None,
                   preceding: str = "") -> tuple[bool, bool]:
    """(is DOB, ambiguous). A dated sentence about a payment is not a birth date."""
    ctx = context_class(text, start, window=80, preceding=preceding, end=end)
    if ctx == "DOB":
        return True, False
    if year is None or ctx == "AMOUNT":
        return False, False
    return year <= _TODAY.year - _MIN_AGE_YEARS, True


def _detect_dates(text: str, preceding: str = "") -> list[Detection]:
    found: list[Detection] = []
    for rx in _DATE_PATTERNS:
        for m in rx.finditer(text):
            g = m.groupdict()
            month = int(g["mnum"]) if g.get("mnum") else _MONTH_NUM[g["mon"].lower()]
            year = int(g["year"]) if g.get("year") else None
            if not _date_ok(int(g["day"]), month, year):
                continue
            is_dob, ambiguous = _is_birth_date(text, m.start(), m.end(), year, preceding)
            if is_dob:
                value = f"{year or '????'}-{month:02d}-{int(g['day']):02d}"
                found.append(Detection(m.start(), m.end(), "DOB", m.group(), value,
                                       ambiguous=ambiguous))
    for m in _LONE_ORDINAL.finditer(text):
        if context_class(text, m.start(), window=80, preceding=preceding,
                         end=m.end()) == "DOB":
            found.append(Detection(m.start(), m.end(), "DOB", m.group(),
                                   f"????-??-{int(m['day']):02d}", ambiguous=True))
    found.extend(_detect_spoken_sinhala_dates(text, preceding))
    return found


def _detect_spoken_sinhala_dates(text: str, preceding: str = "") -> list[Detection]:
    """
    "දෙදාස් දහතුනේ මාර්තු දොළහ": a spoken year, a month name, a spoken day.
    The year carries the locative -ඒ (දහතුනේ = දහතුන + ේ), which is removed
    before decoding; the day may carry වෙනිදා / වැනිදා.
    """
    found = []
    tokens = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]
    for i, (s, e, tok) in enumerate(tokens):
        month = _MONTH_NUM.get(tok.strip(",."))
        if month is None or not re.search(r"[඀-෿]", tok):
            continue
        start, end, year, day = s, e, None, None

        for k in (2, 1):                              # spoken year before the month
            if i - k < 0:
                continue
            words = [t[2] for t in tokens[i - k : i]]
            words[-1] = words[-1].rstrip(",").removesuffix("ේ")
            dec = decode_number_words(" ".join(words))
            if dec and 1900 <= int(dec[0]) <= _TODAY.year:
                start, year = tokens[i - k][0], int(dec[0])
                break

        if i + 1 < len(tokens):                       # spoken day after the month
            raw = tokens[i + 1][2].rstrip(",.")
            stem = re.sub(r"(?:වැනිදා|වෙනිදා)$", "", raw)
            dec = decode_number_words(stem) if stem else None
            if dec and 1 <= int(dec[0]) <= 31 and not stem.isdigit():
                day = int(dec[0])
                end = tokens[i + 1][0] + len(raw)
                if (i + 2 < len(tokens) and stem == raw
                        and tokens[i + 2][2].rstrip(",.") in ("වැනිදා", "වෙනිදා")):
                    end = tokens[i + 2][0] + len(tokens[i + 2][2].rstrip(",."))

        if year is None or day is None:
            continue
        is_dob, _ = _is_birth_date(text, start, end, year, preceding)
        if is_dob:
            found.append(Detection(start, end, "DOB", text[start:end],
                                   f"{year}-{month:02d}-{day:02d}", ambiguous=True))
    return found


# ---------------------------------------------------------------------------
# Numeric identifiers
# ---------------------------------------------------------------------------


def _with_prefix(text: str, start: int) -> int | None:
    """Start of a letter prefix ("DLG ", "D L G ", "N ") directly before `start`."""
    m = _PREFIX_BEFORE.search(text[max(0, start - 12) : start])
    if not m or m.group().replace(" ", "") in _PREFIX_STOP:
        return None
    return start - (len(m.group()))


def _with_extension(text: str, end: int) -> int:
    m = _EXTENSION_AFTER.match(text, end)
    return m.end() if m else end


def _numeric_detections(text: str, preceding: str = "") -> list[Detection]:
    found: list[Detection] = []
    candidates = []   # (start, end, digits, letter, ambiguous)

    for m in _DIGIT_RUN.finditer(text):
        digits = re.sub(r"\D", "", m.group(2))
        letter = m.group(4) or ""
        start = m.start()
        end = m.end() if letter else m.start(2) + len(m.group(2))
        candidates.append((start, end, digits, letter, False))

    for s, e, digits, ambiguous in find_number_word_runs(text, min_digits=4):
        candidates.append((s, e, digits, "", ambiguous))

    # Dotted groups: "123.456.78.90". Dates are excluded (they have their own
    # patterns and at most 8 digits in three groups with a year).
    for m in _DOTTED.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if len(digits) >= 8 and not _looks_like_date(m.group()):
            candidates.append((m.start(), m.end(), digits, "", False))

    for start, end, digits, letter, amb in candidates:
        prefix_start = _with_prefix(text, start)
        if prefix_start is not None and len(digits) >= 4:
            # A lettered reference is an account/customer/policy number, never
            # a phone or NIC.
            found.append(Detection(prefix_start, end, "ACCOUNT", text[prefix_start:end],
                                   re.sub(r"\s", "", text[prefix_start:start]) + digits,
                                   ambiguous=amb))
            continue
        if len(digits) < 6:
            continue
        ctx = context_class(text, start, preceding=preceding, end=end)
        result = classify_digits(digits, ctx, letter)
        if result is None and ctx != "AMOUNT" and len(digits) >= 7 and _read_out(text[start:end]):
            # People say amounts as numbers ("twenty-five thousand") but read an
            # identifier out digit by digit. A digit-by-digit run is an
            # identifier even with no keyword; ACCOUNT is the safe default.
            result = ("ACCOUNT", digits, True)
        if result is None:
            continue
        label, value, ambiguous = result
        role = "PRIVATE_INDIVIDUAL"
        if label == "PHONE":
            end = _with_extension(text, end)
            if _is_hotline(text, start):
                role = "ORGANISATION"
        found.append(Detection(start, end, label, text[start:end], value, role,
                               ambiguous or amb))

    for m in _REF_TOKEN.finditer(text):
        token = m.group()
        if sum(ch.isdigit() for ch in token) >= 4 and not token[:3] in _PREFIX_STOP:
            found.append(Detection(m.start(), m.end(), "ACCOUNT", token, token))

    for m in _EXTENSION_ALONE.finditer(text):
        found.append(Detection(m.start(), m.end(), "PHONE", m.group(),
                               re.sub(r"\D", "", m.group()), "ORGANISATION_REP",
                               ambiguous=True))
    return found


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Tie-break for spans of equal length over the same text.
_PRIORITY = {"EMAIL": 0, "DOB": 1, "NIC": 2, "PHONE": 3, "ACCOUNT": 4}


def _resolve_overlaps(found: list[Detection]) -> list[Detection]:
    """Keep the longest span wherever candidates overlap (maximal span, §4)."""
    ranked = sorted(found, key=lambda d: (-(d.end - d.start), _PRIORITY[d.label], d.start))
    kept: list[Detection] = []
    for d in ranked:
        if all(d.end <= k.start or d.start >= k.end for k in kept):
            kept.append(d)
    return sorted(kept, key=lambda d: d.start)


def detect(text: str, preceding: str = "") -> list[Detection]:
    """
    All structured identifiers in `text`, non-overlapping, in text order.

    `preceding` is the previous utterance of the same document stream, used
    only as keyword context for answers to a question; it is never searched
    for identifiers and offsets always index into `text`.
    """
    found = [Detection(m.start(), m.end(), "EMAIL", m.group(), m.group().lower())
             for m in _EMAIL.finditer(text)]
    found += _detect_dates(text, preceding)
    found += _numeric_detections(text, preceding)
    return _resolve_overlaps(found)
