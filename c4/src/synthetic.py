"""
Synthetic Sinhala-English PII corpus for development and unit testing.

The shared corpus is still being recorded, so the rule layer, the evaluation
harness and the cross-script resolver are built against synthetic recordings
that follow the same schema (docs/PII_SCHEMA.md) and the same document layout:

    <rid>_transcript_uNNN_v1     Singlish transcript, mixed script, spoken numbers
    <rid>_c2_uNNN_en_v1          clean English, compact identifiers
    <rid>_c2_uNNN_si_v1          clean Sinhala, Sinhala-script names
    <rid>_c2_summary_en_v1       English summary

Every value is generated, never copied from the real corpus, and every row is
marked annotation_source "synthetic". Synthetic data is used for development
and unit tests ONLY; every reported metric comes from real held-out recordings.

What is realistic by construction:
  * NIC numbers encode the holder's birth year, day of year and sex exactly as
    issued, so NIC and DOB of one person agree.
  * Identifiers in transcripts appear in every spoken form the corpus shows:
    compact, spaced, grouped, Sinhala number words, romanized number words.
  * One person carries one entity_id across all four documents and both
    scripts -- the property Contribution 2 depends on.
  * Hard negatives are included and left unannotated: money amounts, counting
    words, transaction dates, a branch town (LOCATION, not redacted) and an
    organisation hotline (PHONE owned by an ORGANISATION, not redacted).

What it cannot give you: the disfluency, ASR errors and unpredictable phrasing
of real speech. A model trained only on this learns the templates.

Two splits, both written by default:

    train   the templates above; used to build and to train
    test    held-out templates, held-out names/places/organisations,
            question-and-answer turns and speech noise; used only to score

    python synthetic.py                  # -> data/synthetic/{train,test}/
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from numerals import encode_number_words
from validate import expected_redact, validate_recording

# ---------------------------------------------------------------------------
# Lexicon: (Latin, Sinhala) pairs. Common Sri Lankan forms, not corpus names.
# ---------------------------------------------------------------------------

FIRST_NAMES = [  # (latin, sinhala, sex)
    ("Nimal", "නිමල්", "M"), ("Kamal", "කමල්", "M"), ("Sunil", "සුනිල්", "M"),
    ("Kasun", "කසුන්", "M"), ("Tharindu", "තරිඳු", "M"), ("Ruwan", "රුවන්", "M"),
    ("Pradeep", "ප්‍රදීප්", "M"), ("Chathura", "චතුර", "M"), ("Dinesh", "දිනේෂ්", "M"),
    ("Lahiru", "ලහිරු", "M"), ("Sampath", "සම්පත්", "M"), ("Nuwan", "නුවන්", "M"),
    ("Chamari", "චමරි", "F"), ("Dilani", "දිලානි", "F"), ("Sanduni", "සඳුනි", "F"),
    ("Nadeesha", "නදීෂා", "F"), ("Hiruni", "හිරුනි", "F"), ("Malsha", "මල්ෂා", "F"),
    ("Ishara", "ඉෂාරා", "F"), ("Kavindi", "කවින්දි", "F"), ("Tharushi", "තරුෂි", "F"),
    ("Nethmi", "නෙත්මි", "F"), ("Sachini", "සචිනි", "F"), ("Dulani", "දුලානි", "F"),
]

SURNAMES = [
    ("Perera", "පෙරේරා"), ("Fernando", "ප්‍රනාන්දු"), ("Silva", "සිල්වා"),
    ("Jayasinghe", "ජයසිංහ"), ("Bandara", "බණ්ඩාර"), ("Dissanayake", "දිසානායක"),
    ("Wickramasinghe", "වික්‍රමසිංහ"), ("Rathnayake", "රත්නායක"),
    ("Gunawardena", "ගුණවර්ධන"), ("Herath", "හේරත්"), ("Karunaratne", "කරුණාරත්න"),
    ("Senanayake", "සේනානායක"), ("Wijesinghe", "විජේසිංහ"), ("Mendis", "මෙන්ඩිස්"),
]

TOWNS = [
    ("Colombo", "කොළඹ"), ("Kandy", "මහනුවර"), ("Galle", "ගාල්ල"),
    ("Maharagama", "මහරගම"), ("Nugegoda", "නුගේගොඩ"), ("Kurunegala", "කුරුණෑගල"),
    ("Matara", "මාතර"), ("Negombo", "මීගමුව"), ("Kegalle", "කෑගල්ල"),
    ("Dehiwala", "දෙහිවල"), ("Moratuwa", "මොරටුව"), ("Gampaha", "ගම්පහ"),
]

STREETS = [
    ("Galle Road", "ගාලු පාර"), ("Temple Road", "පන්සල් පාර"),
    ("Station Road", "දුම්රියපොළ පාර"), ("Lake Road", "වැව් පාර"),
    ("Main Street", "ප්‍රධාන වීදිය"), ("Flower Road", "මල් පාර"),
    ("Hospital Road", "රෝහල් පාර"), ("School Lane", "පාසල් පටුමග"),
]

ORGS = [  # public organisation names: ORG is not personal data
    ("Sampath Bank", "සම්පත් බැංකුව"), ("Commercial Bank", "කොමර්ෂල් බැංකුව"),
    ("Bank of Ceylon", "ලංකා බැංකුව"), ("People's Bank", "මහජන බැංකුව"),
    ("Dialog", "ඩයලොග්"), ("Mobitel", "මොබිටෙල්"), ("HNB", "එච්එන්බී"),
]

# Held-out lexicon for the test split: no name, place or organisation here
# occurs in the train split, so a model scored on synthetic-test cannot pass
# by memorising train entities.
TEST_FIRST_NAMES = [
    ("Amal", "අමල්", "M"), ("Saman", "සමන්", "M"), ("Gayan", "ගයාන්", "M"),
    ("Buddhika", "බුද්ධික", "M"), ("Isuru", "ඉසුරු", "M"), ("Janaka", "ජනක", "M"),
    ("Madhavi", "මාධවී", "F"), ("Piumi", "පියුමි", "F"), ("Shanika", "ශානිකා", "F"),
    ("Yasodha", "යශෝධා", "F"), ("Rashmi", "රශ්මි", "F"), ("Anjali", "අංජලී", "F"),
]
TEST_SURNAMES = [
    ("Ranasinghe", "රණසිංහ"), ("Samarakoon", "සමරකෝන්"), ("Weerasinghe", "වීරසිංහ"),
    ("Abeysekara", "අබේසේකර"), ("Liyanage", "ලියනගේ"), ("Kumara", "කුමාර"),
    ("Pathirana", "පතිරණ"), ("Ekanayake", "ඒකනායක"),
]
TEST_TOWNS = [
    ("Panadura", "පානදුර"), ("Kalutara", "කළුතර"), ("Anuradhapura", "අනුරාධපුරය"),
    ("Ratnapura", "රත්නපුර"), ("Badulla", "බදුල්ල"), ("Jaffna", "යාපනය"),
    ("Chilaw", "හලාවත"), ("Homagama", "හෝමාගම"),
]
TEST_STREETS = [
    ("Kandy Road", "මහනුවර පාර"), ("Church Street", "පල්ලිය වීදිය"),
    ("Market Road", "වෙළඳපොළ පාර"), ("Circular Road", "වටරවුම් පාර"),
    ("Park Avenue", "උද්‍යාන මාවත"),
]
TEST_ORGS = [
    ("Seylan Bank", "සෙලාන් බැංකුව"), ("NSB", "ජාතික ඉතිරිකිරීමේ බැංකුව"),
    ("SLT", "ශ්‍රී ලංකා ටෙලිකොම්"), ("Hutch", "හච්"),
    ("Ceylinco Insurance", "සෙලින්කෝ රක්ෂණ"),
]

LEXICONS = {
    "train": dict(first=FIRST_NAMES, surname=SURNAMES, town=TOWNS, street=STREETS, org=ORGS),
    "test": dict(first=TEST_FIRST_NAMES, surname=TEST_SURNAMES, town=TEST_TOWNS,
                 street=TEST_STREETS, org=TEST_ORGS),
}

MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July",
             "August", "September", "October", "November", "December"]
MONTHS_SI = ["ජනවාරි", "පෙබරවාරි", "මාර්තු", "අප්‍රේල්", "මැයි", "ජූනි", "ජූලි",
             "අගෝස්තු", "සැප්තැම්බර්", "ඔක්තෝබර්", "නොවැම්බර්", "දෙසැම්බර්"]

MOBILE_PREFIXES = ["070", "071", "072", "074", "075", "076", "077", "078"]

_SINHALA_CHAR = re.compile(r"[඀-෿]")

# ---------------------------------------------------------------------------
# Templates. {SLOT} markers become annotated spans (or unannotated distractors
# for AMOUNT / TXDATE / COUNT). One list per topic per document kind.
# ---------------------------------------------------------------------------

TEMPLATES = {
    "greet": {
        "transcript": ["hello, mama {AGENT}, {ORG} call centre eken kathaa karanne.",
                       "ආයුබෝවන්, මම {AGENT}, {ORG} එකෙන් කතා කරන්නේ.",
                       "good morning, {ORG} customer service, mama {AGENT}."],
        "en": ["Hello, this is {AGENT} from the {ORG} call centre.",
               "Good morning, {ORG} customer service, {AGENT} speaking."],
        "si": ["ආයුබෝවන්, මම {AGENT}, {ORG} ඇමතුම් මධ්‍යස්ථානයෙන් කතා කරන්නේ."],
        "summary": ["{CUST_FULL} called {ORG} and spoke to {AGENT}."],
    },
    "name": {
        "transcript": ["ow, mage nama {CUST_FULL}.", "mage nama {CUST}.",
                       "මගේ නම {CUST_FULL}.", "ow miss, mama {CUST}."],
        "en": ["My name is {CUST_FULL}.", "Yes, this is {CUST_HON}."],
        "si": ["මගේ නම {CUST_FULL}.", "ඔව්, මම {CUST}."],
        "summary": [],
    },
    "nic": {
        "transcript": ["mage NIC number eka {NIC}.", "NIC එක {NIC}.",
                       "ow, ID number eka {NIC}."],
        "en": ["My NIC number is {NIC}.", "My national identity card number is {NIC}."],
        "si": ["මගේ ජාතික හැඳුනුම්පත් අංකය {NIC}."],
        "summary": ["The customer's NIC number is {NIC}."],
    },
    "phone": {
        "transcript": ["mage mobile number eka {PHONE}.", "contact number එක {PHONE}.",
                       "call karanna puluwan {PHONE} ta."],
        "en": ["My mobile number is {PHONE}.", "You can reach me on {PHONE}."],
        "si": ["මගේ දුරකථන අංකය {PHONE}."],
        "summary": ["The contact number is {PHONE}."],
    },
    "account": {
        "transcript": ["account number eka {ACCOUNT}.", "මගේ account එක {ACCOUNT}."],
        "en": ["My account number is {ACCOUNT}."],
        "si": ["මගේ ගිණුම් අංකය {ACCOUNT}."],
        "summary": ["The account number is {ACCOUNT}."],
    },
    "dob": {
        "transcript": ["mage birthday eka {DOB}.", "date of birth eka {DOB}."],
        "en": ["My date of birth is {DOB}.", "I was born on {DOB}."],
        "si": ["මගේ උපන් දිනය {DOB}."],
        "summary": ["The date of birth given was {DOB}."],
    },
    "address": {
        "transcript": ["mage address eka {ADDRESS}.", "mama inne {ADDRESS}."],
        "en": ["My address is {ADDRESS}.", "I live at {ADDRESS}."],
        "si": ["මගේ ලිපිනය {ADDRESS}."],
        "summary": ["The customer lives at {ADDRESS}."],
    },
    "email": {
        "transcript": ["email eka {EMAIL}.", "mage email එක {EMAIL}."],
        "en": ["My email address is {EMAIL}."],
        "si": ["මගේ ඊමේල් ලිපිනය {EMAIL}."],
        "summary": ["The email address on file is {EMAIL}."],
    },
    "relative": {
        "transcript": ["mage thaththage nama {REL}.", "account eke nominee {REL}."],
        "en": ["My father's name is {REL}.", "The nominee on the account is {REL}."],
        "si": ["මගේ තාත්තාගේ නම {REL}."],
        "summary": ["The nominee is {REL}."],
    },
    # --- hard negatives: nothing personal here, or nothing to redact -------
    "branch": {
        "transcript": ["mama iye {BRANCH} branch ekata giya.",
                       "{BRANCH} branch එකට ගියා."],
        "en": ["I visited the {BRANCH} branch yesterday."],
        "si": ["මම ඊයේ {BRANCH} ශාඛාවට ගියා."],
        "summary": ["The customer had visited the {BRANCH} branch."],
    },
    "payment": {
        "transcript": ["mama {TXDATE} rupiyal {AMOUNT} gewwa.",
                       "last month {AMOUNT} rupees debit una, {COUNT} parak."],
        "en": ["I paid Rs. {AMOUNT} on {TXDATE}."],
        "si": ["මම {TXDATE} රුපියල් {AMOUNT} ක් ගෙව්වා."],
        "summary": ["A payment of Rs. {AMOUNT} was discussed."],
    },
    "hotline": {
        "transcript": ["{ORG} hotline eka {HOTLINE}.", "oyata {HOTLINE} ta call karanna puluwan."],
        "en": ["You can call the {ORG} hotline on {HOTLINE}."],
        "si": ["{ORG} ක්ෂණික ඇමතුම් අංකය {HOTLINE}."],
        "summary": [],
    },
}

OPTIONAL_TOPICS = ["nic", "phone", "account", "dob", "address", "email",
                   "relative", "branch", "payment", "hotline"]

# Test-split templates: phrasings that never occur in TEMPLATES. A tuple is a
# question turn followed by an answer turn -- the answer carries no keyword,
# which is how real callers reply. Every kind in a variant has the same arity,
# so the three renderings stay utterance-aligned.
TEST_TEMPLATES = {
    "greet": [
        {"transcript": "ayubowan, {ORG} eken {AGENT} kiyala kenek kathaa karanne.",
         "en": "Good afternoon, {AGENT} here from {ORG}.",
         "si": "සුබ දවසක්, මම {ORG} ආයතනයෙන් {AGENT}."},
        {"transcript": "{ORG} ekata call kalata thanks, mage nama {AGENT}.",
         "en": "Thank you for calling {ORG}, my name is {AGENT}.",
         "si": "{ORG} ඇමතීම ගැන ස්තූතියි, මගේ නම {AGENT}."},
    ],
    "name": [
        {"transcript": ("oyaage nama mokakda?", "{CUST_FULL} thamai."),
         "en": ("May I have your name?", "It is {CUST_FULL}."),
         "si": ("ඔබගේ නම කුමක්ද?", "{CUST_FULL}.")},
        {"transcript": "mama {CUST}, account holder.",
         "en": "I am {CUST}, the account holder.",
         "si": "මම {CUST}, ගිණුම් හිමියා."},
    ],
    "nic": [
        {"transcript": ("ID eke number eka kiyanna puluwanda?", "ow, {NIC}."),
         "en": ("Could you read out your ID number?", "Sure, {NIC}."),
         "si": ("ඔබගේ හැඳුනුම්පත් අංකය කියන්න.", "{NIC}.")},
        {"transcript": "identity card eke thiyenne {NIC}.",
         "en": "The number on my identity card is {NIC}.",
         "si": "මගේ හැඳුනුම්පතේ අංකය {NIC}."},
    ],
    "phone": [
        {"transcript": ("oyaawa contact karanna number ekak denna.", "{PHONE} ta gahanna."),
         "en": ("What is the best number to contact you on?", "Please call {PHONE}."),
         "si": ("ඔබව සම්බන්ධ කරගන්න අංකයක් දෙන්න.", "{PHONE} අමතන්න.")},
        {"transcript": "whatsapp karanna {PHONE} ekata.",
         "en": "Message me on WhatsApp at {PHONE}.",
         "si": "{PHONE} අංකයට WhatsApp කරන්න."},
    ],
    "account": [
        {"transcript": ("salli daanna one account eka?", "{ACCOUNT}."),
         "en": ("Which account should the money go to?", "{ACCOUNT}."),
         "si": ("මුදල් බැර කළ යුතු ගිණුම කුමක්ද?", "{ACCOUNT}.")},
        {"transcript": "savings eke number eka {ACCOUNT}.",
         "en": "My savings number is {ACCOUNT}.",
         "si": "මගේ ඉතුරුම් අංකය {ACCOUNT}."},
    ],
    "dob": [
        {"transcript": ("upan dinaya kiyanna puluwanda?", "{DOB}."),
         "en": ("Can you confirm your date of birth?", "Yes, {DOB}."),
         "si": ("ඔබේ උපන් දිනය තහවුරු කරන්න.", "{DOB}.")},
        {"transcript": "mama ipadune {DOB}.",
         "en": "My birthday falls on {DOB}.",
         "si": "මම ඉපදුණේ {DOB}."},
    ],
    "address": [
        {"transcript": ("dan inne koheda?", "{ADDRESS}."),
         "en": ("Where are you living now?", "{ADDRESS}."),
         "si": ("දැන් පදිංචිය කොහේද?", "{ADDRESS}.")},
        {"transcript": "letters ewanna {ADDRESS} ta.",
         "en": "Please send letters to {ADDRESS}.",
         "si": "ලියුම් {ADDRESS} වෙත එවන්න."},
    ],
    "email": [
        {"transcript": ("email ekak thiyenawada?", "ow, {EMAIL}."),
         "en": ("Do you have an email address?", "Yes, {EMAIL}."),
         "si": ("ඊමේල් ලිපිනයක් තිබේද?", "ඔව්, {EMAIL}.")},
    ],
    "relative": [
        {"transcript": "mage ayya {REL} th ekka enawa.",
         "en": "My brother {REL} will come with me.",
         "si": "මගේ අයියා {REL} මා සමඟ එනවා."},
    ],
    "branch": [
        {"transcript": "{BRANCH} office eke idala thamai call karanne.",
         "en": "I am calling from near the {BRANCH} office.",
         "si": "මම {BRANCH} කාර්යාලය අසල සිට කතා කරන්නේ."},
    ],
    "payment": [
        {"transcript": "{TXDATE} {AMOUNT} ක් bill eka gewwa.",
         "en": "On {TXDATE} I transferred {AMOUNT} rupees.",
         "si": "{TXDATE} දින රුපියල් {AMOUNT} ක් මාරු කළා."},
    ],
    "hotline": [
        {"transcript": "prashnayak thiyenam {HOTLINE} ta call karanna.",
         "en": "For any issue, call {HOTLINE}.",
         "si": "ගැටලුවක් ඇත්නම් {HOTLINE} අමතන්න."},
    ],
}

TEST_SUMMARY = {
    "greet": "{AGENT} of {ORG} handled a call from {CUST_FULL}.",
    "nic": "Identity number provided: {NIC}.",
    "phone": "Callback number: {PHONE}.",
    "account": "Funds are to be credited to {ACCOUNT}.",
    "dob": "Customer born {DOB}.",
    "address": "Residential address: {ADDRESS}.",
    "email": "Email: {EMAIL}.",
    "relative": "{REL} was mentioned as a family member.",
    "branch": "The {BRANCH} office was referenced.",
    "payment": "A transfer of Rs. {AMOUNT} was reported.",
}

_FILLERS = ["uh, ", "ammo, ", "ah ", "hmm, ", "ane "]
_EN_DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
                   "eight", "nine"]

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    entity_id: str
    label: str
    role: str
    canonical: str
    latin: str = ""              # Latin display form (names, places)
    sinhala: str = ""            # Sinhala display form
    value: str = ""              # normalised identifier value (digits etc.)
    extra: dict = field(default_factory=dict)
    surfaces: list = field(default_factory=list)

    def observe(self, script: str, form: str) -> None:
        item = {"script": script, "form": form}
        if item not in self.surfaces:
            self.surfaces.append(item)


@dataclass
class Person:
    first: tuple[str, str, str]
    surname: tuple[str, str]
    dob: dt.date


# ---------------------------------------------------------------------------
# Identifier generation
# ---------------------------------------------------------------------------


def nic_day_of_year(dob: dt.date) -> int:
    """
    NIC day number. Issued NICs count every year as if February had 29 days,
    so the day is taken from a leap-year calendar regardless of birth year.
    """
    return dt.date(2000, dob.month, dob.day).timetuple().tm_yday


def make_nic(dob: dt.date, sex: str, rng: random.Random, old: bool) -> str:
    """Old: YY DDD SSS C + V/X (9 digits + letter). New: YYYY DDD SSSS C (12)."""
    day = nic_day_of_year(dob) + (500 if sex == "F" else 0)
    if old:
        return (f"{dob.year % 100:02d}{day:03d}{rng.randint(0, 999):03d}"
                f"{rng.randint(0, 9)}{rng.choice('VVVX')}")
    return f"{dob.year}{day:03d}{rng.randint(0, 9999):04d}{rng.randint(0, 9)}"


def make_mobile(rng: random.Random) -> str:
    return rng.choice(MOBILE_PREFIXES) + "".join(rng.choices("0123456789", k=7))


def make_hotline(rng: random.Random) -> str:
    return "011" + "".join(rng.choices("0123456789", k=7))


def make_account(rng: random.Random) -> str:
    n = rng.choice([10, 12, 12, 12, 14])
    return str(rng.randint(1, 9)) + "".join(rng.choices("0123456789", k=n - 1))


def ordinal(n: int) -> str:
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# ---------------------------------------------------------------------------
# Surface renderers
# ---------------------------------------------------------------------------


def _random_groups(digits: str, rng: random.Random) -> list[int]:
    """A valid spoken grouping: 1-3 digits, no multi-digit group starting 0."""
    groups, i = [], 0
    while i < len(digits):
        size = 1 if digits[i] == "0" else rng.choice([1, 1, 2, 2, 3])
        size = min(size, len(digits) - i)
        groups.append(size)
        i += size
    return groups


def _chunked(digits: str, rng: random.Random) -> str:
    """'0775554433' -> '077 555 4433' style grouping."""
    parts, i = [], 0
    while i < len(digits):
        size = rng.choice([3, 3, 4]) if len(digits) - i > 4 else len(digits) - i
        parts.append(digits[i : i + size])
        i += size
    return " ".join(parts)


def spoken_digits(value: str, rng: random.Random, allow_words: bool = True) -> str:
    """One of the transcript renderings of a digit identifier."""
    digits = re.sub(r"\D", "", value)
    letter = value[-1].lower() if value[-1].isalpha() else ""
    forms = ["compact", "spaced", "grouped"]
    if allow_words and not letter:
        forms += ["sinhala_words", "sinhala_words", "romanized_words"]
    form = rng.choice(forms)
    if form == "compact":
        return digits + letter.upper()
    if form == "spaced":
        return " ".join(digits + letter)
    if form == "grouped":
        return _chunked(digits, rng) + (f" {letter}" if letter else "")
    if form == "romanized_words":
        return encode_number_words(digits, romanized=True)
    return encode_number_words(digits, _random_groups(digits, rng),
                               joined=rng.random() < 0.4)


def dob_surface(d: dt.date, kind: str, rng: random.Random) -> str:
    month_en = MONTHS_EN[d.month - 1]
    if kind == "si":
        return rng.choice([f"{d.year} {MONTHS_SI[d.month - 1]} {d.day}",
                           f"{d.year}.{d.month:02d}.{d.day:02d}"])
    forms = [f"{ordinal(d.day)} of {month_en} {d.year}", f"{month_en} {d.day}, {d.year}",
             f"{d.day} {month_en} {d.year}"]
    if kind == "transcript":
        forms.append(f"{d.day:02d}/{d.month:02d}/{d.year}")
    else:
        forms.append(f"{d.year}-{d.month:02d}-{d.day:02d}")
    return rng.choice(forms)


def script_of(surface: str, kind: str) -> str:
    """
    Sinhala letters make a span SINHALA. A span with no letters of either
    script (a digit sequence) takes the script of its document, matching how
    the real corpus tags digits inside clean_si.
    """
    if _SINHALA_CHAR.search(surface):
        return "SINHALA"
    return "SINHALA" if kind == "si" else "LATIN"


# ---------------------------------------------------------------------------
# Recording generation
# ---------------------------------------------------------------------------


class Recording:
    def __init__(self, rid: str, rng: random.Random, split: str = "train"):
        self.rid = rid
        self.rng = rng
        self.split = split
        self.noisy = split == "test"
        self.lex = LEXICONS[split]
        self.entities: dict[str, Entity] = {}
        self.docs: list[dict] = []
        self.rows: list[dict] = []
        self._pseudo: dict[str, int] = {}
        self._utt: dict[str, object] = {}  # distractor values of the current utterance
        self._noise: list[str] = []        # noise applied to the slot being rendered
        self._lower = False                # lowercase the literal text of this utterance

        self.customer = self._person()
        self.agent_first = rng.choice(self.lex["first"])
        self.relative = self._person(male=True)
        self.org = rng.choice(self.lex["org"])
        self.branch = rng.choice(self.lex["town"])
        self.home = (rng.randint(1, 250), rng.choice(self.lex["street"]),
                     rng.choice(self.lex["town"]), rng.random() < 0.3)
        self.old_nic = self.customer.dob.year < 2000 and rng.random() < 0.5

    # -- entity registry ----------------------------------------------------

    def _person(self, male: bool = False) -> Person:
        names = self.lex["first"]
        firsts = [f for f in names if f[2] == "M"] if male else names
        born = dt.date(1955, 1, 1) + dt.timedelta(days=self.rng.randint(0, 18250))
        return Person(self.rng.choice(firsts), self.rng.choice(self.lex["surname"]), born)

    def entity(self, key: str, label: str, role: str, canonical: str, **kw) -> Entity:
        if key not in self.entities:
            eid = f"{self.rid}_e{len(self.entities) + 1:03d}"
            self.entities[key] = Entity(eid, label, role, canonical, **kw)
        return self.entities[key]

    def _pseudonym(self, ent: Entity) -> str:
        if not expected_redact(ent.label, ent.role):
            return ent.canonical
        if "pseudonym" not in ent.extra:
            self._pseudo[ent.label] = self._pseudo.get(ent.label, 0) + 1
            ent.extra["pseudonym"] = f"[{ent.label}_{self._pseudo[ent.label]}]"
        return ent.extra["pseudonym"]

    # -- slot resolution -----------------------------------------------------

    def _name_form(self, latin: str, sinhala: str, kind: str) -> str:
        if kind == "si":
            return sinhala
        if kind == "transcript":
            roll = self.rng.random()
            if roll < 0.35:
                return sinhala
            if roll < 0.55:
                return latin.lower()
        return latin

    def resolve(self, slot: str, kind: str) -> tuple[str, Entity | None]:
        """Surface text for a slot, and the entity it annotates (None = distractor)."""
        rng, c = self.rng, self.customer

        if slot in ("CUST", "CUST_FULL", "CUST_HON"):
            ent = self.entity("customer", "PERSON", "PRIVATE_INDIVIDUAL",
                              f"{c.first[0]} {c.surname[0]}")
            if slot == "CUST":
                return self._name_form(c.first[0], c.first[1], kind), ent
            if slot == "CUST_HON":
                # Maximal span (section 4): the honorific is part of the mention.
                return f"{'Ms.' if c.first[2] == 'F' else 'Mr.'} {c.surname[0]}", ent
            return self._name_form(f"{c.first[0]} {c.surname[0]}",
                                   f"{c.first[1]} {c.surname[1]}", kind), ent

        if slot == "AGENT":
            a = self.agent_first
            ent = self.entity("agent", "PERSON", "ORGANISATION_REP", a[0])
            return self._name_form(a[0], a[1], kind), ent

        if slot == "REL":
            r = self.relative
            ent = self.entity("relative", "PERSON", "PRIVATE_INDIVIDUAL",
                              f"{r.first[0]} {r.surname[0]}")
            return self._name_form(f"{r.first[0]} {r.surname[0]}",
                                   f"{r.first[1]} {r.surname[1]}", kind), ent

        if slot == "ORG":
            ent = self.entity("org", "ORG", "ORGANISATION", self.org[0])
            return (self.org[1] if kind == "si" else self.org[0]), ent

        if slot == "BRANCH":
            ent = self.entity("branch", "LOCATION", "PUBLIC_PLACE", self.branch[0])
            return self._name_form(self.branch[0], self.branch[1], kind), ent

        if slot == "NIC":
            if "nic" not in self.entities:
                value = make_nic(c.dob, c.first[2], rng, self.old_nic)
                self.entity("nic", "NIC", "PRIVATE_INDIVIDUAL", value, value=value)
            ent = self.entities["nic"]
            return (self._spoken(ent.value) if kind == "transcript"
                    else ent.value), ent

        if slot in ("PHONE", "HOTLINE"):
            key = slot.lower()
            if key not in self.entities:
                value = make_mobile(rng) if slot == "PHONE" else make_hotline(rng)
                role = "PRIVATE_INDIVIDUAL" if slot == "PHONE" else "ORGANISATION"
                self.entity(key, "PHONE", role, value, value=value)
            ent = self.entities[key]
            if kind == "transcript":
                return self._spoken(ent.value), ent
            if kind == "en" and rng.random() < 0.3:
                return f"+94 {ent.value[1:3]} {ent.value[3:6]} {ent.value[6:]}", ent
            return ent.value, ent

        if slot == "ACCOUNT":
            if "account" not in self.entities:
                value = make_account(rng)
                self.entity("account", "ACCOUNT", "PRIVATE_INDIVIDUAL", value, value=value)
            ent = self.entities["account"]
            return (self._spoken(ent.value) if kind == "transcript"
                    else ent.value), ent

        if slot == "DOB":
            ent = self.entity("dob", "DOB", "PRIVATE_INDIVIDUAL", c.dob.isoformat(),
                              value=c.dob.isoformat())
            return self._dob(c.dob, kind), ent

        if slot == "ADDRESS":
            num, street, town, sub = self.home
            number = f"{num}/{(num % 7) + 1}" if sub else str(num)
            ent = self.entity("address", "ADDRESS", "PRIVATE_INDIVIDUAL",
                              f"No. {number}, {street[0]}, {town[0]}")
            if kind == "si" or (kind == "transcript" and rng.random() < 0.3):
                return f"{number}, {street[1]}, {town[1]}", ent
            prefix = "No. " if kind != "transcript" or rng.random() < 0.5 else ""
            return f"{prefix}{number}, {street[0]}, {town[0]}", ent

        if slot == "EMAIL":
            if "email" not in self.entities:
                value = (f"{c.first[0].lower()}.{c.surname[0].lower()}"
                         f"{rng.randint(1, 99)}@{rng.choice(['gmail.com', 'yahoo.com'])}")
                self.entity("email", "EMAIL", "PRIVATE_INDIVIDUAL", value, value=value)
            return self.entities["email"].value, self.entities["email"]

        # Distractors: look like identifiers, are not personal data. Drawn once
        # per utterance so its three renderings state the same facts.
        if slot == "AMOUNT":
            n = self._utt.setdefault(slot, rng.choice([1500, 2500, 12000, 25000, 150000]))
            return (f"{n:,}" if kind != "transcript" else str(n)), None
        if slot == "COUNT":
            return self._utt.setdefault(slot, rng.choice(["deka", "thuna", "hathara"])), None
        if slot == "TXDATE":
            m, d = self._utt.setdefault(slot, (rng.randint(1, 12), rng.randint(1, 28)))
            if kind == "si":
                return f"{MONTHS_SI[m - 1]} {d}", None
            return f"{ordinal(d)} of {MONTHS_EN[m - 1]}", None
        raise KeyError(slot)

    # -- document assembly ---------------------------------------------------

    def render(self, template: str, doc_id: str, kind: str) -> str:
        """Fill a template, recording each span at its exact codepoint offset."""
        text, pos = [], 0
        for part in re.split(r"(\{[A-Z_]+\})", template):
            if part.startswith("{") and part.endswith("}"):
                self._noise = []
                surface, ent = self.resolve(part[1:-1], kind)
                if ent is not None:
                    script = script_of(surface, kind)
                    ent.observe(script, surface)
                    self.rows.append({
                        "ann_id": f"{self.rid}_pii{len(self.rows) + 1:03d}",
                        "recording": self.rid,
                        "doc_id": doc_id,
                        "field": "text",
                        "entity_id": ent.entity_id,
                        "label": ent.label,
                        "start_char": pos,
                        "end_char": pos + len(surface),
                        "surface": surface,
                        "script": script,
                        "role": ent.role,
                        "redact": expected_redact(ent.label, ent.role),
                        "annotation_source": "synthetic",
                        **({"noise": list(self._noise)} if self._noise else {}),
                    })
                part = surface
            elif self._lower:
                part = part.lower()
            text.append(part)
            pos += len(part)
        return "".join(text)

    def build(self) -> "Recording":
        rng = self.rng
        # A person needs both scripts somewhere (validator check [6]); the si
        # rendering guarantees it, so every topic that names a person is
        # rendered in all three utterance documents.
        topics = ["greet", "name"] + rng.sample(OPTIONAL_TOPICS, rng.randint(4, 8))
        topics[2:] = sorted(topics[2:], key=lambda _: rng.random())

        if self.split == "test":
            return self._build_test(topics)

        summary_parts = []
        for n, topic in enumerate(topics, start=1):
            self._utt = {}
            for kind, doc_id in (("transcript", f"{self.rid}_transcript_u{n:03d}_v1"),
                                 ("en", f"{self.rid}_c2_u{n:03d}_en_v1"),
                                 ("si", f"{self.rid}_c2_u{n:03d}_si_v1")):
                text = self.render(rng.choice(TEMPLATES[topic][kind]), doc_id, kind)
                self.docs.append({"doc_id": doc_id, "recording": self.rid,
                                  "kind": kind, "utterance": n, "topic": topic,
                                  "text": text})
            if TEMPLATES[topic]["summary"]:
                summary_parts.append(rng.choice(TEMPLATES[topic]["summary"]))

        # Summary offsets are computed on the joined text, so render it whole.
        self._utt = {}
        doc_id = f"{self.rid}_c2_summary_en_v1"
        text = self.render(" ".join(summary_parts), doc_id, "summary")
        self.docs.append({"doc_id": doc_id, "recording": self.rid, "kind": "summary",
                          "utterance": None, "topic": "summary", "text": text})
        return self

    def _build_test(self, topics: list[str]) -> "Recording":
        """Test split: held-out templates, question/answer turns, speech noise."""
        rng = self.rng
        n = 0
        for topic in topics:
            self._utt = {}
            variant = rng.choice(TEST_TEMPLATES[topic])
            turns = {k: v if isinstance(v, tuple) else (v,) for k, v in variant.items()}
            for i in range(len(turns["transcript"])):
                n += 1
                for kind, doc_id in (("transcript", f"{self.rid}_transcript_u{n:03d}_v1"),
                                     ("en", f"{self.rid}_c2_u{n:03d}_en_v1"),
                                     ("si", f"{self.rid}_c2_u{n:03d}_si_v1")):
                    template = turns[kind][i]
                    if kind == "transcript":
                        # Transcripts are speech: fillers and no capitalisation.
                        if rng.random() < 0.3:
                            template = rng.choice(_FILLERS) + template
                        self._lower = rng.random() < 0.3
                    text = self.render(template, doc_id, kind)
                    self._lower = False
                    self.docs.append({"doc_id": doc_id, "recording": self.rid,
                                      "kind": kind, "utterance": n, "topic": topic,
                                      "text": text})

        self._utt = {}
        doc_id = f"{self.rid}_c2_summary_en_v1"
        summary = " ".join(TEST_SUMMARY[t] for t in topics if t in TEST_SUMMARY)
        text = self.render(summary, doc_id, "summary")
        self.docs.append({"doc_id": doc_id, "recording": self.rid, "kind": "summary",
                          "utterance": None, "topic": "summary", "text": text})
        return self

    # -- speech noise (test split, transcripts only) ------------------------

    def _spoken(self, value: str) -> str:
        """Transcript rendering of a digit identifier; noisy in the test split."""
        rng = self.rng
        if not self.noisy:
            return spoken_digits(value, rng)

        digits = re.sub(r"\D", "", value)
        letter = value[-1].lower() if value[-1].isalpha() else ""
        if letter and rng.random() < 0.5:
            letter = ""                       # the V is not said, or not heard
            self._noise.append("nic_letter_dropped")
        if rng.random() < 0.08:
            k = rng.randrange(len(digits))    # a digit said twice
            digits = digits[:k] + digits[k] + digits[k:]
            self._noise.append("digit_repeated")

        forms = ["compact", "spaced", "grouped"]
        if not letter:
            forms += ["sinhala_words", "romanized_words", "english_words"]
        form = rng.choice(forms)
        if form == "compact":
            return digits + letter.upper()
        if form == "spaced":
            return " ".join(digits + letter)
        if form == "grouped":
            return _chunked(digits, rng) + (f" {letter}" if letter else "")
        if form == "english_words":
            words = [_EN_DIGIT_WORDS[int(d)] for d in digits]
            chunks = [" ".join(words[i : i + 3]) for i in range(0, len(words), 3)]
            return ", ".join(chunks)
        if form == "romanized_words":
            spoken = encode_number_words(digits, romanized=True)
        else:
            spoken = encode_number_words(digits, _random_groups(digits, rng),
                                         joined=rng.random() < 0.4)
        if digits[0] == "0" and rng.random() < 0.3:
            spoken = "zero " + spoken         # "zero binduwai ...": said twice
            self._noise.append("restatement")
        return spoken

    def _dob(self, d: dt.date, kind: str) -> str:
        rng = self.rng
        if not (self.noisy and kind == "transcript"):
            return dob_surface(d, kind, rng)
        month = MONTHS_EN[d.month - 1]
        suffix = ordinal(d.day)[len(str(d.day)):]
        if rng.random() < 0.25:
            # ASR writes the wrong ordinal suffix: "9rd", "21th".
            suffix = rng.choice([x for x in ("st", "nd", "rd", "th") if x != suffix])
            self._noise.append("ordinal_error")
        return rng.choice([f"{month} {d.day}{suffix} {d.year}",
                           f"{d.day}{suffix} {month} {d.year}",
                           f"{d.year} {month} {d.day}{suffix}",
                           f"{d.year} {MONTHS_SI[d.month - 1]} {d.day}"])

    def registry(self) -> dict:
        entities = []
        for ent in self.entities.values():
            entities.append({
                "entity_id": ent.entity_id,
                "canonical": ent.canonical,
                "label": ent.label,
                "role": ent.role,
                "pseudonym": self._pseudonym(ent),
                "redact": expected_redact(ent.label, ent.role),
                **({"value": ent.value} if ent.value else {}),
                "surfaces": ent.surfaces,
            })
        return {"recording": self.rid, "annotation_source": "synthetic",
                "entities": entities}


# split -> (recording id prefix, default size, default seed)
SPLITS = {"train": ("SYN_R", 50, 13), "test": ("SYN_T", 60, 29)}


def generate(n_recordings: int, seed: int = 13, split: str = "train"):
    """Yield (recording_id, docs, rows, registry) for each synthetic recording."""
    prefix = SPLITS[split][0]
    for i in range(1, n_recordings + 1):
        rid = f"{prefix}{i:04d}"
        rec = Recording(rid, random.Random(f"{seed}:{rid}"), split).build()
        yield rid, rec.docs, rec.rows, rec.registry()


def write_corpus(out_dir: Path, n_recordings: int, seed: int = 13,
                 split: str = "train") -> int:
    """Write <rid>.docs.jsonl / .pii.jsonl / .entities.json; returns span count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("SYN_*"):
        old.unlink()                     # a smaller rerun must not leave stale files
    total = 0
    for rid, docs, rows, registry in generate(n_recordings, seed, split):
        report = validate_recording(rows, {d["doc_id"]: d["text"] for d in docs},
                                    registry)
        if not report.ok:
            raise AssertionError(f"{rid} failed validation:\n" + "\n".join(report.errors))
        # newline="\n": identical bytes on every OS, so a regenerated set diffs clean.
        with open(out_dir / f"{rid}.docs.jsonl", "w", encoding="utf-8", newline="\n") as f:
            f.writelines(json.dumps(d, ensure_ascii=False) + "\n" for d in docs)
        with open(out_dir / f"{rid}.pii.jsonl", "w", encoding="utf-8", newline="\n") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        with open(out_dir / f"{rid}.entities.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)
        total += len(rows)
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "data" / "synthetic")
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    ap.add_argument("--recordings", type=int, help="default: 50 train, 60 test")
    ap.add_argument("--seed", type=int, help="default: 13 train, 29 test")
    args = ap.parse_args()
    for split in (["train", "test"] if args.split == "both" else [args.split]):
        _, size, seed = SPLITS[split]
        n = args.recordings or size
        spans = write_corpus(args.out / split, n, args.seed or seed, split)
        print(f"{split}: wrote {n} recordings, {spans} spans -> {args.out / split}")


if __name__ == "__main__":
    main()
