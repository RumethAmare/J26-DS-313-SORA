#!/usr/bin/env python3
"""
make_recording_scripts.py -- generate role-play scripts for new SORA recordings.

Why: the fine-tuned models invent "1 2 3 4 5" when someone reads a number.
26% of the long numbers in the current transcripts contain a counting run
(0771234567, 198512345678 -- the old scripts used placeholder numbers),
against 0.5% for random digits, and few recordings read numbers one digit at
a time with pauses. These scripts fix both: every number is random and has no
counting run or repeated digits, and each script tells the speakers exactly
how to say it -- English or Sinhala digit words, grouped, run together, or
one digit at a time with pauses.

Each script is a realistic Sinhala-English call (Sinhala in Sinhala script,
English in Latin script -- the majority convention in the corpus) and comes as:

  <id>_<scenario>.md          what the two speakers read
  <id>_<scenario>.spec.json   draft input for SORA_Dataset/tools/build_annotations.py:
                              utterances with numbers written as DIGITS (the
                              transcript convention), C4 entities pre-registered,
                              Tamil words marked OTHER. Must be corrected to what
                              was actually said before use.

Usage:
    python make_recording_scripts.py                 # 3 scripts per scenario, seed 2026
    python make_recording_scripts.py --per-scenario 5 --seed 7 --start 101
"""
import argparse
import csv
import json
import os
import random
import re

C1_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(C1_ROOT, "data", "recording_scripts")

EN = "zero one two three four five six seven eight nine".split()
SI = "බිංදුව එක දෙක තුන හතර පහ හය හත අට නවය".split()

# How a number is said. "slow" is the case that fails live: one digit at a
# time with a pause after each.
STYLES = {
    "en_grouped": "English digits, a short pause between groups",
    "en_slow": "English digits ONE AT A TIME, pause after each (…)",
    "en_fast": "English digits run together, no pauses",
    "si_grouped": "Sinhala digits, a short pause between groups",
    "si_slow": "Sinhala digits ONE AT A TIME, pause after each (…)",
}

# --- names, places --------------------------------------------------------------
MALE = ["Kasun Perera", "Nuwan Jayasinghe", "Chamara Silva", "Dilan Fernando",
        "Tharindu Wickramasinghe", "Pradeep Kumara", "Sahan Rajapaksha",
        "Ruwan Bandara", "Lahiru Dissanayake", "Isuru Herath", "Dinesh Gunasekara",
        "Charith Abeyratne"]
FEMALE = ["Nethmi Herath", "Dilini Senanayake", "Sachini Jayawardena",
          "Hiruni Abeysekara", "Kaveesha Ranasinghe", "Tharushi Weerasinghe",
          "Ishani Karunaratne", "Madhavi Liyanage", "Sanduni Rathnayake",
          "Piumi Edirisinghe", "Rashmi Wijesekara", "Oshadi Samarasinghe"]
TAMIL_M = ["Suresh Kumar", "Arun Sivakumar", "Vimal Shanmugam", "Rajan Thevarajah"]
TAMIL_F = ["Priya Nadarajah", "Kavitha Rajendran", "Lakshmi Sritharan", "Meena Ganeshan"]
TOWNS = ["Dehiwala", "Maharagama", "Kadawatha", "Nugegoda", "Kandy", "Galle",
         "Kurunegala", "Malabe", "Battaramulla", "Negombo", "Panadura", "Gampaha",
         "Kottawa", "Kelaniya", "Moratuwa", "Kiribathgoda", "Homagama", "Matara"]
ROADS = ["Galle Road", "High Level Road", "Kandy Road", "Temple Road", "Station Road",
         "Lake Road", "Hospital Road", "School Lane", "Old Road", "Church Road",
         "Main Street", "Park Road"]
LANDMARKS = ["Keells super", "Cargills Food City", "bus stand", "Sathosa",
             "post office", "Bank of Ceylon branch", "temple"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
EMAIL_HOSTS = [("gmail.com", "gmail dot com"), ("yahoo.com", "yahoo dot com"),
               ("hotmail.com", "hotmail dot com")]


# --- random numbers without counting runs -----------------------------------------

def clean(d):
    """No run of 3 ascending/descending digits (123, 987) and no digit 3x in a row."""
    for a, b, c in zip(d, d[1:], d[2:]):
        x, y, z = int(a), int(b), int(c)
        if y - x == z - y == 1 or x - y == y - z == 1 or x == y == z:
            return False
    return True


def rdigits(rng, n, first=None):
    while True:
        d = (first or "") + "".join(rng.choice("0123456789") for _ in range(n - len(first or "")))
        if clean(d):
            return d


def say_digits(groups, style, rng):
    """Spoken form of digit groups, e.g. ['077', '419', '8206']."""
    lang = SI if style.startswith("si") else EN
    word = lambda c: ("oh" if c == "0" and style == "en_grouped" and rng.random() < 0.3 else lang[int(c)])
    if style.endswith("slow"):
        return " … ".join(word(c) for g in groups for c in g)
    if style == "en_fast":
        return " ".join(word(c) for g in groups for c in g)
    return ", ".join(" ".join(word(c) for c in g) for g in groups)


def say_letters(s):
    return " ".join(s)


class Num:
    """A number with how it is SAID (per style) and how it is WRITTEN in the transcript."""

    def __init__(self, written, groups, prefix_letters="", suffix_letters="", kind="ACCOUNT"):
        self.written, self.groups = written, groups
        self.pre, self.suf, self.kind = prefix_letters, suffix_letters, kind

    def spoken(self, style, rng):
        parts = []
        if self.pre:
            parts.append(say_letters(self.pre))
        parts.append(say_digits(self.groups, style, rng))
        if self.suf:
            parts.append(say_letters(self.suf))
        return ", ".join(parts)


def phone(rng):
    d = rdigits(rng, 10, rng.choice(["070", "071", "072", "074", "075", "076", "077", "078"]))
    return Num(f"{d[:3]} {d[3:6]} {d[6:]}", [d[:3], d[3:6], d[6:]], kind="PHONE")


def nic(rng):
    if rng.random() < 0.5:                       # new 12-digit format: YYYY DDD SSSS C
        year = str(rng.randint(1965, 2004))
        day = rng.randint(1, 365) + (500 if rng.random() < 0.5 else 0)
        while True:
            d = year + f"{day:03d}" + "".join(rng.choice("0123456789") for _ in range(5))
            if clean(d):
                return Num(d, [d[:4], d[4:7], d[7:]], kind="NIC")
    while True:                                   # old 9 digits + V
        d = f"{rng.randint(65, 99)}{rng.randint(1, 865):03d}" + "".join(rng.choice("0123456789") for _ in range(4))
        if clean(d):
            return Num(d + "V", [d[:3], d[3:6], d[6:]], suffix_letters="V", kind="NIC")


def code(rng, letters, n, kind="ACCOUNT", dash=True):
    d = rdigits(rng, n)
    return Num(f"{letters}{'-' if dash else ''}{d}", [d[:len(d) // 2], d[len(d) // 2:]],
               prefix_letters=letters, kind=kind)


def plain(rng, n, groups, kind="ACCOUNT"):
    d = rdigits(rng, n)
    gs, i = [], 0
    for g in groups:
        gs.append(d[i:i + g])
        i += g
    return Num(d, gs, kind=kind)


def vehicle(rng):
    prov = rng.choice(["WP", "CP", "SP", "NW"])
    letters = "".join(rng.choice("ABCDEGHJKLMNPQRSTUVWXYZ") for _ in range(3))
    d = rdigits(rng, 4)
    return Num(f"{prov} {letters}-{d}", [d], prefix_letters=prov + letters, kind="ACCOUNT")


def email(rng, name):
    first, last = name.lower().split()[0], name.lower().split()[-1]
    nn = rdigits(rng, 2)
    host, host_said = rng.choice(EMAIL_HOSTS)
    written = f"{first}.{last}{nn}@{host}"
    said = f"{first} dot {last} {EN[int(nn[0])]} {EN[int(nn[1])]} at {host_said}"
    return written, said


def amount(rng, lo, hi, step=50):
    v = rng.randrange(lo, hi, step)
    return f"{v:,}"


def date(rng):
    return f"{rng.choice(MONTHS)} {rng.randint(1, 28)}"


def dob(rng):
    return f"{rng.choice(MONTHS)} {rng.randint(1, 28)}, {rng.randint(1968, 2003)}"


def address(rng):
    no = str(rng.randint(3, 289))
    if rng.random() < 0.35:
        no += f"/{rng.randint(1, 9)}"
    return f"No. {no}, {rng.choice(ROADS)}, {rng.choice(TOWNS)}"


def clock(rng, lo, hi):
    return f"{rng.randint(lo, hi)}.{rng.choice(['00', '15', '30', '45'])}"


# --- scenarios ----------------------------------------------------------------------
# A = the organisation's agent, B = the customer. {x} = slot said as written;
# {#x} = number slot said in the customer's style, {#x|alt} = said back in a
# second style (the other speaker repeating it). {hon} = how A addresses B,
# {ahon} = how B addresses A.

SCENARIOS = {
    "bank_card": {
        "title": "Bank card centre — disputed card transactions",
        "orgs": ["Commercial Bank", "Sampath Bank", "Hatton National Bank", "People's Bank", "Bank of Ceylon"],
        "lines": [
            ("A", "Good morning, {org} card center එකෙන්, මම {agent}. කොහොමද මට ඔයාට help කරන්න පුළුවන්?"),
            ("B", "Good morning {ahon}. මගේ credit card එකෙන් මට නොදන්න transactions දෙකක් වෙලා තියෙනවා."),
            ("A", "හරි {hon}, කලබල වෙන්න එපා. මුලින්ම ඔයාගේ full name එක කියන්න පුළුවන්ද?"),
            ("B", "මගේ නම {cust}."),
            ("A", "ස්තූතියි. Verify කරන්න ඔයාගේ NIC number එක කියන්න."),
            ("B", "NIC එක {#nic}."),
            ("A", "මම ආපහු කියන්නම්, {#nic|alt}. ඒක හරිද?"),
            ("B", "ඔව් හරි."),
            ("A", "Card එකේ අන්තිම digits හතර?"),
            ("B", "{#card4}."),
            ("A", "Registered mobile number එක?"),
            ("B", "Mobile number එක {#phone}."),
            ("A", "ස්තූතියි {hon}. කවදද මේ transactions වුණේ?"),
            ("B", "{date} දවසේ. එකක් rupees {amount1}, අනිත් එක rupees {amount2}."),
            ("A", "ඒ දෙකම online purchases විදිහට තියෙන්නේ. ඔයා එහෙම මොකුත් ගත්තේ නැද්ද?"),
            ("B", "නෑ, මම මුකුත් ගත්තේ නෑ. Card එක මගේ ළඟම තියෙනවා."),
            ("A", "හරි, එහෙනම් අපි card එක දැන්ම block කරනවා. අලුත් card එකක් දවස් හතක් ඇතුළත ගෙදරට එවනවා."),
            ("B", "හරි. Address එක තවමත් {address} තමයි."),
            ("A", "හරි, confirm කළා. මේ complaint එකට reference number එක ලියාගන්න, {#ref}."),
            ("B", "ආයෙත් කියන්න පුළුවන්ද? {#ref|alt} නේද?"),
            ("A", "ඔව් හරි. Working days දහහතරක් ඇතුළත investigate කරලා SMS එකක් එවනවා."),
            ("B", "හරි හරි, ගොඩක් ස්තූතියි {ahon}."),
            ("A", "ප්‍රශ්නයක් නෑ {hon}. තව මොනවා හරි help එකක් ඕනෙද?"),
            ("B", "නෑ, එච්චරයි. Thank you."),
            ("A", "{org} එක තෝරගත්තට ස්තූතියි. සුභ දවසක්."),
        ],
        "slots": lambda r, c: {"nic": nic(r), "card4": plain(r, 4, [4]), "phone": phone(r),
                               "ref": code(r, r.choice(["CMP", "DSP", "CRD"]), 6),
                               "date": date(r), "amount1": amount(r, 2500, 48000),
                               "amount2": amount(r, 900, 15000), "address": address(r)},
        "action": "Bank blocks the card, sends a new one within 7 days, and investigates the dispute within 14 working days.",
    },
    "hospital_channelling": {
        "title": "Hospital channelling — booking a doctor appointment",
        "orgs": ["Asiri Hospital", "Nawaloka Hospital", "Lanka Hospitals", "Durdans Hospital", "Hemas Hospital"],
        "lines": [
            ("A", "{org} channelling center එකෙන්, මම {agent} කතා කරන්නේ."),
            ("B", "Hello, මට doctor {doctor} ගේ appointment එකක් ගන්න ඕනේ. Cardiologist කෙනෙක්."),
            ("A", "හරි {hon}. මොන දවසටද ඕනේ?"),
            ("B", "පුළුවන් නම් ලබන {weekday}."),
            ("A", "{weekday} doctor ඉන්නේ හවස {time1} ඉඳන්. දැනට free තියෙන්නේ number {qno}."),
            ("B", "ඒක හරි. ඒක book කරන්න."),
            ("A", "Patient ගේ නම කියන්න."),
            ("B", "{patient}. මගේ අම්මා."),
            ("A", "වයස කීයද?"),
            ("B", "{age}."),
            ("A", "Contact number එක?"),
            ("B", "{#phone}."),
            ("A", "මම ආපහු කියන්නම්, {#phone|alt}. හරිද?"),
            ("B", "ඔව් හරි."),
            ("A", "Registration එකට patient ගේ NIC number එකත් ඕනේ."),
            ("B", "{#nic}."),
            ("A", "Channelling fee එක rupees {amount1}. Hospital charges එක්ක rupees {amount2} වෙනවා."),
            ("B", "Card එකෙන් pay කරන්න පුළුවන්ද?"),
            ("A", "ඔව්, counter එකේදී card එකෙන් pay කරන්න පුළුවන්. Appointment reference එක {#ref}."),
            ("B", "{#ref|alt}. හරි, ලියාගත්තා."),
            ("A", "විනාඩි තිහකට කලින් ඇවිත් reception එකේ report කරන්න. පරණ reports තියෙනවා නම් අරන් එන්න."),
            ("B", "හරි. ECG report එකයි blood test එකයි තියෙනවා."),
            ("A", "ඒක හොඳයි. Appointment එක confirm කරලා SMS එකක් එවන්නම්."),
            ("B", "ගොඩක් ස්තූතියි."),
            ("A", "ස්තූතියි {hon}, සුභ දවසක්."),
        ],
        "slots": lambda r, c: {"phone": phone(r), "nic": nic(r), "ref": code(r, "CH", 6),
                               "doctor": r.choice(["Fernando", "Wijeratne", "Senaratne", "Kumarasinghe"]),
                               "weekday": r.choice(WEEKDAYS), "time1": clock(r, 3, 6),
                               "qno": str(r.randint(4, 28)),
                               "patient": r.choice(FEMALE + TAMIL_F), "age": str(r.randint(55, 79)),
                               "amount1": amount(r, 2000, 4500), "amount2": amount(r, 3500, 6500)},
        "action": "Appointment booked; arrive 30 minutes early with previous reports; SMS confirmation.",
    },
    "telecom_bill": {
        "title": "Mobile operator — disputed bill and data add-on",
        "orgs": ["Dialog", "Mobitel", "Hutch", "Airtel"],
        "lines": [
            ("A", "{org} customer care, මම {agent}. ඔයාට කොහොමද help කරන්න පුළුවන්?"),
            ("B", "Hi, මගේ මේ මාසේ bill එක හරිම වැඩියි. ඒ ගැන check කරන්න ඕනේ."),
            ("A", "හරි {hon}. Connection number එක කියන්න."),
            ("B", "{#phone}."),
            ("A", "Account එක කාගේ නමින්ද?"),
            ("B", "{cust}."),
            ("A", "Verify කරන්න NIC number එක කියන්න පුළුවන්ද?"),
            ("B", "{#nic}."),
            ("A", "ස්තූතියි. මේ මාසේ bill එක rupees {amount1}. සාමාන්‍යයෙන් එන්නේ rupees {amount2} වගේ නේද?"),
            ("B", "ඔව්, ඒක තමයි මට තේරෙන්නේ නැත්තේ."),
            ("A", "{date} දවසේ data add-on එකක් activate වෙලා තියෙනවා. GB {gb} ක package එකක්."),
            ("B", "මම එහෙම එකක් activate කළේ නෑ."),
            ("A", "සමහර වෙලාවට promotion SMS එකකට reply කළාම activate වෙනවා. අපි ඒ charge එක reverse කරන්නම්."),
            ("B", "හරි, ඒක හොඳයි. ඊළඟ bill එකේ adjust වෙනවද?"),
            ("A", "ඔව්, ඊළඟ bill එකේ rupees {amount3} credit එකක් විදිහට පෙන්නයි. Reference number එක {#ref}."),
            ("B", "ආයෙත් කියන්න, {#ref|alt}?"),
            ("A", "ඔව් හරි. ඔයාගේ email එකට e-bill එක එවන්නද?"),
            ("B", "ඔව්, email එක {email_said}."),
            ("A", "හරි, ඒක update කළා. Add-on activate වෙන එක block කරන්නත් පුළුවන්. කරන්නද?"),
            ("B", "ඔව්, කරන්න. තව මොකුත් නෑ, ස්තූතියි."),
            ("A", "{org} එකට call කළාට ස්තූතියි. සුභ දවසක්."),
        ],
        "slots": lambda r, c: {"phone": phone(r), "nic": nic(r), "ref": code(r, "TKT", 6),
                               "amount1": amount(r, 4500, 12000), "amount2": amount(r, 1500, 3500),
                               "amount3": amount(r, 1200, 4000), "date": date(r),
                               "gb": str(r.choice([10, 15, 25, 40, 50]))},
        "action": "Operator reverses the add-on charge as a credit on the next bill, switches to e-bill, blocks add-on activation.",
    },
    "insurance_claim": {
        "title": "Motor insurance — reporting an accident claim",
        "orgs": ["Ceylinco Insurance", "Allianz Insurance", "Softlogic Life", "Fairfirst Insurance", "LOLC Insurance"],
        "lines": [
            ("A", "{org} motor claims, මම {agent}."),
            ("B", "Hello, මගේ car එක accident එකකට පැටලුණා. Claim එකක් report කරන්න ඕනේ."),
            ("A", "අයියෝ, කාටවත් තුවාල වුණාද?"),
            ("B", "නෑ, කාටවත් කිසි දෙයක් නෑ. Front bumper එකයි headlight එකයි කැඩුණා."),
            ("A", "හරි. ඔයාගේ නම කියන්න."),
            ("B", "{cust}."),
            ("A", "Vehicle number එක?"),
            ("B", "{#vehicle}."),
            ("A", "Policy number එක ළඟ තියෙනවද?"),
            ("B", "ඔව්, {#policy}."),
            ("A", "මම ආපහු කියන්නම්, {#policy|alt}."),
            ("B", "ඔව්, ඒක තමයි."),
            ("A", "කොහෙද accident එක වුණේ?"),
            ("B", "{town} junction එක ළඟ, අද උදේ {time1} විතර."),
            ("A", "අනිත් වාහනයේ number එක ගත්තද?"),
            ("B", "ඔව්, ඒක {#vehicle2}."),
            ("A", "Police complaint එකක් දැම්මද?"),
            ("B", "ඔව්, {town} police එකේ. Complaint number එක {#ref}."),
            ("A", "හොඳයි. ඔයාගේ contact number එක?"),
            ("B", "{#phone}."),
            ("A", "අපේ assessor කෙනෙක් පැය දෙකක් ඇතුළත call කරයි. වාහනය දැන් කොහෙද තියෙන්නේ?"),
            ("B", "තවම junction එක ළඟ පැත්තකට කරලා තියෙන්නේ."),
            ("A", "හරි. Claim reference number එක {#claim}. Assessor එනකම් photos ටිකක් ගන්න."),
            ("B", "{#claim|alt}. හරි, ලියාගත්තා. ස්තූතියි."),
            ("A", "Take care {hon}, අපි ඉක්මනටම contact කරන්නම්."),
        ],
        "slots": lambda r, c: {"vehicle": vehicle(r), "vehicle2": vehicle(r), "phone": phone(r),
                               "policy": code(r, "MV", 8, dash=False), "ref": plain(r, 6, [3, 3]),
                               "claim": code(r, "CLM", 6), "town": r.choice(TOWNS),
                               "time1": clock(r, 7, 10)},
        "action": "Assessor calls within 2 hours; customer takes photos of the vehicle meanwhile.",
    },
    "university_results": {
        "title": "University student services — results not released",
        "orgs": ["SLIIT", "University of Moratuwa", "University of Colombo", "NSBM", "University of Kelaniya"],
        "lines": [
            ("A", "Good morning, {org} student services. මම {agent}."),
            ("B", "Good morning {ahon}. මගේ exam results portal එකේ පේන්නේ නෑ."),
            ("A", "හරි. ඔයාගේ student ID එක කියන්න."),
            ("B", "{#student_id}."),
            ("A", "මම ආපහු කියන්නම්, {#student_id|alt}. හරිද?"),
            ("B", "ඔව්. මගේ නම {cust}."),
            ("A", "කොයි semester එකද?"),
            ("B", "Year {year_no} semester {sem}."),
            ("A", "System එකේ semester fee එකේ balance එකක් පෙන්නනවා, rupees {amount1}."),
            ("B", "මම ඒක {date} දවසේ pay කළා. Receipt එක ළඟ තියෙනවා."),
            ("A", "Receipt number එක කියන්න පුළුවන්ද?"),
            ("B", "{#ref}."),
            ("A", "ආ හරි, payment එක ආවා, ඒත් අපේ system එකට update වෙලා නෑ. මම ඒක finance එකට යවන්නම්."),
            ("B", "කවදද results පේන්නේ?"),
            ("A", "Working days තුනක් ඇතුළත. University email එකට message එකක් එයි."),
            ("B", "මගේ personal email එකටත් එවන්න පුළුවන්ද? {email_said}."),
            ("A", "හරි, ඒකත් add කළා. Contact number එකත් කියන්න."),
            ("B", "{#phone}."),
            ("A", "Ticket number එක {#ticket}. ආයෙත් call කරනවා නම් ඒක කියන්න."),
            ("B", "හරි, ස්තූතියි {ahon}."),
            ("A", "සුභ දවසක්."),
        ],
        "slots": lambda r, c: {"student_id": code(r, r.choice(["IT", "EN", "BM"]), 8, kind="ACCOUNT", dash=False),
                               "year_no": str(r.randint(1, 4)), "sem": str(r.randint(1, 2)),
                               "amount1": amount(r, 15000, 95000, 500), "date": date(r),
                               "ref": plain(r, 8, [4, 4]), "phone": phone(r),
                               "ticket": code(r, "SS", 6)},
        "action": "Student services forwards the payment to finance; results visible within 3 working days.",
    },
    "courier_delivery": {
        "title": "Courier — missed parcel delivery",
        "orgs": ["Pronto Lanka", "DHL Sri Lanka", "Domex", "Koombiyo", "Prompt Xpress"],
        "lines": [
            ("A", "{org} customer service, මම {agent}."),
            ("B", "Hello, මගේ parcel එක තවම ආවේ නෑ. Track කරලා බලන්න පුළුවන්ද?"),
            ("A", "Tracking number එක කියන්න."),
            ("B", "{#tracking}."),
            ("A", "මම ආපහු කියන්නම්, {#tracking|alt}."),
            ("B", "ඔව් හරි."),
            ("A", "Parcel එක දැන් තියෙන්නේ {town} hub එකේ. ඊයේ deliver කරන්න ගියාම ගෙදර කවුරුත් හිටියේ නෑ කියලා තියෙනවා."),
            ("B", "අනේ, මම මුළු දවසම ගෙදර හිටියා."),
            ("A", "සමාවෙන්න {hon}. Delivery address එක confirm කරමු."),
            ("B", "{address}."),
            ("A", "ළඟ landmark එකක් තියෙනවද?"),
            ("B", "ඔව්, {landmark} එක ළඟ, දෙවෙනි පාරේ."),
            ("A", "Receiver ගේ නම?"),
            ("B", "{cust}."),
            ("A", "Contact number එක?"),
            ("B", "{#phone}. Rider ට එන්න කලින් call කරන්න කියන්න."),
            ("A", "හරි, ලබන {weekday} උදේ {time1} ත් {time2} ත් අතර deliver කරනවා. Cash on delivery amount එක rupees {amount1}."),
            ("B", "Card එකෙන් pay කරන්න පුළුවන්ද?"),
            ("A", "ඔව්, rider ළඟ card machine එකක් තියෙනවා. Complaint reference එක {#ref}."),
            ("B", "හරි, ගොඩක් ස්තූතියි."),
            ("A", "Thank you, සුභ දවසක්."),
        ],
        "slots": lambda r, c: {"tracking": code(r, "PX", 9, kind="ACCOUNT", dash=False),
                               "town": r.choice(TOWNS), "address": address(r),
                               "landmark": r.choice(LANDMARKS), "phone": phone(r),
                               "weekday": r.choice(WEEKDAYS), "time1": clock(r, 8, 9),
                               "time2": clock(r, 11, 12), "amount1": amount(r, 1500, 18000),
                               "ref": code(r, "CS", 6)},
        "action": "Redelivery booked with a call before arrival; card payment accepted on delivery.",
    },
    "electricity_outage": {
        "title": "Electricity board — power outage and high bill",
        "orgs": ["Ceylon Electricity Board", "LECO"],
        "lines": [
            ("A", "{org} hotline, මම {agent}."),
            ("B", "Hello, අපේ ගෙදර current නෑ, පැය තුනක් විතර තිස්සේ."),
            ("A", "හරි {hon}. Electricity account number එක කියන්න."),
            ("B", "{#acc}."),
            ("A", "මම ආපහු කියන්නම්, {#acc|alt}. Account එක {cust} ගේ නමින්ද?"),
            ("B", "ඔව්, ඒ මම."),
            ("A", "Address එක?"),
            ("B", "{address}."),
            ("A", "ඒ area එකේ transformer එකක fault එකක් report වෙලා තියෙනවා. Team එක දැනටමත් ගිහින්."),
            ("B", "කීයටද ආයෙත් current එන්නේ?"),
            ("A", "හවස {time1} වෙනකොට එයි කියලා හිතනවා."),
            ("B", "හරි. තව එකක්, මගේ අන්තිම bill එකේ units {units} ක් පෙන්නනවා. සාමාන්‍යයෙන් {units2} වගේ."),
            ("A", "Meter reading එක check කරන්න request එකක් දාන්නම්. Contact number එක?"),
            ("B", "{#phone}."),
            ("A", "හරි. Breakdown reference එක {#ref}. Bill complaint එකේ reference එක {#ref2}."),
            ("B", "{#ref|alt} සහ {#ref2|alt}. හරි, ස්තූතියි."),
            ("A", "Inconvenience එකට සමාවෙන්න, සුභ දවසක්."),
        ],
        "slots": lambda r, c: {"acc": plain(r, 10, [2, 4, 4]), "address": address(r),
                               "time1": clock(r, 4, 7), "units": str(r.randint(260, 480)),
                               "units2": str(r.randint(90, 180)), "phone": phone(r),
                               "ref": code(r, "BD", 5), "ref2": code(r, "BC", 5)},
        "action": "Repair team on site; meter reading check requested for the high bill.",
    },
    "birth_certificate": {
        "title": "Divisional secretariat — birth certificate copy",
        "orgs": ["Divisional Secretariat"],
        "lines": [
            ("A", "{town} {org} එකේ registration counter එක. මොකක්ද ඕනේ?"),
            ("B", "මට උප්පැන්න සහතිකයේ copy එකක් ගන්න ඕනේ. Passport එකට."),
            ("A", "හරි. ඔයාගේ නම?"),
            ("B", "{cust}."),
            ("A", "උපන් දිනය?"),
            ("B", "{dob}."),
            ("A", "උපන් තැන?"),
            ("B", "{town2} hospital එකේ."),
            ("A", "Birth certificate number එක දන්නවද?"),
            ("B", "ඔව්, {#birth_no}."),
            ("A", "අම්මගේ නම?"),
            ("B", "{mother}."),
            ("A", "ඔයාගේ NIC number එක?"),
            ("B", "{#nic}."),
            ("A", "මම ආපහු කියන්නම්, {#nic|alt}."),
            ("B", "ඔව් හරි."),
            ("A", "Application fee එක rupees {amount1}. Copy එක working days පහකින් ලැබෙයි."),
            ("B", "Post එකෙන් එවන්න පුළුවන්ද?"),
            ("A", "ඔව්, postal charges rupees {amount2} එකතු වෙනවා. Address එක කියන්න."),
            ("B", "{address}."),
            ("A", "Contact number එක?"),
            ("B", "{#phone}."),
            ("A", "Application number එක {#ref}. Status එක check කරන්න ඒක ඕනේ."),
            ("B", "{#ref|alt}. හරි, ස්තූතියි."),
            ("A", "ස්තූතියි."),
        ],
        "slots": lambda r, c: {"town": r.choice(TOWNS), "town2": r.choice(TOWNS), "dob": dob(r),
                               "birth_no": plain(r, 4, [4]), "mother": r.choice(FEMALE),
                               "nic": nic(r), "amount1": amount(r, 100, 500, 50),
                               "amount2": amount(r, 150, 400, 50), "address": address(r),
                               "phone": phone(r), "ref": code(r, "DS", 6)},
        "action": "Certified copy posted to the applicant within 5 working days.",
    },
}

NUM_RE = re.compile(r"\{#(\w+)(\|alt)?\}")
SLOT_RE = re.compile(r"\{(\w+)\}")


def render(scn, rng, script_id):
    gender = rng.choice("MF")
    tamil = rng.random() < 0.25
    pool = (TAMIL_M if gender == "M" else TAMIL_F) if tamil else (MALE if gender == "M" else FEMALE)
    cust = rng.choice(pool)
    a_gender = rng.choice("MF")
    agent = rng.choice(MALE if a_gender == "M" else FEMALE).split()[0]
    org = rng.choice(scn["orgs"])
    slots = scn["slots"](rng, cust)
    slots.update({"cust": cust, "agent": agent, "org": org,
                  "hon": "sir" if gender == "M" else "madam",
                  "ahon": "sir" if a_gender == "M" else "miss"})
    if any("email_said" in t for _, t in scn["lines"]):
        slots["email"], slots["email_said"] = email(rng, cust)

    # number styles: the customer gets one, the repeat-back another; every
    # script has at least one digit-at-a-time ("slow") number.
    styles = list(STYLES)
    main, alt = rng.sample(styles, 2)
    if "slow" not in main and "slow" not in alt:
        alt = rng.choice(["en_slow", "si_slow"])
    used = set()

    spoken_lines, written_lines = [], []
    for spk, tpl in scn["lines"]:
        def num(m):
            n = slots[m.group(1)]
            st = alt if m.group(2) else main
            used.add(st)
            return n.spoken(st, rng)
        said = NUM_RE.sub(num, tpl)
        said = SLOT_RE.sub(lambda m: slots["email_said"] if m.group(1) == "email_said" else str(slots[m.group(1)]), said)
        wrote = NUM_RE.sub(lambda m: slots[m.group(1)].written, tpl)
        wrote = SLOT_RE.sub(lambda m: slots["email"] if m.group(1) == "email_said" else str(slots[m.group(1)]), wrote)
        spoken_lines.append((spk, said))
        written_lines.append((spk, wrote))

    if tamil:   # a Tamil-speaking customer greets and thanks in Tamil
        i_b = next(i for i, (s, _) in enumerate(spoken_lines) if s == "B")
        spoken_lines[i_b] = ("B", "වනක්කම්. " + spoken_lines[i_b][1])
        written_lines[i_b] = ("B", "වනක්කම්. " + written_lines[i_b][1])
        j_b = max(i for i, (s, _) in enumerate(spoken_lines) if s == "B")
        spoken_lines[j_b] = ("B", spoken_lines[j_b][1] + " රොම්බ නන්ද්‍රි.")
        written_lines[j_b] = ("B", written_lines[j_b][1] + " රොම්බ නන්ද්‍රි.")

    return {"script_id": script_id, "scenario": scn["title"], "org": org, "customer": cust,
            "customer_gender": gender, "agent": agent, "agent_gender": a_gender,
            "tamil_customer": tamil, "styles": sorted(used), "slots": slots,
            "spoken": spoken_lines, "written": written_lines, "action": scn["action"]}


def entities(s):
    ents = [{"canonical": s["customer"], "label": "PERSON", "role": "PRIVATE_INDIVIDUAL",
             "redact": True, "surfaces": [{"script": "LATIN", "form": s["customer"]},
                                          {"script": "LATIN", "form": s["customer"].split()[0]}]},
            {"canonical": s["agent"], "label": "PERSON", "role": "ORGANISATION_REP",
             "redact": True, "surfaces": [{"script": "LATIN", "form": s["agent"]}]},
            {"canonical": s["org"], "label": "ORG", "role": "ORGANISATION", "redact": False,
             "surfaces": [{"script": "LATIN", "form": s["org"]}]}]
    for k, v in s["slots"].items():
        if isinstance(v, Num):
            ents.append({"canonical": v.written, "label": v.kind, "role": None, "redact": True,
                         "surfaces": [{"script": "LATIN", "form": v.written}]})
        elif k in ("patient", "mother"):
            ents.append({"canonical": v, "label": "PERSON", "role": "PRIVATE_INDIVIDUAL",
                         "redact": True, "surfaces": [{"script": "LATIN", "form": v}]})
        elif k == "address":
            ents.append({"canonical": v, "label": "ADDRESS", "role": None, "redact": True,
                         "surfaces": [{"script": "LATIN", "form": v}]})
        elif k == "dob":
            ents.append({"canonical": v, "label": "DOB", "role": None, "redact": True,
                         "surfaces": [{"script": "LATIN", "form": v}]})
        elif k == "email":
            ents.append({"canonical": v, "label": "EMAIL", "role": None, "redact": True,
                         "surfaces": [{"script": "LATIN", "form": v}]})
    return ents


def write_script(s, out_dir, slug):
    sid = s["script_id"]
    n_words = sum(len(t.split()) for _, t in s["spoken"])
    minutes = n_words / 110                       # conversational pace incl. pauses
    roles = {"A": f"{s['org']} — {s['agent']} ({'man' if s['agent_gender'] == 'M' else 'woman'})",
             "B": f"Customer — {s['customer']} ({'man' if s['customer_gender'] == 'M' else 'woman'})"}
    L = [f"# {sid} — {s['scenario']}\n\n",
         f"**Speaker A:** {roles['A']}  \n**Speaker B:** {roles['B']}  \n"
         f"**Length:** ~{minutes:.1f} min · {len(s['spoken'])} turns\n\n",
         "**How numbers are said in this script:**\n"]
    for st in s["styles"]:
        L.append(f"- {STYLES[st]}\n")
    L.append("\nRead numbers **exactly as written** — word by word. `…` means a short pause "
             "(about half a second). Everything else: speak naturally; small changes are fine.\n\n"
             "---\n\n")
    for i, (spk, said) in enumerate(s["spoken"], 1):
        L.append(f"**{i}. {spk}:** {said}\n\n")
    if s["tamil_customer"]:
        L.append("---\n\n_Speaker B greets and thanks in Tamil (වනක්කම් / රොම්බ නන්ද්‍රි)._\n")
    path = os.path.join(out_dir, f"{sid}_{slug}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(L)

    spec = {
        "recording_id": "J26DS313_R____",
        "_note": ("DRAFT generated from the recording script. Correct every utterance to what "
                  "was ACTUALLY said, fill clean_en / clean_si and the summary, assign the real "
                  "recording id, then run tools/c1/realign_mms.py and tools/build_annotations.py."),
        "_script": f"{sid}_{slug}.md",
        "utterances": [{"speaker": "S1" if spk == "A" else "S2", "text": t,
                        "clean_en": "", "clean_si": ""} for spk, t in s["written"]],
        "summary": {"en": "", "si": ""},
        "action_items": [{"intent": s["action"], "owner": s["org"], "receiver": s["customer"],
                          "deadline": None, "confidence": 0.8, "source_spans": []}],
        "entities": entities(s),
        "lang_overrides": {"වනක්කම්.": "OTHER", "වනක්කම්": "OTHER", "රොම්බ": "OTHER",
                           "නන්ද්‍රි.": "OTHER", "නන්ද්‍රි": "OTHER"} if s["tamil_customer"] else {},
    }
    with open(os.path.join(out_dir, f"{sid}_{slug}.spec.json"), "w", encoding="utf-8") as fh:
        json.dump(spec, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return minutes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scenario", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--start", type=int, default=1, help="first script number (S001, ...)")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, n = [], a.start
    for k in range(a.per_scenario):
        for slug, scn in SCENARIOS.items():
            sid = f"S{n:03d}"
            s = render(scn, rng, sid)
            minutes = write_script(s, OUT_DIR, slug)
            rows.append({"script_id": sid, "file": f"{sid}_{slug}.md", "scenario": scn["title"],
                         "org": s["org"], "speaker_A": s["agent"], "speaker_B": s["customer"],
                         "tamil_words": "yes" if s["tamil_customer"] else "",
                         "number_styles": " + ".join(s["styles"]),
                         "est_minutes": round(minutes, 1), "recording_id": "", "recorded_by": ""})
            n += 1
    with open(os.path.join(OUT_DIR, "index.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    total = sum(r["est_minutes"] for r in rows)
    print(f"wrote {len(rows)} scripts to {OUT_DIR} (~{total:.0f} min of speech)")


if __name__ == "__main__":
    main()
