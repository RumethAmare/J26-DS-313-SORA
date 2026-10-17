#!/usr/bin/env python3
"""
update_manifest.py -- rebuild SORA_Dataset/manifests/recordings_current.csv.

One row per recording found in annotations/c1..c4 or processed/audio.

What is MEASURED from the files (filled for every row):
  duration_s, orig_start/orig_end  WAV length (orig_* = 0..duration)
  wav_sha256                       of processed/audio/<rid>.wav
  languages                        from C1 gold `lang`: SI/EN present, OTHER
                                   only when a non-numeric OTHER token exists
                                   (numerals are tagged OTHER by convention in
                                   some recordings and say nothing about Tamil)
  speakers                         distinct speakers in the C3 RTTM, else in
                                   the C1 transcript's `speaker` field
  acquisition_date                 first git commit adding the recording's
                                   files -- an upload date, not a recording date
  notes                            layers present, validate.py result, C1 issues

What is NOT measurable and is never invented:
  owner_creator, source_group, privacy_risk  -> FILL_IN for new rows
  licence_consent                            -> CONSENT_PENDING for new rows
  status: new rows are ANNOTATED_DRAFT when audio exists, REGISTERED when it
  does not. Nothing is set to VALIDATED: per DATA_CONTRACT.md that needs
  validate.py to pass AND a second member's approval, which only the team can
  record.

Rows already in the COMMITTED manifest keep their committed owner, consent,
status and source fields; only the measured fields are refreshed, and a dated
refresh note is appended. The script reads the committed version (git HEAD),
not the working copy, so uncommitted edits are not built on.

Usage:
    python update_manifest.py            # writes the manifest
    python update_manifest.py --dry-run  # prints a summary only
"""
import argparse
import csv
import datetime
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import corpus

ROOT = corpus.DATASET_ROOT
MANIFEST = corpus.MANIFEST_PATH
FIELDS = ["recording_id", "source_url", "owner_creator", "licence_consent",
          "acquisition_date", "orig_start", "orig_end", "duration_s", "languages",
          "speakers", "privacy_risk", "source_group", "status", "wav_sha256", "notes"]
LAYER_FILES = {
    "C1": ["c1:tokens.jsonl", "c1:transcript.json"],
    "C2": ["c2:translation.json", "c2:summary.json"],
    "C3": ["c3:rttm", "c3:uem"],
    "C4": ["c4:pii.jsonl", "c4:redacted.json"],
}
ID_RE = re.compile(r"^(?:J26DS313_)?(R\d{4})\.(.+)$")
NUMERIC_RE = re.compile(r"^[\d.,:/%+\-]+$")


def git(*args):
    return subprocess.run(["git", "-C", ROOT, *args], capture_output=True,
                          text=True, encoding="utf-8").stdout


def inventory():
    files = defaultdict(dict)                 # rid -> {"c1:tokens.jsonl": path}
    for layer in ("c1", "c2", "c3", "c4"):
        d = os.path.join(ROOT, "annotations", layer)
        for f in os.listdir(d):
            m = ID_RE.match(f)
            if m:
                files["J26DS313_" + m.group(1)][f"{layer}:{m.group(2)}"] = os.path.join(d, f)
    for f in os.listdir(corpus.AUDIO_DIR):
        m = ID_RE.match(f)
        if m and m.group(2) == "wav":
            files["J26DS313_" + m.group(1)]["audio"] = os.path.join(corpus.AUDIO_DIR, f)
    return files


def languages(tokens_path):
    present, other_real = set(), False
    with open(tokens_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            t = json.loads(line)
            lang = t.get("lang")
            if lang in ("SI", "EN"):
                present.add(lang)
            elif lang == "OTHER" and not NUMERIC_RE.match(t.get("token", "").strip(".,:;!?\"'()")):
                other_real = True
    out = [l for l in ("SI", "EN") if l in present] + (["OTHER"] if other_real else [])
    return "+".join(out)


def speakers(f):
    if "c3:rttm" in f:
        spk = set()
        with open(f["c3:rttm"], encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) > 7 and parts[0] == "SPEAKER":
                    spk.add(re.sub(r"-BC$", "", parts[7]))
        return len(spk), "C3 RTTM"
    with open(f["c1:transcript.json"], encoding="utf-8") as fh:
        utts = json.load(fh).get("utterances", [])
    return len({u.get("speaker") for u in utts if u.get("speaker")}), "C1 transcript"


def first_commit_date(paths):
    rel = [os.path.relpath(p, ROOT).replace(os.sep, "/") for p in paths]
    out = git("log", "--diff-filter=A", "--format=%ad", "--date=short", "--", *rel)
    dates = sorted(d for d in out.split() if d)
    return dates[0] if dates else ""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(rid, audio):
    cmd = [sys.executable, "validate.py", rid] + (["--audio", audio] if audio else [])
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    fails = re.findall(r"^\s*\[FAIL\]\s*(.+)$", p.stdout, re.MULTILINE)
    if p.returncode == 0 and not fails:
        return "validate.py: passed"
    if "Traceback" in p.stderr:
        # validate.py itself crashed (e.g. no C3 files, or a C2 summary in a
        # different shape) -- not a data FAIL, so say what actually happened.
        err = p.stderr.strip().splitlines()[-1][:90]
        prefix = f"{len(fails)} FAIL before " if fails else ""
        return f"validate.py: {prefix}crashed ({err})"
    first = fails[0][:80] if fails else "non-zero exit"
    return f"validate.py: {len(fails)} FAIL ({first}{'; …' if len(fails) > 1 else ''})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import soundfile as sf

    committed = {r["recording_id"]: r for r in csv.DictReader(
        io.StringIO(git("show", "HEAD:manifests/recordings_current.csv")))}
    files = inventory()
    today = datetime.date.today().isoformat()
    rows, changes = [], defaultdict(list)

    for rid in sorted(files):
        f = files[rid]
        audio = f.get("audio")
        dur = sf.info(audio).duration if audio else None
        layers = [L for L, need in LAYER_FILES.items() if all(k in f for k in need)]
        missing = [L for L in LAYER_FILES if L not in layers]
        n_spk, spk_src = speakers(f)
        canonical_names = all(os.path.basename(p).startswith("J26DS313_") for p in f.values())

        notes = [f"layers: {'/'.join(layers) or 'none'}"
                 + (f" (missing {'/'.join(missing)})" if missing else ""),
                 f"speakers from {spk_src}"]
        if audio:
            notes.append(validate(rid, audio))
        else:
            notes.append("no audio in processed/audio/")
        if not canonical_names:
            notes.append("files lack the J26DS313_ prefix (data contract)")
        if rid in corpus.EXCLUDED_RECORDINGS:
            notes.append(f"C1 language labels under review: {corpus.EXCLUDED_RECORDINGS[rid]}")

        measured = {
            "orig_start": "0.00" if dur else "",
            "orig_end": f"{dur:.2f}" if dur else "",
            "duration_s": f"{dur:.2f}" if dur else "",
            "languages": languages(f["c1:tokens.jsonl"]) if "c1:tokens.jsonl" in f else "",
            "speakers": str(n_spk) if n_spk else "",
            "wav_sha256": sha256(audio) if audio else "",
        }

        if rid in committed:
            row = dict(committed[rid])
            for k, v in measured.items():
                if k == "languages" and row.get(k) and v and set(row[k].split("+")) != set(v.split("+")):
                    changes[rid].append(f"languages {row[k]} -> {v}")
                elif row.get(k) != v and v:
                    if k == "wav_sha256" and row.get(k):
                        changes[rid].append("wav_sha256 updated (audio re-processed)")
                    elif k in ("duration_s", "speakers") and row.get(k):
                        changes[rid].append(f"{k} {row[k]} -> {v}")
                row[k] = v or row.get(k, "")
            row["notes"] = (row.get("notes", "").rstrip()
                            + f" | {today} refresh: " + "; ".join(notes))
        else:
            row = {
                "recording_id": rid,
                "source_url": "FILL_IN",
                "owner_creator": "FILL_IN",
                "licence_consent": "CONSENT_PENDING",
                "acquisition_date": first_commit_date(f.values()),
                **measured,
                "privacy_risk": "FILL_IN",
                "source_group": "FILL_IN",
                "status": "ANNOTATED_DRAFT" if audio else "REGISTERED",
                "notes": "; ".join(notes + ["acquisition_date = first git upload, not "
                                            "the recording date"]),
            }
        rows.append(row)

    n_new = sum(1 for r in rows if r["recording_id"] not in committed)
    status = defaultdict(int)
    for r in rows:
        status[r["status"]] += 1
    passed = sum(1 for r in rows if "validate.py: passed" in r["notes"])
    print(f"{len(rows)} rows ({len(committed)} committed kept, {n_new} added)")
    print(f"status: {dict(status)}")
    print(f"validate.py passed: {passed}/{sum(1 for r in rows if r['wav_sha256'])} with audio")
    for rid, ch in changes.items():
        print(f"  {rid}: {'; '.join(ch)}")
    if a.dry_run:
        return
    with open(MANIFEST, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
