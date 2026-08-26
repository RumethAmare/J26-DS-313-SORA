#!/usr/bin/env python3
"""
rerun_validate_all.py -- Section D.1 of the C1 implementation plan.

Re-runs SORA_Dataset/validate.py against every recording in
manifests/recordings_current.csv and compares its pass/fail verdict against
the manifest's recorded `status` column. Writes a full audit CSV so we know
exactly which recordings are mislabeled VALIDATED before any WER/LID numbers
built on top of them get quoted.

This script lives in J26-DS-313-SORA/c1/ (per project convention: all C1
deliverables live here, not in the shared SORA_Dataset repo) but reads data
from, and shells out to, SORA_Dataset -- that repo is treated as a read-only
data + tooling source.

Usage:
    python3 rerun_validate_all.py
    python3 rerun_validate_all.py --dataset-root /mnt/F/SLIIT/Research/SORA_Dataset
"""
import argparse
import csv
import os
import re
import subprocess
import sys

DEFAULT_DATASET_ROOT = "/mnt/F/SLIIT/Research/SORA_Dataset"
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "validate_status_audit.csv")

FAIL_RE = re.compile(r"^\s*\[FAIL\]\s*(.+)$", re.MULTILINE)
WARN_RE = re.compile(r"^\s*\[warn\]\s*(.+)$", re.MULTILINE)


def read_manifest(dataset_root):
    path = os.path.join(dataset_root, "manifests", "recordings_current.csv")
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_validate(dataset_root, rid, python_bin):
    audio_path = os.path.join(dataset_root, "processed", "audio", f"{rid}.wav")
    audio_present = os.path.exists(audio_path)
    cmd = [python_bin, "validate.py", rid]
    if audio_present:
        cmd += ["--audio", audio_path]
    proc = subprocess.run(
        cmd, cwd=dataset_root, capture_output=True, text=True
    )
    return proc, audio_present


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument(
        "--python-bin",
        default=os.path.join(DEFAULT_DATASET_ROOT, ".venv", "bin", "python3"),
        help="interpreter to run validate.py with (defaults to SORA_Dataset's .venv)",
    )
    a = ap.parse_args()

    if not os.path.exists(a.python_bin):
        a.python_bin = sys.executable

    rows = read_manifest(a.dataset_root)
    out_rows = []
    mismatches = []

    for r in rows:
        rid = r["recording_id"]
        manifest_status = r["status"]
        proc, audio_present = run_validate(a.dataset_root, rid, a.python_bin)
        stdout = proc.stdout
        fails = FAIL_RE.findall(stdout)
        warns = WARN_RE.findall(stdout)
        passed = proc.returncode == 0
        computed_status = "VALIDATED" if passed else "ANNOTATED_DRAFT"
        mismatch = (manifest_status == "VALIDATED") and not passed

        out_rows.append({
            "recording_id": rid,
            "manifest_status": manifest_status,
            "audio_present": audio_present,
            "validate_exit_code": proc.returncode,
            "computed_status": computed_status,
            "status_mismatch": mismatch,
            "num_fails": len(fails),
            "num_warns": len(warns),
            "fail_messages": " | ".join(fails),
        })
        if mismatch:
            mismatches.append(rid)

        tag = "MISMATCH" if mismatch else "ok"
        print(f"{rid}: manifest={manifest_status} computed={computed_status} "
              f"fails={len(fails)} warns={len(warns)} [{tag}]")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print(f"\nWrote {len(out_rows)} rows to {a.out}")
    print(f"{len(mismatches)} recording(s) marked VALIDATED in the manifest but fail validate.py:")
    for rid in mismatches:
        print(f"  - {rid}")


if __name__ == "__main__":
    main()
