"""Tests for the schema validator: each check must fire on its own defect."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from validate import normalise_label, validate_recording  # noqa: E402

TEXT = "mage nama Nimal. මගේ නම නිමල්."
DOC = "SYN_R0001_transcript_u001_v1"
SINHALA = dict(ann_id="a2", start_char=24, end_char=29, surface="නිමල්", script="SINHALA")


def row(**kw):
    base = {"ann_id": "a1", "recording": "SYN_R0001", "doc_id": DOC,
            "entity_id": "SYN_R0001_e001", "label": "PERSON",
            "start_char": 10, "end_char": 15, "surface": "Nimal",
            "script": "LATIN", "role": "PRIVATE_INDIVIDUAL", "redact": True}
    return {**base, **kw}


def check(rows, registry=None):
    return validate_recording(rows, {DOC: TEXT}, registry)


def codes(report):
    return {e[:3] for e in report.errors}


def test_valid_cross_script_person_passes():
    rep = check([row(), row(**SINHALA)])
    assert rep.ok, rep.errors


def test_offset_mismatch_is_an_error():
    assert "[4]" in codes(check([row(end_char=14), row(**SINHALA)]))


def test_out_of_range_offset_is_an_error():
    assert "[4]" in codes(check([row(start_char=90, end_char=95), row(**SINHALA)]))


def test_edge_whitespace_is_an_error():
    assert "[5]" in codes(check([row(start_char=9, surface=" Nimal"), row(**SINHALA)]))


def test_person_in_one_script_only_is_an_error():
    """Check [6] -- the failure Contribution 2 exists to prevent."""
    assert "[6]" in codes(check([row()]))


def test_overlapping_spans_are_an_error():
    inner = row(ann_id="a3", entity_id="SYN_R0001_e002", end_char=13, surface="Nim")
    assert "[7]" in codes(check([row(), row(**SINHALA), inner]))


def test_redacted_org_is_an_error():
    org = row(label="ORG", role="ORGANISATION", redact=True, **SINHALA)
    assert "[8]" in codes(check([org]))


def test_organisation_hotline_is_not_redacted():
    text = "hotline eka 0112345678."
    hot = row(label="PHONE", role="ORGANISATION", redact=False,
              start_char=12, end_char=22, surface="0112345678")
    assert validate_recording([hot], {DOC: text}).ok


def test_undeclared_entity_is_an_error():
    assert "[9]" in codes(check([row(), row(**SINHALA)], registry={"entities": []}))


def test_role_value_in_label_is_normalised_with_a_warning():
    """Section 6 defect: tolerated on read, but reported."""
    legacy = [row(label="PRIVATE_INDIVIDUAL"), row(label="ORGANISATION_REP", **SINHALA)]
    rep = check(legacy)
    assert rep.ok, rep.errors
    assert len(rep.warnings) == 2
    assert normalise_label(legacy[0])["label"] == "PERSON"


def test_legacy_role_is_a_warning_not_a_guess():
    rep = check([row(role="person"), row(role="person", **SINHALA)])
    assert rep.ok and any("legacy role" in w for w in rep.warnings)


def test_missing_recording_is_a_warning_and_checks_still_run():
    """Legacy rows lack `recording`; it is derivable, the offsets are not."""
    legacy = {k: v for k, v in row(end_char=14).items() if k != "recording"}
    rep = check([legacy, row(**SINHALA)])
    assert any("derivable" in w for w in rep.warnings)
    assert "[4]" in codes(rep), "offset check must still run on the legacy row"


def test_missing_offsets_is_an_error():
    broken = {k: v for k, v in row().items() if k != "start_char"}
    assert "[1]" in codes(check([broken, row(**SINHALA)]))


def test_unknown_label_is_an_error():
    assert "[2]" in codes(check([row(label="NAME"), row(label="NAME", **SINHALA)]))
