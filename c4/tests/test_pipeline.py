"""
Tests for the end-to-end pipeline (FR1, FR9), person roles (FR7), the offline
guarantee (NFR1) and the demo's analysis. The rule layer is used as the
detector so the tests need no trained model.
"""
import json
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pipeline  # noqa: E402
from roles import ROLES, entity_features  # noqa: E402

TEXTS = {
    "T_transcript_u001_v1": "mage NIC eka 953201456V, number eka 0771234567.",
    "T_c2_u001_en_v1": "My NIC is 953201456V and my number is 0771234567.",
    "T_c2_summary_en_v1": "NIC 953201456V, phone 0771234567.",
}


def test_offline_guard_blocks_every_connection(monkeypatch):
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, getattr(socket.socket, name))
    monkeypatch.setattr(socket, "create_connection", socket.create_connection)
    attempts = pipeline.block_network()
    with pytest.raises(pipeline.NetworkBlocked):
        socket.create_connection(("example.com", 80))
    assert attempts


def test_pipeline_writes_shareable_output_with_no_leaks(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "OUT_DIR", tmp_path)
    report = pipeline.run("T", TEXTS, model=None, save_map=False, measure=True)
    assert report["leaks"] == 0 and report["by_label"] == {"NIC": 1, "PHONE": 1}
    redacted = json.loads((tmp_path / "T.redacted.json").read_text(encoding="utf-8"))
    text = " ".join(d["text"] for d in redacted["documents"])
    assert "953201456" not in text and "0771234567" not in text
    assert {d["source_doc_id"] for d in redacted["documents"]} == set(TEXTS)
    assert report["ms_per_document"] >= 0 and report["peak_python_memory_mb"] >= 0


def test_detections_file_carries_no_surfaces(tmp_path, monkeypatch):
    """The detections file is metadata; the identifiers stay in the re-id map only."""
    monkeypatch.setattr(pipeline, "OUT_DIR", tmp_path)
    pipeline.run("T", TEXTS, model=None, save_map=False)
    rows = (tmp_path / "T.detections.jsonl").read_text(encoding="utf-8")
    assert "953201456" not in rows and "0771234567" not in rows


def test_c2_files_from_a_loose_directory(tmp_path):
    c2 = tmp_path / "c2"
    c2.mkdir()
    (c2 / "X.translation.json").write_text(json.dumps(
        {"utterances": [{"utt_id": "X_u001", "clean_en": "hi", "clean_si": "හායි"}]}), encoding="utf-8")
    (c2 / "X.summary.json").write_text(json.dumps({"summary": "S"}), encoding="utf-8")
    texts = pipeline.load_inputs("X", c2, None)
    assert set(texts) == {"X_c2_u001_en_v1", "X_c2_u001_si_v1", "X_c2_summary_en_v1"}


def test_role_features_capture_how_a_name_is_given():
    agent = entity_features([{"doc_id": "R_transcript_u001_v1", "text": "hello, mama Amal, Seylan Bank eken",
                              "start": 12, "end": 16, "surface": "Amal"}])
    customer = entity_features([{"doc_id": "R_transcript_u004_v1", "text": "Mage nama Nimal Perera.",
                                 "start": 10, "end": 22, "surface": "Nimal Perera"}])
    assert agent["ntok=1"] == 1 and customer["ntok=2"] == 1
    assert agent["first_utt"] < customer["first_utt"]


def test_roles_are_the_two_contract_values():
    assert set(ROLES) == {"PRIVATE_INDIVIDUAL", "ORGANISATION_REP"}
