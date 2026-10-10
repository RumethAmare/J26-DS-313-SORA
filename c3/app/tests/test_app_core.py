"""Unit tests for the C3 demo helpers: output contract, scoring, error strip, live stitching.
Run:  python -m pytest app/tests -q   (no model download needed)"""
import json
import numpy as np
import pytest
from pyannote.core import Annotation, Segment, Timeline

import app
from live import LiveDiarizer


def ann(rows, uri="t"):
    a = Annotation(uri=uri)
    for i, (s, e, lab) in enumerate(rows):
        a[Segment(s, e), i] = lab
    return a


REF = ann([(0, 4, "spk1"), (4, 8, "spk2"), (7, 9, "spk1")])
UEM = Timeline([Segment(0, 10)], uri="t")


def test_export_contract(tmp_path):
    hyp = app.friendly(ann([(0, 4, "S0"), (4, 8, "S1"), (7, 9, "S0")]))
    rttm, js = app.export(hyp, "clip")
    data = json.loads(open(js, encoding="utf-8").read())
    assert data["uri"] == "clip" and len(data["segments"]) == 3
    seg = data["segments"][0]
    assert set(seg) == {"speaker_id", "start", "end", "overlap_flag"}           # shared contract with C1/C2/C4
    assert " " not in seg["speaker_id"]                                         # RTTM-safe labels
    assert [s["overlap_flag"] for s in data["segments"]] == [False, True, True]
    assert all(len(l.split()) >= 8 for l in open(rttm) if l.strip())


def test_perfect_hypothesis_scores_zero():
    c = app.components(REF, REF.rename_labels({"spk1": "A", "spk2": "B"}), UEM)
    assert c["DER"] == pytest.approx(0.0, abs=1e-9)


def test_component_split():
    hyp = ann([(0, 4, "A"), (4, 6, "A"), (6, 8, "B")])          # 4-6 wrong person, 7-9 overlap spk1 missed
    c = app.components(REF, hyp, UEM)
    assert c["conf"] > 0 and c["miss"] > 0 and c["fa"] == pytest.approx(0.0, abs=1e-9)
    assert c["DER"] == pytest.approx(c["miss"] + c["fa"] + c["conf"])


def test_error_strip_categories():
    refn = app.friendly(REF.crop(UEM))
    hyp = app.friendly(ann([(0, 4, "A"), (4, 8, "B"), (9.5, 10, "B")]), REF.crop(UEM), UEM)
    cats, step = app.error_frames(refn, hyp, UEM, 10.0, step=0.1)
    assert cats[int(1 / step)] == "correct"
    assert cats[int(8.5 / step)] == "missed"            # ref spk1 talks 8-9, model silent
    assert cats[int(9.7 / step)] == "false alarm"       # model talks, nobody did


def test_live_matching_keeps_identity():
    eA, eB = np.array([1, 0, 0.]), np.array([0, 1, 0.])
    d = LiveDiarizer(None, tau=0.5)
    d.commit(5, 0, ann([(0, 4, "L0")]), np.stack([eA]))
    d.commit(10, 0, ann([(0, 4, "x"), (5, 9, "y")]), np.stack([eA, eB]))   # labels sorted: x=A, y=B
    d.commit(15, 5, ann([(6, 9, "p"), (0, 4, "q")]), np.stack([eA, eB]))   # p (11-14) is A again
    res = d.result()
    assert len(d.centroids) == 2
    labels = {round(s.start): l for s, _, l in res.itertracks(yield_label=True)}
    assert labels[0] == labels[11] != labels[5]
