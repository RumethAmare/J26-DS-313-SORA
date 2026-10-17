import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import score, score_actions, score_summaries  # noqa: E402


def test_identical_translation_scores_100():
    assert score(["mama gedara yanawa"], ["mama gedara yanawa"])["chrf++"] == 100.0


def test_zero_width_joiner_does_not_cost_points():
    with_zwj, without = "වෛද්‍ය", "වෛද්ය"
    assert score([without], [with_zwj])["chrf++"] == 100.0


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        score(["a", "b"], ["a"])


def test_rouge_identical_summary():
    assert score_summaries(["A customer lost a card."], ["A customer lost a card."])["rougeL"] == 100.0


def test_action_matches_despite_wording_and_filler_words():
    gold = [[{"intent": "Call Suresh to confirm the booking", "owner": "S1"}]]
    pred = [[{"intent": "Confirm booking", "owner": "S1 (agent)"}]]
    res = score_actions(pred, gold)
    assert res["intent"]["f1"] == 100.0
    assert res["intent+owner"]["f1"] == 100.0


def test_wrong_owner_counts_for_intent_only():
    gold = [[{"intent": "Transfer the advance payment", "owner": "Saman"}]]
    pred = [[{"intent": "Transfer advance payment", "owner": "S1 (agent)"}]]
    res = score_actions(pred, gold)
    assert res["intent"]["f1"] == 100.0
    assert res["intent+owner"]["f1"] == 0.0


def test_one_prediction_matches_one_gold_only():
    gold = [[{"intent": "Pay the fee", "owner": "S2"}, {"intent": "Pay the fee", "owner": "S2"}]]
    pred = [[{"intent": "Pay the fee", "owner": "S2"}]]
    res = score_actions(pred, gold)
    assert res["intent"]["p"] == 100.0 and res["intent"]["r"] == 50.0


def test_no_actions_anywhere():
    res = score_actions([[]], [[]])
    assert res["intent"]["f1"] == 0.0 and res["pred"] == res["gold"] == 0
