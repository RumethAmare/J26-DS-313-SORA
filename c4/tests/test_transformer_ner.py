"""
Tests for the transformer NER plumbing: BIO labels must line up with the
annotated character spans, in Latin and Sinhala script. Uses only the
tokenizer (cached locally); no trained model is needed.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

transformers = pytest.importorskip("transformers")

from transformer_ner import BASE_MODEL, LABELS, encode  # noqa: E402


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    except OSError:
        pytest.skip("tokenizer not cached locally")


def _tagged(tokenizer, text, spans):
    enc = encode(tokenizer, text, spans)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    return [(t, LABELS[l]) for t, l in zip(tokens, enc["labels"]) if l != -100]


def test_latin_name_gets_b_then_i(tokenizer):
    text = "mage nama Nimal Perera."
    tags = [tag for _, tag in _tagged(tokenizer, text, [(10, 22, "PERSON")])]
    assert tags[0] == "O" and "B-PERSON" in tags
    first = tags.index("B-PERSON")
    assert all(t == "I-PERSON" for t in tags[first + 1:tags.index("O", first)])


def test_sinhala_name_is_labelled(tokenizer):
    text = "මගේ නම නිමල් පෙරේරා."
    tags = [tag for _, tag in _tagged(tokenizer, text, [(7, 19, "PERSON")])]
    assert tags.count("B-PERSON") == 1 and "I-PERSON" in tags


def test_text_outside_spans_is_o(tokenizer):
    tags = [tag for _, tag in _tagged(tokenizer, "hello there", [])]
    assert set(tags) == {"O"}


def test_label_set_is_bio_over_model_labels():
    assert LABELS[0] == "O" and len(LABELS) == 9


# --- Span post-processing (general rules, no model needed) ---------------------

from transformer_ner import _split_at_sentence, _without_case_ending  # noqa: E402


def test_names_are_split_at_a_sentence_break_but_not_after_a_title():
    text = "Perera. Nimal came. Dr. Silva"
    parts = [text[s:e] for s, e in _split_at_sentence(text, 0, len(text))]
    assert parts[0] == "Perera" and "Dr. Silva" in parts[-1]


def test_sinhala_case_ending_is_trimmed_from_a_name():
    span = "නිමල් පෙරේරාට"
    assert span[:_without_case_ending(span)] == "නිමල් පෙරේරා"
    assert _without_case_ending("Nimal") == len("Nimal")
