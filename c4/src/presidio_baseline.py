"""
External comparison baseline: Microsoft Presidio (SO2).

Presidio is the established open-source PII detector. It accepts one declared
language per request, so code-mixed Singlish cannot be described to it; it is
run as English, which is how it would be deployed on this data today.

To keep the comparison fair to Presidio:

  * it runs with its standard large English model (en_core_web_lg),
  * its entity types are mapped GENEROUSLY onto ours -- it has no Sri Lankan
    NIC and no date-of-birth type, so its nearest types are credited:

        PERSON                                  -> PERSON
        PHONE_NUMBER                            -> PHONE
        EMAIL_ADDRESS                           -> EMAIL
        LOCATION                                -> ADDRESS   (its only place type)
        ORGANIZATION                            -> ORG
        DATE_TIME                               -> DOB       (any date)
        CREDIT_CARD, IBAN_CODE, US_BANK_NUMBER  -> ACCOUNT
        US_SSN, US_PASSPORT, US_DRIVER_LICENSE,
        US_ITIN, UK_NHS, ...                    -> NIC       (any national ID)

  * it is scored by the same strict-span harness as every other system.

    python evaluate.py --source real-eval --system presidio --labels proposal --save
"""
from __future__ import annotations

from functools import lru_cache

from rules import Detection

PRESIDIO_TO_C4 = {
    "PERSON": "PERSON",
    "PHONE_NUMBER": "PHONE",
    "EMAIL_ADDRESS": "EMAIL",
    "LOCATION": "ADDRESS",
    "ORGANIZATION": "ORG",
    "DATE_TIME": "DOB",
    "CREDIT_CARD": "ACCOUNT", "IBAN_CODE": "ACCOUNT", "US_BANK_NUMBER": "ACCOUNT",
    "US_SSN": "NIC", "US_PASSPORT": "NIC", "US_DRIVER_LICENSE": "NIC", "US_ITIN": "NIC",
    "UK_NHS": "NIC", "UK_NINO": "NIC", "IN_AADHAAR": "NIC", "IN_PAN": "NIC",
    "SG_NRIC_FIN": "NIC", "AU_TFN": "NIC", "IT_FISCAL_CODE": "NIC", "ES_NIF": "NIC",
}

SPACY_MODEL = "en_core_web_lg"


@lru_cache(maxsize=1)
def _analyzer(model: str = SPACY_MODEL):
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": model}],
        "ner_model_configuration": {
            "model_to_presidio_entity_mapping": {
                "PERSON": "PERSON", "PER": "PERSON", "GPE": "LOCATION", "LOC": "LOCATION",
                "FAC": "LOCATION", "ORG": "ORGANIZATION", "NORP": "NRP",
                "DATE": "DATE_TIME", "TIME": "DATE_TIME",
            },
            "labels_to_ignore": ["CARDINAL", "EVENT", "LANGUAGE", "LAW", "MONEY",
                                 "ORDINAL", "PERCENT", "PRODUCT", "QUANTITY", "WORK_OF_ART"],
        },
    }
    nlp_engine = NlpEngineProvider(nlp_configuration=config).create_engine()
    return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])


class PresidioDetector:
    """Presidio's findings, mapped to C4 labels, as Detections."""

    def __init__(self, model: str = SPACY_MODEL, score_threshold: float = 0.0):
        self.model = model
        self.score_threshold = score_threshold

    def __call__(self, text: str, preceding: str = "") -> list[Detection]:
        results = _analyzer(self.model).analyze(text=text, language="en",
                                                score_threshold=self.score_threshold)
        candidates = []
        for r in results:
            label = PRESIDIO_TO_C4.get(r.entity_type)
            if label is None:
                continue
            s, e = r.start, r.end
            while e > s and text[e - 1] in " .,;:?!":
                e -= 1
            while s < e and text[s] == " ":
                s += 1
            if e > s:
                candidates.append((r.score, e - s, Detection(
                    s, e, label, text[s:e], role="ORGANISATION" if label == "ORG"
                    else "PRIVATE_INDIVIDUAL", source="presidio")))
        # Presidio can return overlapping findings; keep the most confident,
        # then the longest, as a deployment would.
        kept: list[Detection] = []
        for _, _, d in sorted(candidates, key=lambda c: (-c[0], -c[1])):
            if all(d.end <= k.start or d.start >= k.end for k in kept):
                kept.append(d)
        return sorted(kept, key=lambda d: d.start)
