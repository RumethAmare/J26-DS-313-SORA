# C4 — PP1 Results: Offline PII Detection and Redaction for Sinhala-English Text

J26-DS-313 · Component 4 · O. S. Jayathilaka (IT23247390) · final, 10 October 2026

## Summary

- **Detection.** On five held-out real recordings, the full C4 pipeline
  reaches **micro F1 0.755, macro F1 0.801** across the seven proposal labels,
  against the baseline's 0.231 / 0.289 and Microsoft Presidio's 0.235 / 0.182.
  **Recall, the primary metric, rises from 0.174 to 0.775.** Both proposal
  targets are met: overall F1 above 0.60, and PERSON F1 above 0.50 (**0.757**).
- **Sinhala script.** Presidio finds 3 of 96 Sinhala-script identifiers (F1
  0.025); C4 reaches F1 **0.738**, with recall 0.792.
- **Cross-script entity resolution** (Contribution 2) links a Latin and a
  Sinhala-script mention of the same person with accuracy **0.933** on
  held-out recordings, against a 0.85 target. String matching scores 0.000.
- **Contribution 2 also drives detection.** A name found once is found again
  in the other script, case or inflection. This raised name recall from 0.497
  to **0.765** on held-out data.
- **End to end,** the redacted output hides **82%** of all personal data: every
  NIC, phone number, date of birth and email, and **76% of names** (up from
  54%).
- **Person roles** (FR7): **0.938** accuracy on held-out people (0.833 in
  cross-validation).
- **Runs offline on a laptop:** 0 network attempts, 14 ms per document,
  338 MB peak memory.
- **Synthetic data**, used at the supervisor's direction, improves the model
  when added to real data (F1 0.657 → 0.699 for the detector). It is not enough
  on its own.

## 1. Data and evaluation protocol

| Data | Recordings | Used for |
|---|---|---|
| Synthetic train | 50 (1,722 spans) | building rules, training models |
| Synthetic test | 60 (1,873 spans) | development score; frozen, never tuned on |
| Real, training and tuning | 73 complete recordings | rule development, model training |
| Real, development fold | 15 of those 73 | error analysis with a model trained without them |
| **Real, held-out** | **5** (R0011, R0019, R0021, R0022, R0023; 289 gold spans) | **reported results only** |

- **Complete recordings only.** A recording is used only if its C1, C2 and C4
  layers exist and at least 95% of its spans match their text: 78 of 85
  qualify. Seven still in progress are excluded.
- **Same held-out set as the baseline**, never trained or tuned on. Every
  method change was developed on the development fold, then scored on the
  held-out set once.
- **Strict matching**: start, end and label must all equal the gold span.
- Every number below is read from a JSON file in `c4/eval/`.

### Result history

Reported for transparency:

| Step | Held-out micro F1 |
|---|---|
| Model trained on 22 real recordings; original gold | 0.570 |
| Same model; gold corrected to the schema (`SORA_Dataset@d8981fc`) | 0.617 |
| Retrained on 73 real recordings | 0.684 |
| Retrained on the 73 complete recordings | 0.699 |
| **+ cross-script name propagation (developed on the development fold)** | **0.755** |

The gold correction was made by a rule-based, public script
(`scripts/c4/fix_c4_annotations.py`) applied uniformly to every recording. A
second team member's review is pending, as the data contract requires.

## 2. Main result: held-out real recordings, seven proposal labels

| System | Recall | Precision | Micro F1 | Macro F1 |
|---|---|---|---|---|
| Microsoft Presidio (`en_core_web_lg`) | 0.294 | 0.196 | 0.235 | 0.182 |
| Baseline (proposal, spaCy CNN) | 0.174 | 0.341 | 0.231 | 0.289 |
| Rule layer only | 0.270 | 0.736 | 0.395 | 0.503 |
| Hybrid detector (rules + model) | 0.630 | 0.784 | 0.699 | 0.778 |
| **Full C4 pipeline (+ propagation and cross-script names)** | **0.775** | **0.737** | **0.755** | **0.801** |

The baseline was scored on transcript utterances and summaries only. On those
documents alone, C4 reaches F1 **0.711**, macro **0.783**.

**Per label (full pipeline):**

| Label | Recall | Precision | F1 |
|---|---|---|---|
| NIC | 1.000 | 1.000 | 1.000 |
| DOB | 1.000 | 1.000 | 1.000 |
| PHONE | 0.967 | 0.935 | 0.951 |
| ADDRESS | 0.688 | 0.846 | 0.759 |
| PERSON | 0.765 | 0.750 | 0.757 |
| ORG | 0.529 | 0.621 | 0.571 |
| ACCOUNT | 0.759 | 0.458 | 0.571 |

**Per script:**

| Script | Presidio F1 | C4 F1 |
|---|---|---|
| Latin | 0.342 | 0.765 |
| Sinhala | **0.025** | **0.738** |

## 3. End-to-end protection

This measures what the shareable output actually hides. It counts every gold
personal span in the 5 held-out recordings:

| Label | Gold spans | Fully hidden |
|---|---|---|
| NIC | 15 | 15 (100%) |
| PHONE | 30 | 30 (100%) |
| DOB | 12 | 12 (100%) |
| EMAIL | 3 | 3 (100%) |
| ADDRESS | 16 | 13 (81%) |
| ACCOUNT | 29 | 22 (+7 partly hidden) |
| PERSON | 153 | 117 (76%) (+3 partly hidden) |
| **All** | **258** | **212 (82%)**; 86% at least partly |

## 4. Cross-script name propagation

The detector alone misses a name it does not recognise in a given sentence,
even when it recognised the same person elsewhere in the call. The resolver's
phonetic matching (Contribution 2) closes that gap. Once a person is found, the
same name is found again:
- in another case ("nimal")
- in the other script (නිමල්)
- with a Sinhala case ending (නිමල්ට, with the ending kept visible)

It was developed on the 15-recording development fold, using a model trained
without them:

| Development fold (unseen recordings) | PERSON recall | PERSON precision | PERSON F1 | Micro F1 |
|---|---|---|---|---|
| Detector only | 0.534 | 0.832 | 0.651 | 0.662 |
| **+ name propagation** | **0.803** | 0.681 | **0.737** | **0.701** |

Safeguards were added after error analysis on the development fold:
- A consonant match alone is not trusted for short words, so කැරට් ("carrot")
  is not taken for a name.
- In clean English and summaries a name must be capitalised.
- Neighbouring name tokens are joined into one span.

Some remaining "false positives" are real names the annotators did not mark.

**Limit:** propagation needs the person to be found at least once. A person the
detector never finds anywhere in a call, such as 2 of the 3 people in R0022,
stays exposed. That is the detector's job, and the reason for the transformer
upgrade.

## 5. Comparison with Presidio (SO2)

Presidio accepts one declared language per request, so it is run as English,
which is how it would be deployed on this data today. It was given its
standard large model and a deliberately generous label mapping.

- **Sinhala script:** recall 0.031.
- **NIC and ADDRESS:** 0 detected; Presidio has no Sri Lankan formats.
- **Precision:** 0.196, consistent with the 0.14 reported in the literature
  cited by the proposal.

## 6. Does synthetic data help? (model ablation)

The same CPU spaCy NER model was trained on three data sets and combined with
the same rules; this is the hybrid detector, without propagation:

| Trained on | PERSON F1 | ADDRESS F1 | ORG F1 | Hybrid micro F1 |
|---|---|---|---|---|
| Real only (73 complete) | 0.597 | 0.560 | 0.415 | 0.657 |
| Synthetic only (50) | 0.331 (precision 0.27) | 0.150 | 0.133 | — |
| **Synthetic + real** | **0.639** | **0.714** | **0.571** | **0.699** |

## 7. Structured identifiers: rule layer (SO3)

| Data | Recall | Precision | F1 |
|---|---|---|---|
| Synthetic test (held-out templates + noise) | 0.984 | 0.983 | 0.984 |
| Real, 73 tuning recordings | 0.713 | 0.679 | 0.696 |
| **Real, held-out** | **0.910** | **0.743** | **0.818** |

The rules handle every spoken form in the corpus:
- spaced and grouped digits
- Sinhala, romanised and English number words
- prefixed reference numbers and phone extensions
- ASR ordinal errors
- keywords after the number or in the previous turn

## 8. Cross-script entity resolution (Contribution 2, SO5)

| System | Synthetic test (unseen names) | Real, 73 tuning | **Real, held-out** |
|---|---|---|---|
| Exact string match | 0.000 | 0.000 | 0.000 |
| Transliteration + exact | 0.601 | 0.427 | 0.400 |
| **Transliteration + phonetic similarity** | **0.899** | **0.924** (185 people) | **0.933** (15 people) |

- The method romanises Sinhala, reduces both scripts to one phonetic key and
  consonant skeleton, and clusters mentions per recording. The threshold was
  tuned on synthetic train only.
- **Linguistic handling:**
  - Sinhala case endings
  - titles in both scripts (Doctor, ආචාර්ය, මහාචාර්ය)
  - surname particles (de / ද)
  - prenasalised ඳ / ඹ (Osadi = ඔසඳි)
  - English -ee = Sinhala ී
  - male/female name pairs (Mahesh ≠ Maheshi)
  - shared surnames
- **Pairwise precision is 1.000:** two different people were never merged.

## 9. Person roles (FR7)

| | People | Accuracy | Macro F1 |
|---|---|---|---|
| **Held-out real** | 16 | **0.938** | 0.935 |
| Cross-validation, real (grouped by recording) | 203 | 0.833 | 0.829 |
| Majority-class baseline (held-out) | 16 | 0.375 | — |

Role never changes whether a person is redacted; both roles are. It is a
required schema field, not a novelty claim.

## 10. Non-functional requirements

Measured by running the full pipeline on all 5 held-out recordings:

| Requirement | Measured | Target |
|---|---|---|
| NFR1 offline | **0** network attempts (all connections blocked) | none |
| NFR4 memory | **338 MB** peak process memory | below 8 GB |
| NFR5 latency | **14 ms** per document (median); 1.0 s per recording | no perceptible delay |
| NFR2 re-identification map | written to git-ignored storage only; refused elsewhere | no copy in version control |
| Leak check | **0** known identifiers left in the output | 0 |

Hardware: a consumer laptop with 8 CPU cores and no GPU.

## 11. Limitations

- **A person never detected anywhere in a call stays exposed.** 36 of 153 name
  mentions (24%) remain visible end to end.
- **ACCOUNT precision is 0.458.** Reference numbers the annotators did not
  mark count as false positives.
- **Small evaluation set:** 5 recordings, 289 spans, and 16 people for roles.
- **CPU spaCy model.** The proposal's xlm-roberta-base transformer needs a GPU.
- **Gold correction by the component owner.** It is rule-based and public, but
  awaits a second team member's approval.

## 12. Next steps

1. **Transformer model (SO4)** on GPU (Colab), behind the same detector
   interface, to find each person at least once.
2. **More complete real recordings** as the corpus grows, then retrain with
   one command.
3. **Noise and robustness evaluation** on real C1 ASR output, with the team.
4. **Integration** of the four components using the documented input–output
   contract.

## Reproducing these results

    cd c4
    python -m pytest tests                                    # 315 tests
    python src/ner.py train --data both                       # ~30 min on CPU
    python src/roles.py train && python src/roles.py evaluate
    python src/evaluate.py --source real-eval --system pipeline:both --labels proposal
    python src/evaluate.py --source real-eval --system presidio --labels proposal
    python src/resolve.py --source real-eval
    python src/pipeline.py --recording J26DS313_R0022 --offline-check --measure
    python src/demo.py                                        # live demo
