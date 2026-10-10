# C4 — PP1 Results: Offline PII Detection and Redaction for Sinhala-English Text

J26-DS-313 · Component 4 · O. S. Jayathilaka (IT23247390) · final, 11 October 2026 (accuracy round included)

## Summary

- **Detection.** On five held-out real recordings, the full C4 pipeline (a
  fine-tuned XLM-RoBERTa transformer, rules and cross-script name propagation)
  reaches **micro F1 0.851, macro F1 0.872**. The baseline scores 0.231 / 0.289
  and Microsoft Presidio 0.235 / 0.182. **Recall, the primary metric, rises
  from 0.174 to 0.938.** Both proposal targets are exceeded: overall F1 above
  0.60, and PERSON F1 above 0.50 (**0.900**).
- **End to end, no personal identifier is left fully exposed.** The redacted
  output fully hides **95.7%** of personal data and at least partly hides
  **100%**: every NIC, phone number, date of birth, email and address, and
  149 of 153 name mentions in full (the other 4 partly).
- **Sinhala script.** Presidio finds 3 of 96 Sinhala-script identifiers (F1
  0.025); C4 reaches recall **0.948**, F1 0.809.
- **Cross-script entity resolution** (Contribution 2) links a Latin and a
  Sinhala-script mention of the same person with accuracy **0.933** on
  held-out recordings, against a 0.85 target. String matching scores 0.000.
  The same matching also drives detection: a name found once is found again
  in the other script.
- **SO4 delivered as proposed.** The multilingual transformer was fine-tuned
  on this laptop's CPU, so the un-consented recordings never left the machine.
- **Runs offline on a laptop:** 0 network attempts, 99 ms per document, about
  1 GB of memory.

## 1. Data and evaluation protocol

| Data | Recordings | Used for |
|---|---|---|
| Synthetic train | 50 (1,722 spans) | building rules, training models |
| Synthetic test | 60 (1,873 spans) | development score; frozen, never tuned on |
| Real, training and tuning | 73 complete recordings | rule development, model training |
| Real, development fold | 15 of those 73 | method decisions, with models trained without them |
| **Real, held-out** | **5** (R0011, R0019, R0021, R0022, R0023; 289 gold spans) | **reported results only** |

- **Complete recordings only.** A recording is used only if its C1, C2 and C4
  layers exist and at least 95% of its spans match their text: 78 of 85
  qualify.
- **Privacy of training.** 81 of 85 recordings are marked CONSENT_PENDING in
  the manifest. All training ran locally, and no recording text was uploaded
  to any cloud service (NFR2).
- **Method discipline.** Every method choice (spaCy vs transformer,
  propagation rules) was made on the development fold, with models trained
  without those recordings. The held-out five were scored once, at the end.
- **Same held-out set as the baseline. Strict matching:** start, end and
  label must all equal the gold span.
- Every number below is read from a JSON file in `c4/eval/`.

### Result history

Reported for transparency:

| Step | Held-out micro F1 | Recall |
|---|---|---|
| spaCy model, 22 real recordings, original gold | 0.570 | 0.485 |
| Same model; gold corrected to the schema (`SORA_Dataset@d8981fc`) | 0.617 | 0.529 |
| spaCy retrained on the 73 complete recordings | 0.699 | 0.630 |
| + cross-script name propagation | 0.755 | 0.775 |
| XLM-RoBERTa transformer + propagation | 0.822 | 0.917 |
| **+ accuracy round: account formats, span rules, confidence threshold (final)** | **0.851** | **0.938** |

The gold correction was made by a rule-based, public script
(`scripts/c4/fix_c4_annotations.py`) applied uniformly to every recording. A
second team member's review is pending, as the data contract requires.

## 2. Main result: held-out real recordings, seven proposal labels

| System | Recall | Precision | Micro F1 | Macro F1 |
|---|---|---|---|---|
| Microsoft Presidio (`en_core_web_lg`) | 0.294 | 0.196 | 0.235 | 0.182 |
| Baseline (proposal, spaCy CNN) | 0.174 | 0.341 | 0.231 | 0.289 |
| Rule layer only | 0.270 | 0.736 | 0.395 | 0.503 |
| Rules + spaCy + propagation | 0.775 | 0.737 | 0.755 | 0.801 |
| Rules + transformer (detector only) | 0.910 | 0.817 | 0.861 | 0.875 |
| **Full C4 pipeline: rules + transformer + propagation** | **0.938** | **0.779** | **0.851** | **0.872** |

The full pipeline trades a little precision for recall, the primary metric,
and it is what produces the shareable output. On transcript utterances and
summaries only (the baseline's documents), it reaches F1 **0.801**, recall
**0.939**.

**Per label (full pipeline):**

| Label | Recall | Precision | F1 |
|---|---|---|---|
| NIC | 1.000 | 1.000 | 1.000 |
| DOB | 1.000 | 1.000 | 1.000 |
| PHONE | 0.967 | 0.935 | 0.951 |
| ADDRESS | 1.000 | 1.000 | 1.000 |
| PERSON | 0.974 | 0.837 | 0.900 |
| ORG | 0.824 | 0.583 | 0.683 |
| ACCOUNT | 0.759 | 0.458 | 0.571 |

**Per script:**

| Script | Presidio F1 | C4 recall | C4 F1 |
|---|---|---|---|
| Latin | 0.342 | 0.933 | 0.874 |
| Sinhala | **0.025** | **0.948** | **0.809** |

## 3. End-to-end protection

This measures what the shareable output actually hides. It counts every gold
personal span in the 5 held-out recordings:

| Label | Gold spans | Fully hidden |
|---|---|---|
| NIC | 15 | 15 (100%) |
| PHONE | 30 | 30 (100%) |
| DOB | 12 | 12 (100%) |
| EMAIL | 3 | 3 (100%) |
| ADDRESS | 16 | 16 (100%) |
| ACCOUNT | 29 | 22, and 7 partly |
| PERSON | 153 | 149, and 4 partly |
| **All** | **258** | **247 (95.7%); 258 (100%) at least partly; 0 fully exposed** |

## 4. Transformer model (SO4)

**xlm-roberta-base**, whose pre-training covers Sinhala, was fine-tuned for
PERSON / ADDRESS / ORG / LOCATION:
- **Data:** 50 synthetic and 73 complete real recordings.
- **Settings:** 3 epochs; 30% of entity-free documents kept as negatives.
- **Training cost:** about 2 hours on CPU, made feasible by **freezing the
  250k-word embedding table**, which cut each training step from 75 s to 4 s.

| Development fold (unseen recordings), full pipeline | PERSON recall | PERSON F1 | Sinhala recall | Micro F1 |
|---|---|---|---|---|
| spaCy CNN | 0.803 | 0.737 | 0.599 | 0.701 |
| **XLM-RoBERTa** | **0.932** | **0.814** | **0.791** | **0.753** |

Combining both models added recall +0.006 at a precision cost, so the
transformer alone was kept.

## 4a. Accuracy round (development fold only)

After error analysis on the 15 development recordings, general rules were
added. No name or number lists were taken from the data, and each rule was
kept only if it improved the development fold:

- **Account formats.** Dotted groups ("123.456.78.90") are identifiers, but
  dates are not. A number read out digit by digit, 7 or more digits, is an
  identifier: people say amounts as numbers, but read identifiers digit by
  digit.
- **Span rules.**
  - A span never starts or ends inside a word.
  - A name never runs across a sentence break (except after Mr./Dr.).
  - A Sinhala case ending is trimmed from a name.
- **Confidence threshold.** A name or address the model is less than 85%
  confident about is dropped. F1 is flat between 0.80 and 0.90, so 0.85 is
  not a knife-edge value.

| Development fold | Recall | Precision | F1 |
|---|---|---|---|
| Before | 0.824 | 0.693 | 0.753 |
| **After** | **0.841** | **0.740** | **0.787** |

Every measure improved and none got worse. The held-out five were then scored
once: F1 0.822 → **0.851**, recall 0.917 → **0.938**.

## 5. Cross-script name propagation

Once a person is found, the same name is found again in another case, in the
other script, or with a Sinhala case ending (kept visible: "[PERSON_1]ට").
This uses the resolver's phonetic matching (Contribution 2). It was developed
on the development fold, with safeguards:
- a consonant match alone is not trusted for short words
- names must be capitalised in clean English
- neighbouring name tokens are joined into one span

With the spaCy model, it raised development-fold PERSON recall from 0.534 to
0.803.

## 6. Comparison with Presidio (SO2)

Presidio accepts one declared language per request, so it is run as English,
which is how it would be deployed today. It was given its standard large model
and a deliberately generous label mapping.

- **Sinhala-script recall:** 0.031.
- **NIC and ADDRESS:** 0 detected.
- **Precision:** 0.196, consistent with the 0.14 reported in the literature
  cited by the proposal.

## 7. Does synthetic data help? (spaCy ablation)

| Trained on | PERSON F1 | ADDRESS F1 | ORG F1 | Hybrid micro F1 |
|---|---|---|---|---|
| Real only (73 complete) | 0.597 | 0.560 | 0.415 | 0.657 |
| Synthetic only (50) | 0.331 (precision 0.27) | 0.150 | 0.133 | — |
| **Synthetic + real** | **0.639** | **0.714** | **0.571** | **0.699** |

Synthetic data helps as additional training data; on its own a model learns
the templates. The final transformer is also trained on synthetic + real.

## 8. Structured identifiers: rule layer (SO3)

| Data | Recall | Precision | F1 |
|---|---|---|---|
| Synthetic test (held-out templates + noise) | 0.984 | 0.983 | 0.984 |
| **Real, held-out** | **0.910** | **0.743** | **0.818** |

The rules handle every spoken form in the corpus:
- spaced and grouped digits
- Sinhala, romanised and English number words
- prefixed reference numbers and phone extensions
- ASR ordinal errors
- keywords after the number or in the previous turn

## 9. Cross-script entity resolution (Contribution 2, SO5)

| System | Synthetic test (unseen names) | Real, 73 tuning | **Real, held-out** |
|---|---|---|---|
| Exact string match | 0.000 | 0.000 | 0.000 |
| Transliteration + exact | 0.601 | 0.427 | 0.400 |
| **Transliteration + phonetic similarity** | **0.899** | **0.924** (185 people) | **0.933** (15 people) |

- **Linguistic handling:**
  - Sinhala case endings
  - titles in both scripts
  - surname particles (de / ද)
  - prenasalised ඳ / ඹ
  - English -ee = Sinhala ී
  - male/female name pairs
  - shared surnames
- **Pairwise precision is 1.000:** two different people were never merged.

## 10. Person roles (FR7)

| | People | Accuracy |
|---|---|---|
| Held-out real | 16 | 0.938 |
| Cross-validation, real | 203 | 0.833 |

This is a required schema field, not a novelty claim. Role never changes
whether a person is redacted.

## 11. Non-functional requirements

Measured by running the full pipeline on all 5 held-out recordings:

| Requirement | Measured | Target |
|---|---|---|
| NFR1 offline | **0** network attempts (all connections blocked) | none |
| NFR4 memory | **about 1 GB** peak process memory | below 8 GB |
| NFR5 latency | **99 ms** per document (median); about 7 s per recording | no perceptible delay in batch use |
| NFR2 data and re-identification map | training local only; map written to git-ignored storage only | no copy in version control |
| Leak check | **0** known identifiers left in the output | 0 |

Hardware: a consumer laptop with 8 CPU cores and no GPU.

## 12. Limitations

- **Precision 0.779.** C4 errs towards over-redaction, the safe direction.
  ACCOUNT precision is 0.458, partly because of reference numbers the
  annotators did not mark.
- **Small evaluation set:** 5 recordings, 289 spans, and 16 people for roles.
- **CPU training takes about 2 hours per model;** retraining as the corpus
  grows is slow but feasible.
- **Gold correction by the component owner.** It is rule-based and public,
  but awaits a second team member's approval.

## 13. Next steps

1. **Re-evaluate on a larger held-out set** as the corpus grows, and report
   performance against corpus size.
2. **Precision:** account and organisation false positives.
3. **Noise and robustness evaluation** on real C1 ASR output, with the team
   (Nov–Dec).
4. **Integration** of the four components through the documented
   input–output contract (Jan 2027).

## Reproducing these results

    cd c4
    python -m pytest tests                                       # 327 tests
    python src/transformer_ner.py train --data both              # ~2 h on CPU
    python src/roles.py train && python src/roles.py evaluate
    python src/evaluate.py --source real-eval --system pipeline:xlmr_both --labels proposal
    python src/evaluate.py --source real-eval --system presidio --labels proposal
    python src/resolve.py --source real-eval
    python src/pipeline.py --recording J26DS313_R0022 --offline-check --measure
    python src/demo.py                                           # live demo
