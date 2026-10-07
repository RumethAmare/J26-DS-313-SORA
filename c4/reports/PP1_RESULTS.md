# C4 — PP1 Results: Offline PII Detection and Redaction for Sinhala-English Text

J26-DS-313 · Component 4 · O. S. Jayathilaka (IT23247390) · updated 7 October 2026

## Summary

- On five **held-out real recordings**, the hybrid system (rules + learned model)
  reaches **micro F1 0.684, macro F1 0.786** across the seven proposal labels,
  against the established baseline of 0.231 / 0.289 and Microsoft Presidio's
  0.235 / 0.182. Recall, the primary metric, rises from 0.174 to **0.637**.
  Both proposal targets are met: overall F1 above 0.60, PERSON F1 above 0.50
  (0.587).
- On **Sinhala-script** identifiers, Presidio finds 3 of 96 (F1 0.025); this
  system reaches F1 **0.549**. That is the gap this component exists to close.
- **Cross-script entity resolution** (Contribution 2) links a Latin-script and
  a Sinhala-script mention of the same person with accuracy **0.933** on
  held-out recordings, above the 0.85 target. String matching scores 0.000.
- **Synthetic data**, used at the supervisor's direction while the real corpus
  grows, adds **+0.109 F1** (0.575 → 0.684) when combined with real data. It is
  not enough on its own (see §4).

## 1. Data and evaluation protocol

| Data | Recordings | Used for |
|---|---|---|
| Synthetic train | 50 (1,722 spans) | building rules, training models |
| Synthetic test | 60 (1,873 spans) | development score; frozen, never tuned on |
| Real, training and tuning | 73 | rule development, model training |
| **Real, held-out** | **5** (R0011, R0019, R0021, R0022, R0023; 289 gold spans) | **reported results only** |

- **Real data:** `SORA_Dataset` at commit `d8981fc`, 78 recordings with a C4
  layer.
- **Synthetic data** is generated from Sri Lankan identifier formats. NIC
  numbers encode the holder's birth date and sex, so NIC and DOB agree. The
  test split is held out from train in three ways:
  - its sentence templates never appear in train
  - its names, places and organisations never appear in train
  - it adds real speech effects: answers given in a separate turn, fillers,
    a dropped NIC "V", repeated digits, "zero binduwai" restatements and ASR
    ordinal errors
- **The held-out real recordings are the same five the baseline was scored
  on.** No rule, threshold or model was tuned on them, and no model was
  trained on them.
- **Matching is strict**: a prediction counts only if its start, end and label
  all equal the gold span. Recall is the primary metric, because a missed
  identifier is a disclosure.
- Every number below is read from a JSON file in `c4/eval/`, which records the
  code commit it was produced at.

### Result history

Reported for transparency:

| Step | Held-out micro F1 |
|---|---|
| Model trained on 22 real recordings; original gold annotations | 0.570 |
| Same model; gold corrected to the schema (`d8981fc`, see §9) | 0.617 |
| **Model retrained on all 73 non-held-out real recordings** | **0.684** |

The gold correction was made by a rule-based script
(`scripts/c4/fix_c4_annotations.py`) applied uniformly to all 78 recordings
against `DATA_CONTRACT.md` and the C4 schema. It was not tuned to the system,
and its second-member review is pending, as the contract requires.

## 2. Main result: held-out real recordings, seven proposal labels

| System | Recall | Precision | Micro F1 | Macro F1 |
|---|---|---|---|---|
| Microsoft Presidio (`en_core_web_lg`) | 0.294 | 0.196 | 0.235 | 0.182 |
| Baseline (proposal, spaCy CNN) | 0.174 | 0.341 | 0.231 | 0.289 |
| Rule layer only | 0.270 | 0.736 | 0.395 | 0.503 |
| **Hybrid: rules + model (synthetic + real)** | **0.637** | **0.739** | **0.684** | **0.786** |

The baseline was scored on transcript utterances and summaries only. On
exactly those documents, the hybrid reaches **F1 0.651, macro 0.751**, and
Presidio reaches 0.207 / 0.161.

**Per label, hybrid system:**

| Label | Recall | Precision | F1 |
|---|---|---|---|
| NIC | 1.000 | 1.000 | 1.000 |
| DOB | 1.000 | 1.000 | 1.000 |
| PHONE | 0.967 | 0.935 | 0.951 |
| ADDRESS | 0.625 | 0.909 | 0.741 |
| ORG | 0.735 | 0.581 | 0.649 |
| PERSON | 0.464 | 0.798 | 0.587 |
| ACCOUNT | 0.759 | 0.458 | 0.571 |

**Per script:**

| Script | Presidio F1 | Hybrid F1 |
|---|---|---|
| Latin | 0.342 | 0.743 |
| Sinhala | **0.025** | **0.549** |

## 3. Comparison with Presidio (SO2)

Presidio accepts one declared language per request, so it is run as English,
which is how it would be deployed on this data today. It was given its
standard large model and a deliberately **generous** label mapping: any
national-ID type counts as NIC, any date counts as DOB, and any location
counts as ADDRESS.

- **Sinhala script:** recall 0.031. Presidio does not see Sinhala-script
  identifiers.
- **Precision:** 0.196. This is consistent with the 0.14 reported for
  Presidio on numerically dense conversation in the literature cited by the
  proposal.
- **NIC and ADDRESS:** 0 detected. Presidio has no Sri Lankan NIC format, and
  its location type does not match personal addresses.

## 4. Does synthetic data help? (model ablation)

The same CPU spaCy NER model was trained on three different data sets and
scored on the same held-out recordings. Model F1 per label is shown, then the
hybrid micro F1 with the same rule layer:

| Trained on | PERSON F1 | ADDRESS F1 | ORG F1 | Hybrid micro F1 |
|---|---|---|---|---|
| Real only (73 recordings) | 0.398 | 0.200 | 0.517 | 0.575 |
| Synthetic only (50 recordings) | 0.331 (precision 0.27) | 0.150 | 0.133 | — |
| **Synthetic + real** | **0.580** | **0.741** | **0.649** | **0.684** |

**Conclusion:** synthetic data adds value as additional training data, and it
still does so with 73 real recordings available. On its own it is not enough:
a model trained only on synthetic data learns the templates, not real speech,
and produces many false positives.

## 5. Structured identifiers: rule layer (SO3)

| Data | Recall | Precision | F1 |
|---|---|---|---|
| Synthetic test (held-out templates + noise) | 0.984 | 0.983 | 0.984 |
| Real, 73 tuning recordings | 0.778 | 0.735 | 0.756 |
| **Real, held-out** | **0.910** | **0.743** | **0.818** |

- The rules handle every spoken form found in the corpus:
  - compact, spaced and grouped digits
  - Sinhala and romanised number words ("බිංදුවයි හතයි …")
  - English number words
  - reference numbers with letter prefixes
  - phone extensions
  - dates with ASR ordinal errors
  - keywords that come after the number (Sinhala is verb-final) or in the
    previous turn
- Every injected speech-noise type is detected at recall 1.000 on the
  synthetic test split.
- Held-out recall is similar for Latin (0.910) and Sinhala (0.909) script.

## 6. Cross-script entity resolution (Contribution 2, SO5)

| System | Synthetic test (unseen names) | Real, 73 tuning recordings | **Real, held-out** |
|---|---|---|---|
| Exact string match | 0.000 | 0.000 | 0.000 |
| Transliteration + exact match | 0.601 | — | 0.400 |
| **Transliteration + phonetic similarity** | **0.899** | **0.839** (168 entities) | **0.933** (15 entities) |

- **Method:**
  1. Sinhala script is romanised character by character, honouring
     al-lakuna and conjuncts (ප්‍රනාන්දු → pranaandu).
  2. Both scripts are reduced to one phonetic key and consonant skeleton, so
     "Fernando" and ප්‍රනාන්දු both become p-r-n-n-d.
  3. Mentions are clustered per recording.
  4. The similarity threshold was tuned on the synthetic **train** split only,
     then frozen.
- **Linguistic handling:** Sinhala case endings (අමාශිට "to Amashi"),
  male/female name pairs (Mahesh ≠ Maheshi), and shared surnames (Kamal
  Perera ≠ Nimal Perera).
- **Pairwise precision is 1.000 on held-out real data:** two different people
  were never merged.
- **Every synthetic failure is an inherent ambiguity:** two people in the call
  share a first name.
- **On the larger tuning set (73 recordings)** accuracy is 0.839. The newer
  recordings contain name patterns not yet handled; this is the next
  improvement target and is worked on with tuning data only.

## 7. Redaction and re-identification (FR6, FR8, NFR2)

- One entity receives one placeholder in the transcript, both C2 renderings
  and the summary. ORG, LOCATION and organisation hotlines stay visible.
- An identifier found in one document is redacted wherever it recurs in the
  recording (propagation).
- A leak check confirms that no known identifier survives in the output.
- The re-identification map reverses redaction exactly. It is written only to
  git-ignored storage; the code refuses any location git would commit.

## 8. Limitations

- **PERSON recall is 0.464.** It is the largest remaining gap: PERSON accounts
  for 82 of the 105 missed spans.
- **Small evaluation set.** The held-out set is 5 recordings (289 gold spans);
  DOB (12) and ADDRESS (16) have few instances.
- **The model is a CPU spaCy CNN.** The proposal's multilingual transformer
  (xlm-roberta-base) needs a GPU, which this machine lacks.
- **The resolver was evaluated on annotated name spans,** which isolates
  linking as the proposal's ablation requires. End-to-end linking depends on
  PERSON recall.
- **The gold correction was made by the component owner.** It is rule-based
  and public, but awaits a second team member's approval.

## 9. Data quality

**Fixed in `SORA_Dataset@d8981fc`**, by `scripts/c4/fix_c4_annotations.py`,
which writes only `annotations/c4/`:

- label/role interchange (317 rows)
- summary spans recorded against a non-contract doc_id (503 rows)
- role vocabulary and redact flags
- stale offsets (40)
- trailing punctuation (51)
- overlapping spans (6)
- out-of-schema ID labels (7)
- emails labelled ACCOUNT (9)
- two privacy leaks in published `redacted.json` files (R0018, R0027), with
  all 78 regenerated in one format and no surviving identifier

**Still open, needing human judgement:**

| Item | Scale |
|---|---|
| Identifiers with no recorded owner (role left empty by decision) | 1,434 rows |
| Spans whose surface occurs more than once, so the position is ambiguous | 9 |
| R0061 rows not convertible from token offsets | 3 |
| PERSON never annotated in Sinhala script | 15 recordings |
| Recording IDs not in `J26DS313_R####` form, in C1, C2 and C4 alike | R0062–R0065 (team-wide fix) |

## 10. Next steps

1. **Transformer model (SO4):** fine-tune xlm-roberta-base on GPU (Colab) on
   the same data. It plugs in behind the existing detector interface.
2. **PERSON recall and resolver accuracy on the new recordings,** improved
   using tuning data only.
3. **End-to-end pipeline command** taking C2's output and producing redacted
   files plus the map (FR1, FR9), and person-role classification (FR7).
4. **Re-run the full evaluation as the corpus grows,** reporting performance
   against corpus size.

## Reproducing these results

    cd c4
    python -m pytest tests                                   # 301 tests
    python src/synthetic.py                                  # regenerate synthetic data
    python src/ner.py train --data both                      # ~20 min on CPU
    python src/evaluate.py --source real-eval --system hybrid:both --labels proposal
    python src/evaluate.py --source real-eval --system presidio --labels proposal
    python src/resolve.py --source real-eval
    python src/redact.py --synthetic SYN_T0002 --model both  # live demo
