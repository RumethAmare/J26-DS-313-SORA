# C4 — PP1 Results: Offline PII Detection and Redaction for Sinhala-English Text

J26-DS-313 · Component 4 · O. S. Jayathilaka (IT23247390) · 1 October 2026

## Summary

- On five **held-out real recordings**, the hybrid system (rules + learned model)
  reaches **micro F1 0.570, macro F1 0.640** across the seven proposal labels,
  against the established baseline of 0.231 / 0.289 and Microsoft Presidio's
  0.228 / 0.175. Recall, the primary metric, rises from 0.174 to 0.485.
- On **Sinhala-script** identifiers, Presidio finds 3 of 98 (F1 0.025); this
  system reaches F1 0.397. That is the gap this component exists to close.
- **Cross-script entity resolution** (Contribution 2) links a Latin-script and
  a Sinhala-script mention of the same person with accuracy **0.933** on
  held-out recordings, above the 0.85 target. String matching scores 0.000.
- Following the supervisor's direction, the component was built on
  **synthetic data** while the real corpus grows. Adding synthetic data to
  training raised held-out F1 from 0.454 to 0.570. Synthetic data alone is
  not enough (see §4).

## 1. Data and evaluation protocol

| Data | Recordings | Used for |
|---|---|---|
| Synthetic train | 50 (1,722 spans) | building rules, training models |
| Synthetic test | 60 (1,873 spans) | development score; frozen, never tuned on |
| Real, tuning | 22 | rule development, model training |
| **Real, held-out** | **5** (R0011, R0019, R0021, R0022, R0023) | **reported results only** |

- **Synthetic data** is generated from Sri Lankan identifier formats. NIC
  numbers encode the holder's birth date and sex, so NIC and DOB agree. The
  test split is held out from train in three ways:
  - its sentence templates never appear in train
  - its names, places and organisations never appear in train
  - it adds real speech effects: answers given in a separate turn, fillers,
    a dropped NIC "V", repeated digits, "zero binduwai" restatements and ASR
    ordinal errors
- **The held-out real recordings are the same five the baseline was scored
  on**, so the numbers are directly comparable. No rule, threshold or model
  was tuned on them.
- **Matching is strict**: a prediction counts only if its start, end and label
  all equal the gold span. Recall is the primary metric, because a missed
  identifier is a disclosure.
- Every number below is read from a JSON file in `c4/eval/`, which records
  the code commit it was produced at.

## 2. Main result: held-out real recordings, seven proposal labels

| System | Recall | Precision | Micro F1 | Macro F1 |
|---|---|---|---|---|
| Microsoft Presidio (`en_core_web_lg`) | 0.281 | 0.192 | 0.228 | 0.175 |
| Baseline (proposal, spaCy CNN) | 0.174 | 0.341 | 0.231 | 0.289 |
| Rule layer only | 0.237 | 0.660 | 0.349 | 0.441 |
| **Hybrid: rules + model (synthetic + real)** | **0.485** | **0.691** | **0.570** | **0.640** |

The baseline was scored on transcript utterances and summaries only. On
exactly those documents, the hybrid reaches **F1 0.537, macro 0.604**, and
Presidio reaches 0.200 / 0.154.

**Per label, hybrid system:**

| Label | Recall | F1 | Gold spans |
|---|---|---|---|
| NIC | 1.000 | 1.000 | 15 |
| DOB | 0.750 | 0.750 | 12 |
| PHONE | 0.800 | 0.787 | 30 |
| ACCOUNT | 0.688 | 0.550 | 32 |
| PERSON | 0.340 | 0.493 | 156 |
| ADDRESS | 0.375 | 0.480 | 16 |
| ORG | 0.412 | 0.418 | 34 |

**Per script:**

| Script | Presidio F1 | Hybrid F1 |
|---|---|---|
| Latin | 0.331 | 0.640 |
| Sinhala | **0.025** | **0.397** |

## 3. Comparison with Presidio (SO2)

Presidio accepts one declared language per request, so it is run as English,
which is how it would be deployed on this data today. It was given its
standard large model and a deliberately **generous** label mapping: any
national-ID type counts as NIC, any date counts as DOB, and any location
counts as ADDRESS.

- **Sinhala script:** recall 0.031. Presidio does not see Sinhala-script
  identifiers.
- **Precision:** 0.192. This is consistent with the 0.14 reported for
  Presidio on numerically dense conversation in the literature cited by the
  proposal.
- **PERSON:** Presidio's recall (0.353) is comparable to this system's
  (0.340). The advantage on names comes from precision (0.898 vs 0.470), not
  from finding more of them.

## 4. Does synthetic data help? (model ablation)

The same CPU spaCy NER model was trained on three different data sets and
scored on the same held-out recordings, combined with the same rule layer:

| Trained on | PERSON F1 | ADDRESS F1 | ORG F1 | Hybrid micro F1 |
|---|---|---|---|---|
| Real only | 0.284 | 0.400 | 0.000 | 0.454 |
| Synthetic only | 0.328 (precision 0.27) | 0.050 | 0.133 | — |
| **Synthetic + real** | **0.493** | **0.480** | **0.418** | **0.570** |

**Conclusion:** synthetic data adds value as additional training data. On its
own it is not enough: a model trained only on synthetic data learns the
templates, not real speech, and produces many false positives.

## 5. Structured identifiers: rule layer (SO3)

| Data | Recall | F1 |
|---|---|---|
| Synthetic test (held-out templates + noise) | 0.984 | 0.984 |
| Real, held-out | 0.786 | 0.707 |
| Real, held-out, gold trailing punctuation ignored | 0.876 | 0.788 |

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
- The third row shows how much of the gap is annotation noise (gold spans
  ending in "."). The strict row is the reported figure.

## 6. Cross-script entity resolution (Contribution 2, SO5)

| System | Synthetic test (unseen names) | Real, tuning | **Real, held-out** |
|---|---|---|---|
| Exact string match | 0.000 | 0.000 | 0.000 |
| Transliteration + exact match | 0.601 | 0.566 | 0.400 |
| **Transliteration + phonetic similarity** | **0.899** | **1.000** | **0.933** |

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
  share a first name, so a bare "Buddhika" cannot be attributed from the name
  alone.
- **Effect on redaction:** "Nimal Perera" and "නිමල් පෙරේරා" receive the same
  placeholder, `[PERSON_1]`, in every document.

## 7. Redaction and re-identification (FR6, FR8, NFR2)

- One entity receives one placeholder in the transcript, both C2 renderings
  and the summary. ORG, LOCATION and organisation hotlines stay visible.
- An identifier found in one document is redacted wherever it recurs in the
  recording (propagation). This raised synthetic-test recall from 0.984 to
  0.993; it has no measured recall effect on real data yet.
- A leak check confirms that no known identifier survives in the output.
- The re-identification map reverses redaction exactly. It is written only to
  git-ignored storage; the code refuses any location git would commit.

## 8. Limitations

- **PERSON recall is 0.340.** The model is precise but misses about two names
  in three. This is the largest remaining gap: PERSON accounts for 103 of the
  152 missed spans.
- **Small evaluation set.** The held-out set is 5 recordings (295 gold spans).
  ADDRESS (16) and DOB (12) have few instances, so their scores move a lot
  with each prediction.
- **The model is a CPU spaCy CNN.** The proposal's multilingual transformer
  (xlm-roberta-base) needs a GPU: this machine has none, and CPU training
  measured 110–130 s per step.
- **The resolver was evaluated on annotated name spans,** which isolates
  linking as the proposal's ablation requires. End-to-end linking depends on
  PERSON recall.
- **Real-data scores are preliminary.** The corpus is still being built.

## 9. Data-quality findings for the dataset team

The validator written for this component (`src/validate.py`) checked all 1,071
real annotation rows and found:

| Finding | Scale |
|---|---|
| Stale offsets (text edited after annotation) | R0008, 16 spans |
| Gold spans ending in "." (against §4 of the schema) | about 20 spans |
| Email addresses labelled ACCOUNT | 9 spans |
| Reference numbers annotated inconsistently (POL/MASIT yes, CLM/ADM never) | schema decision needed |
| Phone extensions sometimes inside the PHONE span, sometimes not | convention needed |
| Missing `recording` field, legacy role values | 525 rows |
| PERSON never seen in Sinhala script | R0011, R0027 |
| Overlapping spans | R0021, R0027, 5 spans |

## 10. Next steps

1. **Transformer model (SO4):** fine-tune xlm-roberta-base on GPU (Colab) on
   the same data. It plugs in behind the existing detector interface.
2. **PERSON recall:** tune on a validation slice of the real tuning
   recordings (never the held-out set), and add more name-context variety to
   the synthetic train split.
3. **Re-run the full evaluation as the corpus grows**, and report the
   performance trajectory against corpus size, as the proposal plans.
4. **Inter-annotator agreement** and resolution of the data-quality findings
   above, with the dataset team.

## Reproducing these results

    cd c4
    python -m pytest tests                                   # 288 tests
    python src/synthetic.py                                  # regenerate synthetic data
    python src/ner.py train --data both                      # ~5 min on CPU
    python src/evaluate.py --source real-eval --system hybrid:both --labels proposal
    python src/evaluate.py --source real-eval --system presidio --labels proposal
    python src/resolve.py --source real-eval
    python src/redact.py --synthetic SYN_T0002 --model both  # live demo
