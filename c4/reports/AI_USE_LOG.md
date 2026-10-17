# C4 AI-Use Record (DRAFT — for the student to review, correct and complete)

J26-DS-313 · Component 4 · O. S. Jayathilaka (IT23247390)

> This draft was prepared by the AI tool itself from its session record. It
> lists what the tool did as fact. Everything marked **[student]** must be
> completed by the student, truthfully, before submission. Do not submit any
> line you have not verified.

## Tool

| | |
|---|---|
| Tool | Claude Code (Anthropic), model Claude Opus 5.5, in Visual Studio Code |
| Period | 29 September – 10 October 2026 |
| Access given | Read and run commands in the local `J26-DS-313-SORA` and `SORA_Dataset` repositories |
| Not given | Git commits: the student made every commit and push |

## Purpose and what the tool produced

| Area | What the tool did | Verification done |
|---|---|---|
| Synthetic data | Wrote `synthetic.py` (train split and frozen held-out test split with speech noise) | 60+ tests; validator run on every generated recording |
| Rule layer | Wrote `rules.py`, extended `numerals.py` | Tests; strict evaluation on synthetic test and held-out real data |
| Evaluation | Wrote `evaluate.py`, `validate.py`, `corpus.py` | Hand-worked scorer tests (e.g. P = R = 0.5 example) |
| Redaction | Wrote `redact.py`, `pipeline.py` | Exact restore test, leak check, offline network-block test |
| Cross-script resolution | Wrote `resolve.py` | Threshold tuned on synthetic train only; held-out accuracy 0.933 |
| Learned models | Wrote `ner.py`, `roles.py`; ran training on CPU | Held-out evaluation; cross-validation for roles |
| Baseline | Wrote `presidio_baseline.py` | Same strict harness as the system |
| Dataset | Audited the C4 layer; wrote `fix_c4_annotations.py` | Dry run before every apply; re-audit after |
| Reports | Drafted `PP1_RESULTS.md`, `ARCHITECTURE.md`, `RISK_REGISTER.md` | Numbers read from `eval/*.json` |
| Demo | Wrote `demo.py` | Manual run on sample text |

## Decisions taken by the student

Recorded in the session:

- Build on synthetic data first, as directed by the supervisor.
- Role classification is not a novelty claim.
- The student makes all commits.
- Identifiers with no recorded owner keep an empty role.
- STUDENT_ID / BOOKING_REF / INDEX_NO are mapped to ACCOUNT.
- All `redacted.json` files are regenerated in one format.
- Only C4 data is corrected; C1–C3 are untouched.
- Training uses completed recordings only.

## Accepted and rejected

**[student]** Which suggestions did you accept, change or reject, and why?

## Prompt summary

**[student]** Summarise in your own words what you asked the tool to do at each
stage.

## Your own modifications

**[student]** List every change you made yourself to code, data or reports.

## Understanding check before PP1

**[student]** For each module, confirm you can explain it without notes. Tick
only when true.

- [ ] numerals.py — how a spoken Sinhala number becomes digits
- [ ] rules.py — how a 12-digit number is classified as NIC or ACCOUNT
- [ ] ner.py — what the model learns and what data it was trained on
- [ ] resolve.py — how "Fernando" and "ප්‍රනාන්දු" are matched
- [ ] roles.py — which clues separate an agent from a customer
- [ ] redact.py — why the re-identification map cannot be committed
- [ ] evaluate.py — what strict span matching means, and why recall is primary
