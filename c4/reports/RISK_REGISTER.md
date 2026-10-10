# C4 Risk Register

J26-DS-313 · Component 4 · O. S. Jayathilaka (IT23247390) · October 2026

Likelihood and impact are rated High (H), Medium (M) or Low (L).

| # | Risk | L | I | Mitigation | Status |
|---|---|---|---|---|---|
| 1 | **Missed identifier is disclosed** in shared output | M | H | Recall is the primary metric (NFR3). Known identifiers are propagated to every occurrence, a leak check runs on every output, and system output on real data is never committed. | Mitigated; 82% of personal data hidden end to end, names 76% |
| 2 | **Re-identification map leaks** (it reverses redaction) | L | H | Written only to git-ignored storage; the code refuses any path git would commit; `*.reid.json` is blocked repo-wide (NFR2). | Mitigated, tested |
| 3 | **Real corpus too small or still being built** | H | H | Built synthetic-first with a frozen held-out synthetic test split. Trains on complete real recordings only. Synthetic + real beats real only by +0.042 F1 (0.657 → 0.699). | Mitigated |
| 4 | **Annotation quality** in the shared corpus | H | M | Validator with 9 checks; deterministic fix script for the C4 layer; open items reported to the team; a second-member review is required by the contract. | In progress |
| 5 | **No GPU** for the proposal's transformer (SO4) | H | M | CPU pipeline meets both F1 targets (0.755 overall, 0.757 PERSON). Transformer fine-tuning planned on Colab behind the same interface. | Mitigated for PP1 |
| 6 | **Upstream format drift** (C2 output changes shape) | H | M | Loader reads all 7 translation and 5 summary layouts found; each layout has a test; unusable rows are reported, not guessed. | Mitigated |
| 7 | **Small held-out set** (5 recordings, 289 spans) makes scores noisy | H | M | Strict matching, per-label and per-script reporting, cross-validation for roles; re-evaluate as the corpus grows. | Accepted; re-evaluate |
| 8 | **Cross-script linking errors** merge two people | L | M | Weakest-token scoring, male/female name rule, shared-surname rule. Pair precision 1.000 on held-out and tuning data. | Mitigated |
| 9 | **Bias from tuning on the test set** | M | H | Held-out recordings never trained or tuned on; thresholds tuned on synthetic train only; result history reported openly. | Mitigated |
| 10 | **Upstream ASR errors** (C1) corrupt identifiers | M | M | Rules handle spoken digits, restatements, dropped NIC letter, ordinal errors; robustness evaluation planned with the team (Nov–Dec). | Partly mitigated |
| 11 | **Integration with C1–C3** later fails | M | M | Input–output contract documented; `pipeline.py` consumes C2 files directly; the loader tolerates layout variants. | Plan exists (Jan 2027) |
| 12 | **Academic integrity: AI-assisted development** | M | H | AI-use log kept; every module tested; the student reviews and must be able to explain each component before PP1. | In progress |
