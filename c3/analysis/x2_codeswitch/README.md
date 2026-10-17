# X2: are speaker errors concentrated near Sinhala-English switches?

Inputs: C1 token language labels (dataset repo `annotations/c1/*.tokens.jsonl`), C3 truth and
community-1 drafts. `align.py` checks the two layers share a time base (97-100 % of token midpoints
fall inside labelled speech on all 22 matching clips). `x2.py` measures confusion within ±1 s of a
switch point vs elsewhere, excluding ±1 s around true speaker changes. Rule written before running.

Result (20 clips, 1,612 switch points): confusion 13.29 % near switches vs 11.25 % elsewhere
(ratio 1.18; clip-bootstrap 95 % CI of the difference −4.89 to +8.64 pp). **Not supported.**
