# C1 Data-Quality Notes (Section D of the implementation plan)

Findings from re-running `validate.py` against every recording in
`SORA_Dataset/manifests/recordings_current.csv` and auditing the gold C1
annotations, before any WER/LID numbers are quoted downstream. Source data
lives in the shared `SORA_Dataset` repo; per project convention nothing was
edited there directly — corrected artifacts are kept here in `c1/results/`
and `c1/docs/` for the team to review and apply.

## D.1 — Manifest status vs validate.py truth

All 27 recordings are marked `VALIDATED` in `manifests/recordings_current.csv`,
but re-running `validate.py <rid> --audio ...` for all 27 (see
`c1/scripts/rerun_validate_all.py`, full results in
`c1/results/validate_status_audit.csv`) shows **6** actually fail:

| recording | fails | what fails |
|---|---|---|
| J26DS313_R0008 | 17 | 22 malformed C1 token lines (junk in timestamps); 16/21 C4 PII spans land on the wrong text |
| J26DS313_R0019 | 2 | 3 tokens outside audio; UEM runs past audio |
| J26DS313_R0020 | 1 | UEM runs past audio |
| J26DS313_R0021 | 2 | RTTM runs past audio; UEM runs past audio |
| J26DS313_R0025 | 1 | UEM runs past audio |
| J26DS313_R0026 | 1 | UEM runs past audio |

The plan document that scoped this work only anticipated 5 of these
(R0019/20/21/25/26, from notes already present in the manifest); **R0008 was
not previously flagged** and has a more serious problem than the rest — its
C1 token file itself is malformed, not just its C3/C4 timeline. It should be
treated the same as R0017 (excluded from Task 3 train/eval, see D.2) until
re-annotated, not just downgraded to `ANNOTATED_DRAFT`.

A corrected copy of the manifest with `status` fixed for these 6 rows is at
`c1/results/recordings_current_corrected.csv`. To apply: replace
`SORA_Dataset/manifests/recordings_current.csv` with this file (or merge the
6 status cells) once a teammate has reviewed it.

## D.2 — R0017 mislabeling (confirmed)

Every token in `annotations/c1/J26DS313_R0017.tokens.jsonl` is tagged `EN`
(237 EN, 14 OTHER, 0 SI) despite containing obvious romanized-Sinhala words —
confirmed by inspection: `eken`, `mama`, `oneda`, `Mage`, `aluth`, `gedara`,
`thiyenawa` are all tagged EN. This is a real gold-data bug (the recording
was very likely tagged as if it were the same speaker/language context as a
neighboring EN-only recording), not a heuristic failure.

**Action**: exclude R0017 from Task 3's (LID classifier) train/eval sets.
**Also exclude R0008** for the same reason — its token file is structurally
broken (see D.1), so any lang labels pulled from it are unreliable too. Both
exclusions should be stated explicitly in the Task 8 dataset card
(`C1_DATASET_CARD.md`) when it's written. Full re-annotation of both is
future work, not in scope before the proposal.

## D.3 — Numeric-token tagging inconsistency (confirmed)

Corpus-wide scan of purely-numeric tokens (digits only, punctuation
stripped) across all `annotations/c1/*.tokens.jsonl`:

| tag | count | examples |
|---|---|---|
| EN | 192 | `9`, `5`, `3`, `2`, `0`, `1`, `4`, ... |
| OTHER | 61 | `23`, `1990`, `5`, `0777998876`, `44`, `2003`, `8`, ... |
| SI | 1 | `17` |

The same short numeral (e.g. `5`, `8`) appears tagged both `EN` and `OTHER`
in different recordings/contexts, with no consistent rule (phone-number-like
long digit strings lean `OTHER`, short standalone digits lean `EN`, but not
reliably). This is annotation-guideline drift, not something a smarter
downstream model can fix.

**Proposed going-forward rule** (to add to `DATA_CONTRACT.md`'s C1 section,
after the existing `lang` field description):

> **Numeric tokens.** A token consisting only of digits (optionally with
> internal `-`/`/` as in phone numbers or dates) is tagged by how it is
> *read aloud* in the recording, not by its script: digit-by-digit readings
> in Sinhala (e.g. a phone number read "බින්දුව හත හත...") are `OTHER` only
> if the digit string itself functions as a foreign-script insertion (e.g.
> written in Tamil numerals); otherwise tag by the spoken language of the
> reading — Sinhala-spoken digit strings are `SI`, English-spoken digit
> strings are `EN`. Purely-written numerals with no clear spoken-language
> cue (e.g. IDs read digit-by-digit in a neutral tone) default to the
> `lang` of the surrounding utterance. This rule applies **going forward**;
> historical tokens are not force-relabeled — see the `validate.py` warning
> check below instead.

**Proposed `validate.py` check [7]** (warn-only, does not fail the
recording, so historical data isn't invalidated): flag any purely-numeric
token whose `lang` differs from the majority `lang` of its 2 neighboring
tokens, as a disagreement worth a human glance. Sketch:

```python
# ---- numeric-token tagging disagreement (warn only) ----
print("\n[7] numeric-token tagging vs. neighboring context")
numeric_warns = 0
for i, t in enumerate(toks):
    core = t["token"].strip(".,?!").replace("-", "").replace("/", "")
    if not core.isdigit():
        continue
    neighbors = [toks[j]["lang"] for j in (i - 1, i + 1) if 0 <= j < len(toks)]
    if neighbors and t["lang"] not in neighbors:
        numeric_warns += 1
if numeric_warns:
    warn(f"{numeric_warns} numeric token(s) tagged differently from both neighbors — check numeric-tagging rule")
else:
    ok("numeric-token tagging consistent with neighboring context")
```

Both the `DATA_CONTRACT.md` paragraph and the `validate.py` check are left
as proposed text/code here rather than applied directly to `SORA_Dataset`,
per project convention — apply when a teammate has reviewed the rule.
