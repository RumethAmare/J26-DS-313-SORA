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

## D.4 — October 2026 batch (R0029–R0065)

The batch pulled on 2026-10-04 added C1 gold for 34 recordings (R0031–R0065;
R0029, R0030, R0047 have C3 only), taking the corpus from 5,818 to 17,782 gold
tokens. `c1/scripts/audit_gold.py` now checks every gold file automatically
(report: `c1/results/c1_gold_audit.md`). `validate.py` passes all of these
problems, because it checks files against each other, not labels against
language.

**Label errors — 11 more recordings excluded** (the full list, with reasons,
is `EXCLUDED_RECORDINGS` in `c1/scripts/corpus.py`):

| problem | recordings | evidence |
|---|---|---|
| bulk EN (R0017 pattern) | R0053, R0054, R0055, R0056, R0059 | every token EN, zero switches, in code-mixed speech (`ekata/EN kohomada/EN sahaya/EN`) |
| bulk SI | R0058 | 369/373 SI, incl. `Customer service desk` |
| partial | R0051, R0062, R0063, R0064, R0065 | 14–51% of unambiguous Sinhala function words (`eka`, `mata`, `oyata`) tagged EN; R0051 also tags English (`billing`, `documents`, `sir`) SI |

Draft corrections for all 12 label-error recordings (these 11 plus R0017) are
in `c1/predictions/pretag/` (`c1/scripts/pretag_relabel.py`): the Task 3
hybrid LID's labels, with the original kept as `lang_original` and a
`needs_review` flag. They are drafts — a person must check them before they
replace anything in `SORA_Dataset`. The model never predicts OTHER, so Tamil
tokens must be fixed by hand.

**Duplicate.** R0053 is a near-duplicate of R0052 (94% sequence similarity) —
the same conversation annotated twice. One of the two should be removed from
the dataset.

**Numeral convention flipped twice (D.3, much worse).**

| recordings | numerals tagged EN | OTHER | SI |
|---|---|---|---|
| R0002–R0028 | 197 | 58 | 1 |
| R0031–R0049 | 0 | 654 | 0 |
| R0050–R0061 | 78 | 4 | 10 |

One annotator group tags every digit OTHER. This is now the largest source of
LID "error" (~5% of all tokens) and makes `OTHER` 97% numerals. Note that the
proposed check [7] above would **not** catch it: a phone number tagged OTHER
digit by digit has OTHER neighbours on both sides. The check should compare a
numeral run against the nearest *non-numeric* tokens either side instead.

**Tamil tagged inconsistently.** `Vanakkam` is OTHER in R0050/R0051, SI in
R0052, EN in R0053/R0064; `වනක්කම්`, `රොම්බ`, `නන්ද්‍රි` (R0049) are SI.
Curated Tamil ground truth: `c1/data/tamil_ground_truth.json`.

**Bookkeeping** (blocks joins, not labels):

- **No audio** for 33 of the 34 new recordings (only R0049's WAV is
  committed), so Tasks 1, 2 and 6 cannot use them yet.
- **Not in the manifest:** none of R0029–R0065 are in
  `recordings_current.csv`, so `rerun_validate_all.py` does not see them.
- **No prefix:** R0062–R0065 are named `R0062.*` rather than
  `J26DS313_R0062.*` in C1, C2 and C4 (and their `utt_id`s lack it too).
  `c1/scripts/corpus.canonical_rid()` papers over this on read.
- **Null timestamps:** R0052 (185/375 tokens), R0045 (114), R0035 (85),
  R0046 (13), R0043 (1).
- **Script convention** is still mixed, per recording: in the new batch
  R0050–R0052, R0058 and R0062–R0065 write Sinhala romanized and the rest use
  Sinhala Unicode (the first batch is split the same way — see
  `sinhala_script_convention` in `c1_gold_audit.csv`). R0057 writes English
  words in Sinhala script (`කස්ටමර් සර්විස්`) and tags them SI.

## D.5 — Status after the 2026-10-05 update

Pulled 2026-10-05: re-annotated R0052/R0053, C1 gold for R0029, R0030, R0047,
R0068–R0075 and R0100–R0105, and audio for R0029–R0049 and the new
recordings. Corpus now 78 gold files; **66 usable, 25,582 tokens**.

| D.4 item | status |
|---|---|
| R0052 / R0053 duplicate + R0053 bulk EN | **fixed** — both re-annotated, no longer near-duplicates; R0053 back in |
| `Vanakkam` tagged inconsistently | **fixed** in R0052/R0053 (now OTHER); R0049 Sinhala-script Tamil and R0052 `Rombha` still SI |
| bulk mislabel R0054, R0055, R0056, R0058, R0059 | open — unchanged, still excluded |
| partial mislabel R0051, R0062–R0065 | open — unchanged, still excluded |
| numeral convention | open — every new recording tags digits OTHER, so OTHER is now the majority (1,019 of 1,315); R0002–R0028 and R0050–R0061 still tag EN |
| no audio | mostly fixed — 61 WAVs; R0050–R0065 (and R0007) still missing |
| not in manifest | open — no rows for R0029 onward |
| no prefix R0062–R0065 | open |
| null timestamps | open, with new cases: R0073 (217/690), R0103 (172), R0075 (161), R0045 (114), R0035 (85), R0101 (80), R0068 (25), R0046 (13), R0043 (1) |

The new recordings pass the label checks (no bulk or partial mislabelling).
R0100–R0105 are Sinhala-heavy (~85% SI) and write numbers as words, so they
contain no digit tokens.
