# C1 gold-data audit

Produced by `scripts/audit_gold.py`. Report only; nothing is modified.

- gold token files: **78** (28,979 tokens)
- usable after exclusions: **66** (25,582 tokens)
- with audio: **74**
- not in manifest: **0**

## Excluded recordings

| recording | reason |
|---|---|
| J26DS313_R0008 | token file structurally malformed (junk in timestamps) |
| J26DS313_R0017 | every token tagged EN, including obvious romanized Sinhala |
| J26DS313_R0051 | partial: 14% of Sinhala function words tagged EN |
| J26DS313_R0054 | every token tagged EN despite code-mixed speech |
| J26DS313_R0055 | every token tagged EN despite code-mixed speech |
| J26DS313_R0056 | 290/291 tokens tagged EN despite code-mixed speech |
| J26DS313_R0058 | 369/373 tokens tagged SI, including plain English ('Customer service desk') |
| J26DS313_R0059 | every token tagged EN despite code-mixed speech |
| J26DS313_R0062 | partial: 51% of Sinhala function words tagged EN |
| J26DS313_R0063 | partial: 35% of Sinhala function words tagged EN |
| J26DS313_R0064 | partial: 22% of Sinhala function words tagged EN |
| J26DS313_R0065 | partial: 26% of Sinhala function words tagged EN |

## Near-duplicate transcripts

None found.

## Recordings with issues (NO_AUDIO alone omitted)

`SI fw→EN` is the share of unambiguous romanized-Sinhala function words (`eka`, `mata`, `oyata`, …) tagged EN, with their count; it should be 0%.

| recording | tokens | SI | EN | OTHER | switches | SI fw→EN | null ts | SI script | issues |
|---|---|---|---|---|---|---|---|---|---|
| J26DS313_R0006 | 108 | 75 | 33 | 0 | 48 | 0% (0) | 0 | unicode | SINHALA_SCRIPT_AS_EN |
| J26DS313_R0008 | 22 | 18 | 4 | 0 | 6 | 0% (0) | 0 | unicode | EXCLUDED SINHALA_SCRIPT_AS_EN |
| J26DS313_R0009 | 67 | 53 | 14 | 0 | 26 | 0% (0) | 0 | unicode | SINHALA_SCRIPT_AS_EN |
| J26DS313_R0010 | 62 | 44 | 18 | 0 | 27 | 0% (0) | 0 | unicode | SINHALA_SCRIPT_AS_EN |
| J26DS313_R0015 | 278 | 177 | 96 | 5 | 116 | 0% (0) | 0 | unicode | SINHALA_SCRIPT_AS_EN |
| J26DS313_R0017 | 251 | 0 | 237 | 14 | 21 | 100% (49) | 0 | none | EXCLUDED PARTIAL_MISLABEL |
| J26DS313_R0035 | 383 | 189 | 128 | 66 | 148 | 0% (0) | 85 | unicode | NULL_TIMESTAMPS |
| J26DS313_R0043 | 357 | 233 | 85 | 39 | 114 | 0% (0) | 1 | unicode | NULL_TIMESTAMPS |
| J26DS313_R0045 | 398 | 245 | 128 | 25 | 150 | 0% (0) | 114 | unicode | NULL_TIMESTAMPS |
| J26DS313_R0046 | 737 | 468 | 226 | 43 | 251 | 0% (0) | 2 | unicode | NULL_TIMESTAMPS |
| J26DS313_R0051 | 310 | 231 | 78 | 1 | 79 | 14% (74) | 0 | romanized | EXCLUDED PARTIAL_MISLABEL |
| J26DS313_R0054 | 431 | 234 | 194 | 3 | 184 | 0% (0) | 0 | unicode | EXCLUDED |
| J26DS313_R0055 | 392 | 0 | 392 | 0 | 0 | 100% (91) | 0 | none | EXCLUDED BULK_LABEL |
| J26DS313_R0056 | 291 | 1 | 290 | 0 | 2 | 100% (51) | 0 | romanized | EXCLUDED BULK_LABEL |
| J26DS313_R0058 | 373 | 369 | 4 | 0 | 2 | 0% (99) | 0 | romanized | EXCLUDED BULK_LABEL |
| J26DS313_R0059 | 173 | 0 | 173 | 0 | 0 | 100% (50) | 0 | none | EXCLUDED BULK_LABEL |
| J26DS313_R0062 | 255 | 66 | 189 | 0 | 77 | 51% (67) | 0 | romanized | EXCLUDED PARTIAL_MISLABEL NO_AUDIO |
| J26DS313_R0063 | 233 | 56 | 177 | 0 | 63 | 35% (49) | 0 | romanized | EXCLUDED PARTIAL_MISLABEL NO_AUDIO |
| J26DS313_R0064 | 243 | 57 | 186 | 0 | 68 | 22% (50) | 0 | romanized | EXCLUDED PARTIAL_MISLABEL NO_AUDIO |
| J26DS313_R0065 | 423 | 113 | 310 | 0 | 140 | 26% (93) | 0 | romanized | EXCLUDED PARTIAL_MISLABEL NO_AUDIO |
| J26DS313_R0068 | 616 | 336 | 244 | 36 | 221 | 0% (0) | 9 | unicode | NULL_TIMESTAMPS |
| J26DS313_R0103 | 810 | 713 | 97 | 0 | 148 | 0% (0) | 172 | unicode | NULL_TIMESTAMPS |

## Sinhala script convention (usable recordings)

- mixed: 1
- romanized: 10
- unicode: 55
