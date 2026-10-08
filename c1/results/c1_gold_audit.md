# C1 gold-data audit

Produced by `scripts/audit_gold.py`. Report only; nothing is modified.

- gold token files: **78** (28,979 tokens)
- usable after exclusions: **66** (25,582 tokens)
- with audio: **74**
- not in manifest: **51**

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
| J26DS313_R0029 | 258 | 155 | 89 | 14 | 131 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0030 | 282 | 173 | 95 | 14 | 128 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0031 | 658 | 367 | 241 | 50 | 247 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0032 | 288 | 174 | 78 | 36 | 100 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0033 | 382 | 217 | 137 | 28 | 145 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0034 | 397 | 215 | 146 | 36 | 134 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0035 | 383 | 189 | 128 | 66 | 148 | 0% (0) | 85 | unicode | NULL_TIMESTAMPS NOT_IN_MANIFEST |
| J26DS313_R0036 | 333 | 155 | 145 | 33 | 127 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0037 | 367 | 201 | 135 | 31 | 120 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0038 | 353 | 176 | 135 | 42 | 137 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0039 | 338 | 228 | 73 | 37 | 99 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0040 | 290 | 187 | 82 | 21 | 92 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0041 | 389 | 208 | 128 | 53 | 139 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0042 | 350 | 208 | 98 | 44 | 129 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0043 | 357 | 233 | 85 | 39 | 114 | 0% (0) | 1 | unicode | NULL_TIMESTAMPS NOT_IN_MANIFEST |
| J26DS313_R0044 | 387 | 245 | 98 | 44 | 115 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0045 | 398 | 245 | 128 | 25 | 150 | 0% (0) | 114 | unicode | NULL_TIMESTAMPS NOT_IN_MANIFEST |
| J26DS313_R0046 | 737 | 468 | 226 | 43 | 251 | 0% (0) | 13 | unicode | NULL_TIMESTAMPS NOT_IN_MANIFEST |
| J26DS313_R0047 | 852 | 447 | 369 | 36 | 370 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0048 | 214 | 149 | 61 | 4 | 70 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0049 | 445 | 225 | 192 | 28 | 207 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0050 | 356 | 157 | 198 | 1 | 128 | 0% (72) | 0 | romanized | NOT_IN_MANIFEST |
| J26DS313_R0051 | 310 | 231 | 78 | 1 | 79 | 14% (74) | 0 | romanized | EXCLUDED PARTIAL_MISLABEL NOT_IN_MANIFEST |
| J26DS313_R0052 | 411 | 186 | 223 | 2 | 188 | 0% (34) | 0 | mixed | NOT_IN_MANIFEST |
| J26DS313_R0053 | 369 | 208 | 160 | 1 | 147 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0054 | 431 | 234 | 194 | 3 | 184 | 0% (0) | 0 | unicode | EXCLUDED NOT_IN_MANIFEST |
| J26DS313_R0055 | 392 | 0 | 392 | 0 | 0 | 100% (91) | 0 | none | EXCLUDED BULK_LABEL NOT_IN_MANIFEST |
| J26DS313_R0056 | 291 | 1 | 290 | 0 | 2 | 100% (51) | 0 | romanized | EXCLUDED BULK_LABEL NOT_IN_MANIFEST |
| J26DS313_R0057 | 162 | 132 | 26 | 4 | 27 | 100% (3) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0058 | 373 | 369 | 4 | 0 | 2 | 0% (99) | 0 | romanized | EXCLUDED BULK_LABEL NOT_IN_MANIFEST |
| J26DS313_R0059 | 173 | 0 | 173 | 0 | 0 | 100% (50) | 0 | none | EXCLUDED BULK_LABEL NOT_IN_MANIFEST |
| J26DS313_R0060 | 208 | 161 | 47 | 0 | 45 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0061 | 298 | 194 | 104 | 0 | 122 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0062 | 255 | 66 | 189 | 0 | 77 | 51% (67) | 0 | romanized | EXCLUDED PARTIAL_MISLABEL NO_PREFIX NO_AUDIO NOT_IN_MANIFEST |
| J26DS313_R0063 | 233 | 56 | 177 | 0 | 63 | 35% (49) | 0 | romanized | EXCLUDED PARTIAL_MISLABEL NO_PREFIX NO_AUDIO NOT_IN_MANIFEST |
| J26DS313_R0064 | 243 | 57 | 186 | 0 | 68 | 22% (50) | 0 | romanized | EXCLUDED PARTIAL_MISLABEL NO_PREFIX NO_AUDIO NOT_IN_MANIFEST |
| J26DS313_R0065 | 423 | 113 | 310 | 0 | 140 | 26% (93) | 0 | romanized | EXCLUDED PARTIAL_MISLABEL NO_PREFIX NO_AUDIO NOT_IN_MANIFEST |
| J26DS313_R0068 | 616 | 336 | 244 | 36 | 221 | 0% (0) | 25 | unicode | NULL_TIMESTAMPS NOT_IN_MANIFEST |
| J26DS313_R0069 | 610 | 397 | 190 | 23 | 217 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0070 | 562 | 297 | 234 | 31 | 203 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0071 | 613 | 307 | 269 | 37 | 255 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0072 | 467 | 267 | 171 | 29 | 163 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0073 | 690 | 454 | 210 | 26 | 255 | 0% (0) | 217 | unicode | NULL_TIMESTAMPS NOT_IN_MANIFEST |
| J26DS313_R0074 | 752 | 411 | 306 | 35 | 371 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0075 | 512 | 286 | 189 | 37 | 209 | 0% (0) | 161 | unicode | NULL_TIMESTAMPS NOT_IN_MANIFEST |
| J26DS313_R0100 | 779 | 653 | 126 | 0 | 135 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0101 | 839 | 719 | 120 | 0 | 160 | 0% (0) | 80 | unicode | NULL_TIMESTAMPS NOT_IN_MANIFEST |
| J26DS313_R0102 | 822 | 721 | 101 | 0 | 156 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0103 | 810 | 713 | 97 | 0 | 148 | 0% (0) | 172 | unicode | NULL_TIMESTAMPS NOT_IN_MANIFEST |
| J26DS313_R0104 | 838 | 720 | 118 | 0 | 174 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |
| J26DS313_R0105 | 865 | 720 | 145 | 0 | 217 | 0% (0) | 0 | unicode | NOT_IN_MANIFEST |

## Sinhala script convention (usable recordings)

- mixed: 1
- romanized: 10
- unicode: 55
