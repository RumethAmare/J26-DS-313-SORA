# Draft re-labels for bulk-mislabelled recordings

Produced by `scripts/pretag_relabel.py` using the Task 3 hybrid LID (`models\lid_latin.ftz`). **Drafts, not gold** — files are in `predictions/pretag/` and must be reviewed before replacing anything in `SORA_Dataset/annotations/c1/`.

A token is marked `needs_review` if the suggestion differs from the original label or its confidence is below 0.8. The hybrid's cross-validated token accuracy is ~93%, so roughly 1 suggestion in 15 will be wrong; it never predicts OTHER, so Tamil words (e.g. `Vanakkam`) must be fixed by hand.

| recording | tokens | suggested SI | suggested EN | labels changed | needs review | switches |
|---|---|---|---|---|---|---|
| J26DS313_R0017 | 251 | 118 | 131 | 130 | 142 | 104 |
| J26DS313_R0054 | 426 | 203 | 222 | 204 | 233 | 205 |
| J26DS313_R0055 | 392 | 195 | 196 | 196 | 218 | 176 |
| J26DS313_R0056 | 291 | 113 | 177 | 114 | 124 | 125 |
| J26DS313_R0058 | 373 | 223 | 149 | 146 | 181 | 162 |
| J26DS313_R0059 | 173 | 95 | 78 | 95 | 108 | 71 |
| J26DS313_R0051 | 310 | 137 | 172 | 134 | 155 | 146 |
| J26DS313_R0062 | 255 | 169 | 85 | 126 | 149 | 87 |
| J26DS313_R0063 | 233 | 97 | 136 | 63 | 82 | 92 |
| J26DS313_R0064 | 243 | 91 | 151 | 55 | 79 | 89 |
| J26DS313_R0065 | 423 | 190 | 232 | 114 | 143 | 185 |
