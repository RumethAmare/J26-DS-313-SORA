# Draft re-labels for bulk-mislabelled recordings

Produced by `scripts/pretag_relabel.py` using the Task 3 hybrid LID (`models\lid_latin.ftz`). **Drafts, not gold** — files are in `predictions/pretag/` and must be reviewed before replacing anything in `SORA_Dataset/annotations/c1/`.

A token is marked `needs_review` if the suggestion differs from the original label or its confidence is below 0.8. The hybrid's cross-validated token accuracy is ~93%, so roughly 1 suggestion in 15 will be wrong; it never predicts OTHER, so Tamil words (e.g. `Vanakkam`) must be fixed by hand.

| recording | tokens | suggested SI | suggested EN | labels changed | needs review | switches |
|---|---|---|---|---|---|---|
| J26DS313_R0017 | 251 | 124 | 126 | 137 | 146 | 109 |
| J26DS313_R0053 | 380 | 189 | 191 | 189 | 202 | 166 |
| J26DS313_R0054 | 426 | 210 | 215 | 211 | 236 | 210 |
| J26DS313_R0055 | 392 | 205 | 186 | 206 | 220 | 181 |
| J26DS313_R0056 | 291 | 117 | 173 | 118 | 126 | 121 |
| J26DS313_R0058 | 373 | 238 | 134 | 131 | 162 | 162 |
| J26DS313_R0059 | 173 | 100 | 73 | 100 | 109 | 75 |
| J26DS313_R0051 | 310 | 153 | 157 | 134 | 153 | 147 |
| J26DS313_R0062 | 255 | 178 | 76 | 127 | 154 | 84 |
| J26DS313_R0063 | 233 | 108 | 125 | 66 | 83 | 95 |
| J26DS313_R0064 | 243 | 98 | 145 | 57 | 89 | 91 |
| J26DS313_R0065 | 423 | 201 | 221 | 117 | 142 | 195 |
