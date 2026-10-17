# Annotation rules (C3 ground truth)

Ground truth is labelled **by ear** in Audacity, one label per speaker turn, then converted
to RTTM + UEM with `scripts/04_audacity_to_rttm.py`.

## Labels

| Heard | Label |
|---|---|
| Speaker 1, 2, 3 … | `spk01`, `spk02`, `spk03` … (numbering is per clip) |
| Back-channel *while someone else is talking* ("mm", "ow", "hari") | `spkNN-BC` |
| Short acknowledgement in its own gap ("ow ow", "yes yes") | `spkNN-ACK` |
| Music, noise, unintelligible | `NOSCORE` (excluded from scoring) |

`-BC` / `-ACK` tags are written to a separate `events.jsonl`, never into the RTTM speaker
field, so a scorer never treats `spk02` and `spk02-BC` as two people.

## Rules

1. **Short pause = same turn.** Gaps under ~0.3 s are merged by the converter.
2. **Two people at once = two labels.** Both keep their real start and end; overlap is never
   given to the louder speaker. Use one label track per speaker in Audacity when needed.
3. **Back-channels are speech** and are labelled.
4. **Unusable audio = `NOSCORE`**, used only for genuinely unintelligible regions.

The converter checks that every `-BC` actually overlaps another speaker and warns when an
export contains no overlap at all (a sign that extra label tracks were not exported).
