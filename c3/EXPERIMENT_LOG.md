# C3 experiment log

All numbers: DER %, protocol v1 (`docs/protocol_v1.md`), collar 0.0, pooled. TEST = the two
public panel clips (`IRD_Harshana_Silva`, `IRD_Recording_2_IIT`, 5.1 min, new voices), frozen.
Every experiment's decision rule was written down before it ran. With 3 seeds per arm, "better"
means every seed of one arm beats every seed of the other (exact p = 0.05).

| ID | Date | Question | Result | Verdict |
|---|---|---|---|---|
| E003 | 29 Jul | Zero-shot community-1 on the 7-clip pilot | 16.06 % pooled; error is mostly speaker confusion; under-counts speakers on 6 of 7 | Baseline set |
| E004 | 25 Aug | Zero-shot on 27 clips (52.2 min) | 18.74 %; confusion = 84 % of error; TEST 7.50 % | Baseline |
| E005 | 25 Aug | Does tuning the clustering threshold help unseen audio? | Threshold almost inert (0.1 pp); TEST output identical (7.50 %) | No cheap fix |
| E008 | 25 Aug | Fine-tune segmentation-3.0 on 21 scripted clips | DEV 22.46 → 8.99 (speakers seen in training); TEST 14.35 → 16.76 | Gain does not transfer |
| E004b | 30 Sep | Zero-shot on 51 clips (130.9 min) | 18.07 % | Baseline |
| E009-A | 30 Sep | More training data (36 vs 115 min), E008 recipe | 18.70 vs 20.35 % | No detectable difference |
| E009-B | 30 Sep | Same, community-1's own segmentation swapped in | 15.02 vs 16.05 %; untouched 7.50 % | Fine-tuning hurts |
| E006 | 30 Sep | Does knowing the speaker count fix it? | 51 clips 18.07 → 16.08 %; under-counted clips 22.38 → 21.28 % | Counting explains only part; voices are hard to separate |
| E010 | 1 Oct | 5 h synthetic conversations (+ real) | synthetic + real 17.30 %; synthetic only 15.59 % | No detectable difference vs real only |
| E013 | 1 Oct | Tune clustering (threshold, Fa, Fb, min_duration_off) on the 49 non-TEST clips | in-sample 18.51 → 17.05 %; TEST 7.50 → 10.40 % | Hurts on TEST |
| P3 | 1 Oct | External reference: pyannoteAI precision-3 (public clips only) | 13.79 %; confusion 2.83 vs 4.51 | Reference only |
| E011 | 1 Oct | Gentler fine-tuning (lr 1e-4) on synthetic + real | 14.07 vs 17.30 % | **Gentler is better (p = 0.05)** |
| E012 | 1 Oct | Synthetic set skewed to 5–6 speakers, more overlap | 15.92 % | No detectable difference |
| E014 | 1 Oct | Gentle fine-tuning on synthetic only | 13.48 % (seeds 13.27 / 13.18 / 13.99) | Best fine-tune; still far from 7.50 |

## What the results say

1. **Untouched community-1 is the best system on unseen real audio** (7.50 % TEST). All 28
   fine-tuned models, the tuned clustering and the commercial reference score worse.
2. **Fine-tuning on our recordings learns the training speakers, not diarization skill.** DEV
   (speakers shared with training) improves 22.5 → ~7 %, TEST gets worse.
3. **Two changes reduce the damage, consistently.** A gentler learning rate
   (17.30 → 14.07; 15.59 → 13.48) and leaving out the scripted clips (17.30 → 15.59;
   14.07 → 13.48).
4. **The remaining error is mostly separating similar voices** (E006), not counting or
   speech detection.

## Limits

- TEST is 2 clips / 5.1 min; `IRD_Harshana_Silva` alone is 37.6 s.
- TEST truth was corrected from community-1 drafts, which favours community-1 on speech
  boundaries. Confusion, which this affects less, still favours community-1.
- Training speakers recur across clips, so no speaker-disjoint in-domain test exists yet.

## Decision and next steps

- **Ship `speaker-diarization-community-1` unchanged** as the C3 model.
- Build a larger clean test set: public Sinhala-English panels with new voices, labelled by
  ear from scratch, 20–30 min. It is added as a second test; the current TEST stays frozen.
- Record new speakers in the project's own setting to test whether fine-tuning helps for
  that domain. All trained checkpoints are kept, so they can be re-scored without retraining.
