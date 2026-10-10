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
| E006 | 30 Sep | Does knowing the speaker count fix it? | 51 clips 18.07 → 16.08 %; under-counted clips 22.38 → 21.28 % | Counting explains only part. **Correction (10 Oct):** in community-1, a forced count replaces VBx+PLDA with KMeans, so this did not test whether the voices separate |
| E010 | 1 Oct | 5 h synthetic conversations (+ real) | synthetic + real 17.30 %; synthetic only 15.59 % | No detectable difference vs real only |
| E013 | 1 Oct | Tune clustering (threshold, Fa, Fb, min_duration_off) on the 49 non-TEST clips | in-sample 18.51 → 17.05 %; TEST 7.50 → 10.40 % | Hurts on TEST |
| P3 | 1 Oct | External reference: pyannoteAI precision-3 (public clips only) | 13.79 %; confusion 2.83 vs 4.51 | Reference only |
| E011 | 1 Oct | Gentler fine-tuning (lr 1e-4) on synthetic + real | 14.07 vs 17.30 % | **Gentler is better (p = 0.05)** |
| E012 | 1 Oct | Synthetic set skewed to 5–6 speakers, more overlap | 15.92 % | No detectable difference |
| E014 | 1 Oct | Gentle fine-tuning on synthetic only | 13.48 % (seeds 13.27 / 13.18 / 13.99) | Best fine-tune; still far from 7.50 |
| E015 | 4 Oct | All 28 fine-tunes + controls on TEST2 (3 new YouTube panels, 13.8 min) | community-1 24.08 %; E009-B big 22.79 % (every seed better) | One possible win, to replicate |
| E016 | 10 Oct | Replicate on TEST3 (3 more panels, 13.4 min) | community-1 23.15 %; E009-B big 22.35–23.46 % | Win does not replicate; gentler > harsher replicates (3rd set) |
| E017 | 10 Oct | Does community-1 run on the demo laptop CPU, offline? | DER identical to GPU; ~0.7–1.0× real time | Yes: offline demo is feasible |
| E018 | 10 Oct | Coarser sliding-window step for speed | 0.78× → 3.35× real time, DER 15.90 → 23.66 % | Rejected; default kept |
| X2 | 10 Oct | Are errors concentrated near Sinhala-English switches? | confusion 13.29 vs 11.25 % (CI includes 0) | Not supported |
| E019 X3 | 10 Oct | VBx clustering tuned with leave-one-set-out CV (8 panel clips) | 21.04 vs 21.27 % shipped; hindsight best 20.38 % | No reliable gain; clustering knobs exhausted |
| E019 X1 / E021 | 10 Oct | Other speaker embeddings (agglomerative clustering, CV) | SB ResNet 22.15, WeSpeaker ResNet34 23.51, ECAPA 26.69, x-vector 31.75 % | None beats shipped community-1 (21.27 %) |
| E020 | 10 Oct | Live chunked mode (10 s window, 5 s step) | 40.65 % vs 21.27 % full recording; windows miss 17 % of speech | Not usable; stays experimental |

## What the results say

1. **Untouched community-1 is the best system on unseen real audio** (7.50 % TEST). All 28
   fine-tuned models, the tuned clustering and the commercial reference score worse.
2. **Fine-tuning on our recordings learns the training speakers, not diarization skill.** DEV
   (speakers shared with training) improves 22.5 → ~7 %, TEST gets worse.
3. **Two changes reduce the damage, consistently.** A gentler learning rate
   (17.30 → 14.07; 15.59 → 13.48) and leaving out the scripted clips (17.30 → 15.59;
   14.07 → 13.48).
4. **The remaining error is mostly speaker confusion** (about 12 of 21 points on the 8 real panel
   clips). Clustering settings (E005, E013, E019) and other off-the-shelf embeddings (E021) do not
   reduce it. Whether it comes from the embeddings or from local segmentation is still open (E006
   did not separate the two; see the correction above).
5. **Across 8 real panel clips (32.3 min) community-1 scores 21.27 %**: TEST 7.50, TEST2 24.08,
   TEST3 23.15. Every alternative tried lands within about ±1 point or worse, and which one wins
   depends on the clip.
6. **The system runs offline on an ordinary laptop CPU** with the same accuracy as on a GPU (E017).

## Limits

- TEST is 2 clips / 5.1 min; `IRD_Harshana_Silva` alone is 37.6 s. With all 8 panel clips
  (32.3 min), differences under about 1 point are not reliable.
- TEST2/TEST3 truth was corrected from pyannoteAI drafts (not community-1).
- TEST truth was corrected from community-1 drafts, which favours community-1 on speech
  boundaries. Confusion, which this affects less, still favours community-1.
- Training speakers recur across clips, so no speaker-disjoint in-domain test exists yet.

## Decision and next steps

- **Ship `speaker-diarization-community-1` unchanged** as the C3 model.
- Build a larger clean test set: public Sinhala-English panels with new voices, labelled by
  ear from scratch, 20–30 min. It is added as a second test; the current TEST stays frozen.
- Record new speakers in the project's own setting to test whether fine-tuning helps for
  that domain. All trained checkpoints are kept, so they can be re-scored without retraining.
- Next (from E019–E021): benchmark complete alternative systems on the same clips (e.g. DiariZen,
  which keeps community-1's embeddings and clustering but uses a stronger segmentation front-end);
  split confusion into clustering vs segmentation with oracle clustering; check the embedding filter
  (`min_active_ratio`) and `embedding_exclude_overlap`; add an independent Sinhala test set.
