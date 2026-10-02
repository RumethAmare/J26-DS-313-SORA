# Evaluation protocol v1

Every DER reported for C3 uses this protocol. It is fixed so that numbers from different
experiments can be compared directly.

| Setting | Value |
|---|---|
| Metric | Diarization Error Rate (pyannote.metrics `DiarizationErrorRate`) |
| Overlapped speech | **Included** (`skip_overlap=False`) |
| Collar | Reported at **0.0 s** (primary) and 0.25 s |
| Scored region | Explicit UEM file per clip; nothing approximated |
| Speaker count | **Not given** to the system (blind). Runs that are told the count are labelled as such and never used as a headline |
| Breakdown | DER = missed speech + false alarm + speaker confusion, plus speaker-count accuracy |
| Pooling | Error components summed over clips, then divided by total reference speech |

## Baseline model

`pyannote/speaker-diarization-community-1` on pyannote.audio 4.x. The older
`speaker-diarization-3.1` recipe no longer installs on current Colab (its torchaudio
dependency was removed) and is marked legacy by the maintainers. On our real test clips
community-1 is also about twice as accurate as the 3.1-style pipeline (7.50 % vs 14.35 %,
E008).

## Data splits used from E005 onward

| Split | Clips | Use |
|---|---|---|
| TRAIN | scripted recordings (21 clips, later 45) | fine-tuning only |
| DEV | R0023–R0026 | early stopping only; shares speakers with TRAIN, so its numbers are inflated |
| TEST | `IRD_Harshana_Silva`, `IRD_Recording_2_IIT` (public panels, 5.1 min, new voices) | frozen; scored once per model |

Each experiment states its decision rule **before** it runs. With 3 training seeds per arm,
"A is better than B" is claimed only if every A seed beats every B seed (exact permutation
p = 1/20 = 0.05).
