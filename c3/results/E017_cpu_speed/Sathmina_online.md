# E017 CPU speed test - Sathmina - online

- CPU: Intel64 Family 6 Model 154 Stepping 4, GenuineIntel | logical cores: 12 | torch threads: 10
- GPU available: False (this test forces CPU)
- Python 3.13.3 | torch 2.14.1+cpu | Windows-11-10.0.26200-SP0
- pyannote.audio 4.0.7

Model load: 16.2 s
Warm-up (5 s audio): 3.9 s

## 1. Batch mode (record/upload, then diarize)

| clip | audio (s) | processing (s) | x real time | speakers found | DER c0.0 (CPU) | DER GPU 1 Oct |
|---|---|---|---|---|---|---|
| IRD_Harshana_Silva | 37.6 | 53.1 | 0.71x | 2 | 27.99 % | 27.99 % |
| IRD_Recording_2_IIT | 266.1 | 401.6 | 0.66x | 6 | 4.87 % | 4.87 % |
| R0046 | 396.4 | 490.6 | 0.81x | 2 | 21.95 % | 21.95 % |

x real time > 1 means faster than the audio plays. A 60 s recording at 2x real time waits 30 s.

## 2. Live-mode feasibility (short chunks)

| chunk length | chunks timed | mean processing (s) | worst (s) | keeps up? |
|---|---|---|---|---|
| 5 s | 6 | 1.13 | 1.74 | yes, comfortably |
| 10 s | 6 | 1.35 | 1.99 | yes, comfortably |

'Keeps up' = each chunk must finish before the next one has been recorded; comfortably = under half its length.
