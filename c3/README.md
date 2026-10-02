# C3: Speaker diarization ("who spoke when") for Sinhala-English conversations

Component 3 of the SORA / Kathā pipeline. It segments a conversation recording into speaker
turns (RTTM), which the other components use for attribution.

- Model: `pyannote/speaker-diarization-community-1` (pyannote.audio 4.x), runs offline.
- Evaluation: protocol v1, see `docs/protocol_v1.md`.
- Ground truth: labelled by ear, see `docs/annotation_rules.md`.

Folder layout and results are added phase by phase; see `EXPERIMENT_LOG.md` once present.
