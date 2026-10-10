"""
Live (chunked) diarization for the C3 demo.

How it works (plain words):
  * Audio arrives continuously. Every STEP seconds we diarize only the last WINDOW seconds (cheap: ~1.4 s of CPU
    for a 10 s window on the laptop, E017).
  * Each window gives "local" speakers plus one voice fingerprint (embedding) per local speaker.
  * Local speakers are matched to the "global" speakers seen so far by cosine similarity of fingerprints.
    Similar enough (>= tau) -> same person; otherwise -> a new person. Global fingerprints are running averages.
  * Only the newest STEP seconds of each window are written to the global timeline; the older part of the window
    is context that makes the decision more stable.
The same class is used by the app (microphone) and by sim_live.py (to measure its error rate on benchmark files).
"""
import numpy as np
import torch
from pyannote.core import Annotation, Segment


class LiveDiarizer:
    def __init__(self, pipe, sr=16000, window=10.0, step=5.0, tau=0.5, min_speech=0.3):
        self.pipe, self.sr = pipe, sr
        self.window, self.step, self.tau, self.min_speech = window, step, tau, min_speech
        self.buf = np.zeros(0, dtype=np.float32)
        self.done_until = 0.0                    # global time up to which the timeline is final
        self.centroids, self.counts = [], []     # global speaker fingerprints
        self.timeline = Annotation(uri="live")
        self.cpu_seconds = []

    @property
    def now(self):
        return len(self.buf) / self.sr

    def feed(self, chunk):
        """Append raw mono float32 samples at self.sr. Processes as many steps as are ready. Returns #steps run."""
        self.buf = np.concatenate([self.buf, np.asarray(chunk, dtype=np.float32)])
        n = 0
        while self.now - self.done_until >= self.step:
            self._process(self.done_until + self.step)
            n += 1
        return n

    def flush(self):
        if self.now - self.done_until > 0.5:
            self._process(self.now)

    def _match(self, emb):
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        if self.centroids:
            C = np.stack([c / (np.linalg.norm(c) + 1e-9) for c in self.centroids])
            sims = C @ emb
            j = int(np.argmax(sims))
            if sims[j] >= self.tau:
                self.centroids[j] = (self.centroids[j] * self.counts[j] + emb) / (self.counts[j] + 1)
                self.counts[j] += 1
                return j
        self.centroids.append(emb.copy()); self.counts.append(1)
        return len(self.centroids) - 1

    def window_output(self, t_end):
        """Run the model on the window ending at t_end. Returns (w_start, annotation, embeddings, cpu_seconds)."""
        import time
        t0 = time.time()
        w_start = max(0.0, t_end - self.window)
        a, b = int(w_start * self.sr), int(t_end * self.sr)
        wav = torch.from_numpy(self.buf[a:b].copy()).unsqueeze(0)
        out = self.pipe({"waveform": wav, "sample_rate": self.sr})
        ann = getattr(out, "speaker_diarization", out)
        embs = getattr(out, "speaker_embeddings", None)
        return w_start, ann, embs, time.time() - t0

    def commit(self, t_end, w_start, ann, embs):
        labels = ann.labels()
        mapping = {}
        for i, lab in enumerate(labels):
            if (ann.label_duration(lab) < self.min_speech or embs is None or i >= len(embs)
                    or not np.all(np.isfinite(embs[i])) or np.linalg.norm(embs[i]) < 1e-6):
                continue
            mapping[lab] = self._match(np.asarray(embs[i], dtype=np.float32))
        new = Segment(self.done_until, t_end)
        for seg, _, lab in ann.itertracks(yield_label=True):
            if lab not in mapping:
                continue
            g = Segment(seg.start + w_start, seg.end + w_start) & new
            if g and g.duration > 0:
                self._k = getattr(self, "_k", 0) + 1
                self.timeline[g, self._k] = f"Speaker {chr(65 + mapping[lab] % 26)}"
        self.done_until = t_end

    def _process(self, t_end):
        w_start, ann, embs, secs = self.window_output(t_end)
        self.cpu_seconds.append(secs)
        self.commit(t_end, w_start, ann, embs)

    def result(self):
        return self.timeline.support(collar=0.25)
