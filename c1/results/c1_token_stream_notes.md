# Task 2 — token stream results, and what the pipeline does to code-switching

Produced by `scripts/build_token_stream.py` (faster-whisper `small`,
auto-detect language, `word_timestamps=True`), validated by
`scripts/validate_token_stream.py`: **26 files, 1,280 tokens, 0 errors,
0 warnings, 0 switch-mismatches.**

The schema works and the glue works. The substantive result is what the
end-to-end pipeline does to the code-switching signal it exists to capture.

## End-to-end output vs gold

| | tokens | SI | EN | OTHER | switch points | switch rate |
|---|---|---|---|---|---|---|
| gold | 5,818 | 3,338 | 2,388 | 92 | 2,012 | 34.6% |
| predicted | 1,280 | 1,172 | 108 | 0 | 86 | 6.7% |

Gold is genuinely code-mixed: **41.0% of gold tokens are English** and a third
of all tokens sit at a language switch. The predicted stream is **8.4%
English** and nearly monolingual Sinhala.

## Why: the ASR collapses the signal before LID ever sees it

The LID method mix over the 1,280 predicted tokens:

| method | tokens | share |
|---|---|---|
| `unicode_sinhala` | 1,032 | 80.6% |
| `fallback_si` | 140 | 10.9% |
| `wordfreq_en` | 98 | 7.7% |
| `numeric` | 10 | 0.8% |
| `unicode_tamil` | 0 | 0% |

Four fifths of all language labels come from the Unicode script rule. That
rule is deterministic and near-perfect *as a script detector* — but on ASR
output it is not detecting the language that was **spoken**, it is reporting
the script **Whisper chose to decode into**. Auto-detect picks `si` for 25 of
26 recordings (Task 1), so Whisper renders English speech in Sinhala
characters, the Unicode rule reads those characters, and the token comes back
SI. The text-based LID is downstream of a decision that has already erased the
distinction it is being asked to make.

This is not a tuning problem. No improvement to the LID stage can recover a
language contrast that the ASR stage has already flattened into one script.

## Consequence for Task 4

**The switch-point recall ceiling is 4.3%.** The pipeline emits 86 switch
points against gold's 2,012; even if every single one were correct, end-to-end
switch-point F1 is bounded near zero. Two thirds of that loss is the 22% token
coverage (Task 1's abstention finding), and the rest is the script collapse
above.

So Task 4 should report switch-point F1 **on gold tokens** — measuring the
switch-derivation and LID logic on correct input — and report the end-to-end
number separately and explicitly as an ASR-bounded figure. Quoting a single
end-to-end switch F1 would attribute an ASR failure to the switch-detection
work, which is not where the problem is.

The plan already evaluates Task 3's LID against gold tokens rather than ASR
output. This quantifies why that was the right call: on ASR output the LID
task is close to degenerate.

## Consequence for `OTHER`

Zero `OTHER` tokens were predicted across the whole corpus, against 92 in
gold. `OTHER` currently has exactly one source — the Tamil Unicode rule — and
Whisper never emits Tamil script here. Task 5's romanized-Tamil detection
therefore cannot be evaluated end-to-end at all; it has to be developed and
scored against gold tokens.

## Reading the confidences

Mean `asr_confidence` runs 0.27–0.45 per recording, correctly signalling that
the underlying transcription is unreliable. Mean `lang_confidence` is much
higher, because 80% of labels come from the deterministic script rule.

That gap is the schema working as intended. A token like
`වසදන්ක්ටිමය` (hallucinated, not a real Sinhala word) carries
`asr_confidence=0.35` and `lang_confidence=0.99` simultaneously: the word is
almost certainly wrong, and it is almost certainly written in Sinhala script.
Both statements are true and a consumer needs them separately. A single merged
confidence would have to be either wrong about the text or wrong about the
language.

**Caveat for downstream consumers:** high `lang_confidence` on this stream
does not mean the language tag is right in any deeper sense — it means the
script is unambiguous. Given the ASR chose that script wholesale, the tag
inherits the ASR's error. Trust `lang_confidence` on *gold* tokens; on
predicted tokens read it together with `asr_confidence`.
