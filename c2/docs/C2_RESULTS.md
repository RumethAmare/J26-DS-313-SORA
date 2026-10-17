# C2 results and error analysis (dev split)

All numbers are on the **dev** split (5 recordings, 95 utterances). The test
split is scored once, at the end, with `src/evaluate.py`. With only 95
utterances, differences of 1–2 chrF++ are within noise.

## Task A — translation (chrF++ / BLEU)

| Target | NLLB-600M zero-shot | NLLB-600M fine-tuned |
|---|---|---|
| clean_en | 36.6 / 13.8 | **56.3 / 33.6** |
| clean_si | 28.1 / 6.6 | **51.1 / 22.3** |

Fine-tuning on 2,369 utterance pairs (75 recordings, 10 epochs, best epoch by
dev chrF++). The earlier mT5-small attempt reached BLEU 9.6 on en→si; NLLB is
pretrained on Sinhala, which is why it was chosen over mT5.

**By writing style** (fine-tuned + augmented en model):

| Input | Utterances | chrF++ |
|---|---|---|
| Romanized Singlish | 48 | 63.5 |
| Sinhala script | 47 | 53.6 |

**Ablation — romanized augmentation + normalisation (en):** 56.4 chrF++, no
significant gain over 56.3. The synthetic lines were Sinhala-script training
sentences re-spelled in Latin letters: same meaning, same target, so they add
no new content, and romanized input was not the weak case. New real
conversations, not re-spellings, are what the model needs.

## Task B — summary and action items

| System | ROUGE-L | Actions F1 (intent) | Actions F1 (intent + owner) |
|---|---|---|---|
| Qwen2.5-3B zero-shot | 25.3 | 16.0 | 8.0 |
| QLoRA, direct (reads Singlish) | 34.0 | 17.4 | 8.7 |
| **QLoRA, translate-first (reads NLLB English)** | **43.7** | **32.0** | **24.0** |

Translate-first beats direct on every measure and trains about 2.5× faster
(shorter English input). The direct run was repeated and reproduced exactly
(ROUGE-L 33.95 both times).

## Error analysis

| Error | Example | Cause | Status |
|---|---|---|---|
| Digits altered | NIC "1995 123 089 34" → "1995123809346" | Model rewrites digit groups | **Fixed**: numbers are copied from the source when the digits differ |
| Spoken digits invented | "zero binduwai hatai hayai…" → "zero zero, seven, seven, nine…" | Sinhala number words unseen | **Fixed**: spoken digits → numerals before translation |
| Split Sinhala conjuncts | "සාමාන් ය අවශ් යතාවය" | NLLB tokenizer drops the zero-width joiner | **Fixed**: re-joined when the conjunct occurs in gold text (28/29 dev lines repaired, 0 correct lines changed) |
| Near-identical words confused | "dannawa" (know) → "give"; "kiyada" (how much) → "how can" | Rare romanized spellings, little data | Open — needs more data |
| Name read as a word | "හරි චතුර" → "Right square" (චතුර = square) | Names that are also dictionary words | Open |
| Translation error flows into the summary | "I will call **you**" → summary invents a "job opportunity" | Cascade design: the summariser trusts the translation | Open — the trade-off of translate-first |
| Long podcast sentences degrade | R0102: repeated phrases at the end of 100-word lines | Sentence-level model on very long turns | Open — split long turns |

## Next

- More real recordings (target: test split of 15–20 recordings).
- Retrain, then score the test split once with `src/evaluate.py`.
- PP2: NLLB-1.3B, previous-utterance context, COMET, LLM-judge + human rating.
