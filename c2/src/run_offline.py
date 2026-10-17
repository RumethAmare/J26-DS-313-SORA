"""
The full C2 pipeline on a laptop CPU, with no internet and no GPU.
Same steps and outputs as run_c2.py, using the files from export_offline.py:

    translate  CTranslate2 int8 NLLB (fine-tuned en, si)
    summarise  llama.cpp GGUF Q4 Qwen-3B (translate-first)
    summary_si CTranslate2 int8 base NLLB

    pip install ctranslate2 llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
    python src/run_offline.py --rid J26DS313_R0007
    python src/run_offline.py --transcript path/to/R0123.transcript.json

Models are read from --models (default: ~/sora_offline). Turn Wi-Fi off to show
it really is offline.
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"          # never reach for the internet
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from normalize import postprocess, prepare  # noqa: E402
from run_c2 import LANG_ENGLISH, load_conversations  # noqa: E402
from train_summary import INSTRUCTION, parse  # noqa: E402
from train_translation import LANG, SRC_LANG  # noqa: E402

C2_ROOT = Path(__file__).resolve().parents[1]
THREADS = os.cpu_count() or 4


def load_translator(model_dir):
    import ctranslate2
    from transformers import AutoTokenizer
    return (AutoTokenizer.from_pretrained(str(model_dir)),
            ctranslate2.Translator(str(model_dir), device="cpu", intra_threads=THREADS))


def translate_with(loaded, texts, src_lang, tgt_lang, batch=8):
    tok, model = loaded
    tok.src_lang = src_lang
    out = []
    for i in range(0, len(texts), batch):
        chunk = [tok.convert_ids_to_tokens(tok.encode(t, truncation=True, max_length=256))
                 for t in texts[i:i + batch]]
        res = model.translate_batch(chunk, target_prefix=[[tgt_lang]] * len(chunk),
                                    beam_size=4, max_decoding_length=256)
        for r in res:
            ids = tok.convert_tokens_to_ids(r.hypotheses[0][1:])   # drop the language tag
            out.append(tok.decode(ids, skip_special_tokens=True))
    return out


def translate(model_dir, texts, src_lang, tgt_lang, batch=8):
    return translate_with(load_translator(model_dir), texts, src_lang, tgt_lang, batch)


def load_summariser(gguf):
    from llama_cpp import Llama
    return Llama(model_path=str(gguf), n_ctx=4096, n_threads=THREADS, verbose=False)


def summarise_with(llm, transcripts):
    results = []
    for text in transcripts:
        ids = llm.tokenize(text.encode("utf-8"), add_bos=False)[:3072]
        text = llm.detokenize(ids).decode("utf-8", errors="ignore")
        msg = INSTRUCTION.format(lang=LANG_ENGLISH, transcript=text)
        res = llm.create_chat_completion(messages=[{"role": "user", "content": msg}],
                                         temperature=0.0, max_tokens=512)
        raw = res["choices"][0]["message"]["content"]
        results.append((raw, *parse(raw)))
    return results


def summarise(gguf, transcripts):
    return summarise_with(load_summariser(gguf), transcripts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rid")
    ap.add_argument("--split", choices=["dev", "test", "train"])
    ap.add_argument("--transcript")
    ap.add_argument("--models", default=str(Path.home() / "sora_offline"))
    ap.add_argument("--out", default=str(C2_ROOT / "predictions" / "offline"))
    args = ap.parse_args()
    if not (args.rid or args.split or args.transcript):
        ap.error("give --rid, --split or --transcript")

    models, out_dir = Path(args.models), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    convs = load_conversations(args)
    texts = [prepare(u["text"]) for _, utts, _ in convs for u in utts]
    timing = {}

    t = time.time()
    en = iter(translate(models / "nllb_en", texts, SRC_LANG, LANG["en"]))
    si = iter(translate(models / "nllb_si", texts, SRC_LANG, LANG["si"]))
    src = iter(texts)
    for _, utts, _ in convs:
        for u in utts:
            s = next(src)
            u["clean_en"], u["clean_si"] = postprocess(s, next(en)), postprocess(s, next(si))
    timing["translate"] = time.time() - t

    t = time.time()
    summaries = summarise(models / "qwen3b_cascade_q4.gguf",
                          ["\n".join(f"{u['speaker']}: {u['clean_en']}" for u in utts)
                           for _, utts, _ in convs])
    timing["summarise"] = time.time() - t

    t = time.time()
    sents = [re.split(r"(?<=[.!?])\s+", s[1] or "") for s in summaries]
    flat = iter(translate(models / "nllb_base", [x for ss in sents for x in ss], LANG["en"], LANG["si"]))
    summaries_si = [" ".join(postprocess(x, next(flat)) for x in ss) for ss in sents]
    timing["summary_si"] = time.time() - t

    for (rid, utts, _), (_, summary, items), summary_si in zip(convs, summaries, summaries_si):
        (out_dir / f"{rid}.translation.json").write_text(json.dumps({
            "recording_id": rid,
            "utterance_records": [{"utt_id": u["utt_id"], "clean_en": u["clean_en"],
                                   "clean_si": u["clean_si"]} for u in utts],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / f"{rid}.summary.json").write_text(json.dumps({
            "recording_id": rid,
            "summaries": {"en": {"text": summary}, "si": {"text": summary_si}},
            "action_items": [{"intent": a["intent"], "owner": a["owner"], "receiver": None,
                              "deadline": a["deadline"]} for a in items],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n{'=' * 70}\n{rid}\n{'=' * 70}")
        for u in utts[:8]:
            print(f"{u['speaker']}: {u['text']}\n    EN: {u['clean_en']}\n    SI: {u['clean_si']}")
        if len(utts) > 8:
            print(f"    ... {len(utts) - 8} more utterances in {rid}.translation.json")
        print(f"\nSUMMARY (EN): {summary}\nSUMMARY (SI): {summary_si}\nACTION ITEMS:")
        for a in items or [{"intent": "(none)", "owner": None, "deadline": None}]:
            print(f"  - {a['intent']}  | owner: {a['owner']}  | deadline: {a['deadline']}")

    n = len(convs)
    print(f"\nOffline on CPU ({THREADS} threads), {n} conversation(s), {len(texts)} utterances:")
    for k, v in timing.items():
        print(f"  {k:<10} {v:6.1f} s")
    print(f"  {'total':<10} {sum(timing.values()):6.1f} s  ({sum(timing.values()) / n:.1f} s per conversation)")
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
