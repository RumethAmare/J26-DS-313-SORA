"""
The full C2 pipeline on one conversation (or a whole split):

    code-mixed transcript (C1+C3)
      -> clean English + clean Sinhala per utterance   (fine-tuned NLLB, step 3)
      -> English summary + action items                (QLoRA Qwen, translate-first, step 4)
      -> Sinhala summary                               (base NLLB, English -> Sinhala)
      -> <rid>.translation.json + <rid>.summary.json   (docs/C2_SCHEMA.md, the C4 input)

Needs a GPU (Colab T4). Models are read from Drive, where steps 3-4 saved them.

    python src/run_c2.py --rid J26DS313_R0007                 # one dev/test conversation
    python src/run_c2.py --transcript path/to/R0123.transcript.json   # a raw C1 file
    python src/run_c2.py --split test                         # every test conversation

Outputs go to predictions/pipeline/ (gitignored: they contain un-redacted PII).
"""
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from train_translation import LANG, SPEAKER, SRC_LANG

C2_ROOT = Path(__file__).resolve().parents[1]
DATA = C2_ROOT / "data"
DRIVE_MODELS = Path("/content/drive/MyDrive/SORA_C2/models")
BASE_NLLB = "facebook/nllb-200-distilled-600M"
BASE_LLM = "Qwen/Qwen2.5-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_conversations(args):
    """[(rid, [{utt_id, speaker, text}], gold_record_or_None)]"""
    if args.transcript:
        data = json.loads(Path(args.transcript).read_text(encoding="utf-8"))
        rid = data.get("recording_id") or Path(args.transcript).name.split(".")[0]
        utts = [{"utt_id": u["utt_id"], "speaker": u.get("speaker") or "S?", "text": u["text"]}
                for u in data["utterances"] if (u.get("text") or "").strip()]
        return [(rid, utts, None)]
    out = []
    for split in ([args.split] if args.split else ["dev", "test", "train"]):
        if not (DATA / f"{split}.jsonl").exists():
            continue  # e.g. test.jsonl is deliberately not uploaded during development
        for line in open(DATA / f"{split}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if args.rid and r["rid"] != args.rid:
                continue
            utts = [{"utt_id": u["utt_id"], "speaker": u["speaker"], "text": SPEAKER.sub("", u["src"])}
                    for u in r["utterances"]]
            out.append((r["rid"], utts, r))
    if not out:
        raise SystemExit(f"no conversation found for --rid {args.rid} / --split {args.split}")
    return out


def translate(model_dir, texts, src_lang, tgt_lang, batch=8, max_len=256):
    tok = AutoTokenizer.from_pretrained(model_dir, src_lang=src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32).to(DEVICE).eval()
    bos = tok.convert_tokens_to_ids(tgt_lang)
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(DEVICE)
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=bos, num_beams=4, max_length=max_len)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    del model
    torch.cuda.empty_cache()
    return out


def summarise(adapter_dir, transcripts):
    """English transcripts -> [(raw_text, summary, action_items)] with the step-4 model."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from train_summary import INSTRUCTION, parse

    tok = AutoTokenizer.from_pretrained(BASE_LLM)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_LLM, device_map="auto", torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))
    model = PeftModel.from_pretrained(model, adapter_dir).eval()
    results = []
    for text in transcripts:
        ids = tok(text, add_special_tokens=False)["input_ids"][:3072]
        msg = INSTRUCTION.format(lang=" in English", transcript=tok.decode(ids))
        prompt = tok.apply_chat_template([{"role": "user", "content": msg}],
                                         add_generation_prompt=True, tokenize=True)
        x = torch.tensor([prompt]).to(model.device)
        with torch.no_grad():
            gen = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=512,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        raw = tok.decode(gen[0][x.shape[1]:], skip_special_tokens=True)
        results.append((raw, *parse(raw)))
    del model
    torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rid", help="one conversation from data/*.jsonl")
    ap.add_argument("--split", choices=["dev", "test", "train"], help="every conversation in a split")
    ap.add_argument("--transcript", help="a raw C1 <rid>.transcript.json file")
    ap.add_argument("--models", default=str(DRIVE_MODELS))
    ap.add_argument("--out", default=str(C2_ROOT / "predictions" / "pipeline"))
    ap.add_argument("--no-summary", action="store_true", help="translation only")
    args = ap.parse_args()
    if not (args.rid or args.split or args.transcript):
        ap.error("give --rid, --split or --transcript")

    models, out_dir = Path(args.models), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    convs = load_conversations(args)
    texts = [u["text"] for _, utts, _ in convs for u in utts]
    print(f"{len(convs)} conversation(s), {len(texts)} utterances")

    # Step A: per-utterance clean English and clean Sinhala.
    en = iter(translate(models / "nllb600m_en_frac100", texts, SRC_LANG, LANG["en"]))
    si = iter(translate(models / "nllb600m_si_frac100", texts, SRC_LANG, LANG["si"]))
    for _, utts, _ in convs:
        for u in utts:
            u["clean_en"], u["clean_si"] = next(en), next(si)

    # Step B: summary from the English translation (translate-first won on dev).
    summaries = [(None, None, [])] * len(convs)
    summaries_si = [None] * len(convs)
    if not args.no_summary:
        summaries = summarise(models / "qwen3b_cascade_qlora" / "adapter",
                              ["\n".join(f"{u['speaker']}: {u['clean_en']}" for u in utts)
                               for _, utts, _ in convs])
        # NLLB is a sentence-level model: translate the summary sentence by sentence.
        sents = [re.split(r"(?<=[.!?])\s+", s[1] or "") for s in summaries]
        flat = iter(translate(BASE_NLLB, [x for ss in sents for x in ss], LANG["en"], LANG["si"]))
        summaries_si = [" ".join(next(flat) for _ in ss) for ss in sents]

    for (rid, utts, _), (_, summary, items), summary_si in zip(convs, summaries, summaries_si):
        (out_dir / f"{rid}.translation.json").write_text(json.dumps({
            "recording_id": rid,
            "utterance_records": [{"utt_id": u["utt_id"], "clean_en": u["clean_en"],
                                   "clean_si": u["clean_si"]} for u in utts],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        if not args.no_summary:
            (out_dir / f"{rid}.summary.json").write_text(json.dumps({
                "recording_id": rid,
                "summaries": {"en": {"text": summary}, "si": {"text": summary_si}},
                "action_items": [{"intent": a["intent"], "owner": a["owner"], "receiver": None,
                                  "deadline": a["deadline"]} for a in items],
            }, ensure_ascii=False, indent=2), encoding="utf-8")

        # Readable view for a demo.
        print(f"\n{'=' * 70}\n{rid}\n{'=' * 70}")
        for u in utts[:8]:
            print(f"{u['speaker']}: {u['text']}\n    EN: {u['clean_en']}\n    SI: {u['clean_si']}")
        if len(utts) > 8:
            print(f"    ... {len(utts) - 8} more utterances in {rid}.translation.json")
        if not args.no_summary:
            print(f"\nSUMMARY (EN): {summary}\nSUMMARY (SI): {summary_si}\nACTION ITEMS:")
            for a in items or [{"intent": "(none)", "owner": None, "deadline": None}]:
                print(f"  - {a['intent']}  | owner: {a['owner']}  | deadline: {a['deadline']}")
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
