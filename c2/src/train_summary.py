"""
Step 4 (Task B): whole conversation -> English summary + action items.

QLoRA fine-tune of a small instruct LLM (Qwen2.5-3B-Instruct, 4-bit) on the
75 training conversations. Needs a GPU: run on Google Colab (T4).

    python src/train_summary.py --input direct --zero-shot    # baseline, no training
    python src/train_summary.py --input direct                # fine-tuned on raw Singlish
    python src/train_summary.py --input cascade               # fine-tuned on English translation

--input direct   the model reads the code-mixed transcript as-is.
--input cascade  "translate first": it reads the English translation. Training
                 uses the gold clean_en; dev uses the NLLB output from step 3
                 (predictions/nllb600m_en_frac100/dev.jsonl), as in real use.

Writes dev predictions to predictions/<run_id>/dev.jsonl and scores to
eval/<run_id>.json. The test split is never read here.
"""
import argparse
import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments)

from metrics import score_actions, score_summaries

C2_ROOT = Path(__file__).resolve().parents[1]
DATA = C2_ROOT / "data"
NLLB_DEV = C2_ROOT / "predictions" / "nllb600m_en_frac100" / "dev.jsonl"

INSTRUCTION = (
    "Below is a Sri Lankan conversation{lang}.\n\n{transcript}\n\n"
    "Write a short English summary (2-5 sentences). Then list every action item: "
    "anything someone agreed or promised to do, including indirect commitments "
    "(e.g. 'karannam' = I'll do it, 'balamu' = let's see). Use exactly this format:\n"
    "SUMMARY: <summary>\n"
    "ACTIONS:\n"
    "- <what> | <who> | <deadline or none>\n"
    "If there are no action items, write 'ACTIONS: none'."
)


def load(split, mode):
    records = [json.loads(line) for line in open(DATA / f"{split}.jsonl", encoding="utf-8")]
    hyp = {}
    if mode == "cascade" and split != "train":
        if NLLB_DEV.exists():
            hyp = {json.loads(l)["utt_id"]: json.loads(l)["hyp"] for l in open(NLLB_DEV, encoding="utf-8")}
        else:
            print(f"WARNING: {NLLB_DEV} not found, using gold clean_en for {split} (optimistic)")
    for r in records:
        if mode == "cascade":
            r["input"] = "\n".join(f"{u['speaker']}: {hyp.get(u['utt_id'], u['clean_en'])}"
                                   for u in r["utterances"])
        else:
            r["input"] = r["transcript"]
    return [r for r in records if r["summary_en"]]


def target_text(r):
    lines = [f"SUMMARY: {r['summary_en']}"]
    if r["action_items"]:
        lines.append("ACTIONS:")
        lines += [f"- {a['intent']} | {a['owner'] or 'none'} | {a['deadline'] or 'none'}"
                  for a in r["action_items"]]
    else:
        lines.append("ACTIONS: none")
    return "\n".join(lines)


def parse(text):
    """'SUMMARY: ... ACTIONS: - a | b | c' -> (summary, [{intent, owner, deadline}])."""
    summary, _, rest = text.partition("ACTIONS:")
    summary = summary.replace("SUMMARY:", "").strip()
    items = []
    for line in rest.splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        parts = [p.strip() for p in line[1:].split("|")] + ["", ""]
        none = lambda v: None if v.lower() in ("", "none") else v
        items.append({"intent": parts[0], "owner": none(parts[1]), "deadline": none(parts[2])})
    return summary, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", choices=["direct", "cascade"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-input", type=int, default=3072, help="transcript tokens kept")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--zero-shot", action="store_true", help="no training: score the base model (baseline)")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    run_id = args.run_id or f"qwen3b_{args.input}_{'zeroshot' if args.zero_shot else 'qlora'}"
    out_model = C2_ROOT / "models" / run_id
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train, dev = load("train", args.input), load("dev", args.input)
    print(f"{run_id}: {len(train)} train conversations, {len(dev)} dev")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True))

    def prompt_ids(r):
        ids = tok(r["input"], add_special_tokens=False)["input_ids"][:args.max_input]
        lang = " in English" if args.input == "cascade" else " in mixed Sinhala-English (Singlish)"
        msg = INSTRUCTION.format(lang=lang, transcript=tok.decode(ids))
        return tok.apply_chat_template([{"role": "user", "content": msg}],
                                       add_generation_prompt=True, tokenize=True)

    if not args.zero_shot:
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
        model.print_trainable_parameters()

        def encode(r):
            p = prompt_ids(r)
            t = tok(target_text(r) + tok.eos_token, add_special_tokens=False)["input_ids"]
            # Loss only on the answer, not on the transcript.
            return {"input_ids": p + t, "labels": [-100] * len(p) + t}

        examples = [encode(r) for r in train]

        def collate(batch):
            n = max(len(b["input_ids"]) for b in batch)
            pad = lambda xs, v: [x + [v] * (n - len(x)) for x in xs]
            ids = pad([b["input_ids"] for b in batch], tok.pad_token_id)
            return {"input_ids": torch.tensor(ids),
                    "labels": torch.tensor(pad([b["labels"] for b in batch], -100)),
                    "attention_mask": torch.tensor(pad([[1] * len(b["input_ids"]) for b in batch], 0))}

        trainer = Trainer(
            model=model,
            train_dataset=examples,
            data_collator=collate,
            args=TrainingArguments(
                output_dir=str(out_model),
                num_train_epochs=args.epochs,
                learning_rate=args.lr,
                per_device_train_batch_size=1,  # conversations are long; one at a time
                gradient_accumulation_steps=args.grad_accum,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                optim="paged_adamw_8bit",
                warmup_ratio=0.05,
                fp16=True,
                logging_steps=5,
                save_strategy="no",
                report_to=[],
                remove_unused_columns=False,
                seed=args.seed,
            ),
        )
        trainer.train()
        model.save_pretrained(str(out_model / "adapter"))

    model.eval()
    model.config.use_cache = True
    rows = []
    for r in dev:
        ids = torch.tensor([prompt_ids(r)]).to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                 max_new_tokens=512, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        summary, items = parse(text)
        rows.append({"rid": r["rid"], "raw": text, "summary": summary, "action_items": items,
                     "ref_summary": r["summary_en"], "ref_action_items": r["action_items"]})
        print(f"--- {r['rid']}\n{text}\n")

    pred_dir = C2_ROOT / "predictions" / run_id
    pred_dir.mkdir(parents=True, exist_ok=True)
    with open(pred_dir / "dev.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    result = {"run_id": run_id, "model": args.model, "input": args.input,
              "train_conversations": 0 if args.zero_shot else len(train),
              "epochs": 0 if args.zero_shot else args.epochs, "lr": args.lr,
              "dev": {"summary": score_summaries([x["summary"] for x in rows],
                                                 [x["ref_summary"] for x in rows]),
                      "actions": score_actions([x["action_items"] for x in rows],
                                               [x["ref_action_items"] for x in rows])}}
    (C2_ROOT / "eval").mkdir(exist_ok=True)
    (C2_ROOT / "eval" / f"{run_id}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
