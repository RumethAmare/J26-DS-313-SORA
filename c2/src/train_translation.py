"""
Step 3 (Task A): fine-tune NLLB-200-distilled-600M to turn one code-mixed
utterance into clean English or clean Sinhala.

    input   "S1: Aiyo machan traffic eka, Borella langa hari block."
    target  "Oh man the traffic, it's very blocked near Borella."     (--target en)
            "අයියෝ මචං ට්‍රැෆික් එක, බොරැල්ල ළඟ හරි බ්ලොක්."          (--target si)

One model per target language. Needs a GPU: run on Google Colab (T4).

    python src/train_translation.py --target en --zero-shot      # baseline, no training
    python src/train_translation.py --target en
    python src/train_translation.py --target en --normalize      # normalise input first
    python src/train_translation.py --target en --fraction 0.25  # data-size curve

Trains on data/train.jsonl, picks the best epoch on data/dev.jsonl (chrF++),
then writes dev predictions to predictions/<run_id>/dev.jsonl and scores to
eval/<run_id>.json. The test split is never read here.
"""
import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments)

from metrics import score
from normalize import normalize

C2_ROOT = Path(__file__).resolve().parents[1]
DATA = C2_ROOT / "data"

# The input mixes Sinhala script and romanized Singlish; NLLB has no code for
# that, so the source is tagged as Sinhala and fine-tuning adapts it.
LANG = {"en": "eng_Latn", "si": "sin_Sinh"}
SRC_LANG = "sin_Sinh"

# The targets carry no "S1:" label, so drop it from the input too; otherwise the
# model spends capacity learning to delete it.
SPEAKER = re.compile(r"^S(\d+|\?): ")


def load_pairs(split, target, fraction=1.0, seed=13, norm=False):
    records = [json.loads(line) for line in open(DATA / f"{split}.jsonl", encoding="utf-8")]
    if fraction < 1.0:
        # Subsample whole recordings, not utterances, so a 25% run sees 25% of
        # the conversations.
        random.Random(seed).shuffle(records)
        records = records[:max(1, round(len(records) * fraction))]
    pairs = []
    for r in records:
        for u in r["utterances"]:
            if u[f"clean_{target}"]:
                src = SPEAKER.sub("", u["src"])
                pairs.append({"utt_id": u["utt_id"], "src": normalize(src) if norm else src,
                              "tgt": u[f"clean_{target}"]})
    return pairs, len(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["en", "si"], required=True)
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--epochs", type=int, default=10, help="upper bound; stops early if dev chrF++ stalls")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2, help="effective batch = batch x grad-accum")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--fraction", type=float, default=1.0, help="share of training recordings to use")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--normalize", action="store_true", help="run normalize.py on the input first")
    ap.add_argument("--zero-shot", action="store_true", help="no training: score the base model (baseline)")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    run_id = args.run_id or (f"nllb600m_{args.target}_zeroshot" if args.zero_shot
                             else f"nllb600m_{args.target}_frac{int(args.fraction * 100)}")
    if args.normalize:
        run_id += "_norm"
    out_model = C2_ROOT / "models" / run_id
    torch.manual_seed(args.seed)

    train, n_train_recs = load_pairs("train", args.target, args.fraction, args.seed, args.normalize)
    dev, _ = load_pairs("dev", args.target, norm=args.normalize)
    print(f"{run_id}: {len(train)} train pairs from {n_train_recs} recordings, {len(dev)} dev pairs")

    tok = AutoTokenizer.from_pretrained(args.model, src_lang=SRC_LANG, tgt_lang=LANG[args.target])
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    tgt_bos = tok.convert_tokens_to_ids(LANG[args.target])

    def encode(batch):
        return tok(batch["src"], text_target=batch["tgt"],
                   max_length=args.max_len, truncation=True)

    train_ds = Dataset.from_list(train).map(encode, batched=True, remove_columns=["utt_id", "src", "tgt"])
    dev_ds = Dataset.from_list(dev).map(encode, batched=True, remove_columns=["utt_id", "src", "tgt"])

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.where(preds < 0, tok.pad_token_id, preds)
        # Padding in labels is -100 (ignored by the loss); swap it back before decoding.
        labels = np.where(labels == -100, tok.pad_token_id, labels)
        hyp = tok.batch_decode(preds, skip_special_tokens=True)
        ref = tok.batch_decode(labels, skip_special_tokens=True)
        return score(hyp, ref)

    targs = Seq2SeqTrainingArguments(
        output_dir=str(out_model),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        # NLLB's 256k vocab makes AdamW + batch 8 overflow a 15 GB T4;
        # Adafactor keeps far less optimizer state.
        optim="adafactor",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,  # no optimizer state: checkpoints 3x smaller, faster
        load_best_model_at_end=True,
        metric_for_best_model="chrf++",
        predict_with_generate=True,
        generation_max_length=args.max_len,
        logging_steps=50,
        report_to=[],
        seed=args.seed,
    )
    model.generation_config.forced_bos_token_id = tgt_bos
    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        # label_pad_token_id=-100 keeps padding out of the loss, so the model is
        # never taught to emit pad tokens.
        data_collator=DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100),
        processing_class=tok,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    if args.zero_shot:
        n_train_recs, train = 0, []
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        trainer.train()
        trainer.save_model(str(out_model / "best"))
        tok.save_pretrained(str(out_model / "best"))
        model = trainer.model

    # Final dev predictions with beam search, kept apart from the gold data.
    model = model.eval()
    hyps = []
    for i in range(0, len(dev), args.batch):
        chunk = dev[i:i + args.batch]
        enc = tok([p["src"] for p in chunk], return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_len).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, forced_bos_token_id=tgt_bos, num_beams=4,
                                 max_length=args.max_len)
        hyps += tok.batch_decode(out, skip_special_tokens=True)

    pred_dir = C2_ROOT / "predictions" / run_id
    pred_dir.mkdir(parents=True, exist_ok=True)
    with open(pred_dir / "dev.jsonl", "w", encoding="utf-8") as f:
        for p, h in zip(dev, hyps):
            f.write(json.dumps({"utt_id": p["utt_id"], "src": p["src"], "hyp": h, "ref": p["tgt"]},
                               ensure_ascii=False) + "\n")

    result = {"run_id": run_id, "model": args.model, "target": args.target,
              "train_recordings": n_train_recs, "train_pairs": len(train),
              "epochs": 0 if args.zero_shot else args.epochs, "lr": args.lr,
              "fraction": args.fraction, "normalize": args.normalize,
              "dev": score(hyps, [p["tgt"] for p in dev])}
    (C2_ROOT / "eval").mkdir(exist_ok=True)
    (C2_ROOT / "eval" / f"{run_id}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
