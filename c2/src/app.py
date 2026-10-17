"""
C2 demo app: paste a Singlish conversation, press the button, get clean
English + clean Sinhala per line, an English + Sinhala summary, action items,
and the JSON that goes to C4. Opens in the browser.

    python src/app.py                       # laptop, offline models (~/sora_offline)
    python src/app.py --backend gpu         # Colab GPU, models from Drive
    python src/app.py --backend preview     # no models: try the screen layout only

Models load once at start-up, then every click reuses them.
"""
import argparse
import json
import re
import time
from pathlib import Path

C2_ROOT = Path(__file__).resolve().parents[1]
LINE = re.compile(r"^\s*([^:\n]{1,20}?):\s+(.+)$")


class Preview:
    """Stand-in backend so the screen can be tried without any model."""
    name = "preview (no models loaded)"

    def translate(self, key, texts, src_lang, tgt_lang):
        return [f"[{key}] {t}" for t in texts]

    def summarise(self, transcript):
        return "raw", "[preview] The summary appears here.", [
            {"intent": "[preview] An action item", "owner": "S1", "deadline": None}]


class Gpu:
    name = "GPU (fine-tuned NLLB-600M + QLoRA Qwen2.5-3B)"

    def __init__(self, models):
        from run_c2 import BASE_NLLB, load_summariser, load_translator
        models = Path(models)
        self.nllb = {"en": load_translator(models / "nllb600m_en_frac100"),
                     "si": load_translator(models / "nllb600m_si_frac100"),
                     "base": load_translator(BASE_NLLB)}
        self.llm = load_summariser(models / "qwen3b_cascade_qlora" / "adapter")

    def translate(self, key, texts, src_lang, tgt_lang):
        from run_c2 import translate_with
        return translate_with(self.nllb[key], texts, src_lang, tgt_lang)

    def summarise(self, transcript):
        from run_c2 import summarise_with
        return summarise_with(self.llm, [transcript])[0]


class Offline:
    name = "offline CPU (CTranslate2 int8 NLLB + llama.cpp Q4 Qwen)"

    def __init__(self, models):
        from run_offline import load_summariser, load_translator
        models = Path(models)
        self.nllb = {"en": load_translator(models / "nllb_en"),
                     "si": load_translator(models / "nllb_si"),
                     "base": load_translator(models / "nllb_base")}
        self.llm = load_summariser(models / "qwen3b_cascade_q4.gguf")

    def translate(self, key, texts, src_lang, tgt_lang):
        from run_offline import translate_with
        return translate_with(self.nllb[key], texts, src_lang, tgt_lang)

    def summarise(self, transcript):
        from run_offline import summarise_with
        return summarise_with(self.llm, [transcript])[0]


def parse_conversation(text):
    """'S1: hello\\nS2: kohomada' -> [{speaker, text}]. Lines without 'X:' go to S1."""
    utts = []
    for line in text.splitlines():
        if not line.strip():
            continue
        m = LINE.match(line)
        speaker, said = (m.group(1).strip(), m.group(2).strip()) if m else ("S1", line.strip())
        utts.append({"speaker": speaker, "text": said})
    return utts


def run_pipeline(backend, text, rid="demo"):
    from normalize import postprocess, prepare
    from train_translation import LANG, SRC_LANG

    utts = parse_conversation(text)
    if not utts:
        return [], "", "", [], {}, "Paste a conversation first."
    timing = {}
    src = [prepare(u["text"]) for u in utts]

    t = time.time()
    en = backend.translate("en", src, SRC_LANG, LANG["en"])
    si = backend.translate("si", src, SRC_LANG, LANG["si"])
    for u, s, e, x in zip(utts, src, en, si):
        u["clean_en"], u["clean_si"] = postprocess(s, e), postprocess(s, x)
    timing["translation"] = time.time() - t

    t = time.time()
    _, summary, items = backend.summarise("\n".join(f"{u['speaker']}: {u['clean_en']}" for u in utts))
    timing["summary"] = time.time() - t

    t = time.time()
    sents = re.split(r"(?<=[.!?])\s+", summary or "")
    summary_si = " ".join(postprocess(s, h) for s, h in
                          zip(sents, backend.translate("base", sents, LANG["en"], LANG["si"])))
    timing["sinhala summary"] = time.time() - t

    rows = [[u["speaker"], u["text"], u["clean_en"], u["clean_si"]] for u in utts]
    actions = [[a["intent"], a["owner"] or "", a["deadline"] or ""] for a in items]
    c4 = {"translation": {"recording_id": rid, "utterance_records": [
              {"utt_id": f"{rid}_u{i:03d}", "speaker": u["speaker"], "clean_en": u["clean_en"],
               "clean_si": u["clean_si"]} for i, u in enumerate(utts, 1)]},
          "summary": {"recording_id": rid, "summaries": {"en": {"text": summary}, "si": {"text": summary_si}},
                      "action_items": [{"intent": a["intent"], "owner": a["owner"], "receiver": None,
                                        "deadline": a["deadline"]} for a in items]}}
    took = " · ".join(f"{k} {v:.1f} s" for k, v in timing.items())
    status = f"{len(utts)} lines · {took} · total {sum(timing.values()):.1f} s · {backend.name}"
    return rows, summary, summary_si, actions, c4, status


def dev_examples():
    """Dev conversations as paste-ready text (never test: that split stays unseen)."""
    path = C2_ROOT / "data" / "dev.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out[r["rid"]] = "\n".join(u["src"] for u in r["utterances"])
    return out


def build(backend):
    import gradio as gr
    examples = dev_examples()

    with gr.Blocks(title="SORA · C2 Singlish translation and summary") as demo:
        gr.Markdown("## SORA · Component 2\nSinhala-English (Singlish) conversation → "
                    "clean English, clean Sinhala, summary and action items.")
        with gr.Row():
            pick = gr.Dropdown(choices=list(examples), label="Example conversation (dev split)",
                               value=None, scale=3)
            load = gr.Button("Load example", scale=1)
        text = gr.Textbox(label="Conversation — one line per turn, e.g.  S1: van ekak onne neda?",
                          lines=10, placeholder="S1: hello sampath bank customer care\n"
                                                "S2: aiyo mage card eka nathi una")
        go = gr.Button("Translate and summarise", variant="primary")
        status = gr.Markdown()
        lines = gr.Dataframe(headers=["Speaker", "Original", "English", "Sinhala"],
                             label="Line-by-line translation", wrap=True)
        with gr.Row():
            sum_en = gr.Textbox(label="Summary (English)", lines=5)
            sum_si = gr.Textbox(label="Summary (Sinhala)", lines=5)
        actions = gr.Dataframe(headers=["Action", "Owner", "Deadline"], label="Action items", wrap=True)
        with gr.Accordion("Output sent to C4 (JSON)", open=False):
            c4 = gr.JSON()

        load.click(lambda rid: examples.get(rid, ""), inputs=pick, outputs=text)
        go.click(lambda t: run_pipeline(backend, t), inputs=text,
                 outputs=[lines, sum_en, sum_si, actions, c4, status])
    return demo


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["offline", "gpu", "preview"], default="offline")
    ap.add_argument("--models", help="model folder (default depends on the backend)")
    ap.add_argument("--share", action="store_true",
                    help="public link via Gradio's servers (needed on Colab); text then leaves this machine")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args(argv)

    print(f"loading {args.backend} models...")
    if args.backend == "gpu":
        from run_c2 import DRIVE_MODELS
        backend = Gpu(args.models or DRIVE_MODELS)
    elif args.backend == "offline":
        backend = Offline(args.models or Path.home() / "sora_offline")
    else:
        backend = Preview()
    build(backend).launch(inbrowser=True, share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
