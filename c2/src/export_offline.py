"""
One-time export of the trained C2 models into small CPU formats, so C2 runs
offline on a laptop (src/run_offline.py). Run on Colab, once per model set.

    NLLB en / si (fine-tuned) + base NLLB  ->  CTranslate2 int8   (~0.6 GB each)
    Qwen-3B + translate-first LoRA         ->  merged -> GGUF Q4_K_M (~1.9 GB)

    python src/export_offline.py

Reads models from Drive (SORA_C2/models), writes SORA_C2/offline/. Download
that folder to the laptop, outside OneDrive (e.g. C:\\Users\\<you>\\sora_offline).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_c2 import BASE_LLM, BASE_NLLB, DRIVE_MODELS

LLAMA_CPP = Path("/content/llama.cpp")


def sh(cmd):
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True)


def export_nllb(src, out):
    import ctranslate2
    if (out / "model.bin").exists():
        print("exists:", out)
        return
    ctranslate2.converters.TransformersConverter(str(src)).convert(str(out), quantization="int8")
    AutoTokenizer.from_pretrained(str(src)).save_pretrained(str(out))
    print("wrote", out)


def export_qwen(adapter, out_gguf, work=Path("/content/qwen_merged")):
    from peft import PeftModel
    if out_gguf.exists():
        print("exists:", out_gguf)
        return
    # Merge the LoRA adapter into full-precision weights (not 4-bit: a merge
    # into quantized weights loses the adapter's precision).
    model = AutoModelForCausalLM.from_pretrained(BASE_LLM, torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.save_pretrained(str(work))
    AutoTokenizer.from_pretrained(BASE_LLM).save_pretrained(str(work))
    del model
    torch.cuda.empty_cache()

    if not LLAMA_CPP.exists():
        sh(f"git clone -q --depth 1 https://github.com/ggml-org/llama.cpp {LLAMA_CPP}")
    if not (LLAMA_CPP / "build" / "bin" / "llama-quantize").exists():
        sh(f"cmake -S {LLAMA_CPP} -B {LLAMA_CPP}/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF > /dev/null")
        sh(f"cmake --build {LLAMA_CPP}/build --target llama-quantize -j 8 > /dev/null")
    f16 = Path("/content/qwen_f16.gguf")
    sh(f"{sys.executable} {LLAMA_CPP}/convert_hf_to_gguf.py {work} --outtype f16 --outfile {f16}")
    sh(f"{LLAMA_CPP}/build/bin/llama-quantize {f16} {out_gguf} Q4_K_M")
    f16.unlink()
    print("wrote", out_gguf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(DRIVE_MODELS))
    ap.add_argument("--out", default=str(DRIVE_MODELS.parent / "offline"))
    ap.add_argument("--en-model", default="nllb600m_en_frac100")
    ap.add_argument("--si-model", default="nllb600m_si_frac100")
    args = ap.parse_args()
    models, out = Path(args.models), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    export_nllb(models / args.en_model, out / "nllb_en")
    export_nllb(models / args.si_model, out / "nllb_si")
    export_nllb(BASE_NLLB, out / "nllb_base")   # translates the summary into Sinhala
    export_qwen(models / "qwen3b_cascade_qlora" / "adapter", out / "qwen3b_cascade_q4.gguf")
    sh(f"du -sh {out}/*")


if __name__ == "__main__":
    main()
