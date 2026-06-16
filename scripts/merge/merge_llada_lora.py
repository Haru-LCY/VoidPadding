#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer


TOKENIZER_AUX_FILES = (
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge a LLaDA PEFT LoRA adapter into its base model for LLaDA eval."
    )
    parser.add_argument("--base_model", required=True, help="LLaDA base/instruct model path or HF id.")
    parser.add_argument("--adapter_path", required=True, help="PEFT LoRA adapter directory.")
    parser.add_argument("--output_dir", required=True, help="Directory to save merged model and tokenizer.")
    parser.add_argument(
        "--tokenizer_path",
        default=None,
        help="Tokenizer path to save with merged model. Defaults to adapter, repo tokenizer_llada, then base model.",
    )
    parser.add_argument(
        "--torch_dtype",
        default="bfloat16",
        choices=("float32", "float16", "bfloat16"),
        help="dtype used while loading and saving the merged model.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for merge: auto, cpu, cuda, or cuda:N. auto uses cuda when available.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code to transformers loaders.",
    )
    return parser.parse_args()


def resolve_dtype(name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def resolve_device(value):
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def repo_root():
    return Path(__file__).resolve().parents[2]


def load_tokenizer(args):
    candidates = []
    if args.tokenizer_path:
        candidates.append(Path(args.tokenizer_path))
    candidates.extend(
        [
            Path(args.adapter_path),
            repo_root() / "tokenizer_llada",
            Path(args.base_model),
        ]
    )

    last_error = None
    for candidate in candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(candidate),
                trust_remote_code=args.trust_remote_code,
            )
            return tokenizer, candidate
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not load tokenizer from candidates: {candidates}") from last_error


def copy_tokenizer_aux_files(tokenizer_source, output_dir):
    if not isinstance(tokenizer_source, Path) or not tokenizer_source.is_dir():
        return
    for name in TOKENIZER_AUX_FILES:
        src = tokenizer_source / name
        dst = output_dir / name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = resolve_dtype(args.torch_dtype)
    device = resolve_device(args.device)

    print(f"[merge] base_model={args.base_model}")
    print(f"[merge] adapter_path={args.adapter_path}")
    print(f"[merge] output_dir={output_dir}")
    print(f"[merge] torch_dtype={args.torch_dtype}")
    print(f"[merge] device={device}")

    base_model = AutoModel.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
    )
    base_model.to(device)
    base_model.eval()

    peft_model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
        is_trainable=False,
    )
    peft_model.eval()

    merged = peft_model.merge_and_unload()
    merged.save_pretrained(str(output_dir), safe_serialization=True)

    generation_config_path = Path(args.base_model) / "generation_config.json"
    if generation_config_path.is_file():
        shutil.copy2(generation_config_path, output_dir / "generation_config.json")

    tokenizer, tokenizer_source = load_tokenizer(args)
    tokenizer.save_pretrained(str(output_dir))
    copy_tokenizer_aux_files(tokenizer_source, output_dir)

    metadata = {
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "tokenizer_source": str(tokenizer_source),
        "torch_dtype": args.torch_dtype,
        "device": device,
    }
    with open(output_dir / "merge_llada_lora.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=True)

    print("[merge] done")


if __name__ == "__main__":
    main()
