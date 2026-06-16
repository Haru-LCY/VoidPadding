import argparse
import random

import numpy as np
import torch

from method import train


SUPPORTED_MODEL_TYPES = {
    "llada_base": "GSAI-ML/LLaDA-8B-Base",
    "llada_instruct": "GSAI-ML/LLaDA-8B-Instruct",
    "dream_base": "Dream-org/Dream-v0-Base-7B",
    "dream_instruct": "Dream-org/Dream-v0-Instruct-7B",
}


def fix_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Rainbow Padding LoRA SFT for LLaDA and Dream.")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--mode", choices=["data"], default="data")
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument(
        "--save_at_update_steps",
        type=str,
        default="",
        help="comma-separated optimizer update steps to checkpoint, e.g. 7000; empty disables step checkpoints",
    )
    parser.add_argument(
        "--stop_at_update_step",
        type=int,
        default=0,
        help="stop training after this optimizer update step; 0 disables early stopping",
    )
    parser.add_argument("--model_type", choices=sorted(SUPPORTED_MODEL_TYPES), default="llada_base")
    parser.add_argument("--method", choices=["sft"], default="sft")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pad_num",
        type=int,
        default=7,
        help="number of Rainbow padding tokens after the first EOS/EOT padding token",
    )
    parser.add_argument(
        "--pad_strategy",
        choices=["rainbow", "eos", "void"],
        default="rainbow",
        help="padding strategy: rainbow cycles pad tokens, eos keeps EOS/EOT padding, void uses VOID padding",
    )
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--sft_cache_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--swanlab_mode",
        choices=["local", "disabled"],
        default="local",
        help="record runs with SwanLab locally or disable it",
    )
    parser.add_argument(
        "--swanlab_logdir",
        type=str,
        default="outputs/swanlab",
        help="base directory for SwanLab local experiment runs",
    )
    parser.add_argument(
        "--resume_dir",
        type=str,
        default=None,
        help="folder that contains adapter_config.json, adapter_model.safetensors, training_state.pt",
    )
    parser.add_argument(
        "--checkpoint_suffix_extra",
        type=str,
        default="",
        help="optional suffix appended to checkpoint directory names",
    )
    return parser.parse_args()


def model_name_from_model_type(model_type):
    return SUPPORTED_MODEL_TYPES[model_type]


def main(args):
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
    print(f"Epochs: {args.epochs}")
    print(f"Mode: {args.mode}")
    print(f"Save Interval: {args.save_interval}")
    print(f"Save At Update Steps: {args.save_at_update_steps or '<disabled>'}")
    print(f"Stop At Update Step: {args.stop_at_update_step or '<disabled>'}")
    print(f"Model Type: {args.model_type}")
    print(f"Model Name: {args.model_name}")
    print(f"Method: {args.method}")
    print(f"Seed: {args.seed}")
    print(f"Pad Num: {args.pad_num}")
    print(f"Pad Strategy: {args.pad_strategy}")
    print(f"Run Name: {args.run_name}")
    print(f"SwanLab Mode: {args.swanlab_mode}")
    print(f"SwanLab LogDir: {args.swanlab_logdir}")

    train(args.method, args)


if __name__ == "__main__":
    args = parse_args()
    if args.model_name is None:
        args.model_name = model_name_from_model_type(args.model_type)
    fix_random_seed(args.seed)

    main(args)
