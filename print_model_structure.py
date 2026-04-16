import argparse
import os
import tempfile
from pathlib import Path

import torch.distributed as dist
from transformers import AutoConfig

from nanovllm.models.qwen3 import Qwen3ForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print the nano-vllm model structure for a local HuggingFace model."
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        default="./Model/Qwen3-4B",
        help="Local model directory, for example ./Model/Qwen3-4B",
    )
    return parser.parse_args()


def format_count(num: int) -> str:
    units = ["", "K", "M", "B", "T"]
    value = float(num)
    for unit in units:
        if abs(value) < 1000 or unit == units[-1]:
            return f"{value:.2f}{unit}"
        value /= 1000.0
    return str(num)


def format_exact_count(num: int) -> str:
    return f"{num:,}"


def count_params(module) -> int:
    return sum(param.numel() for param in module.parameters())


def print_config_summary(hf_config):
    fields = [
        "model_type",
        "architectures",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
        "tie_word_embeddings",
        "torch_dtype",
    ]
    print("=== HuggingFace Config ===")
    for field in fields:
        if hasattr(hf_config, field):
            print(f"{field}: {getattr(hf_config, field)}")
    print()


def print_param_summary(model):
    total_params = count_params(model)
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    print("=== Parameter Summary ===")
    print(f"total_params: {format_exact_count(total_params)}  {format_count(total_params)}")
    print(f"trainable_params: {format_exact_count(trainable_params)}  {format_count(trainable_params)}")
    print()

    print("=== Top-level Module Params ===")
    for name, module in model.named_children():
        count = count_params(module)
        print(f"{name}: {format_exact_count(count)}  {format_count(count)}")
    print()


def print_tree_line(prefix: str, name: str, module, is_last: bool):
    branch = "└─ " if is_last else "├─ "
    count = count_params(module)
    print(
        f"{prefix}{branch}{name} | {module.__class__.__name__} | "
        f"{format_count(count)} params"
    )


def print_module_tree(module, name: str = "root", prefix: str = "", is_last: bool = True):
    if prefix:
        print_tree_line(prefix, name, module, is_last)
    else:
        count = count_params(module)
        print(f"{name} | {module.__class__.__name__} | {format_count(count)} params")

    children = list(module.named_children())
    next_prefix = prefix + ("   " if is_last else "│  ")
    for index, (child_name, child_module) in enumerate(children):
        child_is_last = index == len(children) - 1
        print_module_tree(child_module, child_name, next_prefix, child_is_last)


def main():
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    hf_config = AutoConfig.from_pretrained(model_path)

    fd, store_path = tempfile.mkstemp()
    os.close(fd)
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{store_path}",
            rank=0,
            world_size=1,
        )
        try:
            model = Qwen3ForCausalLM(hf_config)
        finally:
            dist.destroy_process_group()
    finally:
        try:
            os.unlink(store_path)
        except FileNotFoundError:
            pass

    print(f"Model path: {model_path}")
    print()
    print_config_summary(hf_config)
    print_param_summary(model)
    print("=== nano-vllm Module Tree ===")
    print_module_tree(model)


if __name__ == "__main__":
    main()
