import gc
from time import perf_counter

import torch
from nanovllm import LLM, SpeculativeLLM, SamplingParams

BASE_MODEL_PATH = "./Model/Qwen3-4B"
DRAFT_MODEL_PATH = "./Model/Qwen3-0.6B"


def format_chat(user_msg: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


PROMPT_BATCHES = [
    (
        "batch4_mixed",
        [
            format_chat(
                "请用三句话介绍一下 speculative decoding，并解释它为什么可能提升大模型推理速度。"
            ),
            format_chat("What is the capital of France? Answer in one short sentence."),
            format_chat("列出中国的四个直辖市，每个用一行简单介绍。"),
            format_chat(
                "用 Python 写一个函数 fibonacci(n)，返回斐波那契数列前 n 项的列表，只给代码。"
            ),
        ],
    ),
    (
        "batch3_short",
        [
            format_chat("What is the capital of France? Answer in one short sentence."),
            format_chat(
                "用 Python 写一个函数 fibonacci(n)，返回斐波那契数列前 n 项的列表，只给代码。"
            ),
            format_chat("周末天气不错，有没有推荐的户外活动？简短回答即可。"),
        ],
    ),
]

SAMPLING_PARAMS = SamplingParams(
    temperature=1e-5,
    max_tokens=128,
)

BASE_KWARGS = {
    "enforce_eager": True,
    "tensor_parallel_size": 1,
    "dtype": "float16",
    "max_model_len": 1024,
    "kvcache_memory_budget": 6,
}

SPEC_BASE_KWARGS = {
    "enforce_eager": True,
    "tensor_parallel_size": 1,
    "dtype": "float16",
    "max_model_len": 1024,
    "kvcache_memory_budget": 6,
}

DRAFT_KWARGS = {
    "enforce_eager": False,
    "cudagraph_max_bs": 4,
    "tensor_parallel_size": 1,
    "dtype": "float16",
    "max_model_len": 1024,
    "kvcache_memory_budget": 0.5,
}


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _summarize_outputs(outputs: list[dict], elapsed: float) -> dict:
    total_tokens = sum(len(output["token_ids"]) for output in outputs)
    return {
        "elapsed": elapsed,
        "total_tokens": total_tokens,
        "tokens_per_s": total_tokens / elapsed if elapsed > 0 else 0.0,
        "accepted_tokens": sum(output.get("accepted_tokens", 0) for output in outputs),
        "proposed_tokens": sum(output.get("proposed_tokens", 0) for output in outputs),
        "resync_count": sum(output.get("resync_count", 0) for output in outputs),
    }


def _time_baseline_batch(llm: LLM, prompts: list[str]) -> dict:
    outputs = []
    start = perf_counter()
    for prompt in prompts:
        outputs.extend(llm.generate([prompt], SAMPLING_PARAMS, use_tqdm=False))
    elapsed = perf_counter() - start
    return _summarize_outputs(outputs, elapsed)


def _time_single_spec_batch(llm: SpeculativeLLM, prompts: list[str]) -> dict:
    start = perf_counter()
    outputs = llm.generate(prompts, SAMPLING_PARAMS, use_tqdm=False)
    elapsed = perf_counter() - start
    return _summarize_outputs(outputs, elapsed)


def _time_batch_spec_batch(llm: SpeculativeLLM, prompts: list[str]) -> dict:
    start = perf_counter()
    outputs = llm.generate_batch(prompts, SAMPLING_PARAMS, use_tqdm=False)
    elapsed = perf_counter() - start
    return _summarize_outputs(outputs, elapsed)


def run_baseline_suite():
    llm = LLM(BASE_MODEL_PATH, **BASE_KWARGS)
    try:
        return {
            name: _time_baseline_batch(llm, prompts)
            for name, prompts in PROMPT_BATCHES
        }
    finally:
        llm.exit()
        del llm
        _cleanup_cuda()


def run_single_spec_suite(draft_lengths: list[int]):
    llm = SpeculativeLLM(
        BASE_MODEL_PATH,
        DRAFT_MODEL_PATH,
        draft_length=draft_lengths[0],
        base_kwargs=SPEC_BASE_KWARGS,
        draft_kwargs=DRAFT_KWARGS,
    )
    try:
        results = {}
        for k in draft_lengths:
            llm.draft_length = k
            results[k] = {
                name: _time_single_spec_batch(llm, prompts)
                for name, prompts in PROMPT_BATCHES
            }
        return results
    finally:
        llm.exit()
        del llm
        _cleanup_cuda()


def run_batch_spec_suite(draft_lengths: list[int]):
    llm = SpeculativeLLM(
        BASE_MODEL_PATH,
        DRAFT_MODEL_PATH,
        draft_length=draft_lengths[0],
        base_kwargs=SPEC_BASE_KWARGS,
        draft_kwargs=DRAFT_KWARGS,
    )
    try:
        results = {}
        for k in draft_lengths:
            llm.draft_length = k
            results[k] = {
                name: _time_batch_spec_batch(llm, prompts)
                for name, prompts in PROMPT_BATCHES
            }
        return results
    finally:
        llm.exit()
        del llm
        _cleanup_cuda()


def print_summary(
    baseline: dict,
    single_spec: dict[int, dict[str, dict]],
    batch_spec: dict[int, dict[str, dict]],
):
    header = (
        f"{'batch':<18}"
        f"{'base tok/s':>12}"
    )
    for k in single_spec:
        header += (
            f"{'single k=%d' % k:>12}"
            f"{'speedup':>10}"
            f"{'batch k=%d' % k:>12}"
            f"{'speedup':>10}"
        )
    print(header)
    print("-" * len(header))

    for name, _ in PROMPT_BATCHES:
        base = baseline[name]
        row = f"{name:<18}{base['tokens_per_s']:>12.2f}"
        for k in single_spec:
            single = single_spec[k][name]
            batch = batch_spec[k][name]
            row += (
                f"{single['tokens_per_s']:>12.2f}"
                f"{(single['tokens_per_s'] / base['tokens_per_s']):>9.2f}x"
                f"{batch['tokens_per_s']:>12.2f}"
                f"{(batch['tokens_per_s'] / base['tokens_per_s']):>9.2f}x"
            )
        print(row)


def main():
    draft_lengths = [2, 3]

    print("=== Baseline 4B (sequential batch serving) ===")
    baseline = run_baseline_suite()
    for name, prompts in PROMPT_BATCHES:
        result = baseline[name]
        print(
            f"  [{name}] batch={len(prompts)} {result['total_tokens']} tokens in {result['elapsed']:.2f}s "
            f"→ {result['tokens_per_s']:.2f} tok/s"
        )
    print()

    print(f"=== Single-request speculative serving (draft_lengths={draft_lengths}) ===")
    single_spec = run_single_spec_suite(draft_lengths)
    for k, results in single_spec.items():
        print(f"--- single-request draft_length={k} ---")
        for name, prompts in PROMPT_BATCHES:
            result = results[name]
            print(
                f"  [{name}] batch={len(prompts)} {result['total_tokens']} tokens in {result['elapsed']:.2f}s "
                f"→ {result['tokens_per_s']:.2f} tok/s, "
                f"accept={result['accepted_tokens']}/{result['proposed_tokens']}, "
                f"resync={result['resync_count']}"
            )
        print()

    print(f"=== Dynamic batch speculative serving (draft_lengths={draft_lengths}) ===")
    batch_spec = run_batch_spec_suite(draft_lengths)
    for k, results in batch_spec.items():
        print(f"--- dynamic-batch draft_length={k} ---")
        for name, prompts in PROMPT_BATCHES:
            result = results[name]
            print(
                f"  [{name}] batch={len(prompts)} {result['total_tokens']} tokens in {result['elapsed']:.2f}s "
                f"→ {result['tokens_per_s']:.2f} tok/s, "
                f"accept={result['accepted_tokens']}/{result['proposed_tokens']}, "
                f"resync={result['resync_count']}"
            )
        print()

    print("=== Summary ===")
    print_summary(baseline, single_spec, batch_spec)


if __name__ == "__main__":
    main()
