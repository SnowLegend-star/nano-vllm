from time import perf_counter

from nanovllm import LLM, SpeculativeLLM, SamplingParams

BASE_MODEL_PATH = "./Model/Qwen3-4B"
DRAFT_MODEL_PATH = "./Model/Qwen3-0.6B"


def format_chat(user_msg: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# 横向 prompt 集合：覆盖不同「draft 友好度」。
# think_long 是最坏情况（发散、长推理），short_qa / factual 是最好情况（模式化、答案短）。
PROMPT_SUITE = [
    (
        "think_long",
        format_chat(
            "请用三句话介绍一下 speculative decoding，并解释它为什么可能提升大模型推理速度。"
        ),
    ),
    (
        "short_qa",
        format_chat("What is the capital of France? Answer in one short sentence."),
    ),
    (
        "factual_list",
        format_chat("列出中国的四个直辖市，每个用一行简单介绍。"),
    ),
    (
        "template_code",
        format_chat(
            "用 Python 写一个函数 fibonacci(n)，返回斐波那契数列前 n 项的列表，只给代码。"
        ),
    ),
    (
        "casual_chat",
        format_chat("周末天气不错，有没有推荐的户外活动？简短回答即可。"),
    ),
]

SAMPLING_PARAMS = SamplingParams(
    temperature=1e-5,  # 当前 SpeculativeLLM MVP 按 greedy 路径测试更稳定
    max_tokens=128,
)

BASE_KWARGS = {
    "enforce_eager": True,
    "cudagraph_max_bs": 1,
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
    # V4.2: 给 draft engine 打开 CUDA Graph。
    # - speculative decoding 里 draft 永远是 bs=1 decode，吃 Graph 最划算。
    # - 通过 cudagraph_max_bs=1 把 graph 池只捕获一张 [bs=1] 的图，
    #   显存开销和捕获时间都压到最小，也避开了大 bs 无意义捕获。
    # - Case C (全接受 + bonus) 走 prefix-cache prefill，不经 Graph，行为保持不变。
    # - kvcache_memory_budget 从 1 压到 0.5：因为 base 已经吃掉大半显存，
    #   graph capture 还要 ~100-200MB activation workspace，budget=1 时 OOM
    #   会直接 fallback 到 eager。实测 0.5GB≈16 blocks 足够单序列 max_model_len=1024。
    "enforce_eager": False,
    "cudagraph_max_bs": 1,
    "tensor_parallel_size": 1,
    "dtype": "float16",
    "max_model_len": 1024,
    "kvcache_memory_budget": 0.5,
}


def _time_generate(llm, prompt: str) -> dict:
    start = perf_counter()
    output = llm.generate([prompt], SAMPLING_PARAMS, use_tqdm=False)[0]
    elapsed = perf_counter() - start
    num_tokens = len(output["token_ids"])
    output["elapsed"] = elapsed
    output["tokens"] = num_tokens
    output["tokens_per_s"] = num_tokens / elapsed if elapsed > 0 else 0.0
    return output


def run_baseline_suite():
    """在一次模型加载里跑完所有 prompt 的 baseline。"""
    llm = LLM(BASE_MODEL_PATH, **BASE_KWARGS)
    try:
        results = {}
        for name, prompt in PROMPT_SUITE:
            out = _time_generate(llm, prompt)
            results[name] = out
        return results
    finally:
        llm.exit()


def run_speculative_suite(draft_lengths: list[int]):
    """
    在一次 base+draft 加载里跑完所有 (draft_length, prompt) 组合。
    draft_length 通过直接改 llm.draft_length 切换，不重载模型。
    """
    llm = SpeculativeLLM(
        BASE_MODEL_PATH,
        DRAFT_MODEL_PATH,
        draft_length=draft_lengths[0],
        base_kwargs=SPEC_BASE_KWARGS,
        draft_kwargs=DRAFT_KWARGS,
    )
    try:
        results: dict[int, dict[str, dict]] = {}
        for k in draft_lengths:
            llm.draft_length = k
            results[k] = {}
            for name, prompt in PROMPT_SUITE:
                out = _time_generate(llm, prompt)
                results[k][name] = out
        return results
    finally:
        llm.exit()


def _fmt(v, width, spec=""):
    return f"{v:{spec}}".rjust(width)


def print_summary(baseline: dict, spec: dict[int, dict[str, dict]]):
    header = (
        f"{'prompt':<16}{'base tok/s':>12}"
    )
    for k in spec:
        header += f"{'k=%d tok/s' % k:>12}{'speedup':>10}{'accept':>8}{'resync':>8}{'gen':>6}"
    print(header)
    print("-" * len(header))
    for name, _ in PROMPT_SUITE:
        b = baseline[name]
        row = f"{name:<16}{b['tokens_per_s']:>12.2f}"
        for k, results in spec.items():
            r = results[name]
            speedup = r["tokens_per_s"] / b["tokens_per_s"] if b["tokens_per_s"] > 0 else 0.0
            row += (
                f"{r['tokens_per_s']:>12.2f}"
                f"{speedup:>9.2f}x"
                f"{r['acceptance_rate']:>8.2f}"
                f"{r['resync_count']:>8d}"
                f"{r['tokens']:>6d}"
            )
        print(row)


def main():
    print("=== Baseline 4B (prompt suite) ===")
    baseline = run_baseline_suite()
    for name, _ in PROMPT_SUITE:
        b = baseline[name]
        print(
            f"  [{name}] {b['tokens']} tokens in {b['elapsed']:.2f}s → {b['tokens_per_s']:.2f} tok/s"
        )
    print()

    draft_lengths = [2, 3]
    print(f"=== Speculative 4B + 0.6B (draft_lengths={draft_lengths}) ===")
    spec = run_speculative_suite(draft_lengths)
    for k, results in spec.items():
        print(f"--- draft_length={k} ---")
        for name, _ in PROMPT_SUITE:
            r = results[name]
            print(
                f"  [{name}] {r['tokens']} tokens in {r['elapsed']:.2f}s → {r['tokens_per_s']:.2f} tok/s, "
                f"accept={r['acceptance_rate']:.3f}, resync={r['resync_count']}, "
                f"accepted/proposed={r['accepted_tokens']}/{r['proposed_tokens']}"
            )
        print()

    print("=== Summary ===")
    print_summary(baseline, spec)


if __name__ == "__main__":
    main()
