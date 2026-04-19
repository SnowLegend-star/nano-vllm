from time import perf_counter

from nanovllm import LLM, SpeculativeLLM, SamplingParams

BASE_MODEL_PATH = "./Model/Qwen3-4B"
DRAFT_MODEL_PATH = "./Model/Qwen3-0.6B"

PROMPTS = [
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n请用三句话介绍一下 speculative decoding，并解释它为什么可能提升大模型推理速度。<|im_end|>\n<|im_start|>assistant\n",
]

SAMPLING_PARAMS = SamplingParams(
    temperature=1e-5,  # 当前 SpeculativeLLM MVP 按 greedy 路径测试更稳定
    max_tokens=128,
)

BASE_KWARGS = {
    "enforce_eager": True,
    "tensor_parallel_size": 1,
    "dtype": "float16",
    "max_model_len": 1024,
    "kvcache_memory_budget": 6,
}

DRAFT_KWARGS = {
    "enforce_eager": True,
    "tensor_parallel_size": 1,
    "dtype": "float16",
    "max_model_len": 1024,
    "kvcache_memory_budget": 1,
}


def run_baseline(prompt: str):
    llm = LLM(BASE_MODEL_PATH, **BASE_KWARGS)
    try:
        start = perf_counter()
        output = llm.generate([prompt], SAMPLING_PARAMS, use_tqdm=False)[0]
        elapsed = perf_counter() - start
        num_tokens = len(output["token_ids"])
        return {
            "text": output["text"],
            "token_ids": output["token_ids"],
            "elapsed": elapsed,
            "tokens_per_s": num_tokens / elapsed if elapsed > 0 else 0.0,
        }
    finally:
        llm.exit()


def run_speculative(prompt: str):
    llm = SpeculativeLLM(
        BASE_MODEL_PATH,
        DRAFT_MODEL_PATH,
        draft_length=4,
        base_kwargs=BASE_KWARGS,
        draft_kwargs=DRAFT_KWARGS,
    )
    try:
        start = perf_counter()
        output = llm.generate([prompt], SAMPLING_PARAMS, use_tqdm=False)[0]
        elapsed = perf_counter() - start
        num_tokens = len(output["token_ids"])
        output["elapsed"] = elapsed
        output["tokens_per_s"] = num_tokens / elapsed if elapsed > 0 else 0.0
        return output
    finally:
        llm.exit()


def main():
    prompt = PROMPTS[0]
    print("=== Baseline 4B ===")
    baseline = run_baseline(prompt)
    print(f"elapsed: {baseline['elapsed']:.3f}s")
    print(f"generated_tokens: {len(baseline['token_ids'])}")
    print(f"tokens/s: {baseline['tokens_per_s']:.3f}")
    print(f"text: {baseline['text']}")
    print()

    print("=== Speculative 4B + 0.6B ===")
    speculative = run_speculative(prompt)
    print(f"elapsed: {speculative['elapsed']:.3f}s")
    print(f"generated_tokens: {len(speculative['token_ids'])}")
    print(f"tokens/s: {speculative['tokens_per_s']:.3f}")
    print(f"accepted_tokens: {speculative['accepted_tokens']}")
    print(f"proposed_tokens: {speculative['proposed_tokens']}")
    print(f"acceptance_rate: {speculative['acceptance_rate']:.3f}")
    print(f"resync_count: {speculative['resync_count']}")
    print(f"text: {speculative['text']}")
    print()

    if baseline["tokens_per_s"] > 0:
        speedup = speculative["tokens_per_s"] / baseline["tokens_per_s"]
        print(f"speedup_vs_baseline: {speedup:.3f}x")


if __name__ == "__main__":
    main()


