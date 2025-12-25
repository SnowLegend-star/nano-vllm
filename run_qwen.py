from nanovllm import LLM, SamplingParams

MODEL_PATH = "/home/linger2/huggingface/Qwen3-0.6B"

llm = LLM(
    MODEL_PATH,
    enforce_eager=True,        # 1650 上更稳
    tensor_parallel_size=1,
    dtype="float16",           # 降显存
    gpu_memory_utilization = 0.8,
    max_model_len=1024,
)

sampling_params = SamplingParams(
    temperature=0.6,
    max_tokens=64,            # 先别太大
)

prompts = [
    "Hello! Please briefly introduce yourself."
]

outputs = llm.generate(prompts, sampling_params)

print(outputs[0]["text"])


