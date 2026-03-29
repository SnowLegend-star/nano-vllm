from nanovllm import LLM, SamplingParams

MODEL_PATH = "./Model/Qwen3-0.6B"

llm = LLM(
    MODEL_PATH,
    enforce_eager=True,        # 1650 上更稳
    tensor_parallel_size=1,
    dtype="float16",           # 降显存
    gpu_memory_utilization = 0.85,
    max_model_len=500,
)

sampling_params = SamplingParams(
    temperature=0.6,
    max_tokens=200,            # 先别太大
)

prompts = [
    # "Hello! Please briefly introduce yourself.",
    # "今天是圣诞节,圣诞快乐!",
    # "Merry Christmas!",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n我喜欢你。<|im_end|>\n<|im_start|>assistant\n"
]

outputs = llm.generate(prompts, sampling_params)

print(outputs[0]["text"])


