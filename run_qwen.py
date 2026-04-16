import time

from numpy import dtype
from nanovllm import LLM, SamplingParams

MODEL_PATH = "./Model/Qwen3-4B"

llm = LLM(
    "./Model/Qwen3-4B",
    enforce_eager=False,        # 1650 上更稳
    tensor_parallel_size=1,
    dtype="float16",           # 降显存
    gpu_memory_utilization = 0.6,
    max_model_len=4096,
    kvcache_memory_budget=6,
)


draft_llm = LLM(
    "./Model/Qwen3-0.6B",
    enforce_eager=False,
    tensor_parallel_size=1,
    dtype="float16",
    gpu_memory_utilization = 0.3,
    max_model_len=4096,
    kvcache_memory_budget=1,
)

print("Base LLM loaded")

sampling_params = SamplingParams(
    temperature=0.6,
    max_tokens=1024,            # 先别太大
)

prompts = [
    # "Hello! Please briefly introduce yourself.",
    # "今天是圣诞节,圣诞快乐!",
    # "Merry Christmas!",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n我喜欢你。<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n圣诞快乐!<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n我喜欢你,圣诞快乐!<|im_end|>\n<|im_start|>assistant\n"
]

outputs = llm.generate(prompts, sampling_params)

print(outputs[0]["text"])


