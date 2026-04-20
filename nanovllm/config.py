import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    kvcache_keep_last_blocks: int = 4
    kvcache_recompute_chunk_blocks: int = 1
    kvcache_memory_budget: float | None = None
    # CUDA Graph 捕获时允许的最大 batch size。None 表示沿用 max_num_seqs（原行为）。
    # 在 speculative decoding 的 draft engine 这种只跑 bs=1 的场景里设成 1，能把
    # graph pool 占用的显存压到最小，同时避免捕获用不到的大 bs 图。
    cudagraph_max_bs: int | None = None

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.kvcache_keep_last_blocks >= 1
        assert self.kvcache_recompute_chunk_blocks >= 1
        if self.kvcache_memory_budget is not None:
            assert self.kvcache_memory_budget > 0
        if self.cudagraph_max_bs is not None:
            assert self.cudagraph_max_bs >= 1
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
