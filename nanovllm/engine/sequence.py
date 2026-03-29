from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    RECOMPUTE = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)    # 我们预热的时候跑了一次max_seq，所以这里seq_id是32
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.prefix_block_table = []
        self.pending_recompute_block_ids = []
        self.evicted_prefix_blocks = 0
        self.recompute_pending = False
        self.keep_last_blocks = 0
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_prefix_cached_blocks(self):
        return len(self.prefix_block_table)

    @property
    def num_blocks(self):
        '''向上取整'''
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    @property
    def has_evicted_prefix(self):
        return self.evicted_prefix_blocks > 0

    @property
    def recompute_start_block(self):
        return len(self.prefix_block_table)

    @property
    def resident_start_block(self):
        return len(self.prefix_block_table) + self.evicted_prefix_blocks

    @property
    def all_block_ids(self):
        return self.prefix_block_table + self.block_table

    def block(self, i):
        '''获取第 i 个块的 token ID 列表'''
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def mark_recompute(self, evicted_blocks: int):
        if evicted_blocks <= 0:
            return
        self.evicted_prefix_blocks += evicted_blocks
        self.recompute_pending = True
        self.status = SequenceStatus.RECOMPUTE

    def clear_recompute_state(self):
        self.prefix_block_table.clear()
        self.pending_recompute_block_ids.clear()
        self.evicted_prefix_blocks = 0
        self.recompute_pending = False
        if self.status != SequenceStatus.FINISHED:
            self.status = SequenceStatus.RUNNING

    def __getstate__(self):
        return (
            self.seq_id,
            self.status,
            self.token_ids,
            self.last_token,
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            self.prefix_block_table,
            self.pending_recompute_block_ids,
            self.evicted_prefix_blocks,
            self.recompute_pending,
            self.keep_last_blocks,
            self.temperature,
            self.max_tokens,
            self.ignore_eos,
        )

    def __setstate__(self, state):
        (
            self.seq_id,
            self.status,
            self.token_ids,
            self.last_token,
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            self.prefix_block_table,
            self.pending_recompute_block_ids,
            self.evicted_prefix_blocks,
            self.recompute_pending,
            self.keep_last_blocks,
            self.temperature,
            self.max_tokens,
            self.ignore_eos,
        ) = state
