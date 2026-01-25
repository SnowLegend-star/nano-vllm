from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self): #run和wait同时为空
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # Prefill阶段：调度新请求（waiting队列）
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        # 循环条件：有等待的新请求 + 已调度数未达上限
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            # 资源检查：两个条件任意一个不满足，就停止Prefill调度
            # 条件1：新增该序列后，批量token数超过上限（防止显存超限）
            # 条件2：块管理器无法为该序列分配KV缓存块（无空闲块）
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1   # 序列的已调度数+1
            self.block_manager.allocate(seq)
            # 累加「新增token数」：总token数 - 已缓存token数（首次调度时num_cached_tokens=0，即全部token）
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # Decode阶段：调度正在运行的请求（running队列）
        # 循环条件：有运行中的序列 + 已调度数未达上限
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            # 循环检查：当前序列是否能追加新token（资源是否足够）
            while not self.block_manager.can_append(seq):
                if self.running:
                    # 优先抢占运行队列最后一个序列（回收其块）
                    self.preempt(self.running.pop())
                else:
                    # 无其他序列可抢占 → 抢占当前序列，退出检查
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq) # 为追加token预留块资源
                scheduled_seqs.append(seq)
        # 断言：Decode阶段必须调度到至少一个序列（否则程序逻辑异常）
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))   # 先翻转调度列表再添加
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)  
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
