from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        Sequence.block_size = config.kvcache_block_size
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.keep_last_blocks = config.kvcache_keep_last_blocks
        self.recompute_chunk_blocks = config.kvcache_recompute_chunk_blocks
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self): #run和wait同时为空
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        seq.keep_last_blocks = self.keep_last_blocks
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], str]:
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
            return scheduled_seqs, "prefill"

        scheduled_seqs = self.schedule_recompute()
        if scheduled_seqs:
            return scheduled_seqs, "recompute"

        # Decode阶段：调度正在运行的请求（running队列）
        running = list(self.running)
        self.running.clear()
        kept = []
        while running and num_seqs < self.max_num_seqs:
            seq = running.pop(0)
            if seq.recompute_pending:
                kept.append(seq)
                continue
            while not self.block_manager.can_append(seq):
                victim = self._take_victim(running, seq)
                if victim is None:
                    if self.evict_or_preempt(seq):
                        kept.append(seq)
                    seq = None
                    break
                if self.evict_or_preempt(victim):
                    kept.append(victim)
            if seq is None:
                continue
            num_seqs += 1
            self.block_manager.may_append(seq)
            seq.status = SequenceStatus.RUNNING
            scheduled_seqs.append(seq)
            kept.append(seq)
        self.running.extend(kept)
        self.running.extend(running)
        assert scheduled_seqs
        return scheduled_seqs, "decode"

    def schedule_recompute(self) -> list[Sequence]:
        pending = []
        for seq in self.running:
            if not seq.recompute_pending or seq.pending_recompute_block_ids:
                continue
            pending.append(seq)
            num_blocks = min(self.recompute_chunk_blocks, seq.evicted_prefix_blocks)
            if not self.block_manager.can_restore_prefix(seq, num_blocks):
                continue
            self.block_manager.reserve_prefix_restore_blocks(seq, num_blocks)
            seq.status = SequenceStatus.RECOMPUTE
            return [seq]
        if pending:
            seq = pending[-1]
            self.running.remove(seq)
            self.preempt(seq)
        return []

    def _take_victim(self, running: list[Sequence], exclude: Sequence):
        # 只从尚未进入本轮 decode 执行集的 running 中挑 victim，
        # 避免把已经放进 scheduled_seqs/kept 的序列回收掉，导致 block_table 被清空。
        for i in range(len(running) - 1, -1, -1):
            seq = running[i]
            if seq is exclude or seq.recompute_pending:
                continue
            return running.pop(i)
        return None

    def evict_or_preempt(self, seq: Sequence) -> bool:
        evicted = self.block_manager.evict_prefix(seq, seq.keep_last_blocks)
        if evicted:
            return True
        self.preempt(seq)
        return False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess_decode(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)  
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
            else:
                seq.status = SequenceStatus.RUNNING

    def postprocess_recompute(self, seqs: list[Sequence]):
        for seq in seqs:
            self.block_manager.commit_prefix_restore(seq)
            if seq.recompute_pending:
                seq.status = SequenceStatus.RECOMPUTE
            else:
                seq.status = SequenceStatus.RUNNING
