from dataclasses import dataclass

from tqdm.auto import tqdm

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


@dataclass
class _SequenceState:
    seq: Sequence
    prefilled: bool = False


class SpeculativeLLM:
    """
    Speculative decoding (V4.1)。

    Base 侧（沿用 V4.0 fused verify）：
    - 每轮只做一次 prefill(k+1)
    - 输入 `[t_{n-1}, d_1, ..., d_k]`，共 k+1 个 query 位置
    - 输出 `[q_0, q_1, ..., q_k]`
    - `q_i` (i=0..k-1) 用来判决 `d_{i+1}`
    - 全部接受时 `argmax(q_k)` 直接作为 bonus token（greedy 语义）

    Draft 侧（V4.1 新增优化）：
    - Case C（全接受 + bonus）不再额外 decode 一次来「固化 d_k 的 KV」。
    - 改为直接 `append_token(bonus)`，让 draft 末尾保留 2 个 uncached token
      (d_k 和 bonus)。下一轮 propose 的第一次 `_next_logits` 会自动走
      prefix-cache prefill，一次性把这两个 token 的 KV 都补齐。
    - 相比 V4.0 每次 Case C 要多一次 decode kernel launch，V4.1 把它合进了
      下一轮的那个 prefill 里，Case C 的 draft 侧开销净减一次前向。
    """

    def __init__(
        self,
        base_model: str,
        draft_model: str,
        *,
        draft_length: int = 2,
        base_kwargs: dict | None = None,
        draft_kwargs: dict | None = None,
    ):
        assert draft_length >= 1
        self.base_engine = LLMEngine(base_model, **(base_kwargs or {}))
        self.draft_engine = LLMEngine(draft_model, **(draft_kwargs or {}))
        self.draft_length = draft_length
        self._closed = False

        # MVP 先假设是同家族 tokenizer，至少要求词表规模和 eos 一致。
        assert len(self.base_engine.tokenizer) == len(self.draft_engine.tokenizer)
        assert self.base_engine.scheduler.eos == self.draft_engine.scheduler.eos

    def exit(self):
        if self._closed:
            return
        self._closed = True
        self.base_engine.exit()
        self.draft_engine.exit()

    def _make_sequence(self, engine: LLMEngine, token_ids: list[int], sampling_params: SamplingParams) -> _SequenceState:
        seq = Sequence(token_ids, sampling_params)
        block_manager = engine.scheduler.block_manager
        if not block_manager.can_allocate(seq):
            raise RuntimeError("KV cache blocks are insufficient for the speculative decoding prompt.")
        block_manager.allocate(seq)
        return _SequenceState(seq=seq, prefilled=False)

    @staticmethod
    def _common_prefix_length(lhs: list[int], rhs: list[int]) -> int:
        n = min(len(lhs), len(rhs))
        i = 0
        while i < n and lhs[i] == rhs[i]:
            i += 1
        return i

    @staticmethod
    def _copy_block_prefix_kv(engine: LLMEngine, src_block_id: int, dst_block_id: int, num_tokens: int):
        if num_tokens <= 0:
            return
        kv_cache = engine.model_runner.kv_cache
        kv_cache[:, :, dst_block_id, :num_tokens].copy_(
            kv_cache[:, :, src_block_id, :num_tokens]
        )

    def _fork_sequence_from_state(
        self,
        engine: LLMEngine,
        state: _SequenceState,
        target_token_ids: list[int],
        cached_prefix_tokens: int,
    ) -> _SequenceState:
        """
        基于当前 state 做增量 fork：
        - 共享已经 materialize 的整块 KV
        - 若最后一个 cached block 未满，则复制其前缀 KV 到新块
        - 只为真正新增的 suffix 额外分配块
        """
        source_seq = state.seq
        assert not source_seq.prefix_block_table
        assert not source_seq.pending_recompute_block_ids
        assert not source_seq.recompute_pending

        cached_prefix_tokens = min(
            cached_prefix_tokens,
            source_seq.num_cached_tokens,
            len(target_token_ids),
        )

        seq = Sequence(list(target_token_ids), self._sampling_params_from_seq(source_seq))
        seq.num_prompt_tokens = source_seq.num_prompt_tokens
        block_manager = engine.scheduler.block_manager
        shared_blocks = min(cached_prefix_tokens // seq.block_size, len(source_seq.block_table))
        shared_block_ids = list(source_seq.block_table[:shared_blocks])
        partial_cached_tokens = cached_prefix_tokens - shared_blocks * seq.block_size

        try:
            for block_id in shared_block_ids:
                block_manager.blocks[block_id].ref_count += 1
                seq.block_table.append(block_id)
            seq.num_cached_tokens = cached_prefix_tokens

            new_blocks = seq.num_blocks - len(seq.block_table)
            if len(block_manager.free_block_ids) < new_blocks:
                raise RuntimeError("KV cache blocks are insufficient for incremental speculative verify.")

            prefix_hash = block_manager.blocks[seq.block_table[-1]].hash if seq.block_table else -1

            if partial_cached_tokens > 0:
                src_block_id = source_seq.block_table[shared_blocks]
                dst_block_id = block_manager.free_block_ids[0]
                block = block_manager._allocate_block(dst_block_id)
                seq.block_table.append(dst_block_id)
                self._copy_block_prefix_kv(engine, src_block_id, dst_block_id, partial_cached_tokens)
                token_ids = seq.block(shared_blocks)
                if len(token_ids) == seq.block_size:
                    prefix_hash = block_manager.compute_hash(token_ids, prefix_hash)
                    block.update(prefix_hash, token_ids)
                    block_manager.hash_to_block_id[prefix_hash] = dst_block_id

            for logical_block_id in range(len(seq.block_table), seq.num_blocks):
                token_ids = seq.block(logical_block_id)
                block_id = block_manager.free_block_ids[0]
                block = block_manager._allocate_block(block_id)
                seq.block_table.append(block_id)
                if len(token_ids) == seq.block_size:
                    prefix_hash = block_manager.compute_hash(token_ids, prefix_hash)
                    block.update(prefix_hash, token_ids)
                    block_manager.hash_to_block_id[prefix_hash] = block_id

            return _SequenceState(seq=seq, prefilled=False)
        except Exception:
            self._free_sequence(engine, _SequenceState(seq=seq, prefilled=False))
            raise

    def _free_sequence(self, engine: LLMEngine, state: _SequenceState | None):
        if state is None:
            return
        seq = state.seq
        if seq.block_table or seq.prefix_block_table or seq.pending_recompute_block_ids:
            engine.scheduler.block_manager.deallocate(seq)

    @staticmethod
    def _grow_blocks_to_num_tokens(engine: LLMEngine, seq: Sequence):
        """
        保证 seq.block_table 覆盖 seq.num_tokens 个 token 所需的块数，并把
        任何已被填满的 block 补上 rolling hash。

        用于「一次 append 了 >=2 个 token 还没算过 KV」的场景（V4.1 里的 Case C
        走 prefix-cache prefill 前必须先把 block 槽位备好）。没有这个步骤的话，
        后续若再走 decode 路径，`block_manager.may_append` 的
        `assert last_block.hash != -1` 断言会直接炸。
        """
        block_manager = engine.scheduler.block_manager
        need_new = seq.num_blocks - len(seq.block_table)
        if need_new < 0:
            return
        if need_new > 0 and len(block_manager.free_block_ids) < need_new:
            raise RuntimeError("KV cache blocks are insufficient while growing speculative sequence.")

        for _ in range(need_new):
            block_id = block_manager.free_block_ids[0]
            block_manager._allocate_block(block_id)
            seq.block_table.append(block_id)

        # 把所有已填满但尚未 hash 的 block 补上滚动哈希，维持 BlockManager 的
        # 核心不变量：「除了最后一个 block，其它 block 都必须 hash != -1」。
        num_full = seq.num_tokens // seq.block_size
        prev_hash = -1
        for logical_id in range(num_full):
            block = block_manager.blocks[seq.block_table[logical_id]]
            if block.hash != -1:
                prev_hash = block.hash
                continue
            token_ids = seq.block(logical_id)
            prev_hash = block_manager.compute_hash(token_ids, prev_hash)
            block.update(prev_hash, token_ids)
            block_manager.hash_to_block_id[prev_hash] = seq.block_table[logical_id]

    def _next_logits(self, engine: LLMEngine, state: _SequenceState):
        """
        返回「基于当前 seq 末尾上下文预测下一个 token 的 logits」。

        路径分支（按未算 KV 的 token 数分）：
        - 初次调用: 走 prefill, 建立 prompt KV。
        - uncached == 1: 走 decode 快路径（单 token, 可吃 CUDA Graph）。
        - uncached >= 2: 走 prefix-cache prefill, 一次性把多个未算 token 的 KV
          全部写回, 并取最后位置的 logits 作为下一个 token 的预测。
          这是 V4.1 为了消灭 Case C dummy decode 引入的路径。
        """
        seq = state.seq
        if not state.prefilled:
            logits = engine.model_runner.forward_logits([seq], "prefill")
            state.prefilled = True
            seq.num_cached_tokens = seq.num_tokens
            return logits

        uncached = seq.num_tokens - seq.num_cached_tokens
        if uncached <= 0:
            raise RuntimeError("_next_logits called with no uncached token to advance.")

        block_manager = engine.scheduler.block_manager
        if uncached == 1:
            if not block_manager.can_append(seq):
                raise RuntimeError("KV cache blocks are insufficient while advancing speculative decoding.")
            block_manager.may_append(seq)
            logits = engine.model_runner.forward_logits([seq], "decode")
        else:
            # 多 token 未算 KV: 走 prefix-cache prefill
            self._grow_blocks_to_num_tokens(engine, seq)
            logits = engine.model_runner.forward_logits([seq], "prefill")

        seq.num_cached_tokens = seq.num_tokens
        return logits

    def _next_logits_batch(
        self,
        engine: LLMEngine,
        states: list[_SequenceState],
    ) -> list:
        """
        V5.0: 批量版 next_logits。

        同一批 state 里可能混有三种形态：
        - 初次 prompt prefill
        - decode (uncached == 1)
        - prefix-cache prefill (uncached >= 2)

        这里按形态分组后分别 batched forward，避免把 decode/prefill 混到一次
        ModelRunner 调用里。
        """
        logits_by_state = [None] * len(states)
        init_prefill_indices = []
        decode_indices = []
        multi_prefill_indices = []

        for idx, state in enumerate(states):
            seq = state.seq
            if not state.prefilled:
                init_prefill_indices.append(idx)
                continue

            uncached = seq.num_tokens - seq.num_cached_tokens
            if uncached <= 0:
                raise RuntimeError("_next_logits_batch called with no uncached token to advance.")
            if uncached == 1:
                decode_indices.append(idx)
            else:
                multi_prefill_indices.append(idx)

        def _assign_logits(indices: list[int], logits, *, mode: str, init_prefill: bool = False):
            if not indices:
                return
            if len(indices) == 1 and logits.dim() == 1:
                logits = logits.unsqueeze(0)
            for row, idx in enumerate(indices):
                state = states[idx]
                logits_by_state[idx] = logits[row]
                if init_prefill:
                    state.prefilled = True
                state.seq.num_cached_tokens = state.seq.num_tokens

        if init_prefill_indices:
            seqs = [states[idx].seq for idx in init_prefill_indices]
            logits = engine.model_runner.forward_logits(seqs, "prefill")
            _assign_logits(init_prefill_indices, logits, mode="prefill", init_prefill=True)

        if decode_indices:
            block_manager = engine.scheduler.block_manager
            for idx in decode_indices:
                seq = states[idx].seq
                if not block_manager.can_append(seq):
                    raise RuntimeError("KV cache blocks are insufficient while advancing speculative decoding.")
                block_manager.may_append(seq)
            seqs = [states[idx].seq for idx in decode_indices]
            logits = engine.model_runner.forward_logits(seqs, "decode")
            _assign_logits(decode_indices, logits, mode="decode")

        if multi_prefill_indices:
            for idx in multi_prefill_indices:
                self._grow_blocks_to_num_tokens(engine, states[idx].seq)
            seqs = [states[idx].seq for idx in multi_prefill_indices]
            logits = engine.model_runner.forward_logits(seqs, "prefill")
            _assign_logits(multi_prefill_indices, logits, mode="prefill")

        return logits_by_state

    @staticmethod
    def _greedy_from_logits(logits) -> int:
        return int(logits.argmax(dim=-1).reshape(-1)[0].item())

    @staticmethod
    def _sequence_finished(seq: Sequence, token_id: int, eos_token_id: int) -> bool:
        return token_id == eos_token_id or seq.num_completion_tokens >= seq.max_tokens

    @staticmethod
    def _sampling_params_from_seq(seq: Sequence) -> SamplingParams:
        return SamplingParams(
            temperature=seq.temperature,
            max_tokens=seq.max_tokens,
            ignore_eos=seq.ignore_eos,
        )

    def _prompt_to_token_ids(self, prompt: str | list[int]) -> list[int]:
        if isinstance(prompt, str):
            return self.base_engine.tokenizer.encode(prompt)
        return list(prompt)

    def _build_output(
        self,
        state: _SequenceState,
        accepted_tokens: int,
        proposed_tokens: int,
        resync_count: int,
    ) -> dict:
        return {
            "text": self.base_engine.tokenizer.decode(state.seq.completion_token_ids),
            "token_ids": list(state.seq.completion_token_ids),
            "accepted_tokens": accepted_tokens,
            "proposed_tokens": proposed_tokens,
            "acceptance_rate": accepted_tokens / proposed_tokens if proposed_tokens else 0.0,
            "resync_count": resync_count,
        }

    def _truncate_sequence(self, engine: LLMEngine, state: _SequenceState, num_tokens: int):
        seq = state.seq
        if num_tokens >= seq.num_tokens:
            return

        block_manager = engine.scheduler.block_manager
        new_num_blocks = (num_tokens + seq.block_size - 1) // seq.block_size
        if len(seq.block_table) > new_num_blocks:
            block_manager._release_block_ids(seq.block_table[new_num_blocks:])
            del seq.block_table[new_num_blocks:]

        seq.token_ids = seq.token_ids[:num_tokens]
        seq.num_tokens = num_tokens
        seq.last_token = seq.token_ids[-1]
        seq.num_cached_tokens = min(seq.num_cached_tokens, num_tokens)

    def _adopt_base_verify_state(
        self,
        base_state: _SequenceState,
        verify_state: _SequenceState,
        accepted_count: int,
        fallback_token_id: int | None,
    ) -> _SequenceState:
        """
        把 verify_state 升级为新的 base_state：
        - 先截断到 `base_state + accepted_count` 个 token
        - 再 append fallback/bonus（V4 下两者语义一致，都是 base 多往前走的那 1 个 token）
        """
        accepted_num_tokens = base_state.seq.num_tokens + accepted_count
        verify_state.seq.num_prompt_tokens = base_state.seq.num_prompt_tokens
        self._truncate_sequence(self.base_engine, verify_state, accepted_num_tokens)
        verify_state.prefilled = True
        self._free_sequence(self.base_engine, base_state)
        if fallback_token_id is not None:
            verify_state.seq.append_token(fallback_token_id)
        return verify_state

    def _propose_with_draft(self, state: _SequenceState, max_steps: int) -> list[int]:
        '''token by token生成n个draft'''
        proposal_token_ids = []
        eos_token_id = self.draft_engine.scheduler.eos
        for _ in range(max_steps):
            logits = self._next_logits(self.draft_engine, state)
            token_id = self._greedy_from_logits(logits)
            state.seq.append_token(token_id)
            proposal_token_ids.append(token_id)
            if self._sequence_finished(state.seq, token_id, eos_token_id):
                break
        return proposal_token_ids

    def _verify_with_base(self, state: _SequenceState, draft_token_ids: list[int]) -> tuple[_SequenceState, int, int | None]:
        """
        V4.0 fused q0 + verify：每轮 base 只做一次 prefill(k+1)。

        - fork verify_state 时 cached_prefix_tokens = base.num_cached_tokens - 1
          这样 `t_{n-1}` 和 `d_1..d_k` 一起作为 query，位置连续从 n-1 到 n+k-1
        - 一次 forward_verify_logits 取回 k+1 个位置的 logits: `[q_0, q_1, ..., q_k]`
        - 用 verify_draft_tokens(include_bonus_token=True) 做一次性比对：
            - 部分接受 → fallback_token_id = argmax(q_{accepted_count})
            - 全部接受 → fallback_token_id = argmax(q_k)（即 bonus token）

        V4 下 fallback 和 bonus 语义一致，都是「base 已经多往前走的那 1 个 token」，
        上层照样 append 到 verify_state 即可。
        """
        assert draft_token_ids
        assert state.prefilled, "base 必须已经完成 prompt prefill 才能进入 verify 路径"

        k = len(draft_token_ids)
        source_seq = state.seq
        # 留最后 1 个 cached token 给 query，让 t_{n-1} 重新进 FlashAttention 的 query。
        # `t_{n-1}` 的 K/V 会按相同 token + 相同位置重算并写入 verify_state 新块，
        # 等价于原 K/V，对正确性无影响，只是多算 1 个 token 的 attention。
        cached_prefix_tokens = max(0, source_seq.num_cached_tokens - 1)

        verify_state = None
        try:
            target_tokens = list(source_seq.token_ids) + draft_token_ids
            verify_state = self._fork_sequence_from_state(
                self.base_engine,
                state,
                target_tokens,
                cached_prefix_tokens=cached_prefix_tokens,
            )
            # 一次前向拿到 q_0..q_k，总共 k+1 个位置
            verify_logits = self.base_engine.model_runner.forward_verify_logits(
                [verify_state.seq],
                "prefill",
                num_logits_to_keep=k + 1,
            )
            verify_state.seq.num_cached_tokens = verify_state.seq.num_tokens

            # verify_logits[:k] 对齐 draft_token_ids[0..k-1]
            # verify_logits[k] 作为 bonus 的 logits
            accepted_count, _, fallback_token_id = (
                self.base_engine.model_runner.verify_draft_tokens(
                    draft_token_ids,
                    verify_logits,
                    include_bonus_token=True,
                )
            )

            next_state = self._adopt_base_verify_state(
                state,
                verify_state,
                accepted_count=accepted_count,
                fallback_token_id=fallback_token_id,
            )
            verify_state = None
            return next_state, accepted_count, fallback_token_id
        finally:
            self._free_sequence(self.base_engine, verify_state)

    def _propose_batch_with_draft(
        self,
        states: list[_SequenceState],
        max_steps_per_state: list[int],
    ) -> list[list[int]]:
        if len(states) != len(max_steps_per_state):
            raise ValueError("states and max_steps_per_state must have the same length.")

        proposal_token_ids = [[] for _ in states]
        stopped = [False] * len(states)
        eos_token_id = self.draft_engine.scheduler.eos
        max_steps = max(max_steps_per_state, default=0)

        for _ in range(max_steps):
            active_indices = [
                idx for idx, state in enumerate(states)
                if not stopped[idx] and len(proposal_token_ids[idx]) < max_steps_per_state[idx]
            ]
            if not active_indices:
                break

            active_states = [states[idx] for idx in active_indices]
            logits_by_state = self._next_logits_batch(self.draft_engine, active_states)
            for local_idx, idx in enumerate(active_indices):
                token_id = self._greedy_from_logits(logits_by_state[local_idx])
                states[idx].seq.append_token(token_id)
                proposal_token_ids[idx].append(token_id)
                if self._sequence_finished(states[idx].seq, token_id, eos_token_id):
                    stopped[idx] = True

        return proposal_token_ids

    def _verify_batch_with_base(
        self,
        states: list[_SequenceState],
        draft_token_ids_batch: list[list[int]],
    ) -> tuple[list[_SequenceState], list[int], list[int | None]]:
        if len(states) != len(draft_token_ids_batch):
            raise ValueError("states and draft_token_ids_batch must have the same length.")
        if not states:
            return [], [], []
        if len(states) == 1:
            next_state, accepted_count, fallback_token_id = self._verify_with_base(
                states[0],
                draft_token_ids_batch[0],
            )
            return [next_state], [accepted_count], [fallback_token_id]

        proposal_len = len(draft_token_ids_batch[0])
        if proposal_len <= 0:
            raise ValueError("batched verify requires non-empty draft_token_ids.")
        if any(len(token_ids) != proposal_len for token_ids in draft_token_ids_batch):
            raise ValueError("batched verify currently requires all draft_token_ids to have the same length.")

        verify_states = []
        try:
            for state, draft_token_ids in zip(states, draft_token_ids_batch):
                source_seq = state.seq
                cached_prefix_tokens = max(0, source_seq.num_cached_tokens - 1)
                target_tokens = list(source_seq.token_ids) + draft_token_ids
                verify_states.append(
                    self._fork_sequence_from_state(
                        self.base_engine,
                        state,
                        target_tokens,
                        cached_prefix_tokens=cached_prefix_tokens,
                    )
                )

            verify_logits = self.base_engine.model_runner.forward_verify_logits(
                [verify_state.seq for verify_state in verify_states],
                "prefill",
                num_logits_to_keep=proposal_len + 1,
            )
            if verify_logits.dim() == 2:
                verify_logits = verify_logits.unsqueeze(0)
            for verify_state in verify_states:
                verify_state.seq.num_cached_tokens = verify_state.seq.num_tokens

            next_states = []
            accepted_counts = []
            fallback_token_ids = []
            for idx, (state, draft_token_ids) in enumerate(zip(states, draft_token_ids_batch)):
                accepted_count, _, fallback_token_id = self.base_engine.model_runner.verify_draft_tokens(
                    draft_token_ids,
                    verify_logits[idx],
                    include_bonus_token=True,
                )
                next_state = self._adopt_base_verify_state(
                    state,
                    verify_states[idx],
                    accepted_count=accepted_count,
                    fallback_token_id=fallback_token_id,
                )
                verify_states[idx] = None
                next_states.append(next_state)
                accepted_counts.append(accepted_count)
                fallback_token_ids.append(fallback_token_id)

            return next_states, accepted_counts, fallback_token_ids
        finally:
            for verify_state in verify_states:
                self._free_sequence(self.base_engine, verify_state)

    def _generate_batch2(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        if len(prompts) != 2 or len(sampling_params) != 2:
            raise ValueError("V5.0 generate_batch currently only supports exactly 2 prompts.")

        prompt_token_ids_batch = [self._prompt_to_token_ids(prompt) for prompt in prompts]
        base_states = [
            self._make_sequence(self.base_engine, prompt_token_ids, sp)
            for prompt_token_ids, sp in zip(prompt_token_ids_batch, sampling_params)
        ]
        draft_states = [
            self._make_sequence(self.draft_engine, prompt_token_ids, sp)
            for prompt_token_ids, sp in zip(prompt_token_ids_batch, sampling_params)
        ]

        proposed_tokens = [0, 0]
        accepted_tokens = [0, 0]
        resync_count = [0, 0]
        finished = [False, False]
        reported_finished = [False, False]
        pbar = tqdm(total=2, desc="Speculative batch decoding", dynamic_ncols=True) if use_tqdm else None

        def _mark_finished(idx: int):
            finished[idx] = True
            if pbar is not None and not reported_finished[idx]:
                pbar.update(1)
            reported_finished[idx] = True

        try:
            _ = self._next_logits_batch(self.base_engine, base_states)

            while True:
                for idx, state in enumerate(base_states):
                    if finished[idx]:
                        continue
                    remaining = sampling_params[idx].max_tokens - state.seq.num_completion_tokens
                    if remaining <= 0:
                        _mark_finished(idx)

                if all(finished):
                    break

                active_indices = [idx for idx in range(2) if not finished[idx]]
                active_draft_states = [draft_states[idx] for idx in active_indices]
                max_steps_per_state = [
                    min(
                        self.draft_length,
                        sampling_params[idx].max_tokens - base_states[idx].seq.num_completion_tokens,
                    )
                    for idx in active_indices
                ]
                proposal_token_ids_batch = self._propose_batch_with_draft(
                    active_draft_states,
                    max_steps_per_state,
                )

                proposal_by_index = {}
                for local_idx, idx in enumerate(active_indices):
                    proposal_token_ids = proposal_token_ids_batch[local_idx]
                    proposal_by_index[idx] = proposal_token_ids
                    proposed_tokens[idx] += len(proposal_token_ids)

                verify_groups: dict[int, list[int]] = {}
                for idx in active_indices:
                    proposal_len = len(proposal_by_index[idx])
                    if proposal_len > 0:
                        verify_groups.setdefault(proposal_len, []).append(idx)

                if not verify_groups:
                    break

                for group_indices in verify_groups.values():
                    next_states, accepted_counts, fallback_token_ids = self._verify_batch_with_base(
                        [base_states[idx] for idx in group_indices],
                        [proposal_by_index[idx] for idx in group_indices],
                    )
                    for local_idx, idx in enumerate(group_indices):
                        accepted_count = accepted_counts[local_idx]
                        fallback_token_id = fallback_token_ids[local_idx]
                        proposal_token_ids = proposal_by_index[idx]
                        base_states[idx] = next_states[local_idx]
                        accepted_tokens[idx] += accepted_count

                        if self._sequence_finished(
                            base_states[idx].seq,
                            base_states[idx].seq.last_token,
                            self.base_engine.scheduler.eos,
                        ):
                            _mark_finished(idx)
                            continue

                        if fallback_token_id is None:
                            continue

                        is_partial_accept = accepted_count != len(proposal_token_ids)
                        if is_partial_accept:
                            resync_count[idx] += 1
                            target_len = base_states[idx].seq.num_tokens - 1
                            if target_len < draft_states[idx].seq.num_tokens:
                                self._truncate_sequence(self.draft_engine, draft_states[idx], target_len)
                            draft_states[idx].seq.append_token(fallback_token_id)
                        else:
                            draft_states[idx].seq.append_token(fallback_token_id)

            return [
                self._build_output(
                    base_states[idx],
                    accepted_tokens[idx],
                    proposed_tokens[idx],
                    resync_count[idx],
                )
                for idx in range(2)
            ]
        finally:
            if pbar is not None:
                pbar.close()
            for state in base_states:
                self._free_sequence(self.base_engine, state)
            for state in draft_states:
                self._free_sequence(self.draft_engine, state)

    def _generate_one(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
    ) -> dict:
        prompt_token_ids = self._prompt_to_token_ids(prompt)

        base_state = self._make_sequence(self.base_engine, prompt_token_ids, sampling_params)
        draft_state = self._make_sequence(self.draft_engine, prompt_token_ids, sampling_params)

        # V4: 进入 verify 循环前必须先把 base 的 prompt KV 建好。
        # 之后每轮 fused verify 会把 prompt 的最后一个 token 作为 query 位置 n-1 参与前向，
        # 这里返回的 prompt-next logits 用不到直接丢弃（V4 verify 会在 fused 前向里重算 q_0）。
        # draft 这边由 _propose_with_draft 的第一次 decode 自动 prefill，不需要显式处理。
        _ = self._next_logits(self.base_engine, base_state)

        proposed_tokens = 0
        accepted_tokens = 0
        resync_count = 0

        try:
            while True:
                remaining = sampling_params.max_tokens - base_state.seq.num_completion_tokens
                if remaining <= 0:
                    break

                proposal_token_ids = self._propose_with_draft(
                    draft_state,
                    min(self.draft_length, remaining),
                )
                if not proposal_token_ids:
                    break

                proposed_tokens += len(proposal_token_ids)
                base_state, accepted_count, fallback_token_id = self._verify_with_base(base_state, proposal_token_ids)
                accepted_tokens += accepted_count

                if self._sequence_finished(
                    base_state.seq,
                    base_state.seq.last_token,
                    self.base_engine.scheduler.eos,
                ):
                    break

                # V4: 让 draft 追上 base 的最新真实前缀。
                # draft 当前末尾是 `prompt + d_1..d_k`（d_k 的 KV 尚未算，是 draft 的正常稳态），
                # base 现在是 `prompt + d_1..d_{accepted} + (fallback|bonus)`。
                # 只要 fallback_token_id is not None，base 就多往前走了 1 个 token，draft 需要同步。
                if fallback_token_id is not None:
                    is_partial_accept = accepted_count != len(proposal_token_ids)
                    if is_partial_accept:
                        resync_count += 1
                        # Case A/B：直接截断到 accepted 长度（同时丢掉 d_k 的未算 KV），
                        # 再 append fallback，维持「末尾 1 token KV 未算」的 draft 不变量。
                        target_len = base_state.seq.num_tokens - 1
                        if target_len < draft_state.seq.num_tokens:
                            self._truncate_sequence(self.draft_engine, draft_state, target_len)
                        draft_state.seq.append_token(fallback_token_id)
                    else:
                        # Case C (全接受 + bonus)：V4.1 优化——直接 append bonus 就够了。
                        # 此时 draft 末尾有两个 uncached token (d_k 和 bonus), 下一轮
                        # propose 的第一次 _next_logits 会自动走 prefix-cache prefill 一次性
                        # 把两者 KV 都算完，比 V4.0 额外 decode 一次再 append 要便宜。
                        draft_state.seq.append_token(fallback_token_id)

            return self._build_output(
                base_state,
                accepted_tokens,
                proposed_tokens,
                resync_count,
            )
        finally:
            self._free_sequence(self.base_engine, base_state)
            self._free_sequence(self.draft_engine, draft_state)

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        outputs = []
        iterator = zip(prompts, sampling_params)
        if use_tqdm:
            iterator = tqdm(iterator, total=len(prompts), desc="Speculative decoding", dynamic_ncols=True)

        for prompt, sp in iterator:
            outputs.append(self._generate_one(prompt, sp))
        return outputs

    def generate_batch(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        return self._generate_batch2(prompts, sampling_params, use_tqdm=use_tqdm)
