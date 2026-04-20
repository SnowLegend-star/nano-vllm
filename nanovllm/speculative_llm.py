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

    @staticmethod
    def _greedy_from_logits(logits) -> int:
        return int(logits[0].argmax(dim=-1).item())

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

    def _generate_one(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
    ) -> dict:
        if isinstance(prompt, str):
            prompt_token_ids = self.base_engine.tokenizer.encode(prompt)
        else:
            prompt_token_ids = prompt

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

            return {
                "text": self.base_engine.tokenizer.decode(base_state.seq.completion_token_ids),
                "token_ids": list(base_state.seq.completion_token_ids),
                "accepted_tokens": accepted_tokens,
                "proposed_tokens": proposed_tokens,
                "acceptance_rate": accepted_tokens / proposed_tokens if proposed_tokens else 0.0,
                "resync_count": resync_count,
            }
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
