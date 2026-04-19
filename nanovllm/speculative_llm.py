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
    最小可运行版 speculative decoding:
    - 单请求逐条处理，不做 batch
    - 使用 greedy proposal / greedy verify
    - 当前版本已切到“q0 + 整段 verify”路径
    - reject 后直接重建 draft 序列，便于后续替换成高性能 verify
    """

    def __init__(
        self,
        base_model: str,
        draft_model: str,
        *,
        draft_length: int = 4,
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

    def _free_sequence(self, engine: LLMEngine, state: _SequenceState | None):
        if state is None:
            return
        seq = state.seq
        if seq.block_table or seq.prefix_block_table or seq.pending_recompute_block_ids:
            engine.scheduler.block_manager.deallocate(seq)

    def _next_logits(self, engine: LLMEngine, state: _SequenceState):
        '''获取下一个token的logits, 有Prefill和Decode两种情况'''
        if not state.prefilled:
            logits = engine.model_runner.forward_logits([state.seq], "prefill")
            state.prefilled = True
            return logits

        block_manager = engine.scheduler.block_manager
        if not block_manager.can_append(state.seq):
            raise RuntimeError("KV cache blocks are insufficient while advancing speculative decoding.")
        block_manager.may_append(state.seq)
        return engine.model_runner.forward_logits([state.seq], "decode")

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

    def _commit_base_tokens(self, state: _SequenceState, token_ids: list[int]):
        """
        把已经判决完成的 token 真正同步回 base_state。
        当前实现采用顺序 replay：
        - 逐个 append 已知 token
        - 对除最后一个之外的 token，额外跑一次 _next_logits 来 materialize KV

        这样虽然还不够快，但至少能保证：
        - base_state.seq.token_ids
        - base_state 的 KV cache
        在语义上重新一致。
        """
        if not token_ids:
            return

        for idx, token_id in enumerate(token_ids):
            state.seq.append_token(token_id)
            if idx == len(token_ids) - 1:
                continue
            _ = self._next_logits(self.base_engine, state)

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

    def _verify_with_base(self, state: _SequenceState, draft_token_ids: list[int]) -> tuple[int, list[int], int | None]:
        """
        当前版本的整段 verify:
        1. 先用一次 decode logits 得到 q0，用来比较 d1
        2. 若 d1 通过，再对完整 draft suffix 做一次 prefill verify，拿到 q1..qk
        3. 用 q1..q{k-1} 比较 d2..dk
        4. 当前版本先不消费 qk bonus token，后续再接入
        """
        assert draft_token_ids
        eos_token_id = self.base_engine.scheduler.eos
        accepted_count = 0
        accepted_token_ids: list[int] = []

        # 1. 当前真实前缀的 next-token logits: q0
        q0_logits = self._next_logits(self.base_engine, state)
        q0_token_id = self._greedy_from_logits(q0_logits)

        # d1 mismatch，直接走 fallback，无需再构造 verify_seq
        if q0_token_id != draft_token_ids[0]:
            return accepted_count, accepted_token_ids, q0_token_id

        # d1 accepted。注意这里只做“判决”，不在这里直接修改 base_state，
        # 否则 token_ids 和真实 KV 状态会再次脱节。
        first_token_id = draft_token_ids[0]
        accepted_count = 1
        accepted_token_ids.append(first_token_id)
        if self._sequence_finished(state.seq, first_token_id, eos_token_id):
            return accepted_count, accepted_token_ids, None

        if len(draft_token_ids) == 1:
            return accepted_count, accepted_token_ids, None

        # 2. 对完整 draft suffix 做一次 verify prefill，拿到 q1..qk
        verify_state = None
        try:
            verify_state = self._make_sequence(
                self.base_engine,
                list(state.seq.token_ids) + draft_token_ids,
                self._sampling_params_from_seq(state.seq),
            )
            verify_logits = self.base_engine.model_runner.forward_verify_logits(
                [verify_state.seq],
                "prefill",
                num_logits_to_keep=len(draft_token_ids),
            )
            # verify_logits = [q1, q2, ... , qk]
            remaining_draft_token_ids = draft_token_ids[1:]
            if remaining_draft_token_ids:
                accepted_tail_count, accepted_tail_token_ids, fallback_token_id = (
                    self.base_engine.model_runner.verify_draft_tokens(
                        remaining_draft_token_ids,
                        verify_logits[:-1],
                        include_bonus_token=False,
                    )
                )
            else:
                accepted_tail_count, accepted_tail_token_ids, fallback_token_id = 0, [], None
        finally:
            self._free_sequence(self.base_engine, verify_state)

        # 3. 汇总判决结果，真正的 KV 同步交给 _commit_base_tokens 去做
        for token_id in accepted_tail_token_ids:
            accepted_count += 1
            accepted_token_ids.append(token_id)
        return accepted_count, accepted_token_ids, fallback_token_id

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
                accepted_count, accepted_token_ids, fallback_token_id = self._verify_with_base(base_state, proposal_token_ids)
                accepted_tokens += accepted_count

                tokens_to_commit = list(accepted_token_ids)
                if fallback_token_id is not None:
                    tokens_to_commit.append(fallback_token_id)
                self._commit_base_tokens(base_state, tokens_to_commit)

                if self._sequence_finished(
                    base_state.seq,
                    base_state.seq.last_token,
                    self.base_engine.scheduler.eos,
                ):
                    break

                # draft 和 base 出现分歧，重建 draft 状态对齐到最新真实前缀
                if fallback_token_id is not None or accepted_count != len(proposal_token_ids):
                    resync_count += 1
                    self._free_sequence(self.draft_engine, draft_state)
                    draft_state = self._make_sequence(
                        self.draft_engine,
                        list(base_state.seq.token_ids),
                        sampling_params,
                    )

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
