# Speculative Decoding 方案设计

本文档按「版本演进」重新整理了 `nano-vllm` 里 speculative decoding（下文简称 SD）的实现路线，方便后续对比每一版到底解决了什么问题、还剩什么问题。

- 目标：给 `Qwen3-4B`（base）+ `Qwen3-0.6B`（draft）做一套可控、可度量、可渐进优化的 speculative decoding 实现。
- 路线：**保留单模型引擎不动 → 新增 `SpeculativeLLM` → 每个版本只往前推一步，不堆大改**。

## 1. 背景与动机

### 1.1 为什么是 external draft + base，而不是 MTP

- 现有项目结构是单模型 + 单 `LLMEngine` 的经典路线。
- 已经同时下载了 `Qwen3-4B` 和 `Qwen3-0.6B`，非常契合「大模型 + 小草稿」这种经典 SD 配置。
- MTP 需要训练新头，而 external draft 不需要，对这个项目更现实。

### 1.2 项目关键约束

当前生成主循环是单模型思路：

```51:58:nanovllm/engine/llm_engine.py
def step(self):
    seqs, mode = self.scheduler.schedule()
    token_ids = self.model_runner.call("run", seqs, mode)
    if mode == "recompute":
        self.scheduler.postprocess_recompute(seqs)
    else:
        self.scheduler.postprocess_decode(seqs, token_ids)
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
```

而且原来的 `Sequence` 只能承担「一个序列对应一个模型」的状态：

```19:35:nanovllm/engine/sequence.py
def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
    self.seq_id = next(Sequence.counter)
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
```

所以 SD 最大的结构变化不是采样逻辑，而是：**一个用户请求要同时维护 base 和 draft 两套状态**。

### 1.3 总体策略（不变，贯穿所有版本）

- 不改坏 `LLMEngine`，新增 `SpeculativeLLM` 作为上层调度
- 内部持有两个 `LLMEngine`（base / draft），各自管好自己的 KV cache
- 只要求同家族 tokenizer、同 eos、同 vocab size
- 单请求先跑通，然后再考虑 batch / scheduler 融合

## 2. 版本演进全貌

| 版本 | 关键词 | 一句话概括 | 状态 |
| --- | --- | --- | --- |
| V0 | 逐 token verify | MVP，先把闭环跑通 | 已被替代 |
| V1 | q0 + 整段 verify | verify 接口打通，路径正确化 | 已被替代 |
| V2 | 去 replay + adopt verify_state | 削掉最重的 base 额外开销 | 已被替代 |
| V3 | 增量状态复用 | verify / resync 都走增量 fork，partial block 也能复用 | 已被替代 |
| V4.0 | fused q0 + verify | 每轮 base 只做一次 prefill(k+1)，全接受自动吃 bonus | 已被替代 |
| V4.1 | 消灭 Case C dummy decode | draft 侧多 uncached token 走 prefix-cache prefill，一次算完 `[d_k, bonus]` | 已被替代 |
| V4.2 | draft CUDA Graph + graph-friendly `store_kvcache` | 重写 `store_kvcache` 去掉 host sync，顺势打通 draft bs=1 CUDA Graph | 已被替代 |
| V5.0 | fixed batch=2 同步 MVP | 新增独立 `generate_batch()`，draft propose / base verify 都真正 batched | **当前版本** |
| V5.1+ 目标 | 异步 Batch SD | 多 request 动态 batching，共享 base verify，真正把绝对吞吐拉起来 | 规划中 |

后面每一节会按「目标 / 主要改动 / 时序图（可选）/ 性能观察 / 剩余问题」五个维度讲。

## 3. V0：最小闭环（逐 token verify）

### 3.1 目标

先把双模型控制流跑通，验证：

- 单卡能同时装下 4B + 0.6B
- `SpeculativeLLM` 能正确串联两个 `LLMEngine`
- greedy 路径下语义正确

### 3.2 主要改动

- `ModelRunner`
  - 单卡场景不再强制初始化 `torch.distributed` 默认进程组
  - `run()` 拆成 `forward_hidden_state / forward_logits / sample_from_logits`
- `SpeculativeLLM`
  - 每个请求各自维护一条 `base Sequence` 和 `draft Sequence`
  - draft 连续 proposal `K` 个 token
  - base 对这些 proposal **逐 token** 判决
  - mismatch 后直接整轮重建 draft `Sequence`

### 3.3 时序图

```mermaid
sequenceDiagram
    autonumber
    participant U as User Prompt
    participant S as SpeculativeLLM
    participant D as Draft Engine
    participant B as Base Engine

    U->>S: generate(prompt)
    S->>D: 创建 draft Sequence 并 prefill prompt
    S->>B: 创建 base Sequence 并 prefill prompt

    loop 直到达到 max_tokens 或 eos
        S->>D: 连续 proposal 最多 K 个 token
        D-->>S: 返回 draft token 序列

        loop 逐 token verify
            S->>B: 基于当前真实前缀算一次 next-token logits
            B-->>S: base_token
            alt base_token == draft token
                S->>B: 接受并 append 该 token
            else mismatch
                S->>B: append fallback token
                S->>D: 释放并重建 draft Sequence
                Note over S,D: 中断本轮 verify，重新同步 draft
            end
        end
    end

    S-->>U: 输出 text, acceptance_rate
```

### 3.4 限制

- base 前向次数几乎没减少
- 每次 mismatch 都会重建 draft，代价重
- verify 接口能力没真正用上（只是在反复跑 decode）

## 4. V1：q0 + 整段 verify 打通

### 4.1 目标

第一次把 verify 真正改成「**整段 verify**」：base 不再逐 token 决策，而是一次拿到 `q1..qk`。

### 4.2 主要改动

- `Qwen3ForCausalLM.compute_logits()` 新增 `only_last_token` 参数
- `ParallelLMHead.forward()` 同步新增 `only_last_token`
- `ModelRunner` 新增：
  - `forward_verify_logits()`：保留 prefill 全部位置 logits
  - `verify_draft_tokens()`：只做 token 比较，不改外部 `Sequence`
- `SpeculativeLLM._verify_with_base()` 的新形态：
  1. 先跑一次 decode 拿 `q0`，判决 `d1`
  2. 如果 `d1` 接受，再对整段 draft suffix 做一次 verify prefill，拿 `q1..qk`
  3. 用 `q1..q{k-1}` 判决 `d2..dk`

### 4.3 时序图

```mermaid
sequenceDiagram
    autonumber
    participant S as SpeculativeLLM
    participant D as Draft Engine
    participant B as Base Engine
    participant V as Verify 临时状态

    loop 每轮生成
        S->>D: draft 连续 proposal K 个 token
        D-->>S: d1..dk
        S->>B: q0 = base.decode(当前前缀)
        alt d1 == argmax(q0)
            S->>V: 构造 verify_seq = 当前前缀 + d1..dk
            S->>B: forward_verify_logits(verify_seq)
            B-->>S: q1..qk
            S->>S: 一次性比对 d2..dk
        else d1 mismatch
            S->>B: 以 argmax(q0) 作为 fallback
        end
    end
```

### 4.4 性能观察

- 能正确跑通，`acceptance_rate` 合理
- 但 base 前向还是很多：`q0` + verify prefill 基本是双段
- 而且这时还有一个隐藏 bug：**accepted token 的 KV 没同步回 `base_state`**
  - 详见附录 A.1

### 4.5 限制

- verify 阶段 `LMHead` 会先对整段 hidden states 投影，再切片 → 容易 OOM（附录 A.2）
- full-accept 情况下 `qk` 没被复用（bonus token 收益没吃）
- 「mismatch 后整轮重建 draft」还没解决

## 5. V2：去掉最重的 base 额外开销

### 5.1 目标

V1 的正确性修好之后，先砍掉最重的那条 base 开销：**`_commit_base_tokens()` 的顺序 replay**。同时顺手把 full-accept 的 bonus token 收益接回来。

### 5.2 主要改动

- 彻底移除 `_commit_base_tokens()`
- `_verify_with_base()` 不再只返回「判决结果」，而是直接产出新的 `base_state`
  - 如果前缀被接受，就**直接接管 `verify_state`**
  - 对未接受尾部：先截断 `verify_state` 到 accepted 前缀，再 append fallback
- full-accept 时，把 verify 阶段的 `qk` 塞进新状态的 `pending_logits`
  - 下一轮 base 的第一次 `q0` 查询直接命中 pending_logits，省一次 base decode
- `forward_verify_logits()` 改成「**先切 hidden states、再算 logits**」，彻底解掉附录 A.2 的 OOM
- 禁用 `RMSNorm / SiluAndMul` 的 `@torch.compile`，规避 dynamo 重编译
- 修复 `attention.py` 在 paged prefill 分支里把当前步 k/v 误传给 FlashAttention 的问题（附录 A.4）

### 5.3 性能观察

- 从 V1 的 `~0.54x` 提到 `~0.62x`
- acceptance rate 基本稳定，说明正确性没退化

### 5.4 限制

- `verify_state` 依然是**整轮重建**：每次都 `_make_sequence(prefix + draft_suffix)`，从头 allocate 一遍 block
- mismatch 时 `draft_state` 还是整轮重建
- base 真正的 `q0 + verify prefill` 双段结构没变

## 6. V3：增量状态复用

### 6.1 目标

把「**状态重建**」这条剩下的结构性开销也砍掉：

- `base → verify_state` 走增量 fork
- `draft → resync` 也走增量 fork
- 最后未满块的 KV 不再「被迫重算」

### 6.2 主要改动

- 新增 `_fork_sequence_from_state(engine, state, target_token_ids, cached_prefix_tokens)`：
  - 共享 source `block_table` 中完全 cached 的整块（`ref_count += 1`）
  - 如果 `cached_prefix_tokens` 落在最后未满块里，就把这段 KV **拷贝**到一个新块
  - 只为真正新增的 suffix 额外分配块
- `_verify_with_base()` 从 `_make_sequence()` 切换到 `_fork_sequence_from_state()`，复用 `base_state.num_cached_tokens`
- draft resync 也不再 `_make_sequence()`，而是：
  - 计算 `common_prefix_length(draft.token_ids, base.token_ids)`
  - 从 draft_state fork 出新 draft_state，最大限度复用 draft 自己的 KV
- `_next_logits()` 在 prefill/decode 后把 `num_cached_tokens` 更新到「精确 token 数」，不再按整块粗估
- `prepare_prefill()` 支持「部分 cached block」：
  - 从 `uncached_start` 开始构造 `slot_mapping`
  - 对最后未满 cached block，只写未缓存的那段
  - 这样 verify prefill 不会重复计算前面已经命中的 token

### 6.3 时序图

```mermaid
sequenceDiagram
    autonumber
    participant S as SpeculativeLLM
    participant BM as BlockManager(base)
    participant DM as BlockManager(draft)
    participant D as Draft Engine
    participant B as Base Engine

    loop 每轮生成
        S->>D: draft 连续 proposal K 个 token
        D-->>S: d1..dk

        alt 上轮 full accept 留下的 pending_logits 命中
            S->>S: 直接取 pending_logits 作为 q0
        else
            S->>B: q0 = base.decode(当前前缀)
        end

        alt d1 mismatch
            S->>B: append argmax(q0) 作为 fallback
        else d1 accepted
            S->>BM: fork(base_state, suffix=d1..dk, share 整块 prefix)
            BM-->>S: verify_state（KV cache 大部分被共享）
            S->>B: forward_verify_logits(verify_state)
            B-->>S: q1..qk
            S->>S: 比对 d2..dk → accepted_count
            alt full accept
                S->>S: 缓存 qk 为下一轮 pending_logits
            else 部分接受后 reject
                S->>BM: 截断 verify_state 到 accepted 前缀 + fallback
            end
            S->>BM: adopt verify_state 为新 base_state
        end

        alt draft 和 base 出现分歧
            S->>DM: fork(draft_state, target=base.token_ids, 共享公共前缀)
            DM-->>S: 新 draft_state
        end
    end
```

### 6.4 性能观察

典型一次基准（`draft_length=2`、`max_tokens=128`、`temperature=1e-5`、`<think>` 风格 prompt）：

- baseline 4B：`~19.9 tok/s`
- speculative：`~13.3 tok/s`
- `accepted_tokens = 88`
- `proposed_tokens = 150`
- `acceptance_rate ≈ 0.59`
- `resync_count = 40`
- `speedup_vs_baseline ≈ 0.67x`

用这些数字反推一下：

- 一共 ~75 轮 proposal
- 全接受轮 ~35 轮，贡献 70 accepted token
- 其中 `d1 过了但 d2 没过` 的轮次 ~18 轮
- `d1 就没过` 的轮次 ~22 轮
- base 4B 的实际前向次数大约 `q0 + verify ≈ 90+ 次`，而 baseline 是 128 次

也就是说，**base 前向次数确实降了，但降得还不够狠**，这是 V3 的真实瓶颈。

### 6.5 剩余问题

- V3 削掉的主要是「状态管理的固定成本」，不是「base 计算复杂度」
- `q0` 和 verify prefill 还是两段，大量轮次都要跑两次 base
- partial cached block 会触发一次「KV 块拷贝」，比重算便宜，但不是零成本
- 当前 benchmark 的 `<think>` prompt 对 draft 非常不友好，会放大上面的问题

## 7. V4.0：fused q0 + verify（当前版本）

### 7.1 目标

V3 的瓶颈是**每轮 base 还要跑两次前向**（`decode` 拿 `q0` + `prefill(k)` 拿 `q1..qk`）。
V4.0 的目标是把它们合并成**一次** `prefill(k+1)`，并顺手吃到 `bonus token`。

### 7.2 核心洞察

`q0..qk` 是同一个 causal attention 下连续 `k+1` 个 query 位置的输出：

- query at pos `n-1` → 基于 `t_0..t_{n-1}` 预测，即 `q_0`
- query at pos `n` → 基于 `t_0..t_{n-1}, d_1` 预测，即 `q_1`
- ...
- query at pos `n+k-1` → 基于 `t_0..t_{n-1}, d_1..d_k` 预测，即 `q_k`

这些 query 完全可以一次 FlashAttention 前向拿到。区别只是 verify_state 的 query 长度从 `k` 涨到 `k+1`，多出来的那个 query 位置 `n-1` 对应的 token 是 `t_{n-1}`，它的 K/V 需要被重算并写入 verify_state 的新 block（等价于原 K/V，只是多算 1 个 token 的 attention）。

### 7.3 主要改动

- `_verify_with_base()` 重写为 fused 版本：
  - fork verify_state 时 `cached_prefix_tokens = base.num_cached_tokens - 1`
  - 一次 `forward_verify_logits(num_logits_to_keep=k+1)` 拿 `[q_0..q_k]`
  - `verify_draft_tokens(..., include_bonus_token=True)` 做一次性比对
    - 部分接受 → `fallback_token_id = argmax(q_{accepted})`
    - 全部接受 → `fallback_token_id = argmax(q_k)` 即 bonus
- `_adopt_base_verify_state()` 简化：
  - 不再接收 `bonus_logits`
  - `fallback_token_id is not None` 统一 append 到 verify_state（fallback 和 bonus 语义合并）
- `_SequenceState.pending_logits` 字段和 `_next_logits()` 里对应的分支**整条删除**（V4 下每轮都靠 fused verify 生成下一轮的 last_token，不再需要跨轮缓存 logits）
- `_generate_one()` 里的 draft 对齐改走更便宜的 `truncate + append`：
  - **Case A/B（部分接受）**：截断 draft 到 accepted 长度（顺带丢弃 `d_k` 的未算 KV），再 append fallback
  - **Case C（全接受 + bonus）**：先让 draft 做一次 decode 固化 `d_k` 的 KV（logits 丢弃），再 append bonus
  - 这样 draft 维持「末尾 1 token KV 未算」的稳态不变量

### 7.4 时序图

```mermaid
sequenceDiagram
    autonumber
    participant S as SpeculativeLLM
    participant BM as BlockManager(base)
    participant D as Draft Engine
    participant B as Base Engine

    loop 每轮生成
        S->>D: draft 连续 proposal K 个 token
        D-->>S: d_1..d_k

        S->>BM: fork verify_state(cached_prefix = n-1)
        BM-->>S: verify_state（共享整块 prefix，只重算 t_{n-1} 的 KV）
        S->>B: forward_verify_logits(verify_state, num_logits_to_keep=k+1)
        B-->>S: [q_0, q_1, ..., q_k]

        S->>S: verify_draft_tokens(include_bonus_token=True) → accepted_count, fallback_or_bonus
        S->>BM: 截断 verify_state 到 accepted 前缀 + append fallback/bonus
        S->>BM: adopt verify_state 为新 base_state

        alt 部分接受
            S->>D: truncate(draft, accepted_len) + append fallback
        else 全接受 + bonus
            S->>D: decode 一步固化 d_k 的 KV + append bonus
        end
    end
```

### 7.5 每轮 base 成本对比

| 场景 | V3 的 base 成本 | V4.0 的 base 成本 | 备注 |
|---|---|---|---|
| A: `d_1` miss | 1 × decode(1) | 1 × prefill(k+1) | 多算 k 个 token（主要的亏点） |
| B: 部分接受 | 1 × decode(1) + 1 × prefill(k) | 1 × prefill(k+1) | 少一次 launch |
| C: 全接受 | 1 × decode(1) + 1 × prefill(k) | 1 × prefill(k+1)，顺手拿 bonus | 少一次 launch + 多接 1 个 token |

每轮 base forward 永远只有一次调用，CUDA launch 次数显著减少。

### 7.6 实测结果（k=2, `<think>` 长推理 prompt, enforce_eager=True）

| 指标 | V3 | V4.0 | 变化 |
|---|---|---|---|
| baseline 4B | 19.9 tok/s | 21.5 tok/s | 基线略有波动 |
| speculative | 13.31 tok/s | **16.54 tok/s** | +24% |
| `speedup_vs_baseline` | 0.668x | **0.771x** | +0.10 |
| `accepted_tokens` / `proposed_tokens` | 88 / 150 | 71 / 115 | 轮数减少 |
| `acceptance_rate` | 0.587 | 0.617 | 基本持平 ✅ |
| `resync_count`（只记部分接受） | 40 | 28 | -30% |
| `generated_tokens` | 128 | **129** | bonus token 真的被接受了 ✅ |

从 `(accepted=71, proposed=115, resync=28, generated=129)` 反推轮次分布：

- 总轮数 R = 129 - 71 ≈ 58 轮（V3 是 75 轮，**直接少 17 轮 base prefill**）
- 按 `{X=全接受+bonus, Y=部分接受, Z=零接受}` 解方程 → X=30, Y=11, Z=17
- 30 轮全接受每轮产出 3 个 token（d_1, d_2, bonus），是 V4 相对 V3 多出来的主要增益来源

### 7.7 剩余 overhead 的来源分析

baseline 47ms/tok，V4.0 实测 60ms/tok，平均每轮 ~134ms 产 2.22 个 token。相对 baseline 的等量输出多花 ~30ms/轮：

| 来源 | 估计耗时 / 轮 |
|---|---|
| draft 2 次 decode (0.6B) | ~10ms |
| Case C 下多跑的那次 dummy decode（52% 命中率） | ~3ms |
| verify_state fork（含 partial block KV 拷贝） | ~3-5ms |
| Python 控制流 + `.tolist()` + tensor 准备 | ~10ms |
| base `prefill(k+1)` vs baseline `decode(1)` 的单次差 | ~5ms |
| 合计 | ~30ms ✅ |

没有隐藏退化，剩余的都是「小批量前向 + Python 开销 + draft cost」构成的结构性天花板。

### 7.8 横向 prompt 矩阵 & `draft_length` 敏感度分析

V4.0 落地后做过一次多维度对比实验，目标是验证两个假设：

- H1：`<think>` 长推理是最坏情况，换短答案 / 模板化 prompt 应该能显著提升 speedup
- H2：`draft_length = 3` 能在 draft 友好的段落里吃到更多 bonus，是动态调节的入口

#### 7.8.1 测试方法

- 5 种 prompt，同一份 `<|im_start|>` chat 模板：`think_long` / `short_qa` / `factual_list` / `template_code` / `casual_chat`
- `max_tokens = 128`, `temperature = 1e-5`
- 引擎加载 1 次，跑完全部 prompt（避免反复 warmup 引入偏差）
- `draft_length ∈ {2, 3}`，通过直接修改 `SpeculativeLLM.draft_length` 切换，不重载模型

#### 7.8.2 结果矩阵

| prompt | baseline tok/s | k=2 tok/s | k=2 speedup | k=2 accept | k=3 tok/s | k=3 speedup | k=3 accept |
|---|---|---|---|---|---|---|---|
| think_long     | 22.46 | 16.86 | 0.75x | 0.62 | 16.80 | 0.75x | 0.50 |
| short_qa       | 25.36 | 18.44 | 0.73x | 0.65 | 17.22 | 0.68x | 0.53 |
| factual_list   | 25.43 | 17.14 | 0.67x | 0.54 | 16.09 | 0.63x | 0.47 |
| template_code  | 25.29 | 17.97 | 0.71x | 0.67 | **18.34** | **0.73x** | 0.61 |
| casual_chat    | 25.37 | 16.95 | 0.67x | 0.54 | 16.94 | 0.67x | 0.52 |

#### 7.8.3 关键发现

**① H1 几乎不成立：prompt 类型的差异远小于预期**

5 个 prompt 的 acceptance rate 全部落在 `0.54 ~ 0.67` 这个狭窄区间。原因是 Qwen3 是经过 chat-tuning 的 think-style 模型，不管问什么都倾向于先 `<think>...` 走一段推理链。我们其实是在测「5 种前缀下 Qwen3 思考同一件事的前 128 token」，本质上都是开放式推理。

→ **想通过换 prompt 绕开 speculative 瓶颈，收益有限**。真正想大幅提升 acceptance，要从模型侧走（EAGLE / MTP 头 / 蒸馏后的 draft），已经不是工程问题。

**② H2 只在 `template_code` 上成立：k=3 不是通用增益**

k=2 → k=3 的效果：

- `template_code`：+2%（唯一正向，acceptance 0.67 → 0.61 降得少）
- `think_long` / `casual_chat`：基本持平
- `short_qa` / `factual_list`：-6% ~ -7%（acceptance 大幅下滑到 0.47-0.53）

用朴素 Bernoulli 模型反推单 token 接受概率 `p`：

- `k=2 think_long`：`(p + p²)/2 ≈ 0.62` → `p ≈ 0.72`
- `k=3` 同 prompt 理论预测 `≈ 0.54`，实测 `0.50`（略低，说明后位 draft 条件概率更差）

→ **临界点：acceptance rate ≥ 0.6 左右 k=3 才正向**，否则 k=3 纯增加开销。

**③ acceptance rate 的天花板解释**

`template_code` 的 0.67 差不多就是 `Qwen3-0.6B 预测 Qwen3-4B` 在同家族同分词器下的结构天花板。对比公开基准：

- Llama 1.1B + 70B 在 greedy 下 acceptance 通常 0.7-0.8
- Llama 7B + 70B 可以到 0.85+

我们这里 0.6B + 4B 的 0.67 属于「小 draft 合理水平」，继续推高要靠训练（EAGLE / Medusa / MTP）。

**④ baseline 本身存在 warmup 偏差**

`think_long` 是第一个 prompt，baseline = 22.46 tok/s；后面 4 个 prompt 复用 warmup 后的状态，稳定在 25.3-25.4 tok/s。这意味着：

- 以前单 prompt 跑的 baseline 数字（19.9 / 21.5）都包含 warmup，偏低
- V4.0 的真实 speedup 大约 `17 / 25 ≈ 0.68x`，比「0.77x」更接近结构真相
- 之前「越来越接近 1.0x」的感觉，一部分是 warmup 摊薄带来的错觉

#### 7.8.4 这次实验对 V4.1 优先级的影响

基于以上发现，原计划里「动态 draft_length」的收益预估要下调。

- `template_code` 之外的 4 个 prompt，k=3 都不增益或负收益
- 即使上动态策略，收益窗口只在 acceptance ≥ 0.6 的短段落里
- 全套件加权预期增益 **< 1%**
- 实现成本：需维护 running acceptance 估计器 + 调度策略

相比之下：

- Case C dummy decode 是所有 prompt 都跑的结构性成本
- Python / CPU-GPU 同步也是所有 prompt 都吃的开销

→ **动态 `draft_length` 从「中优先」降到「暂缓」**，V4.1 应该先做普适性的结构优化。

### 7.9 V4.0 的遗留问题 & 未来方向（V4.1+）

综合 7.6-7.8 的实测，按**加权预期收益**排了下面这张表，是 V4.1+ 的工作清单：

| 优先级 | 候选项 | 预期收益（全套件加权） | 复杂度 | 备注 |
|---|---|---|---|---|
| P0 | **消灭 Case C dummy decode** | +3~5% | 中 | 让 `_next_logits` 支持 multi-uncached → 走 prefix-cache prefill 一次补算 `[d_k, bonus]` 的 KV，并返回最后位置的 logits |
| P1 | **Python / CPU-GPU 同步清理** | +1~2% | 低 | `verify_draft_tokens` 里 `.tolist()` 改成 tensor 比较后只在拒绝时 `.item()`；fork 里的 Python 循环尽量 tensor 化 |
| P2 | 动态 `draft_length` | <1%（按本次套件） | 高 | 只 `template_code`-like 段落受益；只有 P0/P1 做完还有 headroom 才考虑 |
| P2 | CUDA Graph for fused verify | 未知 | 高 | 每轮 `prefill(k+1)` shape 固定但 `block_tables` 动态，实现难度大 |
| P3 | benchmark 扩展 | 无直接 speedup | 低 | 加更多 prompt 类型 / 长度分布，体系化跑分 |
| P3 | 采样路径支持（temperature > 0） | 无直接 speedup | 中 | V4.0 bonus token 目前只在 greedy 下合法；采样版需要拒绝采样 |

- P0 已于 V4.1 落地（见下一章 §8）
- P1 进行中，跑完数据再决定要不要碰 P2

## 8. V4.1：消灭 Case C dummy decode（当前版本）

### 8.1 目标

P0 里最硬的结构性冗余：**V4.0 里每发生一次「全接受 + bonus」都要额外给 draft 跑一次 decode**，只为了把 `d_k` 这个已经被 draft 自己采样过、但 KV 还没入 cache 的 token 固化下来。

- 在 acceptance ≈ 0.6 的真实分布下，Case C（全接受）占比 ~20~40%
- 这一次 dummy decode 是纯 draft 成本（前向 + kernel launch + Python 同步），完全没有 logits 被用到
- 所有 prompt 都会吃到，属于**通用型结构冗余**

### 8.2 V4.0 的 Case C 长什么样

```text
round_i 末尾:
  draft:  [prompt | d_1 .. d_{k-1} | d_k ]        ← d_k 已 append, KV 未算
          ^^^^^^^^^^^^^^^^^^^^^^^^^       cached
                                   ^^^^^  uncached (1 个)

  base 判决: 全接受 → 产生 bonus token b

  draft 侧 V4.0 的做法:
    1. decode(d_k)  ← 只是为了把 d_k 的 KV 写进 cache, logits 丢弃
    2. append_token(b)  ← b 自己的 KV 等下一轮 decode 再算
```

每发生一次 Case C，draft 就白白多一次前向 + 一次 Python→CUDA 同步。

### 8.3 V4.1 的做法

核心观察：**draft 末尾有两个连续的 uncached token 并不可怕**，只要下一轮真正需要它们的时候一次性算完就行。

改造后的 Case C：

```text
round_i 末尾 (V4.1):
  draft:  [prompt | d_1 .. d_{k-1} | d_k | b ]
          ^^^^^^^^^^^^^^^^^^^^^^^^^             cached
                                   ^^^^^^^^^    uncached (2 个)

  round_{i+1} propose 的第 1 次 _next_logits:
    - uncached == 2 → 走 prefix-cache prefill
    - 一次前向同时算完 [d_k, b] 的 KV
    - 取最后一个位置（b 的位置）的 logits 作为下一个 draft token 的预测依据
```

相比 V4.0：
- Case C 的 draft 侧从「1 次 decode + 下一轮 1 次 decode」降到「下一轮 1 次 prefill(2)」
- 1 次 prefill(2) 的 latency ≈ 1 次 decode(1)（两者都是 attention 权重一次读入），所以**净省一次前向**
- Case A/B（部分接受）走的依然是 `truncate + append` → 下一轮 decode 单 token 快路径，行为完全不变

### 8.4 代码改动

主要在 `nanovllm/speculative_llm.py`：

**(1) `_next_logits` 按 uncached 数量分路径**

```text
uncached == 0   → 理论上不应该出现, 直接 raise
uncached == 1   → decode 快路径（may_append + forward_logits("decode")）
uncached >= 2   → _grow_blocks_to_num_tokens + forward_logits("prefill")
                  （prepare_prefill 已经支持 cu_seqlens_k > cu_seqlens_q 的
                   prefix-cache prefill）
```

**(2) 新增 `_grow_blocks_to_num_tokens`**

`may_append` 只处理「一次 append 1 个 token」的增长；多 token 增长需要新写一个辅助函数：

- 一次性分配 `seq.num_blocks - len(seq.block_table)` 个新 block
- 把所有已填满的 block 补上滚动 hash，维持 `BlockManager` 的核心不变量（「除最后一块外 hash != -1」），避免后续回到 decode 路径时 `may_append` 断言炸掉

**(3) `_generate_one` 的 Case C 化简**

```python
# V4.0
_ = self._next_logits(self.draft_engine, draft_state)   # dummy decode
draft_state.seq.append_token(fallback_token_id)

# V4.1
draft_state.seq.append_token(fallback_token_id)
```

就这么一行；复杂度完全吸收在 `_next_logits` 的多 uncached 路径里。

### 8.5 正确性 checklist

- [x] Case C：draft 下一轮 propose 第一次 `_next_logits` 走 multi-uncached 分支，`[d_k, b]` 的 KV 在一次 prefill(2) 里被同时写回 `k_cache / v_cache`；后续 decode 读 paged cache 时 slot 已经就位
- [x] Case A/B：仍然走 `_truncate_sequence + append_token`，truncate 后 `num_cached_tokens ≤ num_tokens`，append 后 uncached == 1，走 decode 快路径，行为完全不变
- [x] block_table 不变量：`_grow_blocks_to_num_tokens` 补齐 rolling hash，后续 `may_append` 的 `assert last_block.hash != -1` 不会触发
- [x] prepare_prefill 的 `slot_mapping`：以 `num_cached_tokens` 为起点，对 block_table 里所有未算 token 做 paged 映射；与 `_fork_sequence_from_state` 里的 prefix-cache prefill 共用同一条已验证过的 attention 路径

### 8.6 实测结果（prompt suite × `draft_length ∈ {2, 3}`，enforce_eager=True）

和 V4.0 同一套 `run_test.py` 直接比对。

#### 8.6.1 k=2 全套件（V4.0 vs V4.1）

| prompt | V4.0 tok/s | V4.1 tok/s | V4.0 speedup | V4.1 speedup | Δ speedup | Δ tok/s |
|---|---|---|---|---|---|---|
| think_long    | 16.86 | **17.77** | 0.75x | **0.82x** | +0.07 | +5.4% |
| short_qa      | 18.44 | **19.90** | 0.73x | **0.80x** | +0.07 | +7.9% |
| factual_list  | 17.14 | **17.79** | 0.67x | **0.71x** | +0.04 | +3.8% |
| template_code | 17.97 | **19.37** | 0.71x | **0.78x** | +0.07 | +7.8% |
| casual_chat   | 16.95 | **17.92** | 0.67x | **0.76x** | +0.09 | +5.7% |

- 加权平均增益 **~+6%**，完全落在 P0 预期的「+3-5%」区间（略超）
- 所有 prompt 都是正收益，说明 Case C dummy decode 确实是**通用型结构冗余**
- `think_long`、`short_qa`、`template_code` 已经冲到 0.78-0.82x，离 break-even 还差最后一段

#### 8.6.2 用 Case C 事件数反推，savings 对得上理论

以 `think_long` k=2 为例：

- proposed=113, k=2 → 约 57 rounds
- resync=26 (partial accept) → Case C (full accept) 发生 `57 - 26 = 31` 次
- 实测 elapsed：V4.0 7.65s → V4.1 7.26s，**省 390ms**
- 摊到每次 Case C：`390 / 31 ≈ 12.6 ms/次`

一次 draft decode 的量级（`draft ≈ 80 tok/s → 1 decode ≈ 12ms`）刚好吻合，说明 V4.1 真的把 dummy decode 整个省掉了，新引入的 `prefill(2)` 相对 `decode(1)` 的额外开销在噪声级别，净收益干净。

#### 8.6.3 k=3 没拿到明显收益（符合理论）

| prompt | V4.0 k=3 | V4.1 k=3 | Δ |
|---|---|---|---|
| think_long    | 16.80 | 16.39 | -2.4% |
| short_qa      | 17.22 | 17.24 | +0.1% |
| factual_list  | 16.09 | 15.95 | -0.9% |
| template_code | 18.34 | 18.33 | ±0 |
| casual_chat   | 16.94 | 16.99 | +0.3% |

原因是 k=3 的 Case C 占比显著更低：

- k=2 `think_long`：单 token acceptance `p ≈ 0.72` → `P(Case C) = p² ≈ 0.52`
- k=3 `think_long`：同 `p` → `P(Case C) = p³ ≈ 0.38`

同时 k=3 的 rounds 总数也少 1/3，绝对 Case C 事件数大约只有 k=2 的一半（~15 次 vs ~31 次），savings 对应减半，加上 run-to-run 的 ±2% 噪声，**k=3 的 V4.1 收益被噪声吞了**。

这也侧面印证 7.8 里的结论：**k=3 本身在当前 acceptance 水平下就不是最优选择**，P0 能进一步放大 k=2 的优势、但帮不到 k=3 多少，动态 `draft_length`（P2）的潜在收益依然有限。

#### 8.6.4 baseline 本身抖动的校正

这次 baseline 整体比 V4.0 run 慢 2-5%：

- `think_long`: 22.46 → 21.55 (-4.1%)
- `casual_chat`: 25.37 → 23.52 (-7.3%)

是 run-to-run 的 GPU 状态 / warmup 抖动。因为 V4.1 和 baseline 在同一次 `run_test.py` 里顺序跑，所以 **speedup（ratio）比绝对 tok/s 更可靠**。8.6.1 里的「Δ speedup」列才是真正的 V4.1 净收益。

### 8.7 V4.1 遗留 & 下一步

- **P1 被重估为几乎没有收益**（详见 §9.3）：`.tolist()` vs `.item()` 的 sync wait 是一样的（等 GPU kernel 排空），tensor-ize `verify_draft_tokens` 基本只是代码整洁工作
- **CUDA Graph 还没启用**（当前 `enforce_eager=True`），详见 §9.4，估计 +2~4%
- 采样路径（temperature > 0）下 bonus token 的拒绝采样逻辑也还没动

## 9. 理论上限 & 剩余 headroom 评估

V4.1 之后，有必要回头算一下「在当前 `Qwen3-4B + Qwen3-0.6B` 这个组合下，speedup 的理论天花板在哪」，以便判断剩下的工程优化值不值得继续做。

### 9.1 Speculative decoding 的理论 speedup 公式

Leviathan 2023 给出的经典近似：

$$
\text{speedup} \approx \frac{1 - \alpha^{k+1}}{(1 - \alpha)(k \cdot c + 1)}
$$

- `α`：单 token 接受率（我们的 `acceptance_rate` 可以近似当作它，严格意义下是每个位置上的条件接受率）
- `k`：draft length
- `c = T_draft / T_base`：draft 单次前向延迟 / base 单次前向延迟

分子 `1 - α^(k+1)` ≈ 每轮期望推进的 token 数
分母 `(1 - α)(k c + 1)` ≈ 每轮期望耗时（以 base 单次前向为单位）

### 9.2 代入我们这套组合

实测数据：

| 量 | 值 | 备注 |
|---|---|---|
| base 4B 单 decode | ~50 ms | 从 `baseline 25 tok/s` 反推 |
| draft 0.6B 单 decode | ~12 ms | 从 V4.1 里每 Case C savings ≈ 12.6ms 反推 |
| `c = T_draft / T_base` | **~0.24** | 最关键的数字 |
| `α`（`template_code`, k=2） | ~0.67 → 单 token p ≈ 0.82 |
| `α`（`think_long`, k=2） | ~0.64 → 单 token p ≈ 0.78 |

取比较乐观的 `α = 0.72`（对应 `template_code`），`k = 2`，`c = 0.24`：

$$
\text{speedup}_{\max}
= \frac{1 - 0.72^{3}}{(1 - 0.72)(2 \times 0.24 + 1)}
= \frac{0.627}{0.414}
\approx 1.51\times
$$

**这个 base/draft 组合的理论上限大约 1.5x**。V4.1 现在是 `~0.82x`，还差大概 0.7x。

但这只是 ideal 公式——里面忽略了：

- fork / 状态管理 / Python overhead（~20% 的 per-round 时间）
- base 每轮 `prefill(k+1)` 比单纯 `decode` 更贵（有 k+1 query 位置）
- run-to-run 的噪声和 warmup 抖动

加回这些，**"可实现的上限"大概在 1.2~1.3x**。离 0.82x 还有 40%~50% 的 headroom，但这些 headroom 不是一个函数级优化能吃到的。

### 9.3 V4.1 的 per-round overhead 分解 & P1 重估

把 `think_long k=2` 的 127 ms/round 做一个粗拆（数量级估计，非精确 profiling）：

| bucket | 占比 | 能不能压 |
|---|---|---|
| base 一轮 `prefill(k+1)` compute | ~35% | 不可压（纯 4B GEMM/attention） |
| draft `k` 次 decode compute | ~25% | 不可压（纯 0.6B GEMM/attention） |
| CUDA launch / Python / sync overhead | ~20% | **可压**，是 CUDA Graph 的主要阵地 |
| verify_state fork + prepare_prefill 准备 | ~10% | 可压但 μs 级 |
| append_token / truncate / 其他 Python 逻辑 | ~10% | 可压但 marginal |

**P1「Python/sync 清理」重估：几乎没有收益**，原因：

- `.tolist()` 与 `.item()` 的 sync wait 完全一样 —— 都是等 `cudaStreamSynchronize` 让前面排队的 kernel 排空。区别只在于一次搬几个 int，延迟都是 driver 和 kernel 排空决定的
- `verify_draft_tokens` per-round 只有 **1 次** sync，不管 tensor-ize 与否都还是 1 次
- fork 里的 Python for-loop 是纯 CPU、μs 级，迭代次数 ≤ 3，没有 tensor-ize 空间
- 真正的 sync 大头是 `_greedy_from_logits` 在 propose 里每迭代 1 次 `.item()`（k=2 每轮 2 次），但要消除需要把 `Sequence.token_ids` 改成 GPU tensor 存储，**超出 P1 范围**

因此 P1 在 roadmap 里从「+1~2%」下修为 **「~0%，代码整洁任务」**，不再作为优化优先级。

### 9.4 CUDA Graph 可行性评估

#### 9.4.1 现状

- 仓库已有 `ModelRunner.capture_cudagraph()`，但只覆盖 **decode 路径**
- `run_test.py` 里 base / draft 都显式 `enforce_eager=True`，Graph **完全没启用**

#### 9.4.2 可覆盖面

每轮 per-round 前向调用分类（V4.1, k=2）：

| 前向 | 类型 | 模型 | 可 Graph? |
|---|---|---|---|
| base `_verify_with_base` | `prefill(k+1)` | 4B | ❌（prefill 路径） |
| draft propose iter 1（Case C 后一轮） | `prefix-prefill(2)` | 0.6B | ❌（prefill 路径） |
| draft propose iter 1（Case A/B 后一轮）/ iter 2 | `decode` | 0.6B | ✅ |

平均每轮可 Graph 的 draft decode 次数：

- Case C 占比 ~35%：1 次 decode 可命中
- Case A/B 占比 ~65%：2 次 decode 可命中
- 平均：`0.35×1 + 0.65×2 ≈ 1.65` 次/round

#### 9.4.3 收益估算

- draft 0.6B × 36 层，一次 decode 前向包含 **100+ 个 kernel launches**
- 单次 kernel launch overhead ~5-10μs，总计 ~0.5-1 ms 的 launch 成本能被 Graph 吞掉
- 每轮节省：`1.65 × 0.75ms ≈ 1.2 ms`
- 相对每轮 127ms：**~1% 直接来自 launch 消除**
- 叠加 Python dispatch / tensor 创建 / `prepare_decode` 里的 host→device copy overhead 也被 Graph 吃掉，**实测预期 +2~4%**

#### 9.4.4 三个风险点

1. **capture 本身可能 fail**：`capture_cudagraph()` 已经有 try/except fallback。V3 阶段试过失败（paged KV 的动态索引），V4.1 可能已经修好，**需实测**
2. **显存压力**：现有实现给 `bs ∈ {1, 2, 4, 8, 16, 32, ...}` 全都 capture 一遍，每个 graph 有独立 memory pool。draft engine 目前 `Used 23.2GB / Free 1GB`，**整套 graph pool 很可能 OOM**
3. **base 完全不受益**：最贵的 `prefill(k+1)` 在 prefill 路径，Graph 覆盖不到。要覆盖需要写一套 varlen prefill graph，且 `block_tables` 每轮 shape 变化，复杂度高

#### 9.4.5 推荐路线（按 ROI）

| 方案 | 预期收益 | 成本 | 建议 |
|---|---|---|---|
| **A. 只给 draft 开 Graph + `graph_bs=[1]`** | +2~4% | 低（`capture_cudagraph` 加一个可配 `graph_bs`；SpeculativeLLM 的 draft_kwargs 支持 `enforce_eager=False`；防 OOM） | **首选** |
| B. 给 draft 的 `prefix-prefill(2)` 也建专用 Graph | 额外 +1~2% | 中（要改 `run_hidden_states` 路径判断） | A 跑通后再考虑 |
| C. 给 base `prefill(k+1)` 建专用 Graph | +5~10% 潜力 | 高（flash_attn varlen 对 Graph 支持性未知；`block_tables` shape 动态需 padding） | 风险 + 复杂度都不低，当前性价比不高 |

### 9.5 结论：工程 vs 结构

- **当前 `4B + 0.6B` 这套组合的理论上限 ~1.5x，可实现上限 ~1.2~1.3x**
- **仓库内继续做工程 micro-opt 能拿到的极限：再 +3~5%**（CUDA Graph + 可能的 pinned-buffer 复用），最多把 speedup 从 0.82x 推到 ~0.86-0.88x
- **想冲破 1.0x 必须做结构性改动**，按 ROI 排：
  1. **换更大 base（Qwen3-7B/8B）**：`c` 从 0.24 → ~0.10，上限从 1.5x → **~2.2x**；是最直接的收益来源
  2. **batch SD**：多 request 共享一次 base verify 前向；production 级 SD 的真正杀手锏，但需要和 Scheduler 融合
  3. **换 draft（EAGLE / Medusa / 蒸馏）**：`α` 从 0.72 → 0.85+；已经不是纯工程问题，需要训练

因此 nano-vllm 单请求 speculative decoding 的工程优化线，在 CUDA Graph 落地之后就基本告一段落了。

## 10. 长期路线：从单 seq 到 batch SD

§9 得出了一个有点尴尬的结论——**仓库内工程 micro-opt 的天花板就在 0.86x 左右**，想继续往上必须做结构性改动。这一节记录我们对「往哪走」的一次诚实权衡，并给出 V5 batch SD 的路线图。

### 10.1 单 seq SD 为什么永远「差了点意思」

一个被前九章反复忽视的事实：**bs=1 场景下 GPU 本来就没吃满**。speculative decoding 在单请求下优化的其实是「GPU 空跑的那一部分」，不是 GPU 的"算力总量"。

粗估 4090D (fp16) 下两个模型的 GPU 利用率：

| 场景 | 每 token 延迟 | GPU 利用率 | 吞吐 |
|---|---|---|---|
| base 4B decode, bs=1  | ~50 ms | **~8-15%**   | ~20 tok/s  |
| base 4B decode, bs=32 | ~80 ms | **~70-90%**  | **~400 tok/s** |
| draft 0.6B decode, bs=1  | ~12 ms | **~3-5%**   | ~80 tok/s  |
| draft 0.6B decode, bs=32 | ~18 ms | **~40-60%** | ~1800 tok/s |

bs=1 时 `c = T_draft / T_base ≈ 0.24` 看起来很小，其实是因为**两边都没吃满 GPU**，小模型反而 launch overhead 占比更高。一旦进入 batch 模式，情况会彻底翻转。

### 10.2 Batch SD 的数字优势

batch 下 `c ≈ 0.25`（基本不变，都在吃 tensor core），所以 Leviathan 公式给出的 **speedup 比例仍然 1.5x 左右**——但"基数"完全不同：

| 指标 | 单 seq SD (V4.1) | Batch SD (bs=32, 估算) |
|---|---|---|
| speedup vs baseline | 0.82x | **~1.5x** |
| 绝对吞吐 | ~18 tok/s | **~600 tok/s** |
| 相对 bs=1 baseline serving | 0.8x | **~30x** |

关键洞察：**batch SD 的"比例收益"和单 seq SD 差不多，但把基数拉高 20-30x**。这才是 production serving 真正关心的数字。之前我们跟 "1.0x" 较劲一定程度上是在错误的战场。

### 10.3 换 base / 换 draft：硬件和条件都卡

这两条路在当前条件下都不太划算：

**Qwen3-8B on 4090D（24GB）的显存预算估算：**

| 项 | 估算 |
|---|---|
| 权重 (fp16) | ~16.0 GB |
| activation / workspace | ~1.5 GB |
| 系统 + PyTorch overhead | ~1.5 GB |
| 仅 base 已占 | **~19 GB** |
| draft 0.6B 权重 | ~1.2 GB |
| 双模型权重 + 系统 | **~20.2 GB** |
| 留给双份 KV cache | **~3.5-4 GB** |

3.5 GB 的 KV cache 在 `max_model_len=1024` 下大概**只能存 100-200 个 block**，prompt 稍长或者 batch 起来就容易抖。**4090D 跑 Qwen3-8B base 在极限边缘，不推荐**。

退一步换 `Qwen3-7B`（权重 ~14 GB）是稳的，但 `c` 也只从 0.24 推到 ~0.15，**理论上限从 1.5x 提到 1.7x，不是数量级变化**。

**换 draft（EAGLE / Medusa / 蒸馏）** 能把 α 从 ~0.72 推到 0.85+，但需要训练、需要数据、需要 baseline 的 logits 对齐——**已经不是纯工程问题**，暂时不具备条件。

### 10.4 Batch SD 的工程可行性拆解

batch SD 不需要蒸馏、不需要换硬件、不需要训练，**是当前条件下唯一能让 speedup 和绝对吞吐双双拉起来的路径**。按真实改动拆一下：

| 改动 | 工作量 | 复杂度 | 备注 |
|---|---|---|---|
| `_propose_with_draft` 改成 batch | 中 | 低 | nano-vllm 的 `prepare_decode` 本来就支持 batch，只需改 `SpeculativeLLM` 的串行循环 |
| `_verify_with_base` 改成 batch | 中 | 低 | `prepare_prefill` 底层支持 `cu_seqlens_q` 变长，fused `prefill(k+1)` 可以横向拼 |
| 每个 request 的**并发状态机** | **大** | **高** | 最难的一块：每轮 batch 构成不同（有的在 propose / 有的 verify 完 / 有的 Case C 要 prefix-prefill(2)） |
| `BlockManager` 区分 base/draft 两套 block pool | 中 | 中 | 现在两个 engine 各有自己的 block_manager，batch 下需要细化 |
| 和原生 `Scheduler` 融合 | 大 | 中高 | production 级必须做；MVP 阶段可以先绕过 |

**真正的难点只有第三项**——把单请求的 while 循环翻转成多请求并发状态机。其他几项底层都是现成的。

### 10.5 长期路线图

给一个从"已完成"往"远期"排的 roadmap，每个 milestone 都有明确的完成标志。

#### V4.2（可选，单 seq 工程收官）：draft CUDA Graph

- 目标：把单 seq speedup 从 0.82x → ~0.86x
- 改动面：`capture_cudagraph()` 加 `graph_bs` 参数，`SpeculativeLLM.draft_kwargs` 支持 `enforce_eager=False`，裁剪到 `graph_bs=[1]` 防 OOM
- 价值：**作为 batch SD 的铺路**——CUDA Graph 在 batch 下也能吃到（draft batch decode shape 固定）
- 不是非做不可，可以直接跳到 V5

#### V5.0（batch MVP）：固定 batch=2

- 目标：支持固定 `batch=2` 的 speculative decoding
- 改动面：不碰原生 Scheduler，在 `SpeculativeLLM` 外面套一层 `_generate_batch`
- 实现一个**最小同步状态机**：两个 request 同步进 propose、同步进 verify，不同步就等（有 bubble，但代码简单）
- benchmark 目标：batch=2 下 speedup 和单 seq 相当（~0.8x），绝对吞吐 ~1.8x

#### V5.1（任意 batch + 异步状态机）

- 目标：支持任意 batch size，每个 request 独立推进
- `_SequenceState` 扩展为可多元；每轮 dynamic batching：把当前状态兼容的 request 打成 batch
- 比如同时有 3 个要 propose、2 个要 verify、1 个 Case C 要 prefix-prefill(2)，在合适的轮次里各自 batch 调度
- benchmark 目标：bs=8-16 下 speedup ~1.3-1.5x，绝对吞吐 ~10-20x bs=1 baseline
- 这是从 MVP 升级到「可用系统」的关键一步

#### V5.2（和原生 Scheduler 融合）

- 目标：prefill / decode / speculative 都能在同一个 Scheduler 下调度
- 改动面：`Scheduler` 扩展支持 `SpeculativeSequence` 类型；`BlockManager` 细化成 base/draft 双池
- 做到这一步，nano-vllm 就真的成了「带 SD 的 production inference engine 的开源教学版」，**学习价值远大于 speedup 本身**

### 10.6 决策

综合 §9 和 §10：

- 短期（选做）：**V4.2 draft CUDA Graph**——最后一波单 seq 工程收益，顺带为 V5 铺基建
- 中期（值得投入）：**V5.0 → V5.1 batch SD**——真正能把 speedup × 基数都做起来的路径，也是学习 production SD 的必经之路
- 不做：继续换 base / 蒸馏 draft / 上量化——不是纯工程能解决的，且硬件/条件不支持

§10 的意义不是给出确定的下一步，而是**明确"单 seq 工程线到此为止"这个共识**，后面任何继续投入都应该有意识地偏向 batch SD 这条结构性路线。

## 11. V4.2：draft CUDA Graph（当前版本，并重估 §9 的理论上限）

> **TL;DR**：§9 原本预期 CUDA Graph 只能再挤 +2~4%，V4.2 真正落地后 **k=2 从 0.78x → 1.28x、k=3 从 0.72x → 1.48x**，首次稳定突破 1.0x。
> 收益远超预期的根因是 §9 低估了 `store_kvcache` 的 **per-layer GPU→CPU 同步成本**——这条路径每个 attention layer 都会触发一次 `cudaStreamSynchronize`，在 bs=1 的 draft decode 里占比惊人。CUDA Graph 能捕获的前提是重写 `store_kvcache` 去掉 host sync，这个"前置清障"本身就带来了单独的大头收益。

### 11.1 目标

让 draft engine 的 bs=1 decode 路径走上 CUDA Graph：

- `capture_cudagraph()` 支持按需裁剪 `graph_bs` 列表（只捕获 `[1]`，避免冗余 & OOM）
- `SpeculativeLLM` 的 draft engine 支持 `enforce_eager=False`
- Graph capture 失败时 graceful fallback 到 eager（保持 V4.1 行为）

### 11.2 前置问题：`store_kvcache` 阻塞 Graph capture

第一次尝试启用 draft CUDA Graph 时，capture 直接在 capture_end 阶段报错：

```
CUDA error: operation not permitted when stream is capturing
cudaErrorStreamCaptureUnsupported
  at nanovllm/layers/attention.py: valid_slots = slot_mapping[mask].long()
```

定位根因——`nanovllm/layers/attention.py::store_kvcache` 的原始实现：

```python
mask = slot_mapping >= 0
valid_slots = slot_mapping[mask].long()    # boolean mask indexing → 隐式 GPU→CPU 同步
if valid_slots.numel() > 0:                # host-side branch
    k_cache_flat[valid_slots] = key[mask].to(k_cache_flat.dtype)
    v_cache_flat[valid_slots] = value[mask].to(v_cache_flat.dtype)
```

这里两个问题：

1. `slot_mapping[mask]` 是 boolean mask indexing，会触发 `cudaStreamSynchronize` —— 因为 PyTorch 必须知道 mask 里有多少 True 才能分配输出 tensor 的 shape
2. `valid_slots.numel() > 0` 是 host-side 分支——读 tensor 元素数量本身也是同步点

这两点都在 CUDA Graph capture 白名单之外，直接触发 `cudaErrorStreamCaptureUnsupported`。

### 11.3 核心改动

#### 11.3.1 `store_kvcache` 改写成 graph-friendly 版本

`nanovllm/layers/attention.py`：

```python
def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    _, num_heads, head_dim = key.shape
    k_cache_flat = k_cache.view(-1, num_heads, head_dim)
    v_cache_flat = v_cache.view(-1, num_heads, head_dim)

    # slot=-1 -> 映射到 slot 0（任意合法 slot 即可）
    valid = slot_mapping >= 0
    safe_slots = torch.where(
        valid, slot_mapping, torch.zeros_like(slot_mapping)
    ).long()

    # 对 slot=-1 的位置：先读 cache 里原值，再 where 回退 → 写入变 no-op
    mask = valid.view(-1, 1, 1)
    existing_k = k_cache_flat.index_select(0, safe_slots)
    existing_v = v_cache_flat.index_select(0, safe_slots)
    new_k = torch.where(mask, key.to(k_cache_flat.dtype), existing_k)
    new_v = torch.where(mask, value.to(v_cache_flat.dtype), existing_v)

    k_cache_flat.index_copy_(0, safe_slots, new_k)
    v_cache_flat.index_copy_(0, safe_slots, new_v)
```

整个 function 变成「无 data-dependent 分支、全 element-wise / fixed-shape index op」，CUDA Graph 捕获没有障碍。

正确性：
- 所有 `slot_mapping >= 0` 的位置：`new_k == key`，`index_copy_` 行为等价于原版 `k_cache_flat[slot] = key`
- 所有 `slot_mapping == -1` 的位置：`new_k == existing_k[0]`，写回 slot 0 的内容到 slot 0，对 cache 无影响
- **警告**：只有当 `slot_mapping == -1` 的位置和 `slot_mapping == 0` 的位置**不同时出现**才是严格幂等。当前 nano-vllm 的 `prepare_decode` 永远给出合法 slot（无 -1），`prepare_prefill` 也全是正值，所以这个约束天然成立；唯一会出现 -1 的地方是 Graph replay 时 `graph_vars["slot_mapping"].fill_(-1)` 填充的 `[bs:max_bs]` 区域，但 graph 内部 kernel 压根不读这块，不影响结果

#### 11.3.2 `cudagraph_max_bs` 配置项

`nanovllm/config.py`：

```python
@dataclass
class Config:
    ...
    cudagraph_max_bs: int | None = None  # None = 沿用 max_num_seqs
```

`nanovllm/engine/model_runner.py::capture_cudagraph()` 里按这个 cap 裁剪：

```python
cap = self.config.max_num_seqs
if config.cudagraph_max_bs is not None:
    cap = min(cap, config.cudagraph_max_bs)
max_bs = min(cap, 512)
...
candidate_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
self.graph_bs = [bs for bs in candidate_bs if bs <= max_bs]
```

设 `cudagraph_max_bs=1` 就只捕获一张 `bs=1` 的图。speculative decoding 的 draft engine 永远是 bs=1 decode，这是恰好匹配的配置。

另外加了一行 capture 前 `torch.cuda.empty_cache()`，避免 private pool 分配时触发新的 cudaMalloc 引发 `cudaErrorStreamCaptureInvalidated`。

#### 11.3.3 `run_test.py` 的 draft_kwargs 打开

```python
DRAFT_KWARGS = {
    "enforce_eager": False,
    "cudagraph_max_bs": 1,
    ...
    "kvcache_memory_budget": 0.5,  # 从 1 压到 0.5，给 graph capture 留 activation workspace
}
```

### 11.4 实测结果（同一 prompt suite，k ∈ {2, 3}）

| prompt         | base V4.1 | base V4.2 | k=2 V4.1 | k=2 V4.2 | **V4.2 k=2 speedup** | k=3 V4.1 | k=3 V4.2 | **V4.2 k=3 speedup** |
|---|---|---|---|---|---|---|---|---|
| think_long     | 21.55     | **24.43** | 17.77    | **32.92** | **1.35x** | 16.39 | **40.81** | **1.67x** |
| short_qa       | 24.92     | **28.32** | 19.90    | **36.13** | **1.28x** | 17.24 | **38.99** | **1.38x** |
| factual_list   | 24.93     | **28.07** | 17.79    | **34.01** | **1.21x** | 15.95 | **38.72** | **1.38x** |
| template_code  | 24.98     | **28.32** | 19.37    | **36.18** | **1.28x** | 18.33 | **42.81** | **1.51x** |
| casual_chat    | 23.52     | **28.47** | 17.92    | **34.41** | **1.21x** | 16.99 | **40.80** | **1.43x** |

> 单位：tok/s。base / speedup 都按同一版本内的 baseline tok/s 算。
> V4.2 启用后，baseline 本身 **+12% 左右**（eager 模式也少了 `store_kvcache` 的 host sync），spec k=2 平均 **1.27x**，spec k=3 平均 **1.47x**。

#### 11.4.1 最新完整归档（`run_test.py` 原始数据）

下面这张表归档的是一次**完整 prompt suite** 的最新 `run_test.py` 输出。和上面的“V4.1 vs V4.2 首次对比表”相比，这里保留了更完整的 acceptance / resync / accepted-proposed 统计，便于后续直接对照控制流是否漂移。

| prompt | base tok/s | k=2 tok/s | speedup | accept | resync | accepted/proposed | k=3 tok/s | speedup | accept | resync | accepted/proposed |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| think_long | 24.13 | 32.05 | 1.33x | 0.637 | 26 | 72/113 | 39.53 | 1.64x | 0.500 | 36 | 77/154 |
| short_qa | 27.33 | 34.91 | 1.28x | 0.673 | 24 | 74/110 | 37.27 | 1.36x | 0.534 | 32 | 79/148 |
| factual_list | 27.90 | 33.02 | 1.18x | 0.545 | 35 | 67/123 | 37.59 | 1.35x | 0.469 | 36 | 75/160 |
| template_code | 27.70 | 35.11 | 1.27x | 0.673 | 23 | 74/110 | 41.53 | 1.50x | 0.606 | 24 | 83/137 |
| casual_chat | 27.65 | 33.23 | 1.20x | 0.562 | 32 | 68/121 | 39.39 | 1.42x | 0.537 | 30 | 79/147 |

这次完整归档的几个要点：

- **k=2 平均 speedup ≈ 1.25x**
- **k=3 平均 speedup ≈ 1.45x**
- acceptance / resync / accepted-proposed 与前一轮 V4.2 归档**基本一致**
- 波动主要来自**baseline 绝对 tok/s 的运行时漂移**（warmup、allocator 状态、driver 波动等），不是 speculative 控制流变了

所以从“行为是否稳定”这个角度看，V4.2 现在已经比较稳：

1. **相对排序没变**：`k=3` 仍然系统性优于 `k=2`
2. **最差 prompt 仍然是 `factual_list`**：acceptance 最低，speedup 也最低
3. **`template_code` / `think_long` 仍然是收益最明显的两类**，说明当前 draft/base 组合更吃“模式化连续片段”或“较长连续接受链”

#### 为什么收益会这么大（拆账）

这次 V4.2 的收益不能简单理解成“开了 CUDA Graph，所以快了一点”。它其实是**两笔收益叠加**：

| 来源 | 谁受益 | 量级 |
|---|---|---|
| `store_kvcache` 去掉 per-layer host sync | baseline + base/draft eager 路径 | **~+12%** |
| draft bs=1 decode 走 CUDA Graph | speculative 路径里的 draft engine | **draft 单次 forward 约 2x+** |

先看第一笔：

- 旧版 `store_kvcache` 里的 `slot_mapping[mask]` 和 `valid_slots.numel() > 0` 都会触发 **GPU→CPU 同步**
- 这个函数是 **每个 attention layer 都会走一次**
- 所以在 bs=1 decode 下，base 4B / draft 0.6B 都在为“每层一次同步”持续付钱

这解释了为什么 **baseline 本身也变快了**。虽然 baseline 并没有打开 CUDA Graph，但它一样吃到了新版 `store_kvcache` 去同步的收益。

再看第二笔：

- draft 0.6B 在 bs=1 decode 下，本来就不是“算力打满”的场景
- 单次 forward 的真实 GPU kernel 时间并不长，但 kernel launch / Python dispatch / tensor metadata / context 切换等固定开销占比很高
- CUDA Graph 把这类固定开销整体压平，所以**越小、越短、越 bs=1 的 forward，收益反而越夸张**

也就是说，V4.2 看到的大幅提升，本质上不是“单纯 launch overhead 少了几微秒”，而是：

1. 先把 `store_kvcache` 这条**之前没被识别出来的隐藏同步路径**清掉
2. 再让 draft bs=1 decode 的一整段 op 序列走 Graph replay，把 driver / dispatcher / launch 开销整体摊平

所以它不是“一个 2~4% 的小优化突然神奇变成 60%+”，而是**两个之前被混在一起的大头 overhead 同时被打掉了**。

三个反直觉现象：

1. **baseline 也变快了**：V4.2 只给 draft engine 开了 Graph，但 baseline（只加载 base 4B、`enforce_eager=True`）的 tok/s 也从 ~24 涨到 ~28。
   - 根因：`store_kvcache` 的改写同样作用在 base 的 eager decode 路径上，每层 attention 少一次 host sync。4B 有 36 层，一轮 decode 省 36 次同步，在 bs=1 decode 下这个开销占比不低
2. **k=3 反超 k=2**：V4.0/V4.1 里 k=3 永远更慢（draft 成本 > acceptance 带来的收益），V4.2 里 k=3 全面领先。
   - 根因：Graph 把 draft 单次 forward 的开销降到极低，draft 成本下降后更长的 draft 链重新变得划算（每轮能拿更多 bonus token）
3. **实测 speedup 远超 §9.4.3 的 +2~4% 预测**：
   - 1.28x / 0.78x ≈ **+64%**；1.48x / 0.72x ≈ **+105%**
   - §9.4.3 的建模只考虑了 Graph 替代 kernel launch 的固定开销，**完全漏掉了 `store_kvcache` 每层都在做的 GPU→CPU 同步**

### 11.5 §9 理论上限的修正

§9 中对当前实现的单 seq speedup 天花板给了两个估计：

- **理论上限** `(1 - α^(k+1)) / ((1 - α)(kc + 1)) ≈ 1.5x`
- **可实现上限** `~1.2-1.3x`，工程 micro-opt 的极限是 `~0.86x`

V4.2 实测 **k=3 平均 1.47x**，已经非常接近理论上限 1.5x。这意味着：

1. 理论公式里的 `c = T_draft / T_base` 本来是"纯 GPU kernel 时间比"，§9.2 用端到端延迟估了 `c ≈ 0.24`。**但这个端到端延迟里混着大量非 kernel 的 host sync / Python dispatch overhead**，用来代入理论公式是**高估了 c**
2. V4.2 打通 Graph 后，draft 这边基本只剩纯 kernel 时间，**真实的 c 降到了接近 0.1 左右**。代入公式 `(1 - 0.72^4) / ((1 - 0.72)(3×0.1 + 1)) ≈ 1.77x`——**所以 1.47x 并不是奇迹，而是修正 c 之后公式自己给出的结果**
3. §9.5 里说"继续 micro-opt 最多到 0.86x"这个结论基于错误的 c 估计，**现在应作废**

一句话修正：**"工程线到此为止"的判断在方向上是对的（结构改动（batch SD / 换 base）才是未来），但 V4.2 在单 seq 路线上比 §9 预测的多挤出了一整倍的收益；`store_kvcache` 的 host sync 是之前所有版本都共享但没被识别的隐形 overhead。**

### 11.6 V4.2 剩余 overhead 与下一步

即便 k=3 已经冲到 1.47x，还差 1.5x 理论上限一点点，主要来自：

- `verify_draft_tokens` 里的 `.tolist()` / `.item()` 同步（§9.3 里 P1 方向的遗留，~50-200 μs/轮）
- `_fork_sequence_from_state` 的 Python 循环（逐 block 复制）
- base 端的 `prefill(k+1)` 没走 Graph（§9.4 方案 C，ROI 低）

这些优化的绝对收益量级在 +3~5%，比起 V4.2 这次的 +60~100% 明显是量级更小的"清扫"。**是否还要继续单 seq 优化，取决于是否愿意为每 3% 再投入代码**；按 §10 的结论，更值得的路是直接进 V5 batch SD。

V4.2 的额外价值是**为 V5 铺路**：draft batch decode shape 也固定（`[bs, 1, num_heads, head_dim]`），Graph 机制在 batch 下无痛扩展，`store_kvcache` 已经 graph-friendly。

### 11.7 正确性 checklist

- [x] `graph_bs = [1]`（`cudagraph_max_bs=1` 生效）
- [x] capture 前 `empty_cache()` 成功，enforce_eager 保持 False
- [x] 整个 prompt suite 的输出前 64 token 与 V4.1 eager 模式逐 token 一致（greedy 下应完全一致）
- [x] acceptance rate / accepted / proposed 数字 V4.1 → V4.2 不变（只动了执行路径，没动算法）
- [x] Graph capture 失败时 `[WARNING]` fallback 生效（测试：临时改 `store_kvcache` 加一次 `.item()` 模拟失败）
- [x] `run_test.py` 的 baseline 同样受益（eager 路径也走新版 `store_kvcache`）

## 12. V5.0：fixed batch=2 同步 MVP（当前版本）

V4.2 把单 seq speculative decoding 做到 1.4x+ 之后，下一步已经不是再抠单轮几十微秒，而是把**两个请求一起跑**，让 draft propose 和 base verify 都真正吃到 batch。V5.0 的目标不是一次做成 production 级异步 batch SD，而是先验证：**在不碰原生 Scheduler 的前提下，只靠 `SpeculativeLLM` 上层状态机，能不能把吞吐再拉一截。**

### 12.1 目标与边界

V5.0 刻意只做一个最小版本：

- 新增独立 `generate_batch()` API，不改现有单请求 `generate()`
- 只支持 **fixed batch=2**
- 两条请求按**同步轮次**推进：一起 propose，一起 verify；谁先结束就先退出，另一条允许出现 bubble
- draft propose 和 base verify 都做真正 batched forward
- 不做与 `Scheduler` 融合，不做任意 batch size，不做异步状态机

这版的价值是验证“batch SD 的主收益是不是主要来自上层控制流改造”，而不是先把整个引擎大改一遍。

### 12.2 核心改动

#### 12.2.1 `SpeculativeLLM` 新增批版 API 和同步状态机

`nanovllm/speculative_llm.py` 新增：

- `generate_batch()`
- `_generate_batch2()`
- `_propose_batch_with_draft()`
- `_verify_batch_with_base()`
- `_next_logits_batch()`

设计要点：

1. **draft propose 真正 batch**
   - 每个 proposal step 收集当前活跃请求
   - 按 `uncached` 形态拆成 `decode(uncached==1)` 和 `prefill(uncached>=2)` 两组
   - 对每组分别调一次 batched `forward_logits()`

2. **base verify 真正 batch**
   - 每个槽位先各自 `_fork_sequence_from_state()`
   - 若本轮 proposal 长度相同，则把多条 verify seq 一次性送进 batched `forward_verify_logits(..., num_logits_to_keep=k+1)`
   - 接受/拒绝逻辑仍按槽位逐条判决，继续复用已有 `_adopt_base_verify_state()` / `_truncate_sequence()`

3. **不改单请求路径**
   - 现有 `_generate_one()` 和 `generate()` 保持原样
   - V5.0 的风险被隔离在新 API 内

#### 12.2.2 `forward_verify_logits()` 支持 batched last-K 切片

V5.0 真正必须补的一处底层能力在 `nanovllm/engine/model_runner.py`：

- 旧版 `forward_verify_logits(..., num_logits_to_keep=...)` 只支持单条 seq，直接 `hidden_states[-K:]`
- batched prefill 下 `hidden_states` 是按 seq 顺序拼接的，必须按每条 seq 的 query 长度切出自己的最后 K 个 hidden states

V5.0 改成：

- 单 seq 时沿用旧逻辑
- 多 seq 时按 `query_len = seq.num_tokens - seq.num_cached_tokens` 切片
- 得到 `[B, K, H]` 后再做 LM Head 投影

这一步把 “base verify batched” 从概念变成了真正的单次 GPU forward。

### 12.3 时序图（同步 batch=2）

```mermaid
flowchart TD
    init[initTwoRequests] --> prefill[prefillBasePromptKV batch]
    prefill --> loop[batchLoop]
    loop --> propose[batchDraftPropose]
    propose --> verify[batchBaseVerify]
    verify --> update[perSlotAdoptOrResync]
    update --> doneCheck{allFinished}
    doneCheck -->|no| loop
    doneCheck -->|yes| finish[collectOutputs]
```

注意这里的“batch”仍是 **同步批**：

- 如果两条请求本轮 proposal 长度一致，就一起 verify
- 如果其中一条提前 EOS / `remaining` 更小，就会拆成单独 verify
- 这也是为什么 V5.0 只是 MVP，不是最终形态

### 12.4 实测结果（`run_batch_test.py`）

benchmark 配置：

- baseline：单模型 4B，**顺序 serving 两条请求**
- single speculative：沿用 V4.2 单请求路径，依次处理两条请求
- batch speculative：V5.0 `generate_batch()`，固定 batch=2
- draft 继续开 CUDA Graph，但 `cudagraph_max_bs` 从 1 提到 **2**

| pair | base tok/s | single k=2 | batch k=2 | single k=3 | batch k=3 |
|---|---:|---:|---:|---:|---:|
| think_plus_code | 25.79 | 30.87 | **58.12** | 38.70 | **65.73** |
| qa_plus_chat | 27.94 | 32.41 | **53.14** | 37.91 | **63.87** |

换成 speedup（相对 sequential baseline）：

| pair | single k=2 | batch k=2 | single k=3 | batch k=3 |
|---|---:|---:|---:|---:|
| think_plus_code | 1.20x | **2.25x** | 1.50x | **2.55x** |
| qa_plus_chat | 1.16x | **1.90x** | 1.36x | **2.29x** |

两个直接结论：

1. **V5.0 的 batch=2 MVP 已经明显比 V4.2 单请求更快**
   - `think_plus_code`：58.12 / 30.87 ≈ **1.88x**（k=2），65.73 / 38.70 ≈ **1.70x**（k=3）
   - `qa_plus_chat`：53.14 / 32.41 ≈ **1.64x**（k=2），63.87 / 37.91 ≈ **1.68x**（k=3）

2. **batch 收益已经开始接近“乘法效应”**
   - V4.2 单请求主要赢在“单轮成本变低”
   - V5.0 再往前一步，把两条请求的 draft propose / base verify 合在一起
   - 所以 speedup 已经从单请求的 `~1.2-1.5x` 抬到 batch=2 的 `~1.9-2.6x`

### 12.5 当前限制（为什么它还只是 MVP）

V5.0 明显有收益，但还远没到最终形态：

- 只支持固定 **batch=2**
- 是**同步批**，不是异步状态机；一条先结束，另一条会有 bubble
- 只有当 proposal 长度一致时，base verify 才能真正合成一个 batch；否则会拆成多个 verify group
- 还没和原生 `Scheduler` 融合，所以它本质上还是 `SpeculativeLLM` 外挂出来的一条并行生成路径

因此 V5.0 的定位应该是：**throughput proof-of-concept**，证明 batch SD 的主要收益方向是成立的；而不是已经完成了 production 级的 request 调度。

### 12.6 对后续路线的影响

V5.0 之后，路线比之前更清楚了：

- **V4.2** 证明了单 seq 的 kernel / host sync 优化能把单请求拉到 1.4x+
- **V5.0** 证明了哪怕只做 fixed batch=2，同步批也能把总吞吐再抬到 2x+ 量级

这说明 §10 的大方向是对的：真正值得继续投的不是再抠单请求那几个微优化，而是：

1. **V5.1：任意 batch + 异步状态机**
2. **V5.2：和原生 Scheduler 融合**

从 ROI 来看，V5.0 已经把“batch 这条线值得做”从理论判断变成了实测结论。

## 附录 A：已解决的历史问题

### A.1 accepted draft token 没同步回 `base_state` 的 KV cache

- 现象：V1 改成整段 verify 后，输出开始出现重复符号 / 乱码，`token_ids` 看起来推进了但 next token 不合理
- 根因：accepted token 只追加回了 `base_state.seq.token_ids`，但对应 K/V 没写回 base 自己的 cache
- 修复：V2 里 `_verify_with_base()` 直接接管 `verify_state`；KV 不再需要手动 replay

### A.2 `forward_verify_logits()` 的 vocab projection OOM

- 现象：在 `embed_head.py` 的 `F.linear(x, self.weight)` 处 OOM
- 根因：先算整段 hidden→logits，再切 `[-K:]`，显存大头早就花掉了
- 修复：V2 改成**先切 hidden states、再算 logits**，把 vocab projection 的计算量限制到 K 个位置

### A.3 `RMSNorm / SiluAndMul` 的 `@torch.compile` 重编译

- 现象：dynamo 反复重编译、`rank mismatch. expected 3, actual 2` 告警、甚至 autotune 自身 OOM
- 根因：speculative 路径下频繁出现不同 shape，`@torch.compile` 在 eager + 小步 decode 下表现很差
- 修复：V2 起直接去掉这两个热点函数的 `@torch.compile`，并让 `run_test.py` 走 `enforce_eager=True`

### A.4 attention paged prefill 误传当前步 k/v

- 现象：`RuntimeError: Paged KV cache block size must be divisible by 256`，只有 draft resync 频繁时才触发
- 根因：带 `block_table` 的 prefill 应该读 paged `k_cache/v_cache`，但代码把当前步的连续 `k/v` 直接传给了 FlashAttention
- 修复：V2 在 `attention.py` 里区分「prefix-cache prefill」和「首次 prefill」两条路径

### A.5 `atexit` 双重清理

- 现象：程序退出时 `AttributeError: 'LLMEngine' object has no attribute 'model_runner'`
- 根因：手动 `exit()` 之后 `atexit` 还会再跑一次
- 修复：`LLMEngine.exit()` 和 `SpeculativeLLM.exit()` 都加 `_closed` 幂等标记

## 附录 B：验证指标清单

每次改动都至少记录：

- `acceptance_rate`
- `accepted_tokens`
- `proposed_tokens`
- `resync_count`
- 端到端 `tokens/s`（baseline vs speculative）
- `speedup_vs_baseline`
- 显存峰值（结合 `[KV Alloc Debug]` 和 `nvidia-smi`）

经验上：

- acceptance rate 长期 `<30%`：这个 draft/base 组合在当前 prompt 上基本不划算
- acceptance rate 在 `60%-80%`：很值得继续往下优化
- V3 当前 `~0.59` 属于中间地带，瓶颈主要在「base 前向次数还没真正降下来」

## 附录 C：当前仍然存在的限制

- `SpeculativeLLM` 只有 fixed batch=2 的同步 batch API，还没有任意 batch / 异步 batch SD
- 还没有和原生 `Scheduler` 融合
- 还没把 draft / base 的状态抽成独立 `ModelCacheState`
- `draft_length` 只支持固定值，没做按 acceptance rate 自适应

## 附录 D：下一步计划

（此清单已被 §10 + §11 的结论更新，保留早期视角做对比。）

**已落地**

- 重写 `store_kvcache` 为 graph-friendly 版本
- draft engine 启用 CUDA Graph (`cudagraph_max_bs=1`)
- 顺带让 baseline eager 路径也少每层一次 host sync
- **结果**：k=3 从 0.72x → 1.47x 平均，首次稳定突破 1.0x
- V5.0 fixed batch=2 同步 MVP：新增 `generate_batch()`，draft propose / base verify 都真正 batched
- **结果**：batch=2 总吞吐达到 **1.9x~2.6x** sequential baseline，较 V4.2 单请求再提升 **1.6x~1.9x**

**单 seq 路线剩余可选工程（小 ROI，+3~5%）**

1. `verify_draft_tokens` tensor 化（§9.3 P1 方向），消除 `.tolist()` 同步
2. `_fork_sequence_from_state` 的 Python 循环向量化
3. base 端 `prefill(k+1)` 的 varlen graph（§9.4 方案 C）—— 风险 + 复杂度都高

综合 §10 的结论，这些 micro-opt **不建议再继续做**。

**结构档（真正有量级收益的方向）**

1. **V5.1 / V5.2 batch SD**：从 fixed batch=2 同步 MVP 继续升级到任意 batch + 异步状态机 + Scheduler 融合
2. **换更大 base（Qwen3-7B）**：在 4090D 上是可行的（§10.3），理论上限从 1.5x 推到 1.7x，配合 batch SD 是乘法关系
3. **换 draft（EAGLE / MTP / 蒸馏）**：需要训练，暂不具备条件
