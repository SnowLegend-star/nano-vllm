import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.attention import HAS_FLASH_ATTN
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        # 步骤1：基础参数初始化（配置、设备、分布式相关）
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager  # 控制是否禁用CUDA Graph
        self.world_size = config.tensor_parallel_size  # 张量并行的GPU数（多卡时>1）
        self.rank = rank  # 当前进程/GPU的唯一标识
        self.event = event  # CUDA事件，用于异步同步/计时

        if self.world_size > 1:
        # 步骤2：分布式环境初始化（多卡张量并行的基础）
            dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)  # 绑定当前进程到指定GPU

        # 步骤3：GPU推理环境配置（数据类型/设备）
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)  # 设为模型指定 dtype（如float16）
        torch.set_default_device("cuda")  # 默认设备切到GPU

        # 步骤4：模型加载与初始化
        self.model = Qwen3ForCausalLM(hf_config)  # 实例化Qwen3模型
        load_model(self.model, config.model)  # 加载模型权重
        self.sampler = Sampler()  # 初始化token采样器
        self.warmup_model()  # 预热模型（加载GPU kernel，避免首次推理卡顿）
        self.allocate_kv_cache()  # 预分配KV缓存显存（核心！避免推理时动态分配）

        # 步骤5：性能优化（CUDA Graph捕获，非eager模式下启用）
        if not self.enforce_eager:
            if not HAS_FLASH_ATTN:
                print("[WARNING] FlashAttention is unavailable, skipping CUDA Graph and falling back to eager mode.")
                self.enforce_eager = True
            else:
                # V4.2: 捕获前先 empty_cache 把碎片还给 driver，避免 capture 过程中
                # 触发新的 cudaMalloc 从而引发 `cudaErrorStreamCaptureInvalidated`。
                # 在 draft engine 这种“加载完就快贴满显存”的场景尤为重要。
                torch.cuda.empty_cache()
                try:
                    self.capture_cudagraph()  # 捕获CUDA图，提升Decode阶段吞吐量
                except RuntimeError as exc:
                    # V4.2 之前 store_kvcache 用了 boolean mask 索引，会触发
                    # cudaErrorStreamCaptureUnsupported；V4.2 已重写成 graph-friendly
                    # 版本。这里保留 graceful fallback 以防未来新路径再出现不可捕获的 op。
                    print(f"[WARNING] CUDA Graph capture failed, falling back to eager mode: {exc}")
                    self.enforce_eager = True

        # 步骤6：还原默认环境（避免影响其他逻辑）
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 步骤7：多卡共享内存配置（进程间通信）
        if self.world_size > 1:
            if rank == 0:
                # rank=0（主进程）创建共享内存，供多卡进程共享数据
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()  # 等待所有进程同步
            else:
                dist.barrier()  # 等待主进程创建共享内存
                self.shm = SharedMemory(name="nanovllm")  # 从进程连接共享内存
                self.loop()  # 从进程进入推理循环

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        if self.world_size > 1:
            dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        # 步骤1：清空GPU无用缓存，释放显存（核心：为预热腾出干净的显存环境）
        torch.cuda.empty_cache()
        # 步骤2：重置GPU峰值显存统计（核心：让后续的显存统计更准确，方便排查OOM）
        torch.cuda.reset_peak_memory_stats()

        # 步骤3：计算预热用的“虚拟序列数”（适配配置限制，避免预热负载超限）
        # max_num_batched_tokens：单批次最大token数；max_model_len：模型最大生成长度
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        # 计算逻辑：
        # - 总token数//单序列长度 → 理论最大可并行序列数；
        # - 再和配置的max_num_seqs取最小值 → 保证不超最大序列数限制；
        # 目的：预热负载接近真实场景，但不超限（避免预热时OOM）
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)

        # 步骤4：构造预热用的虚拟序列（模拟真实推理的输入格式，无需真实数据）
        # Sequence([0]*max_model_len)：每个序列用全0的token_ids填充（长度=max_model_len）；
        # 构造num_seqs个这样的序列 → 模拟批量推理的输入，触发完整的模型前向流程
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]

        # 步骤5：执行一次完整的推理（预热核心操作）
        # self.run(seqs, True)：调用推理核心方法，True标记为“预热模式”；
        # 作用：触发GPU侧所有一次性初始化操作（见下文详解）
        self.run(seqs, "prefill")

        # 步骤6：再次清空缓存（核心：释放预热过程中临时分配的无用显存，为正式推理预留空间）
        torch.cuda.empty_cache()

    # def allocate_kv_cache(self):
    #     config = self.config
    #     hf_config = config.hf_config
    #     free, total = torch.cuda.mem_get_info()
    #     used = total - free
    #     peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    #     current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    #     num_kv_heads = hf_config.num_key_value_heads // self.world_size
    #     head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    #     block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
    #     config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
    #     assert config.num_kvcache_blocks > 0
    #     self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
    #     layer_id = 0
    #     for module in self.model.modules():
    #         if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
    #             module.k_cache = self.kv_cache[0, layer_id]
    #             module.v_cache = self.kv_cache[1, layer_id]
    #             layer_id += 1
        
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        
        # --- 1. 获取显存信息 ---
        free, total = torch.cuda.mem_get_info()
        used = total - free
        
        # 获取 PyTorch 内部统计
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        
        # 计算 KV Cache 的维度
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        
        # 计算一个 Block 需要多少字节
        # 注意：这里我们假设之后创建 Cache 使用与模型相同的 dtype (float16)，所以显存计算更精准
        dtype_size = hf_config.dtype.itemsize 
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * dtype_size
        
        # --- 2. 打印调试信息 (看看显存到底去哪了) ---
        print(f"\n[KV Alloc Debug] Total: {total/1024**2:.2f}MB, Free: {free/1024**2:.2f}MB")
        print(f"[KV Alloc Debug] Used (Sys+Weight): {used/1024**2:.2f}MB")
        print(f"[KV Alloc Debug] Peak-Current (Fragmentation): {(peak-current)/1024**2:.2f}MB")
        
        # --- 3. 设置可用显存 ---
        if config.kvcache_memory_budget is not None:
            # 显式预算适合双模型场景，避免第二个模型把前一个模型占用也算成自己的负担
            available_bytes = int(config.kvcache_memory_budget * 1024 * 1024 * 1024)
        else:
            budget = total * config.gpu_memory_utilization
            available_bytes = budget - used - (peak - current)

        print(f"[KV Alloc Debug] Calculated Budget for KV: {available_bytes/1024**2:.2f}MB")

        config.num_kvcache_blocks = int(available_bytes) // block_bytes

        # --- 4. 暴力兜底逻辑 (专治 4GB 显卡) ---
        if config.num_kvcache_blocks <= 0:
            fallback_blocks = 50  # 强制分配 50 个块 (约 800 token)
            print(f"[WARNING] Auto-profiling failed (Blocks={config.num_kvcache_blocks}).")
            print(f"[WARNING] Forcing num_kvcache_blocks = {fallback_blocks} manually to bypass OOM check.")
            config.num_kvcache_blocks = fallback_blocks
        else:
            print(f"[KV Alloc Debug] Successfully allocated {config.num_kvcache_blocks} blocks.")

        assert config.num_kvcache_blocks > 0

        # --- 5. 创建 KV Cache Tensor (关键优化) ---
        # 增加 dtype=hf_config.dtype，强制使用 float16。
        # 原代码没有加 dtype，默认是 float32，显存占用会翻倍！
        self.kv_cache = torch.empty(
            2, 
            hf_config.num_hidden_layers, 
            config.num_kvcache_blocks, 
            self.block_size, 
            num_kv_heads, 
            head_dim,
            dtype=hf_config.dtype,  # <--- 必须加这个，省一半显存
            device="cuda"                 # <--- 显式指定设备
        )
        
        # 指针绑定
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, block_tables: list[list[int]]):
        # 步骤1：获取批量中block_table的最大长度
        max_len = max(len(block_table) for block_table in block_tables)
        # 步骤2：用-1填充，统一所有block_table的长度
        block_tables = [block_table + [-1] * (max_len - len(block_table)) for block_table in block_tables]
        # 步骤3：转换为GPU张量（带1650适配优化）
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 步骤4：返回标准化后的张量
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        # 1. 核心数据收集列表（CPU侧暂存，最后转CUDA张量）
        input_ids = []          # 所有序列的未缓存token ID拼接
        positions = []          # 所有未缓存token的位置编码拼接
        cu_seqlens_q = [0]      # query侧累积序列长度（未缓存token长度）
        cu_seqlens_k = [0]      # key侧累积序列长度（全部token长度）
        max_seqlen_q = 0        # 批量中最长的query序列长度（用于模型并行）
        max_seqlen_k = 0        # 批量中最长的key序列长度
        slot_mapping = []       # token到KV缓存slot的映射列表
        block_tables = None     # 前缀缓存用的块表（初始None）
        for seq in seqs:
            # 1. 获取当前序列的总token数（seq实现了__len__，返回token_ids长度）
            seqlen = len(seq)
            
            # 2. 收集未缓存的input_ids：仅取[已缓存token数 : 末尾]的部分（已缓存的在KV里，无需重复输入）
            input_ids.extend(seq[seq.num_cached_tokens:])
            
            # 3. 生成未缓存token的位置编码：从已缓存位置开始，到总长度结束
            # 比如已缓存256个token，总长度500 → 位置是256,257,...,499
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            
            # 4. 计算query/key侧的序列长度
            seqlen_q = seqlen - seq.num_cached_tokens  # query长度=未缓存token数（模型需要计算的部分）
            seqlen_k = seqlen                          # key长度=总token数（KV缓存要存储的全部）
            
            # 5. 更新累积长度数组（核心：用于批量推理的索引）
            # cu_seqlens_q[-1]是上一个累积值，加当前seqlen_q得到新的累积值
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            
            # 6. 更新批量中的最大长度（模型需要知道最长序列，分配足够的显存）
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            
            # 7. 处理KV缓存的slot映射（仅当序列已有块表时，即非首次warmup）
            if not seq.block_table:    # warmup（首次分配块，无块表）→ 跳过
                continue
            uncached_start = seq.num_cached_tokens
            start_block = uncached_start // self.block_size
            start_offset = uncached_start % self.block_size
            # 遍历序列的「未缓存块」（已缓存块的token在KV里，无需映射）
            for i in range(start_block, seq.num_blocks):
                # 计算当前块在KV缓存中的起始slot：块ID × 块大小
                start = seq.block_table[i] * self.block_size
                token_offset = start_offset if i == start_block else 0
                # 计算当前块的结束slot：完整块=start+block_size，最后一块=start+实际token数
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                start += token_offset
                if start >= end:
                    continue
                # 收集当前块的所有slot索引（每个token对应一个slot）
                slot_mapping.extend(list(range(start, end)))
        if slot_mapping:
            assert len(input_ids) == len(slot_mapping), "prefill input_ids and slot_mapping should align."
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables([seq.block_table for seq in seqs])
        # 将所有收集的列表转成CUDA张量，适配模型推理
        # 关键参数：
        # - dtype：匹配模型要求（int64/int32）；
        # - pin_memory=True：锁页内存，加速CPU→GPU传输（避免内存换页）；
        # - cuda(non_blocking=True)：非阻塞式传输，CPU可继续执行其他逻辑，提升并行性；
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_recompute(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        recompute_tables = []
        use_prefix_cache = False
        for seq in seqs:
            block_ids = seq.pending_recompute_block_ids
            assert seq.recompute_pending and block_ids
            start_block = seq.recompute_start_block
            end_block = start_block + len(block_ids)
            start_token = start_block * self.block_size
            end_token = end_block * self.block_size
            input_ids.extend(seq.token_ids[start_token:end_token])
            positions.extend(range(start_token, end_token))
            seqlen_q = end_token - start_token
            seqlen_k = end_token
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            for block_id in block_ids:
                start = block_id * self.block_size
                slot_mapping.extend(range(start, start + self.block_size))
            recompute_tables.append(seq.prefix_block_table + block_ids)
            use_prefix_cache = use_prefix_cache or bool(seq.prefix_block_table)
        if use_prefix_cache:
            block_tables = self.prepare_block_tables(recompute_tables)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []       # 存储每个序列的「最后一个 Token ID」（Decode 阶段仅输入这个 Token）
        positions = []       # 存储每个序列最后一个 Token 的「位置索引」
        slot_mapping = []    # 存储每个序列最后一个 Token 在 KV 缓存中的「绝对槽位索引」
        context_lens = []    # 存储每个序列的「总长度」
        for seq in seqs:
            assert not seq.recompute_pending
            input_ids.append(seq.last_token)  # 步骤1：取最后一个 Token ID
            positions.append(len(seq) - 1)    # 步骤2：取最后一个 Token 的位置（从0开始计数）
            context_lens.append(len(seq))     # 步骤3：取序列总长度
            # 步骤4：计算最后一个 Token 在 KV 缓存中的绝对槽位（核心！）
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        # 转换 input_ids：Token ID 用 int64（范围大），pin_memory+non_blocking 加速传输
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        # 转换 positions：位置索引用 int64，同优化
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        # 转换 slot_mapping：槽位索引用 int32（足够存储），省显存；同优化
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 转换 context_lens：序列长度用 int32，省显存；同优化
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables([seq.block_table for seq in seqs])
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode() # 禁用所有梯度计算、张量设为只读
    def run_hidden_states(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        '''判断是否需要启动cudaGraph'''
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # Prefill 阶段序列长度 / 批次变化大，编译图复用率低，Eager 模式更灵活
            return self.model(input_ids, positions)
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return graph_vars["outputs"][:bs]

    @torch.inference_mode()
    def run_recompute(self, seqs: list[Sequence]):
        input_ids, positions = self.prepare_recompute(seqs)
        self.run_hidden_states(input_ids, positions, True)
        reset_context()
        return None


    @torch.inference_mode()
    def forward_hidden_state(self, seqs: list[Sequence], mode: str) -> torch.Tensor:
        '''进行hidden_state的前向计算'''
        is_prefill = mode == "prefill"
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs) 
        hidden_states = self.run_hidden_states(input_ids, positions, is_prefill)
        return hidden_states
    
    @torch.inference_mode()
    def forward_logits(self,seqs: list[Sequence],mode:str) -> torch.Tensor:
        '''进行logits的前向计算, 注意清理context'''
        try:
            hidden_states = self.forward_hidden_state(seqs, mode)
            logits = self.model.compute_logits(hidden_states)
            return logits
        finally:
            reset_context()

    @torch.inference_mode()
    def forward_verify_logits(
        self,
        seqs: list[Sequence],
        mode: str,
        num_logits_to_keep: int | None = None,
    ) -> torch.Tensor:
        """
        verify 路径保留 prefill 的全部位置 logits。
        当前主要用于单请求 speculative decoding，调用方自行决定如何切片和比对。
        """
        try:
            hidden_states = self.forward_hidden_state(seqs, mode)
            if num_logits_to_keep is not None:
                # 关键优化：
                # 不再先把整段 hidden_states 全部投影到 vocab 再切片，
                # 而是先截取 verify 真正需要的最后若干个 hidden states，
                # 再做 LM Head 投影。这样能显著降低 verify 阶段的显存占用。
                #
                # V5.0 起支持 batched speculative verify：prefill 的 hidden_states
                # 是按 seq 顺序拼接的，所以这里按每条 seq 的 query 长度切片，
                # 分别取末尾 last-K，再 stack 成 [B, K, H]。
                hidden_states = self._slice_last_hidden_states(
                    seqs,
                    hidden_states,
                    num_logits_to_keep,
                )
            logits = self.model.compute_logits(hidden_states, only_last_token=False)
            return logits
        finally:
            reset_context()

    @staticmethod
    def _slice_last_hidden_states(
        seqs: list[Sequence],
        hidden_states: torch.Tensor,
        num_logits_to_keep: int,
    ) -> torch.Tensor:
        if num_logits_to_keep <= 0:
            raise ValueError("num_logits_to_keep must be positive.")

        if len(seqs) == 1:
            return hidden_states[-num_logits_to_keep:].contiguous()

        # prefill/query hidden 按 seq 顺序拼接：[seq0_q, seq1_q, ...]
        # 每条 seq 在 speculative verify 时真正需要的是自己的最后 K 个 query
        # 位置，而不是整个 batch 的最后 K 个 hidden。
        offset = 0
        sliced_hidden_states = []
        for seq in seqs:
            query_len = seq.num_tokens - seq.num_cached_tokens
            if query_len < num_logits_to_keep:
                raise RuntimeError(
                    "verify query length is smaller than num_logits_to_keep in batched forward_verify_logits."
                )
            seq_hidden_states = hidden_states[offset: offset + query_len]
            sliced_hidden_states.append(seq_hidden_states[-num_logits_to_keep:])
            offset += query_len

        return torch.stack(sliced_hidden_states, dim=0).contiguous()

    @torch.inference_mode()
    def sample_from_logits(self, logits: torch.Tensor, temperatures: torch.Tensor):
        '''从logits中采样token, 只有rank0可以拿到完整的logits'''
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank==0 else None
        return token_ids

    @torch.inference_mode()
    def verify_draft_tokens(
        self,
        draft_token_ids: list[int],
        verify_logits: torch.Tensor,
        include_bonus_token: bool = False,
    ) -> tuple[int, list[int], int | None]:
        """
        仅负责比对 draft proposal 与 base verify logits，不直接修改外部 Sequence。
        verify_logits 约定为单请求逐位置 logits，shape=[N, vocab_size]。
        """
        assert draft_token_ids
        assert verify_logits.dim() == 2
        required = len(draft_token_ids) + int(include_bonus_token)
        assert verify_logits.size(0) >= required

        base_token_ids = verify_logits.argmax(dim=-1).tolist()
        accepted_count = 0
        accepted_token_ids: list[int] = []
        fallback_token_id = None

        for idx, draft_token_id in enumerate(draft_token_ids):
            base_token_id = base_token_ids[idx]
            if draft_token_id != base_token_id:
                fallback_token_id = base_token_id
                return accepted_count, accepted_token_ids, fallback_token_id
            accepted_count += 1
            accepted_token_ids.append(draft_token_id)

        if include_bonus_token and len(base_token_ids) > len(draft_token_ids):
            fallback_token_id = base_token_ids[len(draft_token_ids)]
        return accepted_count, accepted_token_ids, fallback_token_id


    @torch.inference_mode()
    def run(self, seqs: list[Sequence], mode: str):
        if mode == "recompute":
            return self.run_recompute(seqs)
        logits = self.forward_logits(seqs, mode)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        token_ids = self.sample_from_logits(logits, temperatures)
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
    # 1. 读取配置（模型配置/最大序列数/块大小等）
        config = self.config
        hf_config = config.hf_config  # HuggingFace模型配置（比如hidden_size）
        # 确定最大batch size：取配置的最大序列数和512的较小值（避免显存溢出）。
        # V4.2 起额外支持 cudagraph_max_bs 显式封顶：在 speculative decoding 的
        # draft engine 这种只跑 bs=1 的场景里设成 1，graph pool 显存 / 捕获耗时
        # 都能降到最小，避免捕获一堆永远用不到的大 bs 图。
        cap = self.config.max_num_seqs
        if config.cudagraph_max_bs is not None:
            cap = min(cap, config.cudagraph_max_bs)
        max_bs = min(cap, 512)
        # 计算单序列最大块数：(最大模型长度 + 块大小 -1) // 块大小 → 向上取整（比如max_model_len=1024，block_size=256 → 4）
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        # 2. 预分配最大尺寸的张量（CUDA Graph要求张量尺寸固定，因此预分配到max_bs）
        # 模型输入张量：input_ids（token ID）、positions（位置编码）
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        # KV缓存相关张量：slot_mapping（token→缓存槽映射）、context_lens（上下文长度）、block_tables（块表）
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        # 模型输出张量：hidden_size是模型隐藏层维度（比如768/4096）
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        # 3. 定义要捕获的batch size列表（覆盖常用的bs，兼顾性能和显存）
        # 先小批量（1/2/4/8），再按16步长到大批量（16,32,...,max_bs），
        # 然后按 max_bs 封顶，确保 cudagraph_max_bs=1 时只捕获 [1]。
        candidate_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graph_bs = [bs for bs in candidate_bs if bs <= max_bs]
        self.graphs = {}  # 保存不同bs对应的CUDA Graph：key=bs，value=graph
        self.graph_pool = None  # CUDA Graph池（复用显存，减少内存碎片）

        # 逆序遍历batch size（从大到小）：大bs的Graph池可复用给小bs，节省显存
        for bs in reversed(self.graph_bs):
            # 1. 初始化CUDA Graph对象
            graph = torch.cuda.CUDAGraph()

            # 2. 设置推理上下文（适配当前bs的张量切片）
            # False：标记为Decode阶段（无需Prefill的上下文）
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])

            # 3. Warmup（预热）：先执行一次模型前向，让GPU加载kernel、分配资源（避免捕获到初始化操作）
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

            # 4. 捕获CUDA Graph：记录模型前向的所有CUDA操作
            # self.graph_pool：复用Graph的内存池，减少显存碎片
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # 实际捕获的操作

            # 5. 初始化Graph池（仅第一次，大bs的池可复用给小bs）
            if self.graph_pool is None:
                self.graph_pool = graph.pool()

            # 6. 保存当前bs的Graph
            self.graphs[bs] = graph

            # 7. 同步GPU：确保Graph捕获完成，避免后续操作冲突
            torch.cuda.synchronize()

            # 8. 重置上下文：避免不同bs的上下文相互干扰
            reset_context()
        # 保存所有预分配的张量，后续重放Graph时直接修改这些张量的数值，再重放
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

