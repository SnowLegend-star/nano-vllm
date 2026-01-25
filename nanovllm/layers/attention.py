# import torch
# from torch import nn
# import triton
# import triton.language as tl

# from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
# from nanovllm.utils.context import get_context


# @triton.jit
# def store_kvcache_kernel(
#     key_ptr,
#     key_stride,
#     value_ptr,
#     value_stride,
#     k_cache_ptr,
#     v_cache_ptr,
#     slot_mapping_ptr,
#     D: tl.constexpr,
# ):
#     idx = tl.program_id(0)
#     slot = tl.load(slot_mapping_ptr + idx)
#     if slot == -1: return
#     key_offsets = idx * key_stride + tl.arange(0, D)
#     value_offsets = idx * value_stride + tl.arange(0, D)
#     key = tl.load(key_ptr + key_offsets)
#     value = tl.load(value_ptr + value_offsets)
#     cache_offsets = slot * D + tl.arange(0, D)
#     tl.store(k_cache_ptr + cache_offsets, key)
#     tl.store(v_cache_ptr + cache_offsets, value)


# def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
#     N, num_heads, head_dim = key.shape
#     D = num_heads * head_dim
#     assert key.stride(-1) == 1 and value.stride(-1) == 1
#     assert key.stride(1) == head_dim and value.stride(1) == head_dim
#     assert k_cache.stride(1) == D and v_cache.stride(1) == D
#     assert slot_mapping.numel() == N
#     store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


# class Attention(nn.Module):

#     def __init__(
#         self,
#         num_heads,
#         head_dim,
#         scale,
#         num_kv_heads,
#     ):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = head_dim
#         self.scale = scale
#         self.num_kv_heads = num_kv_heads
#         self.k_cache = self.v_cache = torch.tensor([])

#     def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
#         context = get_context()
#         k_cache, v_cache = self.k_cache, self.v_cache
#         if k_cache.numel() and v_cache.numel():
#             store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
#         if context.is_prefill:
#             if context.block_tables is not None:    # prefix cache
#                 k, v = k_cache, v_cache
#             o = flash_attn_varlen_func(q, k, v,
#                                        max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
#                                        max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
#                                        softmax_scale=self.scale, causal=True, block_table=context.block_tables)
#         else:    # decode
#             o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
#                                         cache_seqlens=context.context_lens, block_table=context.block_tables, 
#                                         softmax_scale=self.scale, causal=True)
#         return o



# ------------------------------------------------------------
# 乱码时代
# ------------------------------------------------------------
# import torch
# from torch import nn
# import torch.nn.functional as F
# import math

# from nanovllm.utils.context import get_context

# # ------------------------------------------------------------
# # Optional FlashAttention
# # ------------------------------------------------------------
# try:
#     from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
#     HAS_FLASH_ATTN = True
# except ImportError:
#     HAS_FLASH_ATTN = False

# # ------------------------------------------------------------
# # Pure PyTorch: store KV cache (No Triton)
# # ------------------------------------------------------------
# def store_kvcache(
#     key: torch.Tensor,
#     value: torch.Tensor,
#     k_cache: torch.Tensor,
#     v_cache: torch.Tensor,
#     slot_mapping: torch.Tensor,
# ):
#     """
#     纯 PyTorch 实现显存拷贝，替代 Triton Kernel。
#     """
#     N, num_heads, head_dim = key.shape
    
#     k_cache_flat = k_cache.view(-1, num_heads, head_dim)
#     v_cache_flat = v_cache.view(-1, num_heads, head_dim)
    
#     mask = slot_mapping >= 0
#     valid_slots = slot_mapping[mask].long()
    
#     if valid_slots.numel() > 0:
#         k_cache_flat[valid_slots] = key[mask].to(k_cache_flat.dtype)
#         v_cache_flat[valid_slots] = value[mask].to(v_cache_flat.dtype)

# # ------------------------------------------------------------
# # Optimized Fallback: Prefill (SDPA)
# # ------------------------------------------------------------
# def naive_attention_prefill(q, k, v, scale, num_heads, num_kv_heads, context):
#     if q.dim() == 2:
#         q_3d = q.view(q.shape[0], num_heads, -1)
#     else:
#         q_3d = q

#     if k.dim() == 2:
#         k_3d = k.view(k.shape[0], num_kv_heads, -1)
#     else:
#         k_3d = k
        
#     if v.dim() == 2:
#         v_3d = v.view(v.shape[0], num_kv_heads, -1)
#     else:
#         v_3d = v
        
#     output = torch.empty_like(q)
#     cu_seqlens = context.cu_seqlens_q.cpu().tolist()
    
#     for i in range(len(cu_seqlens) - 1):
#         start = cu_seqlens[i]
#         end = cu_seqlens[i+1]
        
#         q_seq = q_3d[start:end]
#         k_seq = k_3d[start:end]
#         v_seq = v_3d[start:end]
        
#         q_h = q_seq.transpose(0, 1).unsqueeze(0) 
#         k_h = k_seq.transpose(0, 1).unsqueeze(0) 
#         v_h = v_seq.transpose(0, 1).unsqueeze(0)
        
#         if num_kv_heads != num_heads:
#             repeat = num_heads // num_kv_heads
#             k_h = k_h.repeat_interleave(repeat, dim=1)
#             v_h = v_h.repeat_interleave(repeat, dim=1)
            
#         out_h = F.scaled_dot_product_attention(
#             q_h, k_h, v_h,
#             attn_mask=None,
#             dropout_p=0.0,
#             is_causal=True,
#             scale=scale
#         )
        
#         out_h = out_h.squeeze(0).transpose(0, 1).contiguous()
        
#         if output.dim() == 2:
#             output[start:end] = out_h.view(out_h.shape[0], -1)
#         else:
#             output[start:end] = out_h
            
#     return output

# # ------------------------------------------------------------
# # Fallback: Decode (Fixed Shape Logic)
# # ------------------------------------------------------------
# def naive_attention_decode(q, k, v, scale, num_heads, num_kv_heads, context):
#     B = q.shape[0]
    
#     # 1. 智能判断 head_dim
#     if q.dim() == 2:
#         # [Batch, Hidden]
#         head_dim = q.shape[-1] // num_heads
#         q_in = q.view(B, num_heads, 1, head_dim)
#     else:
#         # [Batch, Heads, Dim] -> 直接取 Dim
#         head_dim = q.shape[-1]
#         q_in = q.view(B, num_heads, 1, head_dim)

#     # 2. 处理 K, V
#     if k.dim() == 2:
#         k_step = k.view(B, num_kv_heads, 1, head_dim)
#     else:
#         k_step = k.view(B, num_kv_heads, 1, head_dim)
        
#     if v.dim() == 2:
#         v_step = v.view(B, num_kv_heads, 1, head_dim)
#     else:
#         v_step = v.view(B, num_kv_heads, 1, head_dim)

#     # 3. GQA 广播
#     if num_heads != num_kv_heads:
#         n_rep = num_heads // num_kv_heads
#         k_step = k_step.repeat_interleave(n_rep, dim=1)
#         v_step = v_step.repeat_interleave(n_rep, dim=1)

#     # 4. 计算
#     attn = torch.matmul(q_in, k_step.transpose(-1, -2)) * scale
#     attn = torch.softmax(attn, dim=-1)
#     out = torch.matmul(attn, v_step) 
    
#     out = out.transpose(1, 2).contiguous()
    
#     if q.dim() == 2:
#         output = out.view(B, -1)
#     else:
#         output = out.view(B, num_heads, -1)
        
#     return output

# # ------------------------------------------------------------
# # Attention Module
# # ------------------------------------------------------------
# class Attention(nn.Module):

#     def __init__(self, num_heads, head_dim, scale, num_kv_heads):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = head_dim
#         self.scale = scale
#         self.num_kv_heads = num_kv_heads
#         self.k_cache = torch.tensor([], dtype=torch.float16) 
#         self.v_cache = torch.tensor([], dtype=torch.float16)

#     def forward(self, q, k, v):
#         context = get_context()
#         k_cache, v_cache = self.k_cache, self.v_cache

#         if k_cache.numel() and v_cache.numel():
#             store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

#         if context.is_prefill:
#             if HAS_FLASH_ATTN:
#                 o = flash_attn_varlen_func(
#                     q, k, v,
#                     max_seqlen_q=context.max_seqlen_q,
#                     cu_seqlens_q=context.cu_seqlens_q,
#                     max_seqlen_k=context.max_seqlen_k,
#                     cu_seqlens_k=context.cu_seqlens_k,
#                     softmax_scale=self.scale,
#                     causal=True,
#                     block_table=context.block_tables,
#                 )
#             else:
#                 o = naive_attention_prefill(
#                     q, k, v,
#                     scale=self.scale,
#                     num_heads=self.num_heads,
#                     num_kv_heads=self.num_kv_heads,
#                     context=context
#                 )
#         else:
#             if HAS_FLASH_ATTN:
#                 o = flash_attn_with_kvcache(
#                     q.unsqueeze(1), k_cache, v_cache,
#                     cache_seqlens=context.context_lens,
#                     block_table=context.block_tables,
#                     softmax_scale=self.scale, causal=True,
#                 )
#             else:
#                 o = naive_attention_decode(
#                     q, k, v, 
#                     self.scale,
#                     self.num_heads,
#                     self.num_kv_heads,
#                     context
#                 )

#         return o


import torch
from torch import nn
import torch.nn.functional as F
import math

from nanovllm.utils.context import get_context

# ------------------------------------------------------------
# Optional FlashAttention
# ------------------------------------------------------------
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

# ------------------------------------------------------------
# Pure PyTorch: store KV cache (写入缓存)
# ------------------------------------------------------------
def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """
    纯 PyTorch 实现显存写入。
    """
    N, num_heads, head_dim = key.shape
    
    # k_cache 在这里传入时通常是 [Blocks, BlockSize, H, D]
    # 我们将其视为扁平视图 [Total_Slots, H, D] 以便索引
    k_cache_flat = k_cache.view(-1, num_heads, head_dim)
    v_cache_flat = v_cache.view(-1, num_heads, head_dim)
    
    mask = slot_mapping >= 0
    valid_slots = slot_mapping[mask].long()
    
    if valid_slots.numel() > 0:
        k_cache_flat[valid_slots] = key[mask].to(k_cache_flat.dtype)
        v_cache_flat[valid_slots] = value[mask].to(v_cache_flat.dtype)

# ------------------------------------------------------------
# Optimized Fallback: Prefill (SDPA)
# ------------------------------------------------------------
def naive_attention_prefill(q, k, v, scale, num_heads, num_kv_heads, context):
    """
    Prefill 阶段保持不变，使用 SDPA 优化显存
    """
    # 统一视图
    if q.dim() == 2:
        q_3d = q.view(q.shape[0], num_heads, -1)
    else:
        q_3d = q

    if k.dim() == 2:
        k_3d = k.view(k.shape[0], num_kv_heads, -1)
    else:
        k_3d = k
        
    if v.dim() == 2:
        v_3d = v.view(v.shape[0], num_kv_heads, -1)
    else:
        v_3d = v
        
    output = torch.empty_like(q)
    cu_seqlens = context.cu_seqlens_q.cpu().tolist()
    
    for i in range(len(cu_seqlens) - 1):
        start = cu_seqlens[i]
        end = cu_seqlens[i+1]
        
        q_seq = q_3d[start:end]
        k_seq = k_3d[start:end]
        v_seq = v_3d[start:end]
        
        q_h = q_seq.transpose(0, 1).unsqueeze(0) 
        k_h = k_seq.transpose(0, 1).unsqueeze(0) 
        v_h = v_seq.transpose(0, 1).unsqueeze(0)
        
        if num_kv_heads != num_heads:
            repeat = num_heads // num_kv_heads
            k_h = k_h.repeat_interleave(repeat, dim=1)
            v_h = v_h.repeat_interleave(repeat, dim=1)
            
        out_h = F.scaled_dot_product_attention(
            q_h, k_h, v_h,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=scale
        )
        
        out_h = out_h.squeeze(0).transpose(0, 1).contiguous()
        
        if output.dim() == 2:
            output[start:end] = out_h.view(out_h.shape[0], -1)
        else:
            output[start:end] = out_h
            
    return output

# ------------------------------------------------------------
# Correct Fallback: Decode (从缓存恢复历史 + GQA)
# ------------------------------------------------------------
def naive_attention_decode(q, k_cache, v_cache, scale, num_heads, num_kv_heads, context):
    """
    Decode 阶段：正确地从 Block Table 读取历史 KV,解决乱码问题。
    """
    B = q.shape[0]
    
    # 1. 准备 Query
    # [Batch, Hidden] -> [Batch, Heads, 1, HeadDim]
    if q.dim() == 2:
        head_dim = q.shape[-1] // num_heads
        q_in = q.view(B, num_heads, 1, head_dim)
    else:
        head_dim = q.shape[-1]
        q_in = q.view(B, num_heads, 1, head_dim)

    # 准备输出容器
    output_list = []
    
    # 2. 逐个 Sample 处理 (Batch Loop)
    # 因为每个 Sample 的历史长度不一样，Block 分布也不一样，很难并行，
    # 但在 Batch=1 时这很快。
    for i in range(B):
        # 获取当前样本的元数据
        block_table = context.block_tables[i] # [Block_Indices]
        context_len = context.context_lens[i] # 当前总长度 (包含刚生成的 token)
        
        # ------------------------------------------------
        # 核心修复：从 Cache 中重组 Key/Value History
        # ------------------------------------------------
        # k_cache 形状: [Num_Blocks, Block_Size, KV_Heads, Head_Dim]
        # 我们利用 block_table 索引直接取出相关的 blocks
        
        # 1. 取出所有相关的 Blocks
        # block_table 是一个 tensor 或 list
        if isinstance(block_table, torch.Tensor):
            block_indices = block_table.long()
        else:
            block_indices = torch.tensor(block_table, device=k_cache.device, dtype=torch.long)
            
        # 2. Gather Blocks: [Used_Blocks, Block_Size, H, D]
        # 注意：这里可能会报错如果 block_indices 为空，但 decode 阶段肯定有数据
        k_blocks = k_cache[block_indices] 
        v_blocks = v_cache[block_indices]
        
        # 3. 展平: [Used_Blocks * Block_Size, H, D]
        k_history = k_blocks.view(-1, num_kv_heads, head_dim)
        v_history = v_blocks.view(-1, num_kv_heads, head_dim)
        
        # 4. 截断: 去掉 Padding 的部分，只保留前 context_len 个 token
        # (因为 Block 是按块分配的，最后一个 Block 可能没填满)
        k_seq = k_history[:context_len].unsqueeze(0) # [1, Total_Len, H, D]
        v_seq = v_history[:context_len].unsqueeze(0)
        
        # ------------------------------------------------
        # 维度调整与 GQA
        # ------------------------------------------------
        # 调整为 [1, H, Total_Len, D]
        k_seq = k_seq.transpose(1, 2)
        v_seq = v_seq.transpose(1, 2)
        
        # GQA 广播 (如果 KV Heads 少于 Q Heads)
        if num_heads != num_kv_heads:
            n_rep = num_heads // num_kv_heads
            k_seq = k_seq.repeat_interleave(n_rep, dim=1)
            v_seq = v_seq.repeat_interleave(n_rep, dim=1)
            
        # ------------------------------------------------
        # Attention 计算
        # ------------------------------------------------
        # q_i: [1, H, 1, D]
        q_i = q_in[i].unsqueeze(0) 
        
        # attn: [1, H, 1, Total_Len]
        # 这里的 scaled_dot_product_attention 可以自动处理
        # 但我们手动做其实也一样，因为 Q 长度为 1
        
        attn = torch.matmul(q_i, k_seq.transpose(-1, -2)) * scale
        attn = torch.softmax(attn, dim=-1)
        
        # out: [1, H, 1, D]
        out_i = torch.matmul(attn, v_seq)
        
        output_list.append(out_i.squeeze(0)) # [H, 1, D]

    # 3. 拼接 Batch 结果
    # stack -> [B, H, 1, D] -> [B, 1, H, D]
    out = torch.stack(output_list).transpose(1, 2).contiguous()
    
    # 还原形状
    if q.dim() == 2:
        output = out.view(B, -1)
    else:
        output = out.view(B, num_heads, -1)
        
    return output


# ------------------------------------------------------------
# Attention Module
# ------------------------------------------------------------
class Attention(nn.Module):

    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = torch.tensor([], dtype=torch.float16) 
        self.v_cache = torch.tensor([], dtype=torch.float16)

    def forward(self, q, k, v):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # 1. 写入 KV Cache (必须做，否则 Decode 没数据读)
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        # ------------------------
        # Prefill
        # ------------------------
        if context.is_prefill:
            if HAS_FLASH_ATTN:
                o = flash_attn_varlen_func(
                    q, k, v,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale,
                    causal=True,
                    block_table=context.block_tables,
                )
            else:
                o = naive_attention_prefill(
                    q, k, v,
                    scale=self.scale,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    context=context
                )
        # ------------------------
        # Decode
        # ------------------------
        else:
            if HAS_FLASH_ATTN:
                o = flash_attn_with_kvcache(
                    q.unsqueeze(1), k_cache, v_cache,
                    cache_seqlens=context.context_lens,
                    block_table=context.block_tables,
                    softmax_scale=self.scale, causal=True,
                )
            else:
                # 使用修正后的 Decode 函数，传入 Cache
                o = naive_attention_decode(
                    q, k_cache, v_cache, # <-- 关键：传入 Cache
                    self.scale,
                    self.num_heads,
                    self.num_kv_heads,
                    context
                )

        return o