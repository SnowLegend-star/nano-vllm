from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(x, cos, sin):
    # x: [Batch, SeqLen, Heads, HeadDim]
    
    # 1. 切分 (Chunking)
    # 将向量 x 沿着最后一个维度切成两半。
    # 比如 HeadDim=128，切成 x1=[..., 64], x2=[..., 64]
    # 这相当于把向量看作是复数对 (x1 + i*x2)
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    
    # 2. 旋转 (Rotation)
    # 核心公式：
    # y1 = x1 * cos - x2 * sin
    # y2 = x1 * sin + x2 * cos
    # 这正是 2D 平面旋转矩阵的乘法公式
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin # 注意代码里写的是 x2*cos + x1*sin，加法交换律，一样的
    
    # 3. 拼接与还原
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(self, head_size, rotary_dim, max_position_embeddings, base):
        super().__init__()
        self.head_size = head_size
        # 这里的实现强制要求旋转维度等于头维度 (全旋转)
        # 有些变体只旋转头的一半 (rotary_dim = head_size / 2)，这里是全部
        assert rotary_dim == head_size
        
        # 1. 计算频率 (Theta)
        # 这里的公式是：theta_i = 10000 ^ (-2i / d)
        # base 默认通常是 10000 (LLaMA) 或 1000000 (Qwen/CodeLlama)
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        
        # 2. 生成位置索引 (m)
        # t = [0, 1, 2, ..., max_pos-1]
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        
        # 3. 外积计算所有角度 (m * theta)
        # einsum "i,j -> ij" 相当于做外积
        # 结果 freqs 形状: [MaxPos, Dim/2]
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        
        # 4. 计算 Cos 和 Sin
        # 形状: [MaxPos, Dim/2]
        cos = freqs.cos()
        sin = freqs.sin()
        
        # 5. 缓存 Cache
        # 为了配合 apply_rotary_emb 的 cat((y1, y2))
        # 这里把 cos 和 sin 也拼接起来，形状变成 [MaxPos, Dim]
        # unsqueeze_(1) 是为了增加一个维度兼容 Head 维度广播
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        
        # register_buffer 表示这不是模型参数(不更新梯度)，但它是模型状态的一部分
        # persistent=False 表示保存 checkpoint 时不保存它 (因为可以重新算出来，省空间)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(self, positions, query, key):
        # positions: [Batch, SeqLen] 或 [Total_Tokens] (当前这一批 token 的绝对位置)
        
        # 1. 查表 (Lookup)
        # 根据位置索引，从缓存里拿出对应的 cos/sin
        # 结果形状: [Total_Tokens, 1, Dim] (那个 1 是因为 init 里的 unsqueeze)
        cos_sin = self.cos_sin_cache[positions]
        
        # 2. 拆分 Cos 和 Sin
        # 因为缓存里存的是 cat(cos, sin)，这里拆开
        cos, sin = cos_sin.chunk(2, dim=-1)
        
        # 3. 应用旋转
        # 注意：RoPE 只作用于 Query 和 Key，不作用于 Value！
        # 因为 Q 和 K 需要计算注意力分数 (位置敏感)，而 V 是内容信息
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
