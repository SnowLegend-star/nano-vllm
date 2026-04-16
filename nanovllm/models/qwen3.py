import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.utils.distributed import get_tp_world_size


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = get_tp_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size    # 本卡负责的Q head数
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size  # 本卡负责的kv head数
        self.head_dim = head_dim or hidden_size // self.total_num_heads  # 单个head的维度（比如4096/32=128）
        self.q_size = self.num_heads * self.head_dim  # 单卡Q的总维度（比如8*128=1024）
        self.kv_size = self.num_kv_heads * self.head_dim  # 单卡KV的总维度（比如2*128=256）
        self.scaling = self.head_dim ** -0.5  # Attention分数缩放因子（1/√head_dim，避免分数过大）
        self.qkv_bias = qkv_bias  # QKV投影是否加偏置（Qwen3默认False，省参数/算力）

        # 将输入的隐藏态（shape [N, hidden_size]）一次性投影为 Q+K+V 拼接的张量（替代 3 个独立 Linear 层）
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # 将 Attention 输出（shape [N, num_heads*head_dim]）投影回隐藏层维度（hidden_size）
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(self, positions, hidden_states):
        # 1. 投影 (Projection)
        # 输入: [Batch, SeqLen, Hidden]
        # 输出: [Batch, SeqLen, (Num_Q + Num_K + Num_V) * HeadDim / TP_Size]
        qkv = self.qkv_proj(hidden_states)
        
        # 2. 切分 (Split)
        # 将融合的张量切分为 Q, K, V
        # 注意：这里的 q_size 和 kv_size 是根据当前 GPU 负责的头数计算的
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # 3. 重塑 (Reshape)
        # 变为 [Batch*SeqLen, Num_Heads, Head_Dim] (或者是类似的 3D/4D 形状，视 view 实现而定)
        # 代码中 view(-1, ...) 通常是为了兼容 Batch 和 SeqLen 维度合并的情况，或者是 nano-vllm 特定的 tensor 格式
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        # 4. QK-Norm (Qwen 特有逻辑)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
            
        # 5. 旋转位置编码 (RoPE)
        # 注入位置信息
        q, k = self.rotary_emb(positions, q, k)
        
        # 6. 注意力计算 (Attention Kernel)
        # 这里调用核心的 Attention 算子 (通常封装了 FlashAttention)
        # 这一步会自动处理 GQA 的广播逻辑 (repeat_kv)
        o = self.attn(q, k, v)
        
        # 7. 输出投影 (Output Projection)
        # o.flatten(1, -1) 将多头结果拼回 [Batch, SeqLen, Hidden / TP_Size]
        # o_proj 是 RowParallelLinear，它会在内部执行 All-Reduce
        output = self.o_proj(o.flatten(1, -1))
        
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        # 把gate和up矩阵拼起来，减少cuda kernel的启动次数
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        # 在内部会自动执行 All-Reduce，将所有显卡的结果加起来，输出完整的张量
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        # 1. 融合投影
        # 输入: [Batch, Seq, Hidden] (完整的)
        # 输出: [Batch, Seq, 2 * Intermediate / TP_Size] (分片的)
        # 这一步同时算出了 Gate 和 Up 的分片结果，拼在一起
        gate_up = self.gate_up_proj(x)
        
        # 2. 融合激活与乘法
        # 输入: [Batch, Seq, 2 * Intermediate / TP_Size]
        # 输出: [Batch, Seq, Intermediate / TP_Size]
        # 逻辑: SiLU(gate_up_part1) * gate_up_part2
        # 注意：这里没有通信，各卡算各卡的
        x = self.act_fn(gate_up)
        
        # 3. 下投影与归约
        # 输入: [Batch, Seq, Intermediate / TP_Size] (分片的)
        # 输出: [Batch, Seq, Hidden] (完整的，已 All-Reduce)
        x = self.down_proj(x)
        
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(self, config: Qwen3Config):
        super().__init__()
        # 1. 实例化 Attention
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads, # 支持 GQA
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True), # Qwen 默认通常无 Bias
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000), # Qwen3 长文本优化的关键
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        
        # 2. 实例化 MLP (SwiGLU)
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        
        # 3. 实例化两个 RMSNorm
        # input_layernorm: 用于 Attention 之前
        # post_attention_layernorm: 用于 MLP 之前
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        # 1. Attention 前的 Norm + Residual处理
        if residual is None:
            # 第一层：初始化 residual
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # 后续层：融合了 "加残差" 和 "做Norm"
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
            
        # 2. Attention 计算
        hidden_states = self.self_attn(positions, hidden_states)
        
        # 3. MLP 前的 Norm + Residual处理
        # 这里的 hidden_states 是 Attention 的输出
        # 这里的 residual 是 Attention 之前的输入
        # 这一步内部做了：New_Residual = Old_Residual + Attn_Output
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        
        # 4. MLP 计算
        hidden_states = self.mlp(hidden_states)
        
        # 返回 MLP 的输出和当前的 residual
        # 注意：MLP 的结果还没有加到 residual 上，这留给下一层的 input_layernorm 做
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions):
        # 1. 获取 Embedding
        # 输入: [Batch, SeqLen] (整数)
        # 输出: [Batch, SeqLen, Hidden] (向量)
        hidden_states = self.embed_tokens(input_ids)
        
        # 初始化 residual 为 None
        residual = None
        
        # 2. 穿越所有 Decoder 层
        for layer in self.layers:
            # 每一层都接收 residual，并返回更新后的 hidden_states 和 residual
            # 注意：这里的 residual 包含了直到上一层为止的所有累加和
            hidden_states, residual = layer(positions, hidden_states, residual)
            
        # === 关键点：处理最后一层的残差 ===
        # 循环结束时，最后一层的 MLP 输出赋值给了 hidden_states
        # 但这个 hidden_states 还没有加到 residual 上！
        # (因为我们在 Layer 内部使用了“延迟加法”策略)
        
        # 3. 最终归一化与求和
        # self.norm 在这里做两件事：
        # 1. Final_Add: hidden_states = hidden_states + residual
        # 2. Normalize: result = RMSNorm(hidden_states)
        hidden_states, _ = self.norm(hidden_states, residual)
        
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    # 权重合并的 “核心映射表”
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3Config):
        super().__init__()
        # 1. 实例化骨干网络 (Backbone)
        self.model = Qwen3Model(config)
        
        # 2. 实例化语言模型头 (LM Head)
        # 它的输入是 Hidden Size，输出是 Vocab Size
        # ParallelLMHead 通常意味着使用了列并行 (Column Parallel)，
        # 即每张卡只计算一部分词表的 Logits，最后再 Gather 起来。
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        
        # 3. 权重共享 (Tie Word Embeddings)
        # 很多模型 (如 GPT-2, Qwen) 的 Embedding 层和 LM Head 层共享同一份权重。
        # 这在逻辑上意味着：输入向量和输出向量是在同一个语义空间里的。
        # 在工程上意味着：节省巨大的显存 (Vocab * Hidden 是非常大的矩阵)。
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # 只运行骨干网络，返回 Hidden States
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        only_last_token: bool = True,
    ) -> torch.Tensor:
        # 单独运行 LM Head，返回 Logits
        return self.lm_head(hidden_states, only_last_token=only_last_token)
