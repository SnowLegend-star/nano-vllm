import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()  # 继承nn.Module的基础初始化
        # 步骤1：获取张量并行的核心标识（多卡场景）
        self.tp_rank = dist.get_rank()  # 当前GPU的rank（0开始）
        self.tp_size = dist.get_world_size()  # 张量并行的GPU总数

        # 步骤2：校验词汇表可均分（并行的前提）
        assert num_embeddings % self.tp_size == 0, "词汇表大小必须能被GPU数整除"

        # 步骤3：计算当前GPU负责的词汇表分片范围
        self.num_embeddings = num_embeddings  # 总词汇表大小（如151936）
        # 每个GPU负责的词汇表数量 = 总大小 / GPU数
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        # 当前GPU负责的词汇表起始索引（如2卡时，rank0:0，rank1:75968）
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        # 当前GPU负责的词汇表结束索引（如rank0:75968，rank1:151936）
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition

        # 步骤4：定义当前GPU的嵌入权重（仅存储分片，核心显存优化）
        # 普通nn.Embedding存储完整权重，这里只存分片 → 显存占用降为1/tp_size
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))

        # 步骤5：绑定自定义权重加载器（适配分片权重加载）
        # 关联到下面的weight_loader方法，加载权重时只取当前GPU的分片
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # param：当前GPU的嵌入权重参数（分片）
        # loaded_weight：从safetensors加载的完整嵌入权重（CPU上）
        param_data = param.data  # 当前参数的内存地址
        shard_size = param_data.size(0)  # 当前分片的词汇表大小（如75968）
        # 计算当前GPU权重在完整权重中的起始索引
        start_idx = self.tp_rank * shard_size
        # 截取完整权重的分片部分（narrow：维度0，起始位置，长度）
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        # 把分片权重复制到当前参数中（仅加载自己负责的部分）
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        # x：输入token ID张量，形状[batch_size]（如[100, 80000, 150000]）
        if self.tp_size > 1:  # 多卡并行场景
            # 步骤1：生成mask，筛选出当前GPU负责的token（在[start, end)范围内）
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 步骤2：将token ID转换为分片内的相对索引（如rank1的80000→80000-75968=4032）
            x = mask * (x - self.vocab_start_idx)
        
        # 步骤3：计算嵌入向量（仅处理当前GPU负责的token）
        # F.embedding：用当前分片权重计算，非负责的token会被mask置0，结果为0
        y = F.embedding(x, self.weight)

        if self.tp_size > 1:  # 多卡并行场景
            # 步骤4：用mask过滤，只保留当前GPU计算的结果（非负责的token置0）
            y = mask.unsqueeze(1) * y
            # 步骤5：多卡汇总结果（all_reduce：所有GPU的结果相加，得到完整嵌入）
            dist.all_reduce(y)
        
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias  # 核心约束：Qwen3等大模型的LMHead无偏置，避免错误配置
        super().__init__(num_embeddings, embedding_dim)  # 复用父类初始化逻辑

    def forward(self, x: torch.Tensor):
        # x：模型输出的隐藏状态，形状随阶段变化：
        # - Prefill阶段：[总token数, hidden_size]（比如两个序列，总长度100 → [100, 4096]）
        # - Decode阶段：[batch_size, hidden_size]（比如8个序列 → [8, 4096]）
        
        # 步骤1：获取全局推理上下文（包含阶段标识、序列长度等关键信息）
        context = get_context()

        # 步骤2：Prefill阶段优化——仅计算每个序列最后一个token的logits（核心！）
        if context.is_prefill:
            # context.cu_seqlens_q：累积序列长度（如[0, 50, 100]，表示2个序列，长度50/50）
            # cu_seqlens_q[1:] -1 → 每个序列最后一个token的索引（如49、99）
            last_indices = context.cu_seqlens_q[1:] - 1
            # 只取最后一个token的隐藏状态，且保证内存连续（contiguous）避免后续计算报错
            x = x[last_indices].contiguous()

        # 步骤3：计算logits分片（每个GPU只算自己负责的词汇表部分）
        # F.linear：等价于 x @ self.weight.T，self.weight是分片权重（如2卡时rank0算0-75968）
        logits = F.linear(x, self.weight)

        # 步骤4：多卡场景下收集并拼接logits（仅主卡生成完整logits）
        if self.tp_size > 1:
            # rank0（主卡）初始化列表，存储所有卡的logits分片；其他卡设为None
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            # 把所有卡的logits分片收集到rank0的all_logits中
            dist.gather(logits, all_logits, 0)
            # rank0按最后一维拼接所有分片，得到完整logits；其他卡返回None（无需参与采样）
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None

        return logits
