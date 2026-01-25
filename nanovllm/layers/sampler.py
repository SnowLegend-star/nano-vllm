import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__() # 仅继承nn.Module的基础初始化

    @torch.compile # PyTorch 2.0+ 编译优化：将Python代码编译为高效CUDA/C++代码
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
    # 步骤1：温度缩放（控制生成随机性，温度越高越随机）
        # logits.float()：转float避免精度溢出；div_：原地除法（节省显存，无中间张量）
        # temperatures.unsqueeze(dim=1)：扩维（如[bs]→[bs,1]），匹配logits形状[bs, vocab_size]
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))

        # 步骤2：logits转概率分布（softmax归一化到0-1）
        probs = torch.softmax(logits, dim=-1)  # dim=-1：词汇表维度归一化

        # 步骤3：Gumbel-Max采样（核心！高效生成token）
        # 拆解：
        # 1. torch.empty_like(probs).exponential_(1)：生成和probs同形状的指数分布张量（Gumbel分布的等价变换）
        # 2. clamp_min_(1e-10)：限制最小值为1e-10，避免除以0导致NaN（数值稳定性）
        # 3. probs.div_(...)：原地除法，等价于 log(probs) - Gumbel(0,1)（Gumbel-Max核心公式）
        # 4. argmax(dim=-1)：取词汇表维度的最大值索引，即生成的token ID
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)

        return sample_tokens  # 返回形状[bs]的token ID张量
