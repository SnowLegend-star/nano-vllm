import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数 gamma，初始化为 1
        # RMSNorm 不需要偏置项 bias (beta)，只有 weight
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile  # PyTorch 2.0 编译优化，尝试将算子融合为一个 Kernel
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype # 记录原始精度 (如 float16)
        
        # 1. 精度提升 (Upcast)
        # 归一化涉及平方和累加，在 float16 下容易溢出或精度丢失
        # 因此必须强制转为 float32 进行统计量计算
        x = x.float() 
        
        # 2. 计算均方值 (Mean Square)
        # x.pow(2): 平方
        # .mean(...): 求均值
        var = x.pow(2).mean(dim=-1, keepdim=True)
        
        # 3. 归一化 (Normalization)
        # rsqrt = 1 / sqrt(x) (倒数平方根)
        # 相当于 x = x / sqrt(var + eps)
        x.mul_(torch.rsqrt(var + self.eps))
        
        # 4. 恢复精度并缩放 (Scale)
        # 转回 float16，并乘上可学习参数 weight
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,        # 当前层的输出
        residual: torch.Tensor, # 上一层的输入（残差）
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        
        # 1. 融合加法 (Fused Add)
        # 将输入和残差相加。注意这里都转成了 float32 进行加法，保证精度。
        # x.add_(...) 是原地操作，节省显存
        x = x.float().add_(residual.float())
        
        # 2. 更新残差
        # 计算出的和，就是送往下一层的 "新残差"
        residual = x.to(orig_dtype)
        
        # 3. 标准 RMSNorm 流程
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        
        # 返回：(归一化后的结果, 更新后的残差)
        return x, residual

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        # 根据是否传入 residual，自动选择是否使用融合算子
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
