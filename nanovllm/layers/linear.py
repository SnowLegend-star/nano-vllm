import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        # tp_dim 通常指明是在哪个维度进行切分（0=按行切，1=按列切）
        # 但在这个基类中，它主要作为元数据保存
        self.tp_dim = tp_dim
        
        # 获取当前进程的 Rank（ID）和 World Size（总进程数/GPU数）
        # 这是分布式训练/推理的核心：知道“我是谁”以及“总共有多少人”
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        
        # 初始化权重参数
        # 关键点：这里的 input_size 和 output_size 通常已经是“分片后”的大小
        # 例如原模型输出是 4096，有 2 张卡，传入这里的 output_size 可能是 2048
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        
        # === 核心机制：weight_loader ===
        # 这是一个自定义的属性，并非 PyTorch 原生标准。
        # 作用：当从磁盘加载完整模型权重（如 huggingface 的 .bin 文件）时，
        # 框架会调用这个函数来决定“当前这张卡应该加载大矩阵的哪一部分”。
        self.weight.weight_loader = self.weight_loader
        
        # 偏置的处理逻辑（标准 PyTorch 写法）
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 强制子类必须实现 forward，否则报错
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        # 注意：这里直接传入了完整的 input_size 和 output_size
        # 并没有像 Row/ColumnParallel 那样除以 tp_size (world_size)
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        tp_size = dist.get_world_size() # 获取总卡数 (例如 2)
        
        # 关键逻辑：
        # 1. input_size 保持不变 (因为每一列都需要完整的输入维度做点积)
        # 2. output_size 被除以 tp_size (切分输出维度)
        # 3. 最后一个参数 0 是 tp_dim，表示在第 0 维切分权重
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        
        # 1. 获取分片大小 (shard_size)
        # self.tp_dim 是 0。假设 output 是 4096，卡数 2，这里 size 就是 2048。
        shard_size = param_data.size(self.tp_dim)
        
        # 2. 计算起始位置 (start_idx)
        # Rank 0: 0 * 2048 = 0
        # Rank 1: 1 * 2048 = 2048
        start_idx = self.tp_rank * shard_size
        
        # 3. 切片 (narrow)
        # 在 loaded_weight (完整权重) 的第 0 维，从 start_idx 开始，取 shard_size 长度
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        
        # 4. 复制数据
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    '''将多个输入相同、但权重不同的线性层合并成一个大的线性层执行
    例如合并W^Q、W^K、W^K三个大矩阵为一个矩阵'''
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int], # 注意：这里是一个列表，例如 [d_model, d_model, d_model]
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        # 将多个输出维度加起来，创建一个巨大的 Linear 层
        # 例如：Q, K, V 维度都是 128，则创建一个输出为 384 的层
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        # param: 当前 GPU 上显存里那个巨大的、已经切分过的权重参数
        # loaded_weight: 从磁盘加载进来的某一个完整层的权重（例如完整的 Q）
        # loaded_shard_id: 告诉函数我们正在加载第几个部分 (0=Q, 1=K, 2=V)

        param_data = param.data
        
        # 1. 计算在显存参数中的偏移量 (Offset)
        # 比如我们在加载 K (id=1)，我们需要跳过 Q 的空间。
        # 这里的 sizes 都要除以 tp_size，因为 param_data 已经是切分后的大小了。
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        
        # 2. 计算当前要加载这部分的长度
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        
        # 3. 锁定显存的目标区域 (Target Slot)
        # narrow(维度, 起始位置, 长度)
        # 这相当于在 param_data 中挖出了属于当前层（比如 K）的那一块内存引用
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        
        # 4. 对磁盘读进来的权重进行切分 (Source Slicing)
        # loaded_weight 是完整的 Q/K/V，我们需要按 TP 规则切开，只取当前 GPU 需要的那一段。
        # chunk 将 Tensor 切成 tp_size 份，取 [self.tp_rank] 这一份。
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        
        # 5. 复制数据
        # 将切好的源数据，填入切好的目标槽位
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,     # Query 的总头数 (比如 LLaMA-2-7b 是 32)
        total_num_kv_heads: int | None = None, # KV 的总头数 (GQA场景下可能只有 4 或 8)
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        
        # 1. 处理 GQA/MQA 逻辑
        # 如果没传 kv_heads，默认等于 num_heads (标准 Multi-Head Attention)
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        
        self.head_size = head_size
        
        # 2. 计算当前 GPU 分到的 Head 数量 (Local Heads)
        # 假设 total_q=32, total_kv=8, tp_size=2
        # num_heads = 16
        # num_kv_heads = 4
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        
        # 3. 计算总输出维度 (Global Output Size)
        # Q 的大小 + K 的大小 + V 的大小
        # 注意：这里乘了 2 * total_num_kv_heads，分别代表 K 和 V
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        
        # 4. 调用父类初始化
        # ColumnParallelLinear 会自动将 output_size 除以 tp_size 来申请显存
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        
        # === 核心逻辑：计算内存偏移量 (Offset) ===
        # 显存中的数据布局是连续的： [ Local_Q  |  Local_K  |  Local_V ]
        
        if loaded_shard_id == "q":
            # Q 放在最前面
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
            
        elif loaded_shard_id == "k":
            # K 紧接在 Q 后面
            # 长度是 Local KV Heads * Head Size
            shard_size = self.num_kv_heads * self.head_size
            # 偏移量是 Q 的长度
            shard_offset = self.num_heads * self.head_size
            
        else: # loaded_shard_id == "v"
            # V 放在最后
            shard_size = self.num_kv_heads * self.head_size
            # 偏移量是 Q 的长度 + K 的长度
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size

        # 1. 锁定当前 GPU 显存中的目标区域 (Target Slot)
        # 这里的 narrow 操作是在已经切分好的 param_data 上进行的
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        
        # 2. 切分磁盘上加载进来的完整权重 (Source Slicing)
        # 假设 loaded_weight 是完整的 Q 矩阵。
        # 我们按 TP 把它切开，只取当前 Rank 需要的那部分。
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        
        # 3. 复制
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        tp_size = dist.get_world_size()
        # 关键点 1: input_size 被除以 tp_size (切分输入维度)
        # 关键点 2: output_size 保持不变 (输出维度是完整的)
        # 关键点 3: tp_dim=1。PyTorch Linear权重形状是[out, in]，所以我们在第1维切分。
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        # self.tp_dim 是 1
        shard_size = param_data.size(self.tp_dim) # 2048
        start_idx = self.tp_rank * shard_size
        
        # 在第 1 维 (输入维度) 上进行切片
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # === 第一步：计算局部结果 (Local Gemm) ===
        # 这里的输入 x 是分片的 (Sharded Input)。
        # 这里的 bias 处理非常关键：只在 Rank 0 加 bias，其他 Rank 不加。
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        
        # === 第二步：归约 (All-Reduce) ===
        if self.tp_size > 1:
            dist.all_reduce(y) # 默认操作是 SUM
        return y
