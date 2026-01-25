import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    # 步骤1：获取模型的「权重分片映射表」（适配张量并行）
    # packed_modules_mapping：模型预定义的字典，结构如 {分片权重名片段: (目标参数名, 分片ID)}
    # 示例：{"layers.0.attn.qkv.weight.shard0": ("layers.0.attn.qkv.weight", 0)}
    # 作用：把多卡分片的权重名映射到模型的完整参数名+分片ID，支撑分片拼接
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    # 步骤2：遍历权重目录下所有 .safetensors 文件（大模型权重通常分多个文件存储）
    # glob：匹配路径下所有符合后缀的文件，适配权重分片存储场景
    for file in glob(os.path.join(path, "*.safetensors")):
        # 步骤3：打开 Safetensors 文件，指定「CPU 加载」（核心！适配低显存GPU）
        # safe_open：Safetensors 库的安全打开方式，避免pickle的安全风险；
        # "pt"：以PyTorch张量格式读取；"cpu"：先加载到CPU内存，而非直接加载到GPU，降低显存峰值
        with safe_open(file, "pt", "cpu") as f:
            # 步骤4：遍历当前文件内的所有权重名（每个safetensors文件包含多个权重张量）
            for weight_name in f.keys():
                # 步骤5：处理「分片权重」（张量并行场景）
                # 遍历预定义的分片映射表，匹配当前权重是否为分片权重
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # 5.1 解析映射关系：v=目标参数名（模型中的完整参数名），shard_id=分片ID
                        v, shard_id = packed_modules_mapping[k]
                        # 5.2 替换权重名，匹配模型中的参数名（比如把分片名转成完整参数名）
                        param_name = weight_name.replace(k, v)
                        # 5.3 获取模型中对应的参数（比如 model.layers.0.attn.qkv.weight）
                        param = model.get_parameter(param_name)
                        # 5.4 获取参数的「自定义权重加载器」（每个参数可绑定专属加载逻辑）
                        weight_loader = getattr(param, "weight_loader")
                        # 5.5 调用自定义加载器：传入参数、分片张量、分片ID，完成分片拼接
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break  # 匹配到分片映射后，跳出循环，避免重复处理
                # 步骤6：处理「非分片权重」（普通权重，无张量并行）
                else:
                    # 6.1 直接获取模型中对应的参数（权重名与模型参数名一致）
                    param = model.get_parameter(weight_name)
                    # 6.2 获取自定义加载器，无则用默认加载器（default_weight_loader）
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    # 6.3 调用加载器，加载完整权重（无需分片拼接）
                    weight_loader(param, f.get_tensor(weight_name))
