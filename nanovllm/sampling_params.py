from dataclasses import dataclass


@dataclass
class SamplingParams:
    # 采样参数默认值
    temperature: float = 1.0  # 采样温度（控制随机性）
    max_tokens: int = 64       # 最大生成 Token 数
    ignore_eos: bool = False  # 是否忽略结束符（eos_token）

    def __post_init__(self):
        # 断言：温度必须大于1e-10（远大于0），否则抛异常
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
