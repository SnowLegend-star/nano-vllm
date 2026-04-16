import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        # 步骤1：获取Config类的所有字段名（比如enforce_eager、dtype、max_model_len等）
        config_fields = {field.name for field in fields(Config)}
        # 步骤2：从kwargs中筛选出「仅属于Config类的参数」，过滤无关参数
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # 步骤3：用筛选后的参数实例化Config配置类，统一管理配置
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")  # 获取spawn模式的多进程上下文
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()  # 创建spawn上下文的跨进程事件（用于进程同步）
            process = ctx.Process(target=ModelRunner, args=(config, i, event))  # 创建子进程
            process.start()  # 启动子进程（运行ModelRunner，对应rank=i的GPU）
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)  # 主进程运行rank=0的ModelRunner
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self._closed = False
        atexit.register(self.exit)

    def exit(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True

        model_runner = getattr(self, "model_runner", None)
        if model_runner is not None:
            model_runner.call("exit")
            del self.model_runner

        for p in getattr(self, "ps", []):
            if p.is_alive():
                p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, mode = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, mode)
        if mode == "recompute":
            self.scheduler.postprocess_recompute(seqs)
        else:
            self.scheduler.postprocess_decode(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        if mode == "prefill":
            num_tokens = sum(len(seq) for seq in seqs)
        elif mode == "decode":
            num_tokens = -len(seqs)
        else:
            num_tokens = 0
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)  # 令采样配置与Seq长度一直
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
