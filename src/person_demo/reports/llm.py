"""Qwen/Qwen3.5-9B 本地 Transformers 适配器（spec 7.1 / 7.3）。

- 显式加载本地权重目录；不隐式下载；模型没有任意 SQL/磁盘/网络工具
- do_sample=False；通过 chat template 关闭 thinking（探针验证，不靠 /nothink 文字）
- 记录加载延迟与显存；支持任务后释放
"""
from __future__ import annotations

import gc
import logging
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)


class QwenReportModel:
    def __init__(self, model_dir: Path, device: str = "cuda:0", precision: str = "bfloat16"):
        self.model_dir = Path(model_dir)
        self.device = device
        self.precision = precision
        self.model = None
        self.tokenizer = None
        self._lock = threading.Lock()
        self.load_seconds: float | None = None
        self.peak_memory_mb: int | None = None
        self.thinking_disabled_via_template: bool | None = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

        if self.model is not None:
            return
        t0 = time.monotonic()
        if self.device.startswith("cuda"):
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(self.device)
        dtype = getattr(torch, self.precision)
        load_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": False,
            "local_files_only": True,
        }
        try:
            self.model = AutoModelForCausalLM.from_pretrained(str(self.model_dir), **load_kwargs)
        except Exception as e1:  # noqa: BLE001
            log.warning("AutoModelForCausalLM failed (%s); trying AutoModelForImageTextToText", e1)
            self.model = AutoModelForImageTextToText.from_pretrained(str(self.model_dir), **load_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        self.load_seconds = time.monotonic() - t0
        try:
            _dev = torch.device(self.device)
            self.peak_memory_mb = int(torch.cuda.max_memory_allocated(_dev) / 1024 / 1024)  # 加载峰值
            torch.cuda.reset_peak_memory_stats(_dev)  # 后续生成峰值单独记录
        except Exception:  # noqa: BLE001
            self.peak_memory_mb = None

        # 探针：chat template 是否接受 enable_thinking=False
        self.thinking_disabled_via_template = self._probe_thinking_kwarg()

    def _probe_thinking_kwarg(self) -> bool:
        if self.tokenizer is None:
            return False
        if getattr(self.tokenizer, "chat_template", None) is None:
            return False
        if "enable_thinking" not in self.tokenizer.chat_template:
            return False
        try:
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "probe"}],
                tokenize=False,
                enable_thinking=False,
            )
            return True
        except Exception:  # noqa: BLE001
            try:
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": "probe"}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                return True
            except Exception:  # noqa: BLE001
                return False

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def generate(self, messages: list[dict], max_new_tokens: int = 1536, *, max_time: float | None = None) -> tuple[str, dict]:
        """返回 (text, timing)。do_sample=False 确定性生成。"""
        import torch

        with self._lock:
            self.load()
            kw: dict = {}
            if self.thinking_disabled_via_template:
                kw["enable_thinking"] = False
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kw
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            t0 = time.time()
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    max_time=max_time,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            gen_s = time.time() - t0
            text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            if self.device.startswith("cuda"):
                peak = int(torch.cuda.max_memory_allocated(self.device) / 1024 / 1024)
                self.peak_memory_mb = max(self.peak_memory_mb or 0, peak)
            timing = {
                "generate_s": gen_s,
                "input_tokens": int(inputs["input_ids"].shape[1]),
                "output_tokens": int(out.shape[1]) - int(inputs["input_ids"].shape[1]),
            }
            return text, timing

    def count_tokens(self, messages: list[dict]) -> int:
        self.load()
        kw: dict = {}
        if self.thinking_disabled_via_template:
            kw["enable_thinking"] = False
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kw
        )
        return len(self.tokenizer(prompt)["input_ids"])

    def unload(self) -> dict:
        import torch

        loaded = self.model is not None
        self.model = None
        self.tokenizer = None
        gc.collect()
        allocated = None
        if self.device.startswith("cuda") and torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
                allocated = int(torch.cuda.memory_allocated(self.device) / 1024 / 1024)
        return {"released": loaded, "allocated_after_unload_mb": allocated}
