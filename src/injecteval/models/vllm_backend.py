"""Offline vLLM backend (Linux or WSL2 + CUDA). Optional / experimental.

vLLM gives the strongest inference optimizations for the red-teaming hot loop:
continuous batching, PagedAttention and *automatic prefix caching* (the long,
shared attacker/agent system prompts are computed once). LoRA adapters trained
by :mod:`injecteval.training` are served with ``LoRARequest`` without merging.

Install with ``pip install -e ".[vllm]"``. Because vLLM pre-allocates GPU
memory, set ``gpu_memory_utilization`` (default 0.45) so that the target and the
attacker fit side by side on one GPU.
"""

from __future__ import annotations

from typing import List, Optional

from .backend import Backend, GenConfig, Message


class VLLMBackend(Backend):
    name = "vllm"

    def __init__(self, cfg: GenConfig):
        super().__init__(cfg)
        try:
            from vllm import LLM
        except ImportError as e:  # pragma: no cover
            raise ImportError('VLLMBackend needs: pip install -e ".[vllm]" (Linux/WSL2)') from e
        kwargs = dict(model=cfg.model, enable_prefix_caching=cfg.prefix_cache or True,
                      gpu_memory_utilization=cfg.gpu_memory_utilization, seed=cfg.seed,
                      max_model_len=4096)
        if cfg.quantization == "4bit":
            kwargs["quantization"] = "bitsandbytes"
        if cfg.adapter:
            kwargs.update(enable_lora=True, max_lora_rank=64)
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self._lora = None
        if cfg.adapter:
            from vllm.lora.request import LoRARequest
            self._lora = LoRARequest("attacker", 1, cfg.adapter)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _params(self, cfg: GenConfig, n: int = 1):
        from vllm import SamplingParams
        return SamplingParams(n=n, temperature=cfg.temperature, top_p=cfg.top_p,
                              max_tokens=cfg.max_new_tokens, seed=None)

    def _chat(self, batch: List[List[Message]], cfg: GenConfig, n: int = 1):
        return self.llm.chat(batch, self._params(cfg, n), lora_request=self._lora,
                             use_tqdm=False,
                             chat_template_kwargs=cfg.chat_template_kwargs or None)

    def _generate_batch(self, batch, cfg):
        return [o.outputs[0].text.strip() for o in self._chat(batch, cfg)]

    def _sample_n(self, messages, n, cfg):
        return [c.text.strip() for c in self._chat([messages], cfg, n=n)[0].outputs]

    def peak_memory_gb(self) -> Optional[float]:
        return None

    def close(self) -> None:  # pragma: no cover
        import gc
        del self.llm
        gc.collect()
