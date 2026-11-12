"""Unified model-backend interface with cost accounting.

A :class:`Backend` turns chat messages into string completions and keeps running
statistics (calls, tokens, wall-clock time, $ estimate) so that every experiment
can report *how expensive* a red-teaming campaign was. This is the hook for the
inference-optimization side of the project.

Backends
--------
* ``sim``  -- :class:`~injecteval.models.sim_backend.SimulatedBackend`: a
  deterministic, dependency-free stand-in used for CI, unit tests and for
  developing the pipeline on a laptop. **Not evidence about real models.**
* ``hf``   -- :class:`~injecteval.models.hf_backend.HFBackend`: local
  open-weight models through Hugging Face ``transformers`` (+ LoRA adapters via
  ``peft``), with KV-cache, 4/8-bit quantization, batching, shared-prompt
  sampling and prefix caching.
* ``vllm`` -- :class:`~injecteval.models.vllm_backend.VLLMBackend`: offline vLLM
  engine (Linux/WSL2) with automatic prefix caching and LoRA.
* ``api``  -- :class:`~injecteval.models.api_backend.APIBackend`: any
  OpenAI-compatible endpoint (OpenAI, OpenRouter, vLLM server, Ollama, TGI).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

Message = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}


@dataclass
class GenConfig:
    """Generation + inference-optimization settings for one model."""

    backend: str = "sim"  # sim | hf | vllm | api
    model: str = "sim-agent"
    temperature: float = 0.0
    max_new_tokens: int = 256
    top_p: float = 1.0
    # --- inference-optimization knobs (ignored where not applicable) -------
    use_kv_cache: bool = True
    quantization: str = "none"  # none | 8bit | 4bit
    dtype: str = "auto"  # auto | float16 | bfloat16 | float32
    batch_size: int = 8
    prefix_cache: bool = False  # reuse KV of a shared system prompt (hf, bs=1)
    attn_implementation: Optional[str] = None  # e.g. "sdpa", "flash_attention_2"
    # --- adaptation -------------------------------------------------------
    adapter: Optional[str] = None  # path to a LoRA adapter (hf/vllm) or sim policy json
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)  # e.g. enable_thinking
    # --- API-only ---------------------------------------------------------
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    # --- misc -------------------------------------------------------------
    label: Optional[str] = None  # human-readable name used in result tables
    seed: int = 0
    gpu_memory_utilization: float = 0.45  # vllm only

    def merged(self, **overrides: Any) -> "GenConfig":
        d = asdict(self)
        d.update({k: v for k, v in overrides.items() if v is not None and k in d})
        return GenConfig(**d)

    @property
    def display_name(self) -> str:
        if self.label:
            return self.label
        name = self.model.split("/")[-1]
        if self.adapter:
            name += "+" + self.adapter.rstrip("/").split("/")[-1]
        if self.quantization not in (None, "none"):
            name += f"@{self.quantization}"
        return name

    @classmethod
    def from_dict(cls, d: Optional[Dict], **defaults: Any) -> "GenConfig":
        d = dict(defaults, **(d or {}))
        known = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class CallStats:
    """Accumulated usage counters for a backend instance."""

    n_calls: int = 0
    n_prompt_tokens: int = 0
    n_completion_tokens: int = 0
    wall_time_s: float = 0.0
    est_cost_usd: float = 0.0

    def add(self, prompt_tokens: int, completion_tokens: int, dt: float,
            n_calls: int = 1, cost: float = 0.0) -> None:
        self.n_calls += n_calls
        self.n_prompt_tokens += prompt_tokens
        self.n_completion_tokens += completion_tokens
        self.wall_time_s += dt
        self.est_cost_usd += cost

    @property
    def n_tokens(self) -> int:
        return self.n_prompt_tokens + self.n_completion_tokens

    def snapshot(self) -> Dict[str, Any]:
        return {
            "n_calls": self.n_calls,
            "n_prompt_tokens": self.n_prompt_tokens,
            "n_completion_tokens": self.n_completion_tokens,
            "n_tokens": self.n_tokens,
            "wall_time_s": round(self.wall_time_s, 4),
            "est_cost_usd": round(self.est_cost_usd, 6),
        }

    def diff(self, other: "CallStats") -> Dict[str, Any]:
        return {
            "n_calls": self.n_calls - other.n_calls,
            "n_prompt_tokens": self.n_prompt_tokens - other.n_prompt_tokens,
            "n_completion_tokens": self.n_completion_tokens - other.n_completion_tokens,
            "n_tokens": self.n_tokens - other.n_tokens,
            "wall_time_s": round(self.wall_time_s - other.wall_time_s, 4),
            "est_cost_usd": round(self.est_cost_usd - other.est_cost_usd, 6),
        }

    def copy(self) -> "CallStats":
        return CallStats(self.n_calls, self.n_prompt_tokens, self.n_completion_tokens,
                         self.wall_time_s, self.est_cost_usd)


class Backend:
    """Abstract chat-completion backend.

    Subclasses implement :meth:`_generate_batch` (and optionally
    :meth:`_sample_n`, which draws ``n`` samples for the *same* prompt and lets
    a backend encode the shared prompt only once).
    """

    name: str = "base"

    def __init__(self, cfg: GenConfig):
        self.cfg = cfg
        self.stats = CallStats()

    # -- to implement --------------------------------------------------------
    # A subclass implements EITHER ``_generate`` (one prompt -> one string) OR
    # ``_generate_batch`` (the batched primitive). The base wires them together
    # so simple backends (e.g. the simulator) only need ``_generate``.
    def _generate(self, messages: List[Message], cfg: GenConfig) -> str:
        raise NotImplementedError

    def _generate_batch(self, batch: List[List[Message]], cfg: GenConfig) -> List[str]:
        return [self._generate(m, cfg) for m in batch]

    def _sample_n(self, messages: List[Message], n: int, cfg: GenConfig) -> List[str]:
        # Default: independent calls (distinct completions when temperature>0).
        return self._generate_batch([messages] * n, cfg)

    # -- public API ----------------------------------------------------------
    def generate(self, messages: List[Message], **overrides: Any) -> str:
        return self.generate_batch([messages], **overrides)[0]

    def generate_batch(self, batch: List[List[Message]], **overrides: Any) -> List[str]:
        if not batch:
            return []
        cfg = self.cfg.merged(**overrides)
        t0 = time.perf_counter()
        outs = self._generate_batch(batch, cfg)
        self._account(batch, outs, time.perf_counter() - t0)
        return outs

    def sample_n(self, messages: List[Message], n: int, **overrides: Any) -> List[str]:
        """``n`` i.i.d. samples for one prompt (best-of-N / rollouts)."""
        if n <= 0:
            return []
        cfg = self.cfg.merged(**overrides)
        t0 = time.perf_counter()
        outs = self._sample_n(messages, n, cfg)
        dt = time.perf_counter() - t0
        # The shared prompt is counted once per sample (what an API would bill).
        self._account([messages] * len(outs), outs, dt)
        return outs

    # -- accounting ----------------------------------------------------------
    def _account(self, batch: List[List[Message]], outs: List[str], dt: float) -> None:
        pt = sum(self.count_tokens(self._flatten(m)) for m in batch)
        ct = sum(self.count_tokens(o) for o in outs)
        self.stats.add(pt, ct, dt, n_calls=len(batch), cost=self._cost(pt, ct))

    def _cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return 0.0

    @staticmethod
    def _flatten(messages: List[Message]) -> str:
        return "\n".join(m.get("content", "") for m in messages)

    def count_tokens(self, text: str) -> int:
        """Backend-agnostic estimate (~4 chars/token); real backends override."""
        return max(1, (len(text) + 3) // 4)

    def reset_stats(self) -> None:
        self.stats = CallStats()

    def peak_memory_gb(self) -> Optional[float]:
        return None

    def close(self) -> None:
        """Release model memory (important before training on the same GPU)."""


def get_backend(cfg: GenConfig) -> Backend:
    """Build a backend; heavy dependencies are imported lazily."""
    name = cfg.backend or "sim"
    if name == "sim":
        from .sim_backend import SimulatedBackend
        return SimulatedBackend(cfg)
    if name == "hf":
        from .hf_backend import HFBackend
        return HFBackend(cfg)
    if name == "vllm":
        from .vllm_backend import VLLMBackend
        return VLLMBackend(cfg)
    if name == "api":
        from .api_backend import APIBackend
        return APIBackend(cfg)
    raise ValueError(f"Unknown backend: {name!r} (expected sim|hf|vllm|api)")
