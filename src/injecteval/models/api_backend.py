"""OpenAI-compatible API backend (OpenAI, OpenRouter, vLLM server, Ollama, TGI).

    export OPENAI_API_KEY=sk-...                       # or OPENROUTER_API_KEY, see api_key_env
    export OPENAI_BASE_URL=http://localhost:8000/v1    # optional (local servers)

Install with ``pip install -e ".[api]"``.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

from .backend import Backend, GenConfig, Message

# Rough price table ($ per 1K tokens, input/output). Unknown models -> free.
_PRICES = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
}


class APIBackend(Backend):
    name = "api"

    def __init__(self, cfg: GenConfig):
        super().__init__(cfg)
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError('APIBackend needs: pip install -e ".[api]"') from e
        base_url = cfg.base_url or os.environ.get("OPENAI_BASE_URL")
        api_key = os.environ.get(cfg.api_key_env) or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.client = OpenAI(base_url=base_url, api_key=api_key) if base_url else OpenAI(api_key=api_key)
        self._usage = []

    def _one(self, messages: List[Message], cfg: GenConfig) -> str:
        for attempt in range(4):
            try:
                resp = self.client.chat.completions.create(
                    model=cfg.model, messages=messages, temperature=cfg.temperature,
                    max_tokens=cfg.max_new_tokens, top_p=cfg.top_p)
                u = getattr(resp, "usage", None)
                if u is not None:
                    self._usage.append((u.prompt_tokens or 0, u.completion_tokens or 0))
                return (resp.choices[0].message.content or "").strip()
            except Exception:  # rate limits / transient errors
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        return ""

    def _generate_batch(self, batch, cfg):
        with ThreadPoolExecutor(max_workers=max(1, min(cfg.batch_size, 16))) as ex:
            return list(ex.map(lambda m: self._one(m, cfg), batch))

    def _account(self, batch, outs, dt):
        if len(self._usage) == len(outs):
            pt = sum(u[0] for u in self._usage)
            ct = sum(u[1] for u in self._usage)
            self._usage = []
            self.stats.add(pt, ct, dt, n_calls=len(batch), cost=self._cost(pt, ct))
        else:
            self._usage = []
            super()._account(batch, outs, dt)

    def _cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        p_in, p_out = _PRICES.get(self.cfg.model, (0.0, 0.0))
        return prompt_tokens / 1000.0 * p_in + completion_tokens / 1000.0 * p_out
