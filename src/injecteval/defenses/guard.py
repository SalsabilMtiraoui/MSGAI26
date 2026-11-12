"""Optional model-based input guard (a learned prompt-injection classifier).

``model_guard`` runs the untrusted content through a HuggingFace
text-classification model and redacts it if flagged. This is the "trained
guard" defense: compare a learned detector against the cheap heuristic/regex
filters on both detection and false-positive rate.

Default model: ``protectai/deberta-v3-base-prompt-injection-v2`` (swap via the
``INJECTEVAL_GUARD_MODEL`` env var or ``config``). Needs the ``hf`` extra and a
one-time model download. On the ``sim`` backend this is not used; use the
always-available heuristic ``llm_guard`` instead.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

_DEFAULT = "protectai/deberta-v3-base-prompt-injection-v2"
# Label (upper-cased) that means "injection / unsafe" for common guard models.
_POSITIVE = {"INJECTION", "LABEL_1", "UNSAFE", "JAILBREAK", "TRUE", "1"}


@lru_cache(maxsize=2)
def _pipeline(model_id: str):
    from transformers import pipeline
    return pipeline("text-classification", model=model_id, truncation=True, max_length=512)


class ModelGuard:
    def __init__(self, model_id: Optional[str] = None, threshold: float = 0.5):
        self.model_id = model_id or os.environ.get("INJECTEVAL_GUARD_MODEL", _DEFAULT)
        self.threshold = threshold

    def is_injection(self, text: str) -> bool:
        out = _pipeline(self.model_id)(text)[0]
        label = str(out.get("label", "")).upper()
        score = float(out.get("score", 0.0))
        return label in _POSITIVE and score >= self.threshold
