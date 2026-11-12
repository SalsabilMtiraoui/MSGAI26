"""Deterministic, dependency-free simulated backend.

The ``sim`` backend plays two roles depending on the prompt it receives:

* **Target agent** -- when the prompt carries a ``<<SIM_SPEC>>`` control block
  (added by the agent layer only for this backend), it decides whether to obey
  an injection using the explicit feature/defense model in
  :mod:`injecteval.models.features`, and returns a JSON action plan in the
  agent's action protocol.
* **Attacker model** -- when the prompt is an injection-generation request
  (it contains a ``GOAL:`` line), it synthesizes an injection string by
  *imitating* any example winners in the prompt, *avoiding* any filtered
  patterns, and mutating. This is exactly the behavior the real attacker
  prompts elicit from an LLM, so the attacker strategy code is identical in
  sim and real modes.

Everything is seeded (model name + content + an internal nonce) so runs are
reproducible, while still giving ``temperature`` a controlled effect so that
retrying an attack can occasionally flip the outcome -- which is what gives
search something to do.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import List

from . import features as F
from .backend import Backend, GenConfig, Message

SIM_SPEC_RE = re.compile(r"<<SIM_SPEC>>(.*?)<<END_SIM_SPEC>>", re.DOTALL)


def _seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("||".join(parts).encode()).digest()[:8], "big")


class SimulatedBackend(Backend):
    name = "sim"

    def __init__(self, cfg: GenConfig):
        super().__init__(cfg)
        self._nonce = 0

    # -- dispatch ---------------------------------------------------------
    def _generate(self, messages: List[Message], cfg: GenConfig) -> str:
        self._nonce += 1
        text = self._flatten(messages)
        spec_match = SIM_SPEC_RE.search(text)
        if spec_match:
            return self._act_as_agent(spec_match.group(1), cfg)
        if "GOAL:" in text:
            return self._act_as_attacker(text, cfg)
        # Generic fallback (e.g. an LLM-judge call): echo a neutral verdict.
        return "UNKNOWN"

    # -- target agent -----------------------------------------------------
    def _act_as_agent(self, spec_json: str, cfg: GenConfig) -> str:
        spec = json.loads(spec_json)
        untrusted: str = spec.get("untrusted", "")
        defenses: List[str] = spec.get("defenses", []) or ["none"]
        target_model: str = spec.get("target_model", cfg.model)
        task_id: str = spec.get("task_id", "task")
        benign = spec["benign_action"]
        malicious = spec["malicious_action"]

        feats = set(F.extract_features(untrusted))

        # 1) Detection-style defenses can block the content outright.
        for d in defenses:
            dm = F.DEFENSES.get(d)
            if dm and set(dm.detect) & feats:
                # Injected content stripped; agent completes the benign task.
                return self._plan([benign], blocked=True)

        # 2) Suppression + skepticism.
        suppressed: set = set()
        threshold = F.base_threshold(target_model)
        for d in defenses:
            dm = F.DEFENSES.get(d)
            if dm:
                suppressed |= set(dm.suppress)
                threshold += dm.skepticism

        effective = (feats & set(F.effective_features(target_model, task_id))) - suppressed
        pressure = sum(F.FEATURES[f].weight for f in effective)

        # Controlled stochasticity: temperature lets a retry sometimes flip.
        if cfg.temperature > 0:
            rng = random.Random(_seed(target_model, task_id, untrusted, str(self._nonce)))
            jitter = (rng.random() - 0.5) * 2.0 * min(cfg.temperature, 1.0) * 0.4
            pressure += jitter

        obey = pressure >= threshold
        if not obey:
            return self._plan([benign])

        # Stealth: a "loud" effective feature tends to derail the user's task.
        loud = any(F.FEATURES[f].loud for f in effective)
        actions = [malicious] if loud else [malicious, benign]
        return self._plan(actions)

    @staticmethod
    def _plan(actions: List[dict], blocked: bool = False) -> str:
        payload = {"actions": actions, "final": "Task complete."}
        if blocked:
            payload["note"] = "ignored suspicious content"
        return json.dumps(payload)

    # -- attacker model ---------------------------------------------------
    def _act_as_attacker(self, prompt: str, cfg: GenConfig) -> str:
        rng = random.Random(_seed(prompt, str(self._nonce), str(cfg.temperature)))

        payload = self._field(prompt, "PAYLOAD") or "perform the required follow-up action"
        avoid = self._list_field(prompt, "AVOID")
        examples = self._examples(prompt)

        # Imitate: seed from features of example winners.
        chosen: List[str] = []
        for ex in examples:
            for f in F.extract_features(ex):
                if f not in chosen:
                    chosen.append(f)

        # If nothing to imitate, sample a starter set.
        if not chosen:
            pool = [n for n in F.ALL_FEATURE_NAMES]
            rng.shuffle(pool)
            chosen = pool[: rng.randint(2, 3)]

        # Avoid: drop features whose name or surface form is filtered.
        chosen = [f for f in chosen if not self._is_avoided(f, avoid)]

        # Mutate: occasionally add a fresh feature (more exploration at higher T).
        n_add = 1 + (1 if rng.random() < cfg.temperature else 0)
        candidates = [n for n in F.ALL_FEATURE_NAMES
                      if n not in chosen and not self._is_avoided(n, avoid)]
        rng.shuffle(candidates)
        chosen += candidates[:n_add]
        chosen = chosen[:4]  # keep injections compact

        return self._assemble(chosen, payload, rng)

    @staticmethod
    def _is_avoided(feature: str, avoid: List[str]) -> bool:
        if feature in avoid:
            return True
        # Also avoid if any avoided token appears in the feature's phrases.
        feat = F.FEATURES[feature]
        blob = " ".join(feat.phrases).lower()
        return any(tok and tok.lower() in blob for tok in avoid)

    @staticmethod
    def _assemble(feature_names: List[str], payload: str, rng: random.Random) -> str:
        # Order features so prefixes come first and framing wraps the payload.
        order = ["politeness_wrap", "authority_claim", "citation_bait",
                 "instruction_override", "role_reassign", "delimiter_spoof",
                 "fake_tool_output", "exfil_framing", "obfuscation", "urgency"]
        feature_names = sorted(set(feature_names), key=lambda n: order.index(n)
                               if n in order else 99)
        parts = []
        for name in feature_names:
            phrase = rng.choice(list(F.FEATURES[name].phrases))
            parts.append(phrase)
        # Embed the concrete malicious instruction.
        parts.append(payload if payload.endswith(".") else payload + ".")
        return " ".join(parts)

    # -- tiny prompt parsers ---------------------------------------------
    @staticmethod
    def _field(prompt: str, key: str) -> str:
        m = re.search(rf"^{key}:\s*(.+)$", prompt, flags=re.MULTILINE)
        return m.group(1).strip() if m else ""

    @classmethod
    def _list_field(cls, prompt: str, key: str) -> List[str]:
        raw = cls._field(prompt, key)
        return [t.strip() for t in re.split(r"[;,]", raw) if t.strip()] if raw else []

    @staticmethod
    def _examples(prompt: str) -> List[str]:
        return [m.group(1).strip()
                for m in re.finditer(r"^EXAMPLE:\s*(.+)$", prompt, flags=re.MULTILINE)]
