"""Adaptation of the attacker model (the project's *model adaptation* axis).

Three inference-only / training-based strategies turn a generic small instruct
model into a stronger red-teaming generator, all supervised by the **programmatic
sandbox judge** (never an LLM judge, never a human-written attack corpus):

* :mod:`injecteval.training.collect`  -- run the current attacker against the
  sandbox and log, per attempt, whether it compromised the agent. This yields
  supervised data (successful attempts) and preference pairs (success > failure)
  *entirely from sandbox outcomes*.
* :mod:`injecteval.training.sft`      -- LoRA supervised fine-tuning on the
  successful attempts (behaviour cloning of what worked).
* :mod:`injecteval.training.dpo`      -- Direct Preference Optimization on the
  (success, failure) pairs (the preference-training direction).
* :mod:`injecteval.training.expert_iteration` -- alternate generate->judge->SFT
  so the generator bootstraps from its own best samples (the "train to discover
  new attacks" direction), with the held-out defense suite as the test.

Everything targets the in-memory sandbox agent with fake secrets; the artefact
produced is a small LoRA adapter used to *measure how much a cheap adaptation
raises defense-evasion*, i.e. to quantify defense robustness. Requires the
``train`` extra: ``pip install -e ".[train]"``.
"""

from __future__ import annotations

__all__ = ["collect", "sft", "dpo", "expert_iteration"]
