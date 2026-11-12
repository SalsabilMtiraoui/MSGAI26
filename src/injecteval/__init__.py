"""InjectEval: efficient automated red-teaming for indirect prompt injection.

A research harness that measures how well simple defenses hold up against
automated prompt-injection attackers on a *simulated* tool-using LLM agent, and
how cheaply that red-teaming can be run using a small local attacker model plus
inference-time optimizations.

Everything happens inside a sandbox: the "tools" act on an in-memory fake
workspace, and the attack goal is an *unauthorized action* (e.g. exfiltrating a
fake secret to a honeypot address). No real systems are touched and no harmful
content is produced -- the object of study is control-flow hijacking of the
agent, not the payload.
"""

__version__ = "0.1.0"

from .models.backend import Backend, CallStats, GenConfig, get_backend

__all__ = ["Backend", "GenConfig", "CallStats", "get_backend", "__version__"]
