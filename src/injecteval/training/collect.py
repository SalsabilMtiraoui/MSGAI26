"""Collect training data from sandbox outcomes.

We run an attacker strategy against the (optionally defended) sandbox agent and
record every attempt together with the *ground-truth* judge verdict. Two files
are produced, both derived only from what happened inside the sandbox:

* ``sft.jsonl``  -- ``{"messages": [...], "completion": <injection>}`` for every
  attempt that compromised the agent (optionally: only *stealthy* successes).
* ``dpo.jsonl``  -- ``{"prompt": <chat>, "chosen": <success>, "rejected":
  <failure>}`` pairs for the same task+defense, so the preference is "this
  attempt worked here, that one did not".

The "prompt" is exactly the attacker prompt the strategy would use
(:meth:`Attacker._gen_messages`), so a model fine-tuned on this data slots back
into the identical attacker loop with no prompt drift.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..agent.agent import Agent
from ..attackers import get_attacker
from ..attackers.base import EvalOutcome
from ..defenses import get_defense
from ..judge import judge
from ..models.backend import GenConfig, get_backend
from ..tasks import load_suite


@dataclass
class Attempt:
    task_id: str
    defense: str
    messages: List[Dict]      # the attacker prompt (chat messages)
    injection: str
    attack_success: bool
    utility_success: bool
    blocked: bool


@dataclass
class CollectConfig:
    suite: str = "default"
    defenses: List[str] = field(default_factory=lambda: ["system_prompt", "sandwich", "spotlight"])
    attacker: str = "search"
    target: Optional[dict] = None
    attacker_model: Optional[dict] = None
    max_queries: int = 12
    stealthy_only: bool = False
    seed: int = 0


def collect(cfg: CollectConfig) -> List[Attempt]:
    tasks = load_suite(cfg.suite)
    target = get_backend(GenConfig.from_dict(cfg.target, backend="sim", model="sim-agent"))
    attacker_backend = get_backend(
        GenConfig.from_dict(cfg.attacker_model, backend="sim", model="sim-attacker"))
    agent = Agent(target)
    attacker = get_attacker(cfg.attacker, attacker_backend, seed=cfg.seed)

    attempts: List[Attempt] = []
    for dname in cfg.defenses:
        for task in tasks:
            task._target_model = target.cfg.model
            log: List[Attempt] = []

            def evaluate(injection: str, _task=task, _d=dname, _log=log) -> EvalOutcome:
                env = _task.build_env()
                traj = agent.run(_task, env, get_defense(_d), injection)
                jr = judge(env, _task)
                _log.append(Attempt(
                    task_id=_task.id, defense=_d,
                    messages=attacker.last_messages or [], injection=injection,
                    attack_success=jr.attack_success, utility_success=jr.utility_success,
                    blocked=traj.blocked))
                return EvalOutcome(jr.attack_success, jr.utility_success, traj.blocked, injection)

            attacker.run(task, dname, evaluate, cfg.max_queries)
            attempts.extend(log)
    return attempts


def to_sft(attempts: List[Attempt], stealthy_only: bool = False) -> List[Dict]:
    rows = []
    for a in attempts:
        ok = (a.attack_success and a.utility_success) if stealthy_only else a.attack_success
        if ok and a.messages:
            rows.append({"messages": a.messages, "completion": a.injection})
    return rows


def to_dpo(attempts: List[Attempt]) -> List[Dict]:
    """Pair a success with a failure on the *same* (task, defense)."""
    by_cell: Dict[tuple, Dict[str, List[Attempt]]] = {}
    for a in attempts:
        if not a.messages:
            continue
        cell = by_cell.setdefault((a.task_id, a.defense), {"win": [], "loss": []})
        (cell["win"] if a.attack_success else cell["loss"]).append(a)
    rows = []
    for cell in by_cell.values():
        for w in cell["win"]:
            for loss in cell["loss"]:
                rows.append({"prompt": w.messages, "chosen": w.injection, "rejected": loss.injection})
    return rows


def write_jsonl(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def collect_and_write(cfg: CollectConfig, outdir: str = "data") -> Dict[str, int]:
    attempts = collect(cfg)
    sft = to_sft(attempts, cfg.stealthy_only)
    dpo = to_dpo(attempts)
    write_jsonl(sft, os.path.join(outdir, "sft.jsonl"))
    write_jsonl(dpo, os.path.join(outdir, "dpo.jsonl"))
    counts = {"attempts": len(attempts), "sft": len(sft), "dpo_pairs": len(dpo),
              "successes": sum(a.attack_success for a in attempts)}
    print(f"collect: {counts}  -> {outdir}/sft.jsonl, {outdir}/dpo.jsonl")
    return counts
