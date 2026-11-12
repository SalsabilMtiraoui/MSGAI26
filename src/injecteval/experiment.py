"""Experiment runner: evaluate the (target x defense x attacker) grid.

For every (target model, defense, attacker, task) it runs the attacker's
budgeted search against the agent and records attack success, stealth, queries,
and the compute spent by both the attacker and the target model.

``run_clean`` additionally runs every defense on the *benign* (non-injected)
documents to measure its utility cost and false-positive rate -- the defender's
side of the trade-off.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from .agent.agent import Agent
from .attackers import get_attacker
from .attackers.base import EvalOutcome
from .defenses import get_defense
from .judge import judge
from .models import features as F
from .models.backend import Backend, GenConfig, get_backend
from .tasks import load_suite


@dataclass
class RunConfig:
    seed: int = 0
    suite: str = "default"
    target: Optional[dict] = None
    targets: Optional[List[dict]] = None   # several target models in one sweep
    attacker_model: Optional[dict] = None
    defenses: Optional[List[str]] = None
    attackers: Optional[List[str]] = None
    max_queries: int = 20
    output_dir: str = "results"

    def target_list(self) -> List[dict]:
        if self.targets:
            return self.targets
        return [self.target or {}]


def _mk_cfg(d: Optional[dict], default_backend: str) -> GenConfig:
    return GenConfig.from_dict(d, backend=default_backend)


def _evaluate_factory(agent: Agent, task, defense_name: str):
    def evaluate(injection: str) -> EvalOutcome:
        env = task.build_env()
        traj = agent.run(task, env, get_defense(defense_name), injection)
        jr = judge(env, task)
        return EvalOutcome(jr.attack_success, jr.utility_success, traj.blocked, injection)
    return evaluate


def run_grid(cfg: RunConfig, progress: bool = True) -> List[Dict]:
    tasks = load_suite(cfg.suite)
    attacker_cfg = _mk_cfg(cfg.attacker_model, "sim")
    records: List[Dict] = []

    for tdict in cfg.target_list():
        target_cfg = _mk_cfg(tdict, "sim")
        target = get_backend(target_cfg)
        agent = Agent(target)

        # Fresh attacker instances per target so an archive does not leak across
        # targets; each attacker gets its own backend for clean cost accounting.
        attacker_backends: Dict[str, Backend] = {}
        attacker_objs: Dict[str, object] = {}
        for aname in cfg.attackers:
            ab = get_backend(attacker_cfg)
            attacker_backends[aname] = ab
            attacker_objs[aname] = get_attacker(aname, ab, seed=cfg.seed)

        total = len(cfg.defenses) * len(cfg.attackers) * len(tasks)
        done = 0
        for defense_name in cfg.defenses:
            for aname in cfg.attackers:
                attacker = attacker_objs[aname]
                ab = attacker_backends[aname]
                for task in tasks:
                    task._target_model = target_cfg.model
                    evaluate = _evaluate_factory(agent, task, defense_name)

                    t_before = target.stats.copy()
                    a_before = ab.stats.copy()
                    t0 = time.perf_counter()
                    result = attacker.run(task, defense_name, evaluate, cfg.max_queries)
                    wall = time.perf_counter() - t0

                    records.append({
                        "suite": cfg.suite,
                        "target_model": target_cfg.display_name,
                        "attacker_model": attacker_cfg.display_name,
                        "defense": defense_name,
                        "attacker": aname,
                        "task_id": task.id,
                        "success": result.success,
                        "stealthy_success": result.stealthy_success,
                        "queries": result.queries,
                        "n_attempts": result.n_attempts,
                        "first_success_query": result.first_success_query,
                        "best_injection": result.best_injection,
                        "techniques": F.extract_features(result.best_injection),
                        "wall_time_s": round(wall, 4),
                        "target_cost": target.stats.diff(t_before),
                        "attacker_cost": ab.stats.diff(a_before),
                    })
                    done += 1
                    if progress and (done % max(1, total // 20) == 0 or done == total):
                        print(f"  [{target_cfg.display_name}] [{done}/{total}] "
                              f"{defense_name} x {aname} x {task.id} "
                              f"-> {'HIT' if result.success else 'miss'}")

        mem = target.peak_memory_gb()
        if mem is not None:
            for r in records:
                if r["target_model"] == target_cfg.display_name:
                    r.setdefault("target_peak_mem_gb", mem)
        target.close()
        for ab in attacker_backends.values():
            ab.close()
    return records


def run_clean(cfg: RunConfig, progress: bool = False) -> List[Dict]:
    """Run each defense on benign (non-injected) documents.

    Measures: does the user task still complete (utility), and does a filter
    wrongly block benign content (false positive)? Benign content is the task's
    own template with an empty injection slot.
    """
    tasks = load_suite(cfg.suite)
    records: List[Dict] = []
    for tdict in cfg.target_list():
        target_cfg = _mk_cfg(tdict, "sim")
        target = get_backend(target_cfg)
        agent = Agent(target)
        for defense_name in cfg.defenses:
            defense = get_defense(defense_name)
            for task in tasks:
                task._target_model = target_cfg.model
                env = task.build_env()
                traj = agent.run(task, env, defense, injection="")
                jr = judge(env, task)
                records.append({
                    "suite": cfg.suite, "target_model": target_cfg.display_name,
                    "defense": defense_name, "task_id": task.id,
                    "blocked": traj.blocked, "utility_success": jr.utility_success,
                    "attack_success": jr.attack_success})
                if progress:
                    print(f"  clean {defense_name} x {task.id} "
                          f"blocked={traj.blocked} util={jr.utility_success}")
        target.close()
    return records


def save_records(records: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def load_records(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
