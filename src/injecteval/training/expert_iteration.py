"""Expert iteration for the attacker (generate -> judge -> self-train).

Each round:
  1. generate attempts against the *training* defenses with the current model,
  2. keep only the attempts the sandbox judge marks successful (the "experts"),
  3. LoRA-SFT the model on those experts (starting from the previous adapter),
  4. measure attack success on a *held-out* defense suite.

This is the gradient-based counterpart of the inference-only
``expert_iteration`` attacker, and the "train in a way that can discover new
attacks" direction. The reward is purely the programmatic sandbox judge, so the
loop needs no human labels and no LLM judge. Returns the per-round held-out ASR
so the report can show the learning curve. Requires ``pip install -e ".[train]"``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .collect import CollectConfig, collect, to_sft, write_jsonl
from .sft import SFTArgs, train_sft


@dataclass
class ExpertIterationArgs:
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    suite: str = "default"
    train_defenses: List[str] = field(default_factory=lambda: ["system_prompt", "sandwich"])
    heldout_defenses: List[str] = field(default_factory=lambda: ["spotlight", "regex_filter"])
    rounds: int = 3
    attacker: str = "search"
    max_queries: int = 10
    target: Optional[dict] = None
    attacker_model: Optional[dict] = None
    stealthy_only: bool = False
    out: str = "adapters/attacker-ei"
    workdir: str = "runs/ei"
    epochs: float = 2.0
    quantization: str = "none"
    seed: int = 0


def _heldout_asr(args: ExpertIterationArgs, adapter: Optional[str]) -> Dict[str, float]:
    """ASR of the current model on the held-out defenses (no training)."""
    from ..experiment import RunConfig, run_grid
    am = dict(args.attacker_model or {})
    am["adapter"] = adapter
    recs = run_grid(RunConfig(
        seed=args.seed, suite=args.suite, target=args.target,
        attacker_model=am, defenses=args.heldout_defenses, attackers=[args.attacker],
        max_queries=args.max_queries), progress=False)
    asr = sum(r["success"] for r in recs) / max(1, len(recs))
    stealth = sum(r["stealthy_success"] for r in recs) / max(1, len(recs))
    return {"asr": round(asr, 4), "stealthy_asr": round(stealth, 4)}


def run_expert_iteration(args: ExpertIterationArgs) -> List[Dict]:
    os.makedirs(args.workdir, exist_ok=True)
    history: List[Dict] = []
    adapter: Optional[str] = None

    base_metrics = _heldout_asr(args, adapter=None)
    history.append({"round": 0, "adapter": None, **base_metrics})
    print(f"[EI] round 0 (base)   held-out ASR={base_metrics['asr']}")

    for rnd in range(1, args.rounds + 1):
        am = dict(args.attacker_model or {})
        am["adapter"] = adapter
        attempts = collect(CollectConfig(
            suite=args.suite, defenses=args.train_defenses, attacker=args.attacker,
            target=args.target, attacker_model=am, max_queries=args.max_queries,
            stealthy_only=args.stealthy_only, seed=args.seed + rnd))
        sft_rows = to_sft(attempts, args.stealthy_only)
        data_path = os.path.join(args.workdir, f"round{rnd}_sft.jsonl")
        write_jsonl(sft_rows, data_path)
        n_succ = sum(a.attack_success for a in attempts)
        print(f"[EI] round {rnd}: {n_succ}/{len(attempts)} successful experts")
        if not sft_rows:
            print(f"[EI] round {rnd}: no experts collected, stopping.")
            break

        out = os.path.join(args.out + f"-r{rnd}")
        train_sft(SFTArgs(base_model=args.base_model, data=data_path, out=out,
                          epochs=args.epochs, quantization=args.quantization, seed=args.seed))
        adapter = out
        m = _heldout_asr(args, adapter=adapter)
        history.append({"round": rnd, "adapter": adapter, "n_experts": len(sft_rows), **m})
        print(f"[EI] round {rnd}         held-out ASR={m['asr']}  stealthy={m['stealthy_asr']}")

    # Symlink/copy the final adapter to the stable path.
    if adapter and adapter != args.out:
        try:
            if os.path.islink(args.out) or os.path.exists(args.out):
                os.remove(args.out)
            os.symlink(os.path.abspath(adapter), args.out)
        except OSError:
            pass
    return history
