"""Command-line entry point: ``injecteval <command>``.

Commands
--------
sweep     Run the (target x defense x attacker) grid; write JSONL + clean-doc eval.
summary   Print the aggregated (attacker x defense) table from a JSONL.
figures   Render the report figures from a run (and optional clean JSONL).
bench     Inference-optimization microbenchmark (KV cache / quant / batch / prefix).
collect   Generate SFT + DPO training data from sandbox outcomes.
train     Fine-tune the attacker: sft | dpo | expert-iteration (LoRA).
info      List available defenses, attackers and task suites.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

from . import __version__
from .attackers import list_attackers
from .config import load_yaml, resolve_preset
from .defenses import list_defenses
from .experiment import RunConfig, load_records, run_clean, run_grid, save_records
from .metrics import defense_utility_table, summary_table
from .tasks import list_suites


# --------------------------------------------------------------- model helpers
def _apply_model_overrides(section: Dict, prefix: str, args) -> Dict:
    section = dict(section or {})
    preset = getattr(args, f"{prefix}_preset", None)
    if preset:
        section.update(resolve_preset(preset))
    for attr, key in (("backend", "backend"), ("model", "model"), ("adapter", "adapter")):
        val = getattr(args, f"{prefix}_{attr}", None)
        if val:
            section[key] = val
    return section


def _target_list(cfg_dict: Dict, args) -> List[Dict]:
    presets = getattr(args, "target_presets", None)
    if presets:
        return [resolve_preset(p) for p in presets]
    return [_apply_model_overrides(cfg_dict.get("target", {}), "target", args)]


# ----------------------------------------------------------------------- sweep
def cmd_sweep(args) -> None:
    cfg_dict = load_yaml(args.config) if args.config else {}
    targets = _target_list(cfg_dict, args)
    attacker_model = _apply_model_overrides(cfg_dict.get("attacker_model", {}), "attacker", args)
    grid = cfg_dict.get("grid", {})
    budget = cfg_dict.get("budget", {})

    run_cfg = RunConfig(
        seed=args.seed if args.seed is not None else cfg_dict.get("seed", 0),
        suite=args.suite or cfg_dict.get("suite", "default"),
        targets=targets,
        attacker_model=attacker_model,
        defenses=args.defenses or grid.get("defenses", list_defenses()),
        attackers=args.attackers or grid.get("attackers", list_attackers()),
        max_queries=args.max_queries or budget.get("max_queries", 20),
        output_dir=cfg_dict.get("output_dir", "results"))

    print(f"InjectEval sweep  suite={run_cfg.suite}  "
          f"targets={[t.get('model') for t in targets]}  "
          f"attacker={attacker_model.get('backend')}:{attacker_model.get('model')}")
    print(f"  defenses={run_cfg.defenses}")
    print(f"  attackers={run_cfg.attackers}  budget={run_cfg.max_queries}")

    records = run_grid(run_cfg)
    out = args.out or os.path.join(run_cfg.output_dir, "run.jsonl")
    save_records(records, out)
    print(f"\nWrote {len(records)} records -> {out}")
    _print_table(summary_table(records))

    if not args.no_clean:
        clean = run_clean(run_cfg)
        clean_out = out.replace(".jsonl", "") + ".clean.jsonl"
        save_records(clean, clean_out)
        print(f"\nClean-document evaluation -> {clean_out}")
        _print_clean(defense_utility_table(clean))


# --------------------------------------------------------------------- summary
def cmd_summary(args) -> None:
    _print_table(summary_table(load_records(args.records)))
    if args.clean and os.path.exists(args.clean):
        _print_clean(defense_utility_table(load_records(args.clean)))


def _print_table(rows: List[Dict]) -> None:
    if not rows:
        print("(no records)")
        return
    cols = ["attacker", "defense", "asr", "stealthy_asr",
            "mean_queries_to_success", "attacker_tokens", "cost_per_success"]
    _grid_print(rows, cols, group="attacker")


def _print_clean(rows: List[Dict]) -> None:
    print("\nDefense utility / false positives (benign documents):")
    _grid_print(rows, ["defense", "n_clean", "clean_utility", "false_positive_rate"], group=None)


def _grid_print(rows, cols, group=None) -> None:
    widths = {c: max(len(c), *(len(_fmt(r.get(c))) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print("\n" + header)
    print("-" * len(header))
    last = None
    for r in rows:
        if group and last is not None and r.get(group) != last:
            print()
        last = r.get(group)
        print("  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in cols))


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    if v is None:
        return "-"
    return str(v)


# --------------------------------------------------------------------- figures
def cmd_figures(args) -> None:
    from .figures import make_all_figures
    records = load_records(args.records)
    clean = load_records(args.clean) if args.clean and os.path.exists(args.clean) else None
    paths = make_all_figures(records, args.outdir, clean_records=clean)
    print(f"Wrote {len(paths)} figures to {args.outdir}:")
    for p in paths:
        print(f"  - {p}")


# ----------------------------------------------------------------------- bench
def cmd_bench(args) -> None:
    import csv

    from .inference import run_sweep
    cfg_dict = load_yaml(args.config) if args.config else {}
    attacker = _apply_model_overrides(cfg_dict.get("attacker_model", {}), "attacker", args)
    isweep = cfg_dict.get("inference_sweep", {})
    configs = isweep.get("configs") or [
        {"name": "baseline", "use_kv_cache": False, "quantization": "none", "batch_size": 1},
        {"name": "kv_cache", "use_kv_cache": True, "quantization": "none", "batch_size": 1},
        {"name": "kv_prefix", "use_kv_cache": True, "quantization": "none", "batch_size": 1,
         "prefix_cache": True},
        {"name": "kv_batch", "use_kv_cache": True, "quantization": "none", "batch_size": 16},
        {"name": "kv_batch_4bit", "use_kv_cache": True, "quantization": "4bit", "batch_size": 16},
    ]
    print(f"Inference benchmark on {attacker.get('backend')}:{attacker.get('model')} "
          f"(n={args.n} items/config)")
    rows = run_sweep(attacker, configs, n=args.n)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out}")


# --------------------------------------------------------------------- collect
def cmd_collect(args) -> None:
    from .training.collect import CollectConfig, collect_and_write
    cfg_dict = load_yaml(args.config) if args.config else {}
    collect_and_write(CollectConfig(
        suite=args.suite or cfg_dict.get("suite", "default"),
        defenses=args.defenses or ["system_prompt", "sandwich", "spotlight", "llm_guard"],
        attacker=args.attacker,
        target=_apply_model_overrides(cfg_dict.get("target", {}), "target", args),
        attacker_model=_apply_model_overrides(cfg_dict.get("attacker_model", {}), "attacker", args),
        max_queries=args.max_queries or 12,
        stealthy_only=args.stealthy_only,
        seed=args.seed or 0), outdir=args.outdir)


# ----------------------------------------------------------------------- train
def cmd_train(args) -> None:
    if args.method == "sft":
        from .training.sft import SFTArgs, train_sft
        train_sft(SFTArgs(base_model=args.base_model, data=args.data, out=args.out,
                          epochs=args.epochs, quantization=args.quantization))
    elif args.method == "dpo":
        from .training.dpo import DPOArgs, train_dpo
        train_dpo(DPOArgs(base_model=args.base_model, data=args.data, out=args.out,
                          init_adapter=args.init_adapter, epochs=args.epochs,
                          quantization=args.quantization))
    elif args.method in ("ei", "expert-iteration"):
        import json

        from .training.expert_iteration import ExpertIterationArgs, run_expert_iteration
        hist = run_expert_iteration(ExpertIterationArgs(
            base_model=args.base_model, suite=args.suite or "default", rounds=args.rounds,
            out=args.out, quantization=args.quantization, max_queries=args.max_queries or 10))
        print("\nExpert-iteration held-out ASR by round:")
        for h in hist:
            print(f"  round {h['round']}: asr={h.get('asr')}  stealthy={h.get('stealthy_asr')}")
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(os.path.join(os.path.dirname(args.out) or ".", "ei_history.json"), "w") as fh:
            json.dump(hist, fh, indent=2)


# ------------------------------------------------------------------------ info
def cmd_info(args) -> None:
    print(f"InjectEval {__version__}")
    print(f"  defenses:  {', '.join(list_defenses())}")
    print(f"  attackers: {', '.join(list_attackers())}")
    print(f"  suites:    {', '.join(list_suites())}")


# ------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="injecteval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def add_model_flags(sp, target=True, attacker=True):
        if target:
            sp.add_argument("--target-backend", choices=["sim", "hf", "vllm", "api"])
            sp.add_argument("--target-model")
            sp.add_argument("--target-preset")
            sp.add_argument("--target-presets", nargs="*", help="several targets in one sweep")
        if attacker:
            sp.add_argument("--attacker-backend", choices=["sim", "hf", "vllm", "api"])
            sp.add_argument("--attacker-model")
            sp.add_argument("--attacker-adapter", help="path to a trained LoRA adapter")
            sp.add_argument("--attacker-preset")

    sp = sub.add_parser("sweep", help="run the target x defense x attacker grid")
    sp.add_argument("--config", default="config/default.yaml")
    add_model_flags(sp)
    sp.add_argument("--suite")
    sp.add_argument("--defenses", nargs="*")
    sp.add_argument("--attackers", nargs="*")
    sp.add_argument("--max-queries", type=int)
    sp.add_argument("--seed", type=int)
    sp.add_argument("--out")
    sp.add_argument("--no-clean", action="store_true", help="skip the clean-document eval")
    sp.set_defaults(func=cmd_sweep)

    sp = sub.add_parser("summary", help="print aggregated table from a run")
    sp.add_argument("records")
    sp.add_argument("--clean", help="clean-eval JSONL for the defense utility table")
    sp.set_defaults(func=cmd_summary)

    sp = sub.add_parser("figures", help="render report figures from a run")
    sp.add_argument("records")
    sp.add_argument("--clean", help="clean-eval JSONL (enables the defense trade-off figure)")
    sp.add_argument("--outdir", default="figures")
    sp.set_defaults(func=cmd_figures)

    sp = sub.add_parser("bench", help="inference-optimization microbenchmark")
    sp.add_argument("--config", default="config/default.yaml")
    add_model_flags(sp, target=False)
    sp.add_argument("--n", type=int, default=32)
    sp.add_argument("--out", default="results/bench.csv")
    sp.set_defaults(func=cmd_bench)

    sp = sub.add_parser("collect", help="generate SFT + DPO data from sandbox outcomes")
    sp.add_argument("--config", default="config/default.yaml")
    add_model_flags(sp)
    sp.add_argument("--suite")
    sp.add_argument("--attacker", default="search")
    sp.add_argument("--defenses", nargs="*")
    sp.add_argument("--max-queries", type=int)
    sp.add_argument("--stealthy-only", action="store_true")
    sp.add_argument("--seed", type=int)
    sp.add_argument("--outdir", default="data")
    sp.set_defaults(func=cmd_collect)

    sp = sub.add_parser("train", help="fine-tune the attacker (sft|dpo|ei)")
    sp.add_argument("method", choices=["sft", "dpo", "ei", "expert-iteration"])
    sp.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    sp.add_argument("--data", default="data/sft.jsonl")
    sp.add_argument("--out", default="adapters/attacker")
    sp.add_argument("--init-adapter")
    sp.add_argument("--epochs", type=float, default=3.0)
    sp.add_argument("--rounds", type=int, default=3)
    sp.add_argument("--suite")
    sp.add_argument("--max-queries", type=int)
    sp.add_argument("--quantization", default="none", choices=["none", "8bit", "4bit"])
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("info", help="list defenses, attackers, suites")
    sp.set_defaults(func=cmd_info)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
