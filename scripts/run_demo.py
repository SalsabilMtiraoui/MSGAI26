#!/usr/bin/env python3
"""One-command reproduction of the demo (deterministic ``sim`` backend, no GPU).

Runs the full (defense x attacker) grid on both task suites, the clean-document
evaluation (defense false positives), writes JSONL records, prints the tables,
and renders every report figure. A few seconds total.

    python scripts/run_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from injecteval.attackers import list_attackers  # noqa: E402
from injecteval.cli import _print_clean, _print_table  # noqa: E402
from injecteval.experiment import RunConfig, run_clean, run_grid, save_records  # noqa: E402
from injecteval.figures import make_all_figures  # noqa: E402
from injecteval.metrics import defense_utility_table, summary_table  # noqa: E402

# Every defense except model_guard, which needs a downloaded HF classifier
# (use it only with the hf backend; see README).
SIM_DEFENSES = ["none", "system_prompt", "sandwich", "spotlight",
                "regex_filter", "ppl_filter", "llm_guard", "combo"]


def main() -> None:
    all_records, all_clean = [], []
    for suite in ("default", "hard"):
        print(f"\n=== suite: {suite} ===")
        cfg = RunConfig(
            seed=0, suite=suite,
            target={"backend": "sim", "model": "sim-agent", "temperature": 0.0},
            attacker_model={"backend": "sim", "model": "sim-attacker", "temperature": 0.9},
            defenses=SIM_DEFENSES, attackers=list_attackers(), max_queries=20)
        records = run_grid(cfg, progress=False)
        clean = run_clean(cfg)
        save_records(records, f"results/{suite}.jsonl")
        save_records(clean, f"results/{suite}.clean.jsonl")
        _print_table(summary_table(records))
        _print_clean(defense_utility_table(clean))
        all_records += records
        all_clean += clean

    save_records(all_records, "results/all.jsonl")
    save_records(all_clean, "results/all.clean.jsonl")
    figs = make_all_figures(all_records, "figures", clean_records=all_clean)
    print("\nFigures:")
    for f in figs:
        print("  -", f)
    print("\nDone. Records in results/, figures in figures/.")


if __name__ == "__main__":
    main()
