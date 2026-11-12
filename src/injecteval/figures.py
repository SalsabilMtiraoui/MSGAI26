"""Report figures from a run's JSONL records.

Design choices follow a few visualization non-negotiables: a fixed,
colorblind-safe categorical palette (Okabe-Ito) assigned per attacker in a fixed
order (never cycled), a single-hue sequential colormap for the ASR heatmap,
legends always present for multi-series plots, recessive grids, and no dual
axes. Figures are written as both PNG (for slides) and PDF (for the LaTeX
report).
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .metrics import aggregate, group_by

# Fixed order + colorblind-safe palette (Okabe-Ito).
ATTACKER_ORDER = ["naive_static", "handcrafted", "best_of_n", "search", "expert_iteration"]
ATTACKER_COLOR = {
    "naive_static": "#999999",
    "handcrafted": "#E69F00",
    "best_of_n": "#CC79A7",
    "search": "#56B4E9",
    "expert_iteration": "#009E73",
}
DEFENSE_ORDER = ["none", "system_prompt", "sandwich", "spotlight",
                 "regex_filter", "ppl_filter", "llm_guard", "model_guard", "combo"]
_SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f7fbff", "#08306b"])
INK = "#222222"
MUTED = "#6b6b6b"


def _style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)


def _present(records, key, order):
    have = {r[key] for r in records}
    return [x for x in order if x in have] + sorted(have - set(order))


def _matrix(records, value_key):
    attackers = _present(records, "attacker", ATTACKER_ORDER)
    defenses = _present(records, "defense", DEFENSE_ORDER)
    cells: Dict[tuple, float] = {}
    for (atk, dfn), recs in group_by(records, ["attacker", "defense"]).items():
        cells[(atk, dfn)] = aggregate(recs).get(value_key, 0.0)
    mat = [[cells.get((a, d), 0.0) for d in defenses] for a in attackers]
    return attackers, defenses, mat


# --------------------------------------------------------------------------- 1
def fig_asr_heatmap(records: List[Dict], outdir: str) -> str:
    attackers, defenses, mat = _matrix(records, "asr")
    fig, ax = plt.subplots(figsize=(1.4 + 1.0 * len(defenses), 1.2 + 0.7 * len(attackers)))
    im = ax.imshow(mat, cmap=_SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(defenses)), defenses, rotation=30, ha="right")
    ax.set_yticks(range(len(attackers)), attackers)
    for i in range(len(attackers)):
        for j in range(len(defenses)):
            v = mat[i][j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.55 else INK)
    ax.set_title("Attack success rate by attacker x defense", fontsize=11,
                 color=INK, pad=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("ASR", color=INK, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return _save(fig, outdir, "fig_asr_heatmap")


# --------------------------------------------------------------------------- 2
def fig_stealthy_bars(records: List[Dict], outdir: str) -> str:
    attackers, defenses, mat = _matrix(records, "stealthy_asr")
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(defenses), 4.2))
    _style(ax)
    n = len(attackers)
    width = 0.8 / max(1, n)
    x = range(len(defenses))
    for k, atk in enumerate(attackers):
        offs = [xi + (k - (n - 1) / 2) * width for xi in x]
        ax.bar(offs, mat[k], width=width, label=atk,
               color=ATTACKER_COLOR.get(atk, "#444444"), edgecolor="white",
               linewidth=0.5)
    ax.set_xticks(list(x), defenses, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Stealthy ASR\n(attack succeeds, user task intact)", fontsize=9,
                  color=INK)
    ax.set_title("Stealthy attack success by defense", fontsize=11, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    return _save(fig, outdir, "fig_stealthy_bars")


# --------------------------------------------------------------------------- 3
def fig_cost_tradeoff(records: List[Dict], outdir: str) -> str:
    """ASR vs attacker compute per task -> the quality/efficiency trade-off."""
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    _style(ax)
    ax.grid(axis="both", color="#e6e6e6", linewidth=0.8)
    attackers = _present(records, "attacker", ATTACKER_ORDER)
    for atk in attackers:
        recs = [r for r in records if r["attacker"] == atk]
        agg = aggregate(recs)
        x = agg["attacker_tokens"] / max(1, agg["n_tasks"])   # tokens / task
        y = agg["asr"]
        ax.scatter([x], [y], s=120, color=ATTACKER_COLOR.get(atk, "#444"),
                   edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(atk, (x, y), xytext=(6, 6), textcoords="offset points",
                    fontsize=9, color=INK)
    ax.set_xlabel("Attacker compute per task (tokens)", fontsize=9, color=INK)
    ax.set_ylabel("Attack success rate", fontsize=9, color=INK)
    ax.set_ylim(0, 1)
    ax.set_title("Quality vs. cost: ASR against attacker compute", fontsize=11,
                 color=INK, pad=10)
    fig.tight_layout()
    return _save(fig, outdir, "fig_cost_tradeoff")


# ----------------------------------------------------------------- bench (hf)
def fig_bench(csv_path: str, outdir: str) -> str:
    import csv
    with open(csv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    names = [r["name"] for r in rows]
    tps = [float(r["tokens_per_s"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    _style(ax)
    ax.bar(range(len(names)), tps, color="#56B4E9", edgecolor="white", width=0.6)
    for i, v in enumerate(tps):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(len(names)), names, rotation=20, ha="right")
    ax.set_ylabel("Attacker throughput (tokens/s)", fontsize=9, color=INK)
    ax.set_title("Inference optimizations: attacker throughput", fontsize=11,
                 color=INK, pad=10)
    fig.tight_layout()
    return _save(fig, outdir, "fig_inference_bench")


# --------------------------------------------------------------------------- 5
def fig_asr_at_k(records: List[Dict], outdir: str, defense: str = None) -> str:
    """ASR@k budget curves: success within k target queries, per attacker."""
    from .metrics import asr_at_k
    recs = [r for r in records if defense is None or r["defense"] == defense]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    _style(ax)
    ax.grid(axis="both", color="#e6e6e6", linewidth=0.8)
    for atk in _present(recs, "attacker", ATTACKER_ORDER):
        curve = asr_at_k([r for r in recs if r["attacker"] == atk])
        if curve:
            ax.plot(range(1, len(curve) + 1), curve, marker="o", markersize=4,
                    label=atk, color=ATTACKER_COLOR.get(atk, "#444"), linewidth=1.8)
    ax.set_xlabel("Target queries (budget k)", fontsize=9, color=INK)
    ax.set_ylabel("ASR@k", fontsize=9, color=INK)
    ax.set_ylim(0, 1)
    title = "Attack success vs. query budget"
    if defense:
        title += f"  (defense: {defense})"
    ax.set_title(title, fontsize=11, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    return _save(fig, outdir, "fig_asr_at_k")


# --------------------------------------------------------------------------- 6
def fig_defense_tradeoff(records: List[Dict], clean_records: List[Dict], outdir: str) -> str:
    """Per defense: best attacker ASR it still allows vs. its benign FPR."""
    from .metrics import aggregate, defense_utility_table, group_by
    fpr = {r["defense"]: r["false_positive_rate"] for r in defense_utility_table(clean_records)}
    best_asr = {}
    for (dfn,), recs in group_by(records, ["defense"]).items():
        best_asr[dfn] = max(aggregate([r for r in recs if r["attacker"] == a]).get("asr", 0)
                            for a in set(r["attacker"] for r in recs))
    defenses = _present(records, "defense", DEFENSE_ORDER)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    _style(ax)
    ax.grid(axis="both", color="#e6e6e6", linewidth=0.8)
    for dfn in defenses:
        x, y = fpr.get(dfn, 0.0), best_asr.get(dfn, 0.0)
        ax.scatter([x], [y], s=130, color="#2166ac", edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(dfn, (x, y), xytext=(6, 5), textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlabel("False-positive rate on benign documents", fontsize=9, color=INK)
    ax.set_ylabel("Residual ASR (best attacker)", fontsize=9, color=INK)
    ax.set_ylim(-0.03, 1)
    ax.set_title("Defense trade-off: security (low ASR) vs. utility (low FPR)",
                 fontsize=11, color=INK, pad=10)
    ax.text(0.98, 0.02, "bottom-left = ideal", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=8, color=MUTED, style="italic")
    fig.tight_layout()
    return _save(fig, outdir, "fig_defense_tradeoff")


# --------------------------------------------------------------------------- 7
def fig_technique_hist(records: List[Dict], outdir: str) -> str:
    """Which injection techniques appear in the winning attacks."""
    from .metrics import technique_histogram
    hist = technique_histogram(records)
    items = [(k, v) for k, v in hist.items() if v > 0]
    items.sort(key=lambda kv: kv[1], reverse=True)
    if not items:
        items = [("(none)", 0)]
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(6.4, 0.5 + 0.4 * len(names)))
    _style(ax)
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
    ax.barh(range(len(names)), vals, color="#009E73", edgecolor="white", height=0.7)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlabel("Winning attacks using this technique", fontsize=9, color=INK)
    ax.set_title("Techniques in successful injections", fontsize=11, color=INK, pad=10)
    fig.tight_layout()
    return _save(fig, outdir, "fig_technique_hist")


# ---------------------------------------------------------------------- utils
def _save(fig, outdir: str, name: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, name + ".png")
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(outdir, name + ".pdf"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return png


def make_all_figures(records: List[Dict], outdir: str,
                     clean_records: List[Dict] = None) -> List[str]:
    figs = [
        fig_asr_heatmap(records, outdir),
        fig_stealthy_bars(records, outdir),
        fig_cost_tradeoff(records, outdir),
        fig_asr_at_k(records, outdir),
        fig_technique_hist(records, outdir),
    ]
    if clean_records:
        figs.append(fig_defense_tradeoff(records, clean_records, outdir))
    return figs
