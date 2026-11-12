# Report outline (7–8 pages, ICML template)

Maps the project onto the structure the course suggests. Each section lists what to write and
which artifact/figure to pull from (all produced by `make demo` and the real-model commands in
`GPU_SETUP.md`).

### 1. Motivation and problem statement (~1 p.)
- LLM agents act on untrusted content (emails, docs, web) → **indirect prompt injection**.
- Supervisor's challenge: naive injection already works, so where is the contribution?
  **Reframe to the harder, realistic problem: attacking a *defended* agent, cheaply.**
- Research questions **RQ1–RQ3** (README §1). State the **stealthy-success** objective
  (behaviour change the user doesn't notice).
- **Figure 1 = `docs/architecture.svg`** (the required summary figure).

### 2. Prior art and background (~1–1.5 p.)
- Indirect prompt injection (Greshake et al. 2023); agent benchmarks: AgentDojo
  (Debenedetti et al. 2024) — we build a lightweight self-contained harness in the same spirit.
- Defenses: spotlighting/datamarking (Hines et al. 2024), prompt-level defenses, filters,
  learned guards.
- Automated red-teaming: Red-LM (Perez et al. 2022); RL-based red-teaming and its **query-cost**
  limitation (arXiv:2510.04885); the **insufficiency of static defenses** (arXiv:2510.09023).
- Position: we don't invent a new algorithm; we **systematically evaluate** defense robustness
  with an efficient, adaptable automated attacker (what the rubric rewards).

### 3. Proposed application: architecture & adaptation (~1.5–2 p.)
- The sandbox agent (plan-and-execute, tools, environment) and the threat model (§agent, §judge).
- Defenses implemented, including `combo` (defense-in-depth) and the learned `model_guard`.
- Attackers as the *adaptation*:
  - baselines `naive_static`, `handcrafted`, `best_of_n` (inference-time sampling),
  - adaptive `search` (iterative refinement + filter-avoidance feedback),
  - inference-only `expert_iteration` (archive of winners as exemplars),
  - **trained** attacker: LoRA **SFT** on successes, **DPO** on (success > failure) pairs,
    and **expert iteration** (generate → judge → self-train), all supervised by the
    programmatic sandbox reward.
- Make explicit this is the "train to discover new attacks / preference over (win, loss)"
  direction, done cheaply and reproducibly on student hardware.

### 4. Proposed inference acceleration (~1 p.)
- The attacker is the hot loop (one forward pass per attempt). Apply **KV-cache, 4/8-bit
  quantization, batched generation, prefix caching** on a small attacker model.
- Method: `inference.py` microbenchmark; report tok/s, latency/attempt, peak memory
  (`fig_inference_bench`).
- Framing: throughput multiplies how many `(defense × attack)` cells you evaluate per unit
  compute → a more thorough robustness evaluation at fixed budget.

### 5. Evaluation (~2 p.)
- Setup: ≥4 target models (presets), the attacker model, 9 defenses, 5 attackers, 2 suites,
  budget, seeds. Report **Wilson 95% CIs** on every rate.
- **Quality:** ASR heatmap (`fig_asr_heatmap`), stealthy ASR (`fig_stealthy_bars`). Naive → ~0
  under defenses; adaptive/trained recover (RQ1, RQ2).
- **Budget/cost:** `fig_asr_at_k` (success vs query budget), `fig_cost_tradeoff` (ASR vs
  attacker compute), cost-per-success, expert-iteration query efficiency (RQ3).
- **Adaptation ablation:** base vs SFT vs DPO vs expert-iteration attacker (table +
  `ei_history.json` learning curve).
- **Defense trade-off:** `fig_defense_tradeoff` — residual ASR vs **false-positive rate** on
  benign documents; discuss when/why each defense helps and its utility cost.
- **Inference:** `fig_inference_bench` + one sentence on the net effect ("N× throughput → N×
  more cells per hour").

### 6. Conclusions & future work (~0.5 p.)
- Simple defenses are necessary but not sufficient; adaptive/trained attackers are cheap;
  defense-in-depth + learned guards raise the bar at a utility cost.
- Future: real AgentDojo integration; GRPO/online preference training; adaptive defenses;
  transfer across target models.

### 7. References
- The list in `README.md` §13; add anything you cite.

---

**Checklist for the top rubric band**
- [ ] Working baseline (naive + undefended) ✔
- [ ] ≥1 adaptation — search / expert iteration **and** trained SFT/DPO/EI ✔
- [ ] Inference-time optimization with measured gains ✔ (`inference.py`)
- [ ] ≥4 model variants with comprehensive metrics + CIs ✔ (presets + suites)
- [ ] Reproducible repo + clear README + CI ✔
- [ ] Summary figure ✔ (`docs/architecture.svg`)
