# InjectEval — Efficient Automated Red-Teaming of *Defended* LLM Agents

> **MSGAI26 group project (UNINE / BeNeFri).** A reproducible harness that measures
> **how well cheap, deployable defenses hold up against automated indirect-prompt-injection
> attackers** on a tool-using LLM agent — and **how cheaply that red-teaming can be run**
> with a small *local* attacker model plus inference-time optimizations and lightweight
> fine-tuning (LoRA SFT / DPO / expert iteration).

<p align="center">
  <img src="docs/architecture.svg" alt="InjectEval architecture" width="880">
</p>

> 🇫🇷 **Prise en main rapide.** `pip install -e ".[dev]"` puis `make demo` : tout le pipeline
> tourne en quelques secondes **sans GPU** (backend `sim`), écrit les tables et les figures.
> Pour les *vrais* modèles et l'entraînement sur ton PC à GPU, voir §6–§8 et
> [`docs/GPU_SETUP.md`](docs/GPU_SETUP.md). La **seule partie à faire toi-même** (ta
> contribution de recherche) est décrite dans [`docs/YOUR_CONTRIBUTION.md`](docs/YOUR_CONTRIBUTION.md).

---

## 1. Why this project (how it answers the supervisor's feedback)

Basile's comment on the first idea was, in essence: *why fine-tune an attacker when copy/paste
injection already works?* and *can you show that simple approaches fail?* We **raise the
difficulty** so that a dedicated method is demonstrably necessary: we attack an agent that is
**protected by simple, realistic defenses**, and we show three things.

| Research question | Result the harness produces |
|---|---|
| **RQ1 — Do simple deployable defenses stop *naive* injection?** | `naive_static` scores well against the undefended agent but collapses to **~0** under a protective system prompt, a sandwich reminder, a regex/heuristic filter, or a layered `combo`. |
| **RQ2 — Does an *adaptive/trained* attacker recover the attack surface?** | `search` and `expert_iteration` (and a fine-tuned attacker) **recover most of it** by avoiding filtered patterns and reusing winning techniques. |
| **RQ3 — What is the quality-vs-cost trade-off, and how far do inference optimizations push it?** | Compute is a **first-class metric**: ASR vs attacker tokens, queries-to-success, and a KV-cache / quantization / batching / prefix-cache benchmark on the small attacker. |

This maps onto Basile's three suggested directions: **bypassing simple defenses**,
**expert iteration / preference training over successful-vs-failed attacks** (`collect` →
`sft` → `dpo` → `expert_iteration`), and **agentic setups where the agent's behaviour changes
without the user noticing** — our **stealthy-success** metric (the forbidden action happens
*and* the user's task still completes).

It also integrates the colleague's references: the **query-cost** limitation of RL red-teaming
(arXiv:2510.04885) motivates our efficiency angle, and the critique that **static defenses are
not enough** (arXiv:2510.09023) is exactly what the `naive → defended → adaptive` story shows.

## 2. Threat model & safety (read this)

Everything runs in an **in-memory sandbox**. The agent's tools (`send_email`, `read_document`,
…) act on a fake workspace; there is **no network, no real mailbox, no real secret**. An attack
*goal* is an **unauthorized action inside the sandbox** (e.g. exfiltrating a sandbox-only string
`sk-SANDBOX-…` to a honeypot address). The object of study is **control-flow hijacking of the
agent** (a security property), judged **only from sandbox state** — no injection here tries to
elicit harmful content. This is the standard shape of indirect-prompt-injection research
(Greshake et al. 2023; AgentDojo, Debenedetti et al. 2024), and the purpose is **defensive**:
measuring and improving robustness.

## 3. What's implemented

| Component | Options |
|---|---|
| **Target agent** | Plan-and-execute, tool-using agent over a sandbox workspace (`agent/`) |
| **Defenses** | `none`, `system_prompt`, `sandwich`, `spotlight` (datamarking), `regex_filter`, `ppl_filter`, `llm_guard` (broad heuristic guard), `model_guard` (learned HF classifier), `combo` (defense-in-depth) |
| **Attackers** | `naive_static`, `handcrafted`, `best_of_n` (inference-time sampling), `search` (adaptive refinement), `expert_iteration` (archive of winning attacks) |
| **Adaptation** | LoRA **SFT**, **DPO**, and **expert iteration** of the attacker against the sandbox reward (`training/`) → a fine-tuned attacker plugs back in via `attacker_model.adapter=…` |
| **Backends** | `sim` (deterministic, no GPU), `hf` (local Qwen/Llama + LoRA), `vllm` (offline engine), `api` (OpenAI / OpenRouter / vLLM-server / Ollama) |
| **Judge** | Ground-truth from sandbox state (not an LLM judge) → attack success + *stealthy* success |
| **Metrics** | ASR + Wilson CIs, stealthy ASR, **ASR@k**, queries-to-success, attacker/target tokens·time·$, **cost-per-success**, and defense **false-positive rate / utility** on benign docs |
| **Inference opt** | KV-cache, 4/8-bit quantization, batch size, prefix cache, shared-prompt sampling — benchmarked for tok/s and peak memory |
| **Task suites** | `default` (8 tasks), `hard` (benign text that *looks* instruction-like) — add more as JSON, no code |

## 4. Install

```bash
git clone <your-repo-url> injecteval && cd injecteval
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # core + tests/lint; runs the full sim pipeline, no GPU
```

For real experiments on your GPU (see [`docs/GPU_SETUP.md`](docs/GPU_SETUP.md)):

```bash
pip install -e ".[hf,quant]"       # local models + 4/8-bit (bitsandbytes, CUDA)
pip install -e ".[train]"          # attacker fine-tuning (TRL + PEFT)
pip install -e ".[api]"            # OpenAI-compatible endpoints
pip install -e ".[vllm]"           # offline vLLM engine (Linux/WSL2)
```

## 5. Quickstart — reproduce the demo (no GPU, seconds)

```bash
make demo            # == python scripts/run_demo.py
```

Runs the full `defense × attacker` grid on **both** suites, the **clean-document** evaluation,
writes `results/*.jsonl`, prints the tables, and renders every figure to `figures/`.
Representative `sim` output (attack success rate):

```
attacker          defense        asr    stealthy_asr   q_to_success
naive_static      none           0.38   0.00           1.0
naive_static      regex_filter   0.00   0.00           –        <- naive collapses
naive_static      combo          0.00   0.00           –
search            regex_filter   1.00   1.00           2.75     <- adaptive recovers
expert_iteration  regex_filter   1.00   1.00           1.50     <- fewer queries
best_of_n         llm_guard      0.12   0.00           19.0
search            combo          0.00   0.00           –        <- defense-in-depth holds
expert_iteration  combo          0.00   0.00           –
```

<p align="center">
  <img src="docs/sample_figures/fig_asr_heatmap.png" width="48%">
  <img src="docs/sample_figures/fig_defense_tradeoff.png" width="48%"><br>
  <img src="docs/sample_figures/fig_asr_at_k.png" width="48%">
  <img src="docs/sample_figures/fig_cost_tradeoff.png" width="48%">
</p>

> ⚠️ **The `sim` numbers are scaffolding, not evidence.** `sim` is an explicit, inspectable
> model of technique-vs-defense interactions (`models/features.py`) that exists so the whole
> pipeline is testable with no GPU. The **scientific** results for the report come from the
> `hf` / `vllm` / `api` backends, where the *identical* attacker/agent/judge code drives a real
> LLM and success is decided by real sandbox state. Always report real-model numbers.

## 6. Run with real models (your GPU)

```bash
# One target, small local models:
injecteval sweep --config config/default.yaml \
    --target-preset qwen15 --attacker-preset qwen05 --out results/qwen.jsonl

# FOUR target variants in one sweep (hits the top "extensiveness" rubric band):
injecteval sweep --config config/default.yaml \
    --target-presets qwen05 qwen15 qwen3b qwen3_4b \
    --attacker-preset qwen05 --out results/multi.jsonl

# OpenAI-compatible endpoint (OpenAI / OpenRouter / local vLLM server):
export OPENAI_API_KEY=sk-...        # or OPENROUTER_API_KEY, see models.yaml
injecteval sweep --config config/default.yaml \
    --target-backend api --target-model gpt-4o-mini \
    --attacker-backend api --attacker-model gpt-4o-mini --out results/api.jsonl

injecteval summary results/multi.jsonl --clean results/multi.clean.jsonl
injecteval figures results/multi.jsonl --clean results/multi.clean.jsonl --outdir figures
```

Presets (`config/models.yaml`): `qwen05`, `qwen3_06b`, `qwen15`, `qwen3b`, `qwen3_4b`,
`llama1b`, `llama3b`, `gpt4omini`, `openrouter_qwen`, `local_vllm`, `qwen05_vllm`. To test the
**learned guard**, add `model_guard` to `--defenses` (downloads a HF prompt-injection classifier).

## 7. Fine-tune the attacker — the *adaptation* (your GPU)

This is the "dedicated method" that answers RQ2, and Basile's expert-iteration / preference
direction. All training data is generated **from sandbox outcomes** — no external attack corpus.

```bash
# 1) Collect SFT + DPO data by running the adaptive attacker against defended agents.
injecteval collect --config config/default.yaml --attacker search --outdir data

# 2) LoRA supervised fine-tuning on the attempts that compromised the agent.
injecteval train sft  --base-model Qwen/Qwen2.5-0.5B-Instruct --data data/sft.jsonl \
    --out adapters/attacker-sft --quantization 4bit        # 4bit = QLoRA, fits a T4

# 3) DPO on (success > failure) preference pairs, warm-started from SFT.
injecteval train dpo  --base-model Qwen/Qwen2.5-0.5B-Instruct --data data/dpo.jsonl \
    --init-adapter adapters/attacker-sft --out adapters/attacker-dpo

# 4) Expert iteration: generate -> judge -> self-train, measured on HELD-OUT defenses.
injecteval train ei   --base-model Qwen/Qwen2.5-0.5B-Instruct --rounds 3 \
    --out adapters/attacker-ei                              # writes ei_history.json (learning curve)

# 5) Evaluate the fine-tuned attacker vs the baselines (same grid, new attacker weights):
injecteval sweep --config config/default.yaml --target-preset qwen15 \
    --attacker-preset qwen05 --attacker-adapter adapters/attacker-ei \
    --attackers search --out results/finetuned.jsonl
```

The headline comparison for the report: **base attacker vs SFT vs DPO vs expert-iteration**, each
as ASR / stealthy-ASR / queries-to-success against the defended agent, plus the EI **learning
curve** (`ei_history.json`).

## 8. Inference-optimization benchmark (the systems contribution)

```bash
injecteval bench --config config/default.yaml --attacker-preset qwen05 --n 64 --out results/bench.csv
python -c "from injecteval.figures import fig_bench; fig_bench('results/bench.csv','figures')"
```

Measures attacker throughput and peak memory under `{baseline, kv_cache, kv_prefix, kv_batch,
kv_batch_4bit}`. Framing: every optimization multiplies how many `(defense × attack)` cells you
can evaluate per unit compute → a **more thorough** robustness evaluation at fixed budget.
(Runs on `sim` too, but only `hf`/`vllm` timings are meaningful.)

## 9. Push it to GitHub from VS Code

```bash
cd injecteval
git init && git add . && git commit -m "InjectEval: automated red-teaming of defended LLM agents"
gh repo create injecteval --public --source=. --push      # or create the repo on github.com first:
# git remote add origin https://github.com/<you>/injecteval.git && git branch -M main && git push -u origin main
```

`results/`, `figures/*.png`, `data/`, `adapters/`, and model caches are git-ignored, so only the
code and configs are committed. CI (`.github/workflows/ci.yml`) runs lint + tests + the `sim`
demo on every push. In VS Code you can also use the **Source Control** panel (stage → commit →
"Publish Branch").

## 10. Repository layout

```
injecteval/
├── README.md · pyproject.toml · requirements.txt · Makefile · LICENSE
├── config/            default.yaml · models.yaml · training.yaml
├── src/injecteval/
│   ├── models/        backends: sim · hf (+LoRA, KV/quant/batch/prefix) · vllm · api; features.py (sim model)
│   ├── agent/         sandbox agent, tools, environment
│   ├── defenses/      system_prompt · sandwich · spotlight · filters · guards · combo
│   ├── attackers/     naive · handcrafted · best_of_n · search · expert_iteration
│   ├── training/      collect · sft · dpo · expert_iteration (LoRA adaptation)
│   ├── tasks/         Task suite + data/*.json
│   ├── judge.py · metrics.py · inference.py · experiment.py · figures.py · cli.py
├── scripts/run_demo.py
├── tests/test_pipeline.py      (29 tests, sim-only, no GPU)
└── docs/              architecture.svg · REPORT_OUTLINE.md · GPU_SETUP.md · YOUR_CONTRIBUTION.md
```

## 11. How it maps to the grading rubric

| Rubric criterion | Where |
|---|---|
| **Novelty of GAI application** | Automated red-teaming of a *defended* agent, with **stealth** (behaviour change the user doesn't notice) as an explicit objective. |
| **Difficulty of adaptation** | *Model adaptation*: LoRA **SFT + DPO + expert iteration** of the attacker against a programmatic reward, plus inference-only `search`/`expert_iteration`. *Inference-time optimization*: KV-cache / quantization / batching / prefix-cache on the attacker (`inference.py`). |
| **Extensiveness of evaluation** | Grid over **9 defenses × 5 attackers × ≥4 target models × 2 suites** with quality **and** cost metrics, Wilson CIs, ASR@k, and a defense FPR/utility analysis. |
| **Report & code repository** | This README, deterministic `make demo`, 29-test `pytest` suite, CI, clear module layout, config-driven experiments, report outline. |

See [`docs/REPORT_OUTLINE.md`](docs/REPORT_OUTLINE.md) for the 7–8 page ICML mapping and
[`docs/YOUR_CONTRIBUTION.md`](docs/YOUR_CONTRIBUTION.md) for the part your group authors.

## 12. Extending

- **New defense** — subclass `Defense` in `defenses/library.py`, add to `DEFENSE_REGISTRY`.
- **New attacker** — subclass `Attacker` in `attackers/library.py`, implement `run()`, register it.
- **New tasks** — drop a JSON file in `tasks/data/` (schema of `default.json`); no code.
- **New injection techniques** — your research contribution; see `docs/YOUR_CONTRIBUTION.md`.

## 13. References

- K. Greshake et al., *Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection*, 2023. arXiv:2302.12173
- E. Debenedetti et al., *AgentDojo: A Dynamic Environment to Evaluate Attacks and Defenses for LLM Agents*, 2024. https://github.com/ethz-spylab/agentdojo
- K. Hines et al., *Defending Against Indirect Prompt Injection Attacks With Spotlighting*, 2024. arXiv:2403.14720
- W. Zou et al., *PoisonedRAG*, 2024. arXiv:2402.07867
- E. Perez et al., *Red Teaming Language Models with Language Models*, 2022. arXiv:2202.03286
- *RL-based automated red-teaming & its query cost*, 2025. arXiv:2510.04885
- *On the (in)sufficiency of static defenses*, 2025. arXiv:2510.09023

## License

MIT — see `LICENSE`.
