# The part your group authors (and how to run everything)

The harness in this repo is **complete and tested**: agent, defenses, attackers, judge,
metrics, figures, the three fine-tuning methods, the inference benchmark, and the `sim`
backend for development. What turns it into *your* graded project is the **research
contribution and the empirical study** — which is exactly what the rubric rewards and what you
should align with Basile. None of this requires writing new infrastructure; it is analysis and
targeted extension.

## A. Your research contribution (pick one or two, discuss with Basile)

1. **Extend the injection-technique taxonomy.** `src/injecteval/models/features.py` is the
   *student-authored* catalogue of indirect-injection techniques used by the `sim` model, and
   `attackers/library.py::HandcraftedAttacker` is the seed library used on real models. Propose
   and add techniques (new `Feature` entries + how each defense `suppress`/`detect`s them), then
   **measure** whether they raise ASR against the defended agent *while keeping stealth*. This is
   the "discover new attacks" direction, made measurable.
   - Validate on `sim`: `make demo` and read the ASR heatmap + `fig_technique_hist`.
   - Then confirm on a real model (`--target-preset qwen15`).

2. **Study the training methods.** Compare **base vs SFT vs DPO vs expert-iteration** attackers
   (README §7). Does preference training (DPO) beat behaviour cloning (SFT)? Does expert
   iteration **generalize to held-out defenses** (`ei_history.json` learning curve)? This is the
   preference-training direction Basile suggested.

3. **Defense robustness & the security/utility trade-off.** Which defenses actually stop the
   *adaptive* attacker, and at what **false-positive cost** on benign documents
   (`fig_defense_tradeoff`)? Is `combo` (defense-in-depth) worth its latency? Compare the learned
   `model_guard` against the cheap `regex`/`llm_guard` filters.

4. **Efficiency.** Under a **fixed compute budget**, does a 4-bit attacker with larger best-of-N
   beat an fp16 attacker with small N (`bench` + ASR@k)? This is the course's core
   quality-vs-cost question.

> Keep everything in the sandbox and defensive: the deliverable is *measurements of defense
> robustness*, never a deployable attack tool. Every success is judged from fake sandbox state.

## B. What to write in the report (7–8 pages)

Follow [`REPORT_OUTLINE.md`](REPORT_OUTLINE.md). The figures you need are produced by
`make demo` (sim) and by the real-model commands in [`GPU_SETUP.md`](GPU_SETUP.md):
`fig_asr_heatmap`, `fig_stealthy_bars`, `fig_cost_tradeoff`, `fig_asr_at_k`,
`fig_technique_hist`, `fig_defense_tradeoff`, `fig_inference_bench`, and the
base/SFT/DPO/EI comparison table.

## C. Minimal division of labour (3 people)

| Person | Focus |
|---|---|
| A | Attackers + fine-tuning (collect/SFT/DPO/EI), the base-vs-adapted comparison |
| B | Defenses + clean-document FPR/utility study, the learned `model_guard` |
| C | Inference optimization benchmark + multi-model evaluation + figures/report |

Each axis is independently runnable, so you can work in parallel and merge JSONL results.

## D. Smoke test before you rely on the GPU

```bash
pip install -e ".[dev]" && pytest -q && make demo      # must pass with no GPU
pip install -e ".[hf,quant,train]"                     # then the real runs in GPU_SETUP.md
```

If `pytest` is green and `make demo` writes figures, the pipeline is intact and any real-model
failure is an environment issue (CUDA / VRAM / model access), not the code — see
[`GPU_SETUP.md`](GPU_SETUP.md) §4.
