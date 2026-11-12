# Running InjectEval on your own GPU

The `sim` backend needs nothing. This page is for the **real-model** experiments and the
**attacker fine-tuning** — the numbers that go in the report.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

# Install PyTorch for YOUR CUDA (see https://pytorch.org/get-started/locally/).
# Example, CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Then the project extras: personnaly I didn't put quant because of my limited gpu
pip install -e ".[hf,quant,train,dev]"
```

Check the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> **bitsandbytes (4/8-bit) is CUDA-only and does not run natively on Windows.** On Windows use
> **WSL2 + CUDA**, or skip quantization (`--quantization none`) and use a smaller model. On
> Apple Silicon there is no bitsandbytes; run fp16/fp32 small models on `mps` (set
> `dtype: float16`) or use the `api` backend.

## 2. Which models fit your VRAM

Both the target *and* the small attacker must fit (the attacker is tiny). Rough guide for
inference; add ~1.5–2× for LoRA training (or use `--quantization 4bit` = QLoRA).

| GPU VRAM | Target preset | Attacker preset | Notes |
|---|---|---|---|
| 6–8 GB | `qwen05`, `qwen3_06b`, `qwen15` | `qwen05` | use `--quantization 4bit` for ≥1.5B |
| 10–12 GB | `qwen3b`, `qwen3_4b`, `llama3b` | `qwen05` | fp16 target + fp16 attacker is fine |
| 16–24 GB | up to `qwen3_4b` fp16 + training | `qwen05` | full LoRA SFT/DPO/EI comfortable |
| Colab T4 (15 GB) | `qwen15` / `qwen3b` 4bit | `qwen05` | the whole fine-tuning loop runs here |

Qwen3 models default to "thinking" mode; the presets pass `enable_thinking: false` so the agent
emits a plain JSON plan. Llama presets need a HuggingFace account with access granted.

## 3. The full report pipeline (copy-paste) 

```bash
# (a) Baselines + defenses on 4 target variants -> extensiveness band
injecteval sweep --config config/default.yaml \
  --target-presets qwen05 qwen15 qwen3b qwen3_4b --attacker-preset qwen05 \
  --out results/multi.jsonl
Window:
  injecteval sweep --config config/default.yaml --target-presets qwen05 qwen3_06b qwen15 qwen3b --attacker-preset qwen05 --out results/multi.jsonl

# (b) Fine-tune the attacker (adaptation)
injecteval collect --config config/default.yaml --attacker search --outdir data
injecteval train sft --base-model Qwen/Qwen2.5-0.5B-Instruct --data data/sft.jsonl \
  --out adapters/attacker-sft --quantization 4bit
injecteval train dpo --base-model Qwen/Qwen2.5-0.5B-Instruct --data data/dpo.jsonl \
  --init-adapter adapters/attacker-sft --out adapters/attacker-dpo
injecteval train ei  --base-model Qwen/Qwen2.5-0.5B-Instruct --rounds 3 --out adapters/attacker-ei
Window:
injecteval collect --config config/default.yaml --attacker search --outdir data
injecteval train sft --base-model Qwen/Qwen2.5-0.5B-Instruct --data data/sft.jsonl --out adapters/attacker-sft
injecteval train dpo --base-model Qwen/Qwen2.5-0.5B-Instruct --data data/dpo.jsonl --init-adapter adapters/attacker-sft --out adapters/attacker-dpo
injecteval train ei --base-model Qwen/Qwen2.5-0.5B-Instruct --rounds 3 --out adapters/attacker-ei

# (c) Evaluate base vs SFT vs DPO vs EI attacker
for A in "" "--attacker-adapter adapters/attacker-sft" \
         "--attacker-adapter adapters/attacker-dpo" "--attacker-adapter adapters/attacker-ei"; do
  injecteval sweep --config config/default.yaml --target-preset qwen15 \
    --attacker-preset qwen05 $A --attackers search --out results/atk_$(echo $A | md5sum | cut -c1-6).jsonl
done
Window: 
injecteval sweep --config config/default.yaml --target-preset qwen15 --attacker-preset qwen05 --attackers search --out results/atk_base.jsonl
injecteval sweep --config config/default.yaml --target-preset qwen15 --attacker-preset qwen05 --attacker-adapter adapters/attacker-sft --attackers search --out results/atk_sft.jsonl

# (d) Inference optimization benchmark (systems contribution)
injecteval bench --config config/default.yaml --attacker-preset qwen05 --n 64 --out results/bench.csv

# (e) Figures for the report
injecteval figures results/multi.jsonl --clean results/multi.clean.jsonl --outdir figures
python -c "from injecteval.figures import fig_bench; fig_bench('results/bench.csv','figures')"
```

## 4. Troubleshooting

- **CUDA out of memory** — lower `--quantization 4bit`, reduce `batch_size` in the config, or
  pick a smaller preset. Training: lower `per_device_train_batch_size`, raise `grad_accum`.
- **`bitsandbytes` import error on Windows** — use WSL2, or run without `quant`.
- **vLLM pre-allocates too much memory** — lower `gpu_memory_utilization` in the preset so the
  target and attacker share one GPU.
- **Slow first call** — model load + warm-up; the bench excludes warm-up from its timings.
- **Flash-Attention** — optional; set `attn_implementation: flash_attention_2` in a preset if you
  installed `flash-attn`, otherwise the default SDPA attention is used.
