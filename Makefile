.PHONY: install install-hf install-train demo sweep clean-eval figures bench collect train-sft train-dpo train-ei test lint clean

install:            ## core + dev (sim backend, no GPU)
	pip install -e ".[dev]"

install-hf:         ## + local models and 4/8-bit
	pip install -e ".[hf,quant,dev]"

install-train:      ## + attacker fine-tuning (TRL/PEFT)
	pip install -e ".[hf,quant,train,dev]"

demo:               ## full deterministic demo: grid + clean eval + figures (no GPU)
	python scripts/run_demo.py

sweep:              ## sim grid on the default suite
	injecteval sweep --config config/default.yaml --out results/run.jsonl

figures:            ## render figures from the demo run
	injecteval figures results/all.jsonl --clean results/all.clean.jsonl --outdir figures

bench:              ## inference microbenchmark (meaningful on hf/vllm + a GPU)
	injecteval bench --config config/default.yaml --attacker-preset qwen05 --out results/bench.csv

collect:            ## generate SFT + DPO data from sandbox outcomes
	injecteval collect --config config/default.yaml --attacker search --outdir data

train-sft:          ## LoRA SFT of the attacker (needs a GPU for real models)
	injecteval train sft --base-model Qwen/Qwen2.5-0.5B-Instruct --data data/sft.jsonl --out adapters/attacker-sft

train-dpo:          ## DPO of the attacker, warm-started from SFT
	injecteval train dpo --base-model Qwen/Qwen2.5-0.5B-Instruct --data data/dpo.jsonl --init-adapter adapters/attacker-sft --out adapters/attacker-dpo

train-ei:           ## expert iteration (generate -> judge -> self-train)
	injecteval train ei --base-model Qwen/Qwen2.5-0.5B-Instruct --rounds 3 --out adapters/attacker-ei

test:
	pytest -q

lint:
	ruff check src tests scripts

clean:
	rm -rf results figures data adapters runs *.egg-info src/*.egg-info .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
