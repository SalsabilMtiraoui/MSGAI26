"""Direct Preference Optimization of the attacker model.

Preference signal is the ground-truth sandbox judge: for a given (task, defense),
an attempt that compromised the agent is *chosen* over one that did not
(:func:`injecteval.training.collect.to_dpo`). This is the "preference training
over successful vs. failed attacks" direction from the project brief, used to
measure how much a cheap preference step raises defense robustness pressure.

Thin wrapper over ``trl.DPOTrainer`` + LoRA. Requires ``pip install -e ".[train]"``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DPOArgs:
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    data: str = "data/dpo.jsonl"
    out: str = "adapters/attacker-dpo"
    init_adapter: Optional[str] = None   # e.g. start from the SFT adapter
    epochs: float = 2.0
    lr: float = 5e-6
    beta: float = 0.1
    batch_size: int = 4
    grad_accum: int = 2
    max_length: int = 768
    max_prompt_length: int = 512
    lora_r: int = 16
    lora_alpha: int = 32
    quantization: str = "none"
    seed: int = 0


def _read_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def train_dpo(a: DPOArgs) -> str:
    import torch  # noqa: F401
    from datasets import Dataset
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    from ..models.hf_backend import load_model_and_tokenizer

    model, tok = load_model_and_tokenizer(
        a.base_model, quantization=a.quantization, adapter=a.init_adapter, trainable=True)
    if a.quantization not in (None, "none"):
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    rows = _read_jsonl(a.data)
    if not rows:
        raise SystemExit(f"No DPO pairs in {a.data}; run `injecteval collect` first.")

    def render(r):
        prompt = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "chosen": r["chosen"], "rejected": r["rejected"]}

    ds = Dataset.from_list([render(r) for r in rows])

    peft_cfg = None if a.init_adapter else LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear")

    from .sft import _precision
    bf16, fp16 = _precision()
    cfg = DPOConfig(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, beta=a.beta,
        per_device_train_batch_size=a.batch_size, gradient_accumulation_steps=a.grad_accum,
        max_length=a.max_length, max_prompt_length=a.max_prompt_length,
        logging_steps=5, save_strategy="no", report_to="none", seed=a.seed,
        bf16=bf16, fp16=fp16)
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    print(f"DPO adapter saved -> {a.out}  ({len(rows)} pairs)")
    return a.out
