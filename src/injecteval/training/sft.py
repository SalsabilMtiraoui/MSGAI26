"""LoRA supervised fine-tuning of the attacker model (behaviour cloning).

Trains a small instruct model to reproduce the attack attempts that compromised
the sandbox agent (from :mod:`injecteval.training.collect`). Produces a LoRA
adapter that plugs back into any attacker strategy via
``attacker_model.adapter=<path>``.

Thin wrapper over ``trl.SFTTrainer`` + ``peft``; kept deliberately small and
version-tolerant. Requires ``pip install -e ".[train]"``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SFTArgs:
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    data: str = "data/sft.jsonl"
    out: str = "adapters/attacker-sft"
    epochs: float = 3.0
    lr: float = 2e-4
    batch_size: int = 8
    grad_accum: int = 1
    max_length: int = 768
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    quantization: str = "none"       # "4bit" -> QLoRA
    target_modules: Optional[List[str]] = None
    seed: int = 0


def _read_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def _peft_config(a: SFTArgs):
    from peft import LoraConfig
    return LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=a.target_modules or "all-linear")


def _precision():
    """(bf16, fp16) flags matching the current hardware (fp32 on CPU).

    bf16 only on compute capability >= 8.0 (Ampere+); older CUDA GPUs (e.g.
    RTX 20xx / Turing, cc 7.5) use fp16 -- real bf16 matmuls crash there."""
    import torch

    from ..models.hf_backend import bf16_supported
    if torch.cuda.is_available():
        return (True, False) if bf16_supported(torch) else (False, True)
    return (False, False)


def train_sft(a: SFTArgs) -> str:
    import torch  # noqa: F401
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    from ..models.hf_backend import load_model_and_tokenizer

    model, tok = load_model_and_tokenizer(
        a.base_model, quantization=a.quantization, trainable=True)
    if a.quantization not in (None, "none"):
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    rows = _read_jsonl(a.data)
    if not rows:
        raise SystemExit(f"No SFT rows in {a.data}; run `injecteval collect` first.")

    def render(r):
        msgs = r["messages"] + [{"role": "assistant", "content": r["completion"]}]
        return {"text": tok.apply_chat_template(msgs, tokenize=False)}

    ds = Dataset.from_list([render(r) for r in rows])

    bf16, fp16 = _precision()
    cfg = SFTConfig(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.batch_size, gradient_accumulation_steps=a.grad_accum,
        max_length=a.max_length, logging_steps=5, save_strategy="no",
        report_to="none", seed=a.seed, dataset_text_field="text",
        gradient_checkpointing=False, bf16=bf16, fp16=fp16)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=_peft_config(a))
    trainer.train()
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    print(f"SFT adapter saved -> {a.out}  ({len(rows)} examples)")
    return a.out
