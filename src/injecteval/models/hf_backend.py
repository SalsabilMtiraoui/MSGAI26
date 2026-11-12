"""Local open-weight backend via Hugging Face ``transformers`` (+ ``peft``).

Inference-optimization knobs measured in the systems experiments:

* ``use_kv_cache``  -> ``model.generate(use_cache=...)``
* ``quantization``  -> 4-bit NF4 / 8-bit weights via ``bitsandbytes``
* ``batch_size``    -> left-padded batched generation
* ``sample_n``      -> ``num_return_sequences``: the shared prompt is encoded
                       once and its KV cache is shared by the N samples
* ``prefix_cache``  -> the KV cache of a fixed system prompt is computed once
                       and re-used for every request (batch size 1)

Adaptation: ``adapter`` loads a LoRA adapter produced by
:mod:`injecteval.training` on top of the base model.

Install with ``pip install -e ".[hf]"`` (and ``".[quant]"`` for 4/8-bit).
"""

from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional

from .backend import Backend, GenConfig, Message

_DTYPES = {"float16": "float16", "fp16": "float16", "bfloat16": "bfloat16",
           "bf16": "bfloat16", "float32": "float32", "fp32": "float32"}


def bf16_supported(torch) -> bool:
    """True only on GPUs with *hardware* bf16 (compute capability >= 8.0:
    Ampere/Ada/Hopper). ``torch.cuda.is_bf16_supported()`` also returns True on
    older cards (Turing, e.g. RTX 20xx) via emulation, which then crashes real
    bf16 matmuls -- so we gate on the compute capability instead."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def _torch_dtype(torch, name: str):
    if name in (None, "auto"):
        if torch.cuda.is_available():
            return torch.bfloat16 if bf16_supported(torch) else torch.float16
        return torch.float32
    return getattr(torch, _DTYPES.get(name, name))


def load_model_and_tokenizer(model_id: str, quantization: str = "none", dtype: str = "auto",
                             adapter: Optional[str] = None, attn_implementation=None,
                             trainable: bool = False):
    """Shared loader used by the inference backend *and* the trainers."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Guard: a local adapter path that doesn't exist would otherwise be sent to
    # the HF Hub and fail with a confusing 404. Catch it with a clear message.
    if adapter:
        looks_local = (os.path.isabs(adapter)
                       or adapter.replace("\\", "/").startswith(("adapters/", "runs/", "./", "../")))
        if looks_local and not os.path.isdir(adapter):
            raise FileNotFoundError(
                f"Adapter folder '{adapter}' was not found. Train it first, e.g.\n"
                f"    injecteval train sft --base-model {model_id} --data data/sft.jsonl --out {adapter}\n"
                f"Each training run writes an '{adapter}/adapter_config.json'. If a previous "
                f"training step errored (e.g. 4-bit/bitsandbytes on Windows), re-run it WITHOUT "
                f"--quantization 4bit so the adapter is actually saved.")

    tok_src = adapter if adapter and os.path.exists(os.path.join(adapter, "tokenizer_config.json")) \
        else model_id
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    kwargs: Dict = {"torch_dtype": _torch_dtype(torch, dtype)}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if quantization not in (None, "none"):
        from transformers import BitsAndBytesConfig
        if quantization == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=kwargs["torch_dtype"], bnb_4bit_use_double_quant=True)
        else:
            raise ValueError(f"Unknown quantization: {quantization}")

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
        if not trainable and quantization in (None, "none"):
            model = model.merge_and_unload()  # zero adapter overhead at inference
    if not trainable:
        model.eval()
    return model, tokenizer


class HFBackend(Backend):
    name = "hf"

    def __init__(self, cfg: GenConfig):
        super().__init__(cfg)
        try:
            import torch
        except ImportError as e:  # pragma: no cover
            raise ImportError('HFBackend needs: pip install -e ".[hf]"') from e
        self.torch = torch
        self.model, self.tokenizer = load_model_and_tokenizer(
            cfg.model, cfg.quantization, cfg.dtype, cfg.adapter, cfg.attn_implementation)
        self.device = next(self.model.parameters()).device
        self._prefix_cache: Dict[str, tuple] = {}
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    # -- tokenization --------------------------------------------------------
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _render(self, messages: List[Message], cfg: GenConfig) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **(cfg.chat_template_kwargs or {}))
        except Exception:  # tokenizer without a chat template
            body = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
            return body + "\nASSISTANT:"

    def _gen_kwargs(self, cfg: GenConfig) -> dict:
        kw = dict(max_new_tokens=cfg.max_new_tokens, use_cache=cfg.use_kv_cache,
                  pad_token_id=self.tokenizer.pad_token_id)
        if cfg.temperature and cfg.temperature > 0:
            kw.update(do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        return kw

    def _encode(self, prompts, **kw):
        enc = self.tokenizer(prompts, return_tensors="pt", add_special_tokens=False, **kw)
        enc.pop("token_type_ids", None)  # some fast tokenizers emit this; generate() rejects it
        return enc.to(self.device)

    def _decode(self, out, prompt_len: int) -> List[str]:
        gen = out[:, prompt_len:]
        return [t.strip() for t in self.tokenizer.batch_decode(gen, skip_special_tokens=True)]

    # -- generation ----------------------------------------------------------
    def _generate_batch(self, batch: List[List[Message]], cfg: GenConfig) -> List[str]:
        if cfg.prefix_cache and cfg.use_kv_cache:
            return [self._generate_with_prefix_cache(m, cfg) for m in batch]
        torch = self.torch
        prompts = [self._render(m, cfg) for m in batch]
        outputs: List[str] = []
        bs = max(1, cfg.batch_size)
        for i in range(0, len(prompts), bs):
            enc = self._encode(prompts[i:i + bs], padding=True)
            with torch.inference_mode():
                out = self.model.generate(**enc, **self._gen_kwargs(cfg))
            outputs.extend(self._decode(out, enc["input_ids"].shape[1]))
        return outputs

    def _sample_n(self, messages: List[Message], n: int, cfg: GenConfig) -> List[str]:
        torch = self.torch
        enc = self._encode([self._render(messages, cfg)])
        kw = self._gen_kwargs(cfg)
        outs: List[str] = []
        chunk = max(1, cfg.batch_size)
        for start in range(0, n, chunk):
            k = min(chunk, n - start)
            with torch.inference_mode():
                out = self.model.generate(**enc, num_return_sequences=k, **kw)
            outs.extend(self._decode(out, enc["input_ids"].shape[1]))
        return outs

    def _generate_with_prefix_cache(self, messages: List[Message], cfg: GenConfig) -> str:
        """Re-use the KV cache of the (fixed) system prompt across requests."""
        torch = self.torch
        full = self._render(messages, cfg)
        sys_msgs = [m for m in messages if m["role"] == "system"]
        prefix = self._render(sys_msgs, cfg) if sys_msgs else ""
        # Rendering with add_generation_prompt appends an assistant header; keep
        # only the part that is a true string prefix of the full prompt.
        while prefix and not full.startswith(prefix):
            prefix = prefix[:-1]
        if len(prefix) < 32:
            enc = self.tokenizer([full], return_tensors="pt", add_special_tokens=False).to(self.device)
            with torch.inference_mode():
                out = self.model.generate(**enc, **self._gen_kwargs(cfg))
            return self._decode(out, enc["input_ids"].shape[1])[0]

        if prefix not in self._prefix_cache:
            from transformers import DynamicCache
            p_ids = self.tokenizer([prefix], return_tensors="pt",
                                   add_special_tokens=False).input_ids.to(self.device)
            cache = DynamicCache()
            with torch.inference_mode():
                self.model(input_ids=p_ids, past_key_values=cache, use_cache=True)
            self._prefix_cache = {prefix: (p_ids.shape[1], cache)}  # keep one entry
        p_len, cache = self._prefix_cache[prefix]
        ids = self.tokenizer([full], return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        if ids.shape[1] <= p_len:  # tokenization boundary mismatch -> no reuse
            with torch.inference_mode():
                out = self.model.generate(input_ids=ids, **self._gen_kwargs(cfg))
            return self._decode(out, ids.shape[1])[0]
        with torch.inference_mode():
            out = self.model.generate(input_ids=ids, past_key_values=copy.deepcopy(cache),
                                      **self._gen_kwargs(cfg))
        return self._decode(out, ids.shape[1])[0]

    # -- housekeeping --------------------------------------------------------
    def peak_memory_gb(self) -> Optional[float]:
        if self.torch.cuda.is_available():
            return round(self.torch.cuda.max_memory_allocated() / 1e9, 3)
        return None

    def close(self) -> None:
        import gc
        self._prefix_cache.clear()
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
