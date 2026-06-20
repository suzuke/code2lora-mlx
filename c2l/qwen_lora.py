"""Inject hypernetwork-generated LoRA into a frozen mlx-lm Qwen2.5-Coder-0.5B.

LoRA apply — faithful to code2lora-lite/src/qwen2_lora.rs LoRALinear.forward
(NO extra alpha/rank scaling; the scale is baked into A/B via the hypernetwork's
exp(log_scale)):

    y = base(x) + (x @ A.T) @ B.T        A:(rank, in)   B:(out, rank)
"""

import mlx.core as mx
import mlx.nn as nn

from .config import MODULE_TYPES  # noqa: F401  (kept for parity/iteration)

BASE_MODEL = "Qwen/Qwen2.5-Coder-0.5B"

# LoRA module type -> (submodule, attribute) on an mlx-lm Qwen2 decoder layer.
_PROJ = {
    "q": ("self_attn", "q_proj"),
    "k": ("self_attn", "k_proj"),
    "v": ("self_attn", "v_proj"),
    "o": ("self_attn", "o_proj"),
    "gate": ("mlp", "gate_proj"),
    "up": ("mlp", "up_proj"),
    "down": ("mlp", "down_proj"),
}


class LoRAWrap(nn.Module):
    """Wraps a base Linear; adds `(x @ A.T) @ B.T` when a LoRA pair is set."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self._a = None  # (rank, in)
        self._b = None  # (out, rank)

    def set_lora(self, a: mx.array, b: mx.array):
        # A:(rank,in) B:(out,rank) — ranks must match (cheap guard; full in/out
        # dims are validated at matmul time against the base weight).
        if a.shape[0] != b.shape[1]:
            raise ValueError(f"LoRA rank mismatch: A{tuple(a.shape)} B{tuple(b.shape)}")
        self._a, self._b = a, b

    def clear_lora(self):
        self._a = self._b = None

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base(x)
        if self._a is not None and self._b is not None:
            a = self._a.astype(x.dtype)
            b = self._b.astype(x.dtype)
            y = y + (x @ a.T) @ b.T
        return y


def _layers(model):
    inner = getattr(model, "model", model)
    return inner.layers


def load_base_model(model_id: str = BASE_MODEL):
    """Load the frozen Qwen and wrap every q/k/v/o/gate/up/down projection."""
    from mlx_lm import load

    model, tokenizer = load(model_id)
    for layer in _layers(model):
        for _mt, (sub, attr) in _PROJ.items():
            submod = getattr(layer, sub)
            setattr(submod, attr, LoRAWrap(getattr(submod, attr)))
    return model, tokenizer


def inject_lora(model, all_lora):
    """all_lora: list[num_layers] of {module_type: (A, B)} from the hypernetwork."""
    layers = _layers(model)
    if len(all_lora) != len(layers):
        raise ValueError(f"adapter layers {len(all_lora)} != model layers {len(layers)}")
    for layer, lora in zip(layers, all_lora):
        for mt, (sub, attr) in _PROJ.items():
            a, b = lora[mt]
            getattr(getattr(layer, sub), attr).set_lora(a, b)


def clear_lora(model):
    for layer in _layers(model):
        for _mt, (sub, attr) in _PROJ.items():
            getattr(getattr(layer, sub), attr).clear_lora()


def generate_text(model, tokenizer, prompt: str, max_tokens: int = 64) -> str:
    """Greedy completion (base model, raw prompt — no chat template)."""
    from mlx_lm import generate

    try:
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=0.0)  # greedy, deterministic
        return generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens,
            sampler=sampler, verbose=False,
        )
    except TypeError:
        # older/newer mlx-lm generate signatures
        return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
