"""Hypernetwork — faithful MLX port of code2lora-lite/src/hypernetwork.rs.

repo_embedding (1, 768) -> shared MLP -> per-layer offset (layer embedding)
-> per-module heads -> (A, B) LoRA pairs for all 24 layers x 7 modules.

Generation per pair (matches Rust gen_pair):
    a = tanh(head_a(h)).reshape(rank, in_dim)  * exp(log_scale_a)
    b = tanh(head_b(h)).reshape(out_dim, rank) * exp(log_scale_b)
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .config import MODULE_TYPES, HypernetworkConfig


class Code2LoRAHead(nn.Module):
    def __init__(self, cfg: HypernetworkConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim

        # Shared MLP (Rust: linear "mlp.0" then "mlp.2", gelu in between).
        self.linear1 = nn.Linear(cfg.repo_embed_dim, h)
        self.linear2 = nn.Linear(h, h)

        # Per-layer learned offset added to the shared base.
        self.layer_emb = nn.Embedding(cfg.num_layers, h)

        # Per-module heads + learnable scalar log-scales.
        # Stored in dict attributes; MLX nn.Module tracks params inside dicts.
        heads_a, heads_b, log_scale_a, log_scale_b = {}, {}, {}, {}
        for m in MODULE_TYPES:
            in_dim = cfg.lora_in_dim(m)
            out_dim = cfg.lora_out_dim(m)
            heads_a[m] = nn.Linear(h, cfg.rank * in_dim)   # -> (rank, in_dim)
            heads_b[m] = nn.Linear(h, out_dim * cfg.rank)  # -> (out_dim, rank)
            log_scale_a[m] = mx.zeros((1,))
            log_scale_b[m] = mx.zeros((1,))
        self.heads_a = heads_a
        self.heads_b = heads_b
        self.log_scale_a = log_scale_a
        self.log_scale_b = log_scale_b

    def _base(self, repo_emb: mx.array) -> mx.array:
        h = self.linear1(repo_emb)
        # Candle's Tensor::gelu() is the *tanh approximation* (not erf-exact),
        # so use nn.gelu_approx for faithfulness (verified vs candle-core 0.10.2).
        h = nn.gelu_approx(h)
        h = self.linear2(h)
        # L2-normalize along the feature axis, then scale by sqrt(hidden_dim).
        norm = mx.sqrt(mx.sum(h * h, axis=1, keepdims=True))
        base = h / norm
        base = base * math.sqrt(self.cfg.hidden_dim)
        return base

    def _scale(self, log_scale: mx.array) -> mx.array:
        # exp(log_scale), but clamp log_scale <= log(max_lora_scale) so the
        # effective scale can never blow up (the 10-repo run hit ~2432). The clamp
        # is one-sided: below the cap gradients flow normally; at/above the cap the
        # value is pinned. (TUNING FIX.)
        cap = getattr(self.cfg, "max_lora_scale", 0.0)
        if cap and cap > 0.0:
            log_scale = mx.minimum(log_scale, math.log(cap))
        return mx.exp(log_scale)

    def _gen_pair(self, h: mx.array, m: str):
        in_dim = self.cfg.lora_in_dim(m)
        out_dim = self.cfg.lora_out_dim(m)
        rank = self.cfg.rank
        a = self.heads_a[m](h).reshape(rank, in_dim)
        b = self.heads_b[m](h).reshape(out_dim, rank)
        a = mx.tanh(a) * self._scale(self.log_scale_a[m])
        b = mx.tanh(b) * self._scale(self.log_scale_b[m])
        return a, b

    def forward_all(self, repo_emb: mx.array):
        """Return list[num_layers] of {module: (A, B)} dicts.

        A: (rank, in_dim), B: (out_dim, rank) — same convention as Candle.
        """
        base = self._base(repo_emb)  # (1, hidden_dim)
        layers = []
        for li in range(self.cfg.num_layers):
            le = self.layer_emb(mx.array([li]))  # (1, hidden_dim)
            h_layer = base + le
            layers.append({m: self._gen_pair(h_layer, m) for m in MODULE_TYPES})
        return layers

    def __call__(self, repo_emb: mx.array):
        return self.forward_all(repo_emb)
