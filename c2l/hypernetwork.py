"""Hypernetwork — trainable MLX twin of the OFFICIAL `Code2LoRAHead`.

Mirrors `hypernetwork/code2lora_core.py` from the official anonymous repo
(anonymous.4open.science/r/code2lora-6857). Architecture:

    repo_state_embedding (1, repo_embed_dim)
      -> Linear -> GELU -> Linear -> GELU                    (exact erf GELU)
      -> h = L2norm(h) * sqrt(hidden_dim)
      -> per module type:
           A = tanh(head_A(h)).reshape(rank, in_f)  * clamp(exp(log_scale_A), 1e-5, 0.3)
           B = tanh(head_B(h)).reshape(out_f, rank) * clamp(exp(log_scale_B), 1e-5, 0.3)

ONE (A, B) pair per module type, SHARED across ALL decoder layers — there is no
per-layer embedding (that was the third-party Candle reimpl's design). `forward_all`
returns the same adapter replicated `num_layers` times so the injection path is
unchanged. This is the trainable counterpart of `official_direct.OfficialDirectHead`
(which loads the released weights for inference).
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

        # Exact-GELU trunk (official: Linear -> GELU -> Linear -> GELU).
        self.linear1 = nn.Linear(cfg.repo_embed_dim, h)
        self.linear2 = nn.Linear(h, h)

        # Per-module heads + learnable scalar log-scales.
        # init_log_scale=-3.5 => exp(-3.5)~=0.03, a tiny LoRA delta at init that the
        # head must *earn* magnitude for (official behaviour). MLX nn.Module tracks
        # params stored inside dict attributes.
        heads_a, heads_b, log_scale_a, log_scale_b = {}, {}, {}, {}
        for m in MODULE_TYPES:
            in_dim = cfg.lora_in_dim(m)
            out_dim = cfg.lora_out_dim(m)
            heads_a[m] = nn.Linear(h, cfg.rank * in_dim)   # -> (rank, in_dim)
            heads_b[m] = nn.Linear(h, out_dim * cfg.rank)  # -> (out_dim, rank)
            log_scale_a[m] = mx.full((1,), cfg.init_log_scale)
            log_scale_b[m] = mx.full((1,), cfg.init_log_scale)
        self.heads_a = heads_a
        self.heads_b = heads_b
        self.log_scale_a = log_scale_a
        self.log_scale_b = log_scale_b

    def _trunk(self, repo_emb: mx.array) -> mx.array:
        # Exact erf GELU (official uses nn.GELU, NOT the tanh approximation).
        h = nn.gelu(self.linear1(repo_emb))
        h = nn.gelu(self.linear2(h))
        # L2-normalize along the feature axis, then scale by sqrt(hidden_dim).
        norm = mx.sqrt(mx.sum(h * h, axis=1, keepdims=True))
        return h / norm * math.sqrt(self.cfg.hidden_dim)

    def _scale(self, log_scale: mx.array) -> mx.array:
        # Official: clamp(exp(log_scale), 1e-5, 0.3) — bounds the LoRA magnitude so
        # a bad adapter can never blow up (the prior from-scratch run hit ~2432).
        return mx.clip(
            mx.exp(log_scale), self.cfg.scale_clamp_min, self.cfg.scale_clamp_max
        )

    def _gen_pair(self, h: mx.array, m: str):
        in_dim = self.cfg.lora_in_dim(m)
        out_dim = self.cfg.lora_out_dim(m)
        rank = self.cfg.rank
        a = self.heads_a[m](h).reshape(rank, in_dim)
        b = self.heads_b[m](h).reshape(out_dim, rank)
        a = mx.tanh(a) * self._scale(self.log_scale_a[m])
        b = mx.tanh(b) * self._scale(self.log_scale_b[m])
        return a, b

    def forward_shared(self, repo_emb: mx.array):
        """The single {module: (A, B)} adapter, shared across all layers (official).

        A: (rank, in_dim), B: (out_dim, rank).
        """
        h = self._trunk(repo_emb)  # (1, hidden_dim)
        return {m: self._gen_pair(h, m) for m in MODULE_TYPES}

    def forward_all(self, repo_emb: mx.array):
        """list[num_layers] of the SAME shared adapter (official architecture)."""
        adapter = self.forward_shared(repo_emb)
        return [adapter for _ in range(self.cfg.num_layers)]

    def __call__(self, repo_emb: mx.array):
        return self.forward_all(repo_emb)
