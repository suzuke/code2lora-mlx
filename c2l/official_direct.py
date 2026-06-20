"""Faithful MLX re-implementation of the OFFICIAL Code2LoRA-direct head.

Mirrors `hypernetwork/code2lora_core.py` from the paper repo
(anonymous.4open.science/r/code2lora-6857) so we can run the official released
checkpoint `code2lora/code2lora-direct` (Qwen2.5-Coder-1.5B, rank 16, alpha 32).

Official forward (verified against state_dict + source):
    h = Linear(2048->1024)(ctx) ; GELU ; Linear(1024->1024) ; GELU      # exact erf GELU
    h = F.normalize(h, p=2) * sqrt(1024)
    A[t] = tanh( heads_A[t](h).view(rank, in_f) )  * clamp(exp(log_scale_A[t]), 1e-5, 0.3)
    B[t] = tanh( heads_B[t](h).view(out_f, rank) ) * clamp(exp(log_scale_B[t]), 1e-5, 0.3)
LoRA apply (per module, ALL layers share the same A/B):
    y = base(x) + (alpha/rank) * (x @ A.T) @ B.T        alpha/rank = 32/16 = 2.0
"""

import math

import mlx.core as mx
import mlx.nn as nn

# order matters only for iteration; dims come from the checkpoint config
TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ALPHA_OVER_RANK = 32.0 / 16.0  # = 2.0, the official LoRA `scaling`


class OfficialDirectHead:
    """Loads the official .pt and reproduces its forward in MLX (fp32)."""

    def __init__(self, ckpt_path: str):
        import torch

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck["state_dict"]
        cfg = ck["config"]
        self.rank = int(cfg["rank"])
        self.hidden = int(cfg["hidden_dim"])
        self.type_dims = cfg["type_dims"]  # {type: [in_f, out_f]}

        def m(name):
            return mx.array(sd[name].float().numpy())

        self.t0w, self.t0b = m("trunk.0.weight"), m("trunk.0.bias")
        self.t2w, self.t2b = m("trunk.2.weight"), m("trunk.2.bias")
        self.hAw = {t: m(f"heads_A.{t}.weight") for t in TYPES}
        self.hAb = {t: m(f"heads_A.{t}.bias") for t in TYPES}
        self.hBw = {t: m(f"heads_B.{t}.weight") for t in TYPES}
        self.hBb = {t: m(f"heads_B.{t}.bias") for t in TYPES}
        self.lsA = {t: float(sd[f"log_scale_A.{t}"]) for t in TYPES}
        self.lsB = {t: float(sd[f"log_scale_B.{t}"]) for t in TYPES}

    def forward(self, ctx):
        """ctx: mx.array shape (2048,). Returns {type: (A(rank,in), B(out,rank))}."""
        h = nn.gelu(ctx @ self.t0w.T + self.t0b)          # exact erf GELU
        h = nn.gelu(h @ self.t2w.T + self.t2b)
        h = h / mx.sqrt(mx.sum(h * h)) * math.sqrt(self.hidden)
        out = {}
        for t in TYPES:
            in_f, out_f = int(self.type_dims[t][0]), int(self.type_dims[t][1])
            a_raw = (h @ self.hAw[t].T + self.hAb[t]).reshape(self.rank, in_f)
            b_raw = (h @ self.hBw[t].T + self.hBb[t]).reshape(out_f, self.rank)
            sA = min(max(math.exp(self.lsA[t]), 1e-5), 0.3)
            sB = min(max(math.exp(self.lsB[t]), 1e-5), 0.3)
            out[t] = (mx.tanh(a_raw) * sA, mx.tanh(b_raw) * sB)
        return out
