"""MLX port of the official CommitGRU (Code2LoRA-Evo front-end), faithful to
hypernetwork/code2lora_core.py:
  diff_proj      = Linear(2048->2048) -> LayerNorm
  repo_init_proj = Linear(2048->2048) -> GELU -> LayerNorm
  gru            = 1-layer nn.GRU(2048, 2048)            # torch gate order r,z,n
  output_norm    = LayerNorm(2048)
Rollout: h = repo_init_proj(repo_state_emb_0); for diff: h = gru_step(diff_proj(diff), h)
         ctx_t = output_norm(h)  ->  fed into the SAME Code2LoRAHead (input_dim 2048)
"""

import mlx.core as mx
import mlx.nn as nn


def _ln(x, w, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / mx.sqrt(var + eps) * w + b


class EvoGRU:
    """Holds GRU+proj params as mx arrays (loaded from code2lora_gru.pt)."""

    def __init__(self, P, hidden=2048):
        self.P = P
        self.H = hidden

    def _diff_proj(self, x):
        y = x @ self.P["diff_w"].T + self.P["diff_b"]
        return _ln(y, self.P["diff_ln_w"], self.P["diff_ln_b"])

    def init_hidden(self, repo_emb0):
        y = repo_emb0 @ self.P["repo_w"].T + self.P["repo_b"]
        y = nn.gelu(y)                                  # exact erf GELU
        return _ln(y, self.P["repo_ln_w"], self.P["repo_ln_b"])  # h_0: (H,)

    def step(self, diff_emb, h):
        x = self._diff_proj(diff_emb)                  # (H,)
        H = self.H
        gi = x @ self.P["gru_Wih"].T + self.P["gru_bih"]   # (3H,)
        gh = h @ self.P["gru_Whh"].T + self.P["gru_bhh"]   # (3H,)
        r = mx.sigmoid(gi[:H] + gh[:H])
        z = mx.sigmoid(gi[H:2 * H] + gh[H:2 * H])
        n = mx.tanh(gi[2 * H:] + r * gh[2 * H:])
        return (1 - z) * n + z * h

    def ctx(self, h):
        return _ln(h, self.P["out_ln_w"], self.P["out_ln_b"])   # output_norm(h)
