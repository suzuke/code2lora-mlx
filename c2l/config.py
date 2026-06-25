"""Configuration — the OFFICIAL Code2LoRA head architecture.

Mirrors the official `Code2LoRAHead` (anonymous.4open.science/r/code2lora-6857,
`hypernetwork/code2lora_core.py`): a 2-layer exact-GELU trunk of width `hidden_dim`,
L2-normalised and scaled by sqrt(hidden_dim), then per module type
`tanh(head(h)) * clamp(exp(log_scale), 1e-5, 0.3)`. It emits ONE (A, B) pair per
module type, SHARED across all decoder layers (no per-layer embedding).

Base-model dims default to Qwen2.5-Coder-0.5B; switch to the released-checkpoint
1.5B dims (hidden 1536 / intermediate 8960 / kv 256 / num_layers 28) to match it.
"""

from dataclasses import dataclass

# All 7 LoRA-injected module types.
MODULE_TYPES = ["q", "k", "v", "o", "gate", "up", "down"]


@dataclass
class HypernetworkConfig:
    # ── Official Code2LoRAHead hyperparameters ──
    hidden_dim: int = 1024           # trunk width (official)
    rank: int = 16                   # LoRA rank (official; alpha=32 -> alpha/rank=2.0)
    init_log_scale: float = -3.5     # official init: exp(-3.5) ~= 0.03 -> tiny LoRA at init
    scale_clamp_min: float = 1e-5    # official clamp on exp(log_scale)
    scale_clamp_max: float = 0.3
    repo_embed_dim: int = 2048       # official Qwen3-Embedding-0.6B repo_state_embedding

    # ── Base-model dims (default Qwen2.5-Coder-0.5B). The head emits ONE adapter
    #    shared across all `num_layers`. For the 1.5B released checkpoint use
    #    hidden 1536 / intermediate 8960 / kv 256 / num_layers 28. ──
    num_layers: int = 24
    llm_hidden_dim: int = 896
    llm_intermediate_dim: int = 4864
    kv_proj_dim: int = 128           # GQA: 2 kv_heads * 64 head_dim

    def lora_in_dim(self, module_type: str) -> int:
        # q/k/v/o/gate/up: input is the hidden state; down: the intermediate activation.
        if module_type in ("q", "k", "v", "o", "gate", "up"):
            return self.llm_hidden_dim
        if module_type == "down":
            return self.llm_intermediate_dim
        raise ValueError(f"Unknown module type: {module_type}")

    def lora_out_dim(self, module_type: str) -> int:
        if module_type in ("q", "o"):
            return self.llm_hidden_dim
        if module_type in ("k", "v"):
            return self.kv_proj_dim          # GQA: smaller than hidden
        if module_type in ("gate", "up"):
            return self.llm_intermediate_dim
        if module_type == "down":
            return self.llm_hidden_dim
        raise ValueError(f"Unknown module type: {module_type}")
