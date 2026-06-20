"""Configuration — mirrors code2lora-lite/src/config.rs HypernetworkConfig."""

from dataclasses import dataclass

# All 7 LoRA-injected module types, same order as the Candle MODULE_TYPES.
MODULE_TYPES = ["q", "k", "v", "o", "gate", "up", "down"]


@dataclass
class HypernetworkConfig:
    # Defaults match Qwen2.5-Coder-0.5B (loaded from HF config.json in the Rust version):
    #   hidden_size=896, intermediate_size=4864, num_attention_heads=14,
    #   num_key_value_heads=2 -> kv_proj_dim = 2 * (896/14) = 128
    hidden_dim: int = 384            # shared MLP width
    rank: int = 8                    # LoRA rank
    num_layers: int = 24             # Qwen decoder layers
    # repo_embed_dim: 768 = our 384 mean + 384 max MiniLM encoder. For the REAL
    # RepoPeftBench data we use the paper's own `repo_state_embedding`
    # (Qwen3-Embedding-0.6B, dim=2048) -> set repo_embed_dim=2048 in that run.
    repo_embed_dim: int = 768
    llm_hidden_dim: int = 896
    llm_intermediate_dim: int = 4864
    kv_proj_dim: int = 128           # GQA: 2 kv_heads * 64 head_dim
    # TUNING FIX (scaled run): the prior 10-repo run let exp(log_scale) blow up to
    # ~2432, producing a huge LoRA delta that destroyed the model on held-out repos.
    # Cap the effective per-entry LoRA scale so the delta stays bounded. The A/B
    # entries are tanh(.)*exp(log_scale) in [-scale, scale]; capping scale<=1.0
    # keeps each entry in [-1,1] and the delta sane. <=0 disables the cap.
    max_lora_scale: float = 1.0

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
