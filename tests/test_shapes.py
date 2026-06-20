"""Mirror of the Candle test_hypernetwork_shapes: verify the MLX hypernetwork
produces identical LoRA shapes, distinct per-layer weights, and trainable params."""

import mlx.core as mx
from mlx.utils import tree_flatten

from c2l.config import HypernetworkConfig
from c2l.hypernetwork import Code2LoRAHead

cfg = HypernetworkConfig()
hn = Code2LoRAHead(cfg)
mx.eval(hn.parameters())

x = mx.random.normal(shape=(1, cfg.repo_embed_dim))
layers = hn.forward_all(x)
mx.eval([ab[0] for l in layers for ab in l.values()])

assert len(layers) == cfg.num_layers, f"layers={len(layers)}"

l0 = layers[0]
# A: (rank, in_dim), B: (out_dim, rank) — exactly as Candle.
assert l0["q"][0].shape == (cfg.rank, cfg.llm_hidden_dim)               # q_A (8,896)
assert l0["q"][1].shape == (cfg.llm_hidden_dim, cfg.rank)               # q_B (896,8)
assert l0["k"][0].shape == (cfg.rank, cfg.llm_hidden_dim)               # k_A (8,896)
assert l0["k"][1].shape == (cfg.kv_proj_dim, cfg.rank)                  # k_B (128,8) GQA
assert l0["gate"][1].shape == (cfg.llm_intermediate_dim, cfg.rank)      # gate_B (4864,8)
assert l0["down"][0].shape == (cfg.rank, cfg.llm_intermediate_dim)      # down_A (8,4864)
assert l0["down"][1].shape == (cfg.llm_hidden_dim, cfg.rank)            # down_B (896,8)

# per-layer distinctness (layer embedding offset must differentiate them)
diff = mx.sum(mx.abs(layers[0]["q"][0] - layers[1]["q"][0])).item()
assert diff > 1e-3, f"layers 0/1 q_A should differ, diff={diff}"

# MLX must track heads + log-scales as trainable params
flat = dict(tree_flatten(hn.trainable_parameters()))
nparams = sum(v.size for v in flat.values())
has_scale = any("log_scale_a" in k for k in flat)
has_heads = any("heads_a.q" in k for k in flat)
assert has_scale, "log_scale params not tracked"
assert has_heads, "head params not tracked"

print(f"OK shapes match Candle | layers distinct (diff={diff:.3f}) | "
      f"trainable params={nparams:,} | log_scale tracked={has_scale} | heads tracked={has_heads}")
print("default device:", mx.default_device())
