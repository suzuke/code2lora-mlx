"""P0 end-to-end smoke: load Qwen via mlx-lm -> hypernetwork generates LoRA ->
inject -> generate. With a random (untrained) hypernetwork the output is
expected to be gibberish (same as the Candle toy baseline); this only proves the
MLX inference pipeline runs end-to-end on Metal."""

import time

import mlx.core as mx

from c2l import qwen_lora
from c2l.config import HypernetworkConfig
from c2l.hypernetwork import Code2LoRAHead
from c2l.repo_encoder import load_embedding

cfg = HypernetworkConfig()
hn = Code2LoRAHead(cfg)
mx.eval(hn.parameters())

# reuse the Candle reference repo embedding (768-d)
emb = mx.array(load_embedding("/tmp/candle_ref.embed").reshape(1, -1))
all_lora = hn.forward_all(emb)
mx.eval([ab[0] for layer in all_lora for ab in layer.values()])
print(f"hypernetwork -> {len(all_lora)} layers x 7 modules of LoRA")

t = time.time()
model, tok = qwen_lora.load_base_model()
print(f"load Qwen (mlx-lm): {time.time() - t:.1f}s")

qwen_lora.inject_lora(model, all_lora)
print("LoRA injected into all layers")

prompt = "def test_add():\n    assert add(2, 3) =="

t = time.time()
out_lora = qwen_lora.generate_text(model, tok, prompt, max_tokens=32)
print(f"generate WITH lora: {time.time() - t:.1f}s")
print("OUTPUT(lora):", repr(out_lora[:160]))

qwen_lora.clear_lora(model)
t = time.time()
out_base = qwen_lora.generate_text(model, tok, prompt, max_tokens=32)
print(f"generate NO lora (sanity, should be coherent code): {time.time() - t:.1f}s")
print("OUTPUT(base):", repr(out_base[:160]))
print("default device:", mx.default_device())
