"""Interactive slim code2lora: load the standalone 2.6MB shared adapter and
complete a prompt — base vs slim, side by side. No encoder, no hypernetwork.

  uv run --with torch python slim_complete.py "def test_x():\n    assert f(2) == " [max_tokens]
"""

import sys

import mlx.core as mx
import numpy as np

from c2l import qwen_lora

BASE = "Qwen/Qwen2.5-Coder-1.5B"
TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

prompt = sys.argv[1] if len(sys.argv) > 1 else "def test_add():\n    assert add(2, 3) == "
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 30

d = np.load("slim_code2lora.npz")
adapter = {t: (mx.array(d[f"A_{t}"]), mx.array(d[f"B_{t}"])) for t in TYPES}
model, tok = qwen_lora.load_base_model(BASE)


def inject():
    for layer in qwen_lora._layers(model):
        for _m, (s, a) in qwen_lora._PROJ.items():
            A, B = adapter[a]
            getattr(getattr(layer, s), a).set_lora(A, 2.0 * B)


qwen_lora.clear_lora(model)
base = qwen_lora.generate_text(model, tok, prompt, max_tokens=max_tokens)
inject()
slim = qwen_lora.generate_text(model, tok, prompt, max_tokens=max_tokens)

print("\n=== PROMPT ===")
print(prompt)
print("\n=== [BASE] frozen Qwen2.5-Coder-1.5B ===")
print(base)
print("\n=== [SLIM] + 2.6MB shared adapter ===")
print(slim)
