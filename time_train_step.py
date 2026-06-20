"""Measure ONE real Code2LoRA-direct training step on this machine, at the
official recipe's shape (Qwen2.5-Coder-1.5B, seq_len 4096, micro-batch 2):
  hypernetwork.forward(repo_emb) -> LoRA -> inject all 28 layers
  -> Qwen forward+backward -> next-token CE -> grad w.r.t. hypernetwork.
Reports seconds/step so we can extrapolate the full from-scratch ETA.
"""

import math
import time

import mlx.core as mx
import mlx.nn as mlxnn
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from c2l import qwen_lora

BASE = "Qwen/Qwen2.5-Coder-1.5B"
B, L = 2, 4096          # official args: lm_micro_batch=2, max_seq_len=4096
N_TIMED = 3

# ---- official head shapes/weights (any init works for timing) ----
ck = torch.load(hf_hub_download("code2lora/code2lora-direct", "code2lora_direct.pt"),
                map_location="cpu", weights_only=False)
sd, cfg = ck["state_dict"], ck["config"]
TYPES = sorted(cfg["type_dims"].keys())
TYPE_DIMS = {t: tuple(cfg["type_dims"][t]) for t in TYPES}
RANK, HID = int(cfg["rank"]), int(cfg["hidden_dim"])
g = lambda n: mx.array(sd[n].float().numpy())
P = {"t0w": g("trunk.0.weight"), "t0b": g("trunk.0.bias"),
     "t2w": g("trunk.2.weight"), "t2b": g("trunk.2.bias")}
for t in TYPES:
    P[f"hAw_{t}"] = g(f"heads_A.{t}.weight"); P[f"hAb_{t}"] = g(f"heads_A.{t}.bias")
    P[f"hBw_{t}"] = g(f"heads_B.{t}.weight"); P[f"hBb_{t}"] = g(f"heads_B.{t}.bias")
    P[f"lsA_{t}"] = g(f"log_scale_A.{t}"); P[f"lsB_{t}"] = g(f"log_scale_B.{t}")


def hn_forward(P, ctx):
    h = mlxnn.gelu(ctx @ P["t0w"].T + P["t0b"])
    h = mlxnn.gelu(h @ P["t2w"].T + P["t2b"])
    h = h / mx.sqrt(mx.sum(h * h)) * math.sqrt(HID)
    out = {}
    for t in TYPES:
        in_f, out_f = TYPE_DIMS[t]
        ar = (h @ P[f"hAw_{t}"].T + P[f"hAb_{t}"]).reshape(RANK, in_f)
        br = (h @ P[f"hBw_{t}"].T + P[f"hBb_{t}"]).reshape(out_f, RANK)
        sA = mx.clip(mx.exp(P[f"lsA_{t}"]), 1e-5, 0.3)
        sB = mx.clip(mx.exp(P[f"lsB_{t}"]), 1e-5, 0.3)
        out[t] = (mx.tanh(ar) * sA, mx.tanh(br) * sB)
    return out


print(f"loading {BASE} ...")
model, tok = qwen_lora.load_base_model(BASE)
layers = qwen_lora._layers(model)
print(f"{len(layers)} layers wrapped")

rng = np.random.default_rng(0)
try:
    vocab = int(model.model.embed_tokens.weight.shape[0])
except Exception:
    vocab = 151936
ids = mx.array(rng.integers(0, vocab, size=(B, L)))
ctx = mx.array(rng.standard_normal(2048).astype(np.float32))


def loss_fn(P):
    ad = hn_forward(P, ctx)
    for layer in layers:
        for _m, (s, a) in qwen_lora._PROJ.items():
            A, Bm = ad[a]
            getattr(getattr(layer, s), a).set_lora(A, 2.0 * Bm)
    logits = model(ids)[:, :-1, :].astype(mx.float32)
    tgt = ids[:, 1:]
    return mx.mean(mlxnn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1)))


gfn = mx.value_and_grad(loss_fn)

print("warmup step (includes Metal kernel compile) ...")
t = time.time()
Lv, Gv = gfn(P)
mx.eval(Lv, *Gv.values())
print(f"  warmup: {time.time() - t:.1f}s  (loss={float(Lv):.3f})")

print(f"timing {N_TIMED} steps @ B={B} L={L} ...")
t = time.time()
for _ in range(N_TIMED):
    Lv, Gv = gfn(P)
    mx.eval(Lv, *Gv.values())
sps = (time.time() - t) / N_TIMED
print(f"\n  ==> {sps:.2f} s / training step  (B={B}, L={L}, {L*B} tokens/step)")

# extrapolate to the official recipe: 44,149 train QnA rows, 3 epochs, micro-batch 2
steps_per_epoch = math.ceil(44149 / B)
total = steps_per_epoch * 3
hrs = total * sps / 3600
print(f"  official scale: ~44,149 rows x 3 epochs / B={B} = {total:,} steps")
print(f"  => full from-scratch ETA on THIS machine ~ {hrs:.0f} h ({hrs/24:.1f} days) at {sps:.2f}s/step")
