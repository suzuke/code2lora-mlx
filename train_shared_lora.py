"""Train ONE free shared LoRA (~660k params, repo-independent) directly on the
RepoPeftBench static task, then eval on held-out vs base. The 'slim code2lora':
no encoder, no 680M hypernetwork — just a single shared adapter.

Question: can a freely-trained 660k shared LoRA match (or beat) the official 680M
per-repo hypernetwork? (avg_train centroid already ties it at CE 0.378 / EM 59.4%.)
"""

import json
import random
import re
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from huggingface_hub import hf_hub_download

from c2l import qwen_lora
from c2l.official_direct import OfficialDirectHead, TYPES

BASE = "Qwen/Qwen2.5-Coder-1.5B"
STEPS = 4000
LR = 2e-4
MAXLEN = 640
CLIP = 1.0
EVAL_CE = 12
EVAL_EM = 8

head = OfficialDirectHead(hf_hub_download("code2lora/code2lora-direct", "code2lora_direct.pt"))
DIMS = {t: (int(head.type_dims[t][0]), int(head.type_dims[t][1])) for t in TYPES}
RANK = head.rank

model, tok = qwen_lora.load_base_model(BASE)
print(f"{len(qwen_lora._layers(model))} layers; rank {RANK}")


class SharedLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = {t: mx.random.normal((RANK, DIMS[t][0])) * 0.01 for t in TYPES}
        self.B = {t: mx.zeros((DIMS[t][1], RANK)) for t in TYPES}


slora = SharedLoRA()


def inject(s):
    for layer in qwen_lora._layers(model):
        for _m, (sub, a) in qwen_lora._PROJ.items():
            getattr(getattr(layer, sub), a).set_lora(s.A[a], s.B[a])


def n_params(s):
    return sum(v.size for v in list(s.A.values()) + list(s.B.values()))


print(f"shared LoRA params: {n_params(slora)/1e3:.0f}k")

train = [json.loads(l) for l in open("data/rpb25/train.jsonl")]
random.seed(0)
random.shuffle(train)


def make_batch(ex):
    ip = tok.encode(ex["input_prefix"])
    fu = tok.encode(ex["input_prefix"] + ex["target_value"])
    if len(fu) <= len(ip) or len(fu) < 4:
        return None
    fu = fu[:MAXLEN]
    b = min(len(ip), len(fu) - 1)
    return mx.array(fu), b


sched = optim.cosine_decay(LR, STEPS, LR * 0.05)  # 2e-4 → 1e-5 over training
opt = optim.AdamW(learning_rate=sched, weight_decay=0.01)

t0 = time.time()
step = 0
i = 0
losses = []
while step < STEPS:
    mb = make_batch(train[i % len(train)])
    i += 1
    if mb is None:
        continue
    ids, b = mb

    def closure():
        inject(slora)
        logits = model(ids[None])[0]
        pred = logits[b - 1:ids.shape[0] - 1]
        return mx.mean(nn.losses.cross_entropy(pred, ids[b:]))

    loss, grads = nn.value_and_grad(slora, closure)()
    grads = optim.clip_grad_norm(grads, CLIP)[0]
    opt.update(slora, grads)
    mx.eval(slora.parameters(), opt.state, loss)
    losses.append(float(loss))
    step += 1
    if step % 50 == 0:
        print(f"  step {step}/{STEPS}  loss {np.mean(losses[-50:]):.3f}  "
              f"{(time.time()-t0)/step:.1f}s/step")

print(f"trained {STEPS} steps in {(time.time()-t0)/60:.1f} min")
np.savez("slim_shared_lora.npz",
         **{f"A_{t}": np.array(slora.A[t]) for t in TYPES},
         **{f"B_{t}": np.array(slora.B[t]) for t in TYPES})
print("saved slim_shared_lora.npz")

# ---- eval on held-out ----
rows = [json.loads(l) for l in open("data/rpb25/heldout.jsonl")]
byrepo = {}
for r in rows:
    byrepo.setdefault(r["repo_id"], []).append(r)


def ce(prefix, target):
    ip = tok.encode(prefix)
    fu = tok.encode(prefix + target)
    bb = len(ip)
    if len(fu) <= bb:
        return None
    lg = model(mx.array([fu]))[0]
    return float(nn.losses.cross_entropy(lg[bb - 1:len(fu) - 1], mx.array(fu[bb:]), reduction="mean"))


def _norm(s):
    return re.sub(r"\s+", " ", s.strip()).rstrip(" \t;:,.)]}")


def em(prefix, target):
    g = _norm(qwen_lora.generate_text(model, tok, prefix, max_tokens=24))
    t = _norm(target)
    return bool(t) and g.startswith(t)


cb, cs = [], []
for rid, exs in byrepo.items():
    for ex in exs[:EVAL_CE]:
        qwen_lora.clear_lora(model)
        a = ce(ex["input_prefix"], ex["target_value"])
        inject(slora)
        b2 = ce(ex["input_prefix"], ex["target_value"])
        if a is not None and b2 is not None:
            cb.append(a)
            cs.append(b2)
eb = es = en = 0
for rid, exs in byrepo.items():
    for ex in exs[:EVAL_EM]:
        en += 1
        qwen_lora.clear_lora(model)
        eb += em(ex["input_prefix"], ex["target_value"])
        inject(slora)
        es += em(ex["input_prefix"], ex["target_value"])

print("=" * 50)
print(f"HELD-OUT eval (slim shared LoRA, {n_params(slora)/1e3:.0f}k params)")
print(f"  CE  base={np.mean(cb):.3f}  shared={np.mean(cs):.3f}")
print(f"  EM  base={100*eb/en:.1f}%  shared={100*es/en:.1f}%")
print(f"  (reference: official 680M per-repo → CE 0.377, EM 59.4%; avg_train → 0.378/59.4%)")
