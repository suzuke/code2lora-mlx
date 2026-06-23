"""Over-parameterization test: train ONE adapter, but generated THROUGH the
official hypernetwork architecture (trunk + giant per-module heads, ~680M,
randomly initialized) fed a FIXED input (mean train-repo embedding — no repo
conditioning). Does generating an adapter through this over-parameterized network
beat directly training a shared LoRA (~47%) / approach the official (~59%)?
Isolates whether the architecture/over-parameterization is the source of the gain.
"""

import json
import math
import random
import re
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from huggingface_hub import hf_hub_download
from mlx.utils import tree_map

from c2l import qwen_lora
from c2l.official_direct import OfficialDirectHead, TYPES

BASE = "Qwen/Qwen2.5-Coder-1.5B"
EPOCHS, ACCUM, LR, WD, MAXLEN, CLIP, SCALE = 3, 8, 1e-4, 0.01, 640, 1.0, 2.0
HID, RANK = 1024, 16

ref = OfficialDirectHead(hf_hub_download("code2lora/code2lora-direct", "code2lora_direct.pt"))
DIMS = {t: (int(ref.type_dims[t][0]), int(ref.type_dims[t][1])) for t in TYPES}
model, tok = qwen_lora.load_base_model(BASE)


class OverparamHead(nn.Module):
    """Official Code2LoRAHead architecture, randomly initialized & trainable."""
    def __init__(self):
        super().__init__()
        g = lambda o, i: mx.random.normal((o, i)) * (1.0 / math.sqrt(i))
        self.t0w, self.t0b = g(HID, 2048), mx.zeros((HID,))
        self.t2w, self.t2b = g(HID, HID), mx.zeros((HID,))
        self.hAw = {t: g(RANK * DIMS[t][0], HID) for t in TYPES}
        self.hAb = {t: mx.zeros((RANK * DIMS[t][0],)) for t in TYPES}
        self.hBw = {t: g(DIMS[t][1] * RANK, HID) for t in TYPES}
        self.hBb = {t: mx.zeros((DIMS[t][1] * RANK,)) for t in TYPES}
        self.lsA = {t: mx.array(-3.5) for t in TYPES}
        self.lsB = {t: mx.array(-3.5) for t in TYPES}

    def __call__(self, ctx):
        h = nn.gelu(ctx @ self.t0w.T + self.t0b)
        h = nn.gelu(h @ self.t2w.T + self.t2b)
        h = h / mx.sqrt(mx.sum(h * h)) * math.sqrt(HID)
        out = {}
        for t in TYPES:
            in_f, out_f = DIMS[t]
            ar = (h @ self.hAw[t].T + self.hAb[t]).reshape(RANK, in_f)
            br = (h @ self.hBw[t].T + self.hBb[t]).reshape(out_f, RANK)
            sA = mx.clip(mx.exp(self.lsA[t]), 1e-5, 0.3)
            sB = mx.clip(mx.exp(self.lsB[t]), 1e-5, 0.3)
            out[t] = (mx.tanh(ar) * sA, mx.tanh(br) * sB)
        return out


head = OverparamHead()
nparams = sum(v.size for v in [head.t0w, head.t0b, head.t2w, head.t2b]
              + list(head.hAw.values()) + list(head.hAb.values())
              + list(head.hBw.values()) + list(head.hBb.values()))
print(f"over-param head: {nparams/1e6:.0f}M trainable params")

train = [json.loads(l) for l in open("data/rpb25/train.jsonl")]
# FIXED input = mean of unique train-repo embeddings (the "average repo")
seen, embs = set(), []
for r in train:
    if r["repo_id"] not in seen:
        seen.add(r["repo_id"])
        embs.append(np.array(r["repo_embedding"], dtype=np.float32))
FIXED = mx.array(np.mean(embs, axis=0))
print(f"fixed input = mean of {len(embs)} train-repo embeddings")

random.seed(0)
total_fwd = EPOCHS * len(train)
opt_steps = total_fwd // ACCUM
warmup = max(1, int(opt_steps * 0.03))
sched = optim.join_schedules(
    [optim.linear_schedule(0.0, LR, warmup), optim.cosine_decay(LR, opt_steps - warmup, LR * 0.05)], [warmup])
opt = optim.AdamW(learning_rate=sched, weight_decay=WD)


def inject(ad):
    for layer in qwen_lora._layers(model):
        for _m, (sub, a) in qwen_lora._PROJ.items():
            A, B = ad[a]
            getattr(getattr(layer, sub), a).set_lora(A, SCALE * B)


def closure_for(ids):
    def closure():
        inject(head(FIXED))
        logits = model(ids[None])[0][:-1]
        return mx.mean(nn.losses.cross_entropy(logits, ids[1:]))
    return closure


rows = [json.loads(l) for l in open("data/rpb25/heldout.jsonl")]
byrepo = {}
for r in rows:
    byrepo.setdefault(r["repo_id"], []).append(r)


def ce(p, t):
    ip = tok.encode(p)
    fu = tok.encode(p + t)
    b = len(ip)
    if len(fu) <= b:
        return None
    lg = model(mx.array([fu]))[0]
    return float(nn.losses.cross_entropy(lg[b - 1:len(fu) - 1], mx.array(fu[b:]), reduction="mean"))


def _norm(s):
    return re.sub(r"\s+", " ", s.strip()).rstrip(" \t;:,.)]}")


def em(p, t):
    g = _norm(qwen_lora.generate_text(model, tok, p, max_tokens=24))
    return bool(_norm(t)) and g.startswith(_norm(t))


def evaluate(nper):
    ad = head(FIXED)
    mx.eval([x for ab in ad.values() for x in ab])
    bce, sce, eb, es, en = [], [], 0, 0, 0
    for rid, exs in byrepo.items():
        for ex in exs[:nper]:
            qwen_lora.clear_lora(model)
            a = ce(ex["input_prefix"], ex["target_value"])
            inject(ad)
            b2 = ce(ex["input_prefix"], ex["target_value"])
            if a is not None and b2 is not None:
                bce.append(a)
                sce.append(b2)
            en += 1
            qwen_lora.clear_lora(model)
            eb += em(ex["input_prefix"], ex["target_value"])
            inject(ad)
            es += em(ex["input_prefix"], ex["target_value"])
    return np.mean(bce), np.mean(sce), 100 * eb / en, 100 * es / en


t0 = time.time()
step, fwd, running = 0, 0, []
order = list(range(len(train)))
random.shuffle(order)
while fwd < total_fwd:
    acc, lsum, nacc = None, 0.0, 0
    while nacc < ACCUM and fwd < total_fwd:
        ex = train[order[fwd % len(train)]]
        if fwd % len(train) == len(train) - 1:
            random.shuffle(order)
        ids = tok.encode(ex["input_prefix"] + ex["target_value"])
        fwd += 1
        if len(ids) < 2:
            continue
        ids = mx.array(ids[:MAXLEN])
        loss, g = nn.value_and_grad(head, closure_for(ids))()
        acc = g if acc is None else tree_map(lambda a, b: a + b, acc, g)
        lsum += float(loss)
        nacc += 1
    if acc is None:
        continue
    acc = tree_map(lambda x: x / nacc, acc)
    acc = optim.clip_grad_norm(acc, CLIP)[0]
    opt.update(head, acc)
    mx.eval(head.parameters(), opt.state)
    step += 1
    running.append(lsum / nacc)
    if step % 50 == 0:
        print(f"step {step}/{opt_steps} fwd {fwd}/{total_fwd} loss {np.mean(running[-50:]):.3f} {(time.time()-t0)/60:.0f}min")
    if step % max(1, opt_steps // EPOCHS) == 0:
        bce, sce, beM, seM = evaluate(6)
        print(f"  [eval @ step {step}] CE base {bce:.3f} overparam {sce:.3f} | EM base {beM:.1f}% overparam {seM:.1f}%")

print(f"trained {step} steps / {fwd} forwards in {(time.time()-t0)/60:.0f} min")
bce, sce, beM, seM = evaluate(12)
print("=" * 56)
print(f"OVER-PARAM adapter (held-out, n=12/repo): CE base {bce:.3f} overparam {sce:.3f} | EM base {beM:.1f}% overparam {seM:.1f}%")
print("reference: direct shared LoRA ~47% ; official hypernetwork 59.4%")
print("VERDICT: ≫47% (→ architecture/over-param IS the source) ; ≈47% (→ it's the varied-repo training, not the architecture)")
