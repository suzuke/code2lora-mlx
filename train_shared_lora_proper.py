"""Properly-trained shared LoRA baseline (no hypernetwork) — closest feasible
reproduction of the paper's "Single LoRA" baseline. Settles whether a well-trained
shared LoRA reaches the hypernetwork's ~59% EM (→ hypernetwork adds nothing) or
stays ~47% (→ hypernetwork trains a genuinely better generic adapter).

Recipe: full train set, 3 epochs, grad-accum effective batch 8, lr 1e-4 +
warmup/cosine, weight_decay 0.01, full-sequence next-token CE (official IR-loss
style), alpha/rank=2 apply scaling.
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
from mlx.utils import tree_map

from c2l import qwen_lora
from c2l.official_direct import OfficialDirectHead, TYPES

BASE = "Qwen/Qwen2.5-Coder-1.5B"
EPOCHS, ACCUM, LR, WD, MAXLEN, CLIP, SCALE = 3, 8, 1e-4, 0.01, 640, 1.0, 2.0

head = OfficialDirectHead(hf_hub_download("code2lora/code2lora-direct", "code2lora_direct.pt"))
DIMS = {t: (int(head.type_dims[t][0]), int(head.type_dims[t][1])) for t in TYPES}
RANK = head.rank
model, tok = qwen_lora.load_base_model(BASE)


class SharedLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = {t: mx.random.normal((RANK, DIMS[t][0])) * 0.01 for t in TYPES}
        self.B = {t: mx.zeros((DIMS[t][1], RANK)) for t in TYPES}


slora = SharedLoRA()


def inject(s):
    for layer in qwen_lora._layers(model):
        for _m, (sub, a) in qwen_lora._PROJ.items():
            getattr(getattr(layer, sub), a).set_lora(s.A[a], SCALE * s.B[a])


train = [json.loads(l) for l in open("data/rpb25/train.jsonl")]
random.seed(0)
total_fwd = EPOCHS * len(train)
opt_steps = total_fwd // ACCUM
warmup = max(1, int(opt_steps * 0.03))
sched = optim.join_schedules(
    [optim.linear_schedule(0.0, LR, warmup), optim.cosine_decay(LR, opt_steps - warmup, LR * 0.05)],
    [warmup])
opt = optim.AdamW(learning_rate=sched, weight_decay=WD)
print(f"{len(train)} train ex, {EPOCHS} epochs → {total_fwd} forwards / {opt_steps} opt-steps (accum {ACCUM})")


def closure_for(ids):
    def closure():
        inject(slora)
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
    bce, sce, eb, es, en = [], [], 0, 0, 0
    for rid, exs in byrepo.items():
        for ex in exs[:nper]:
            qwen_lora.clear_lora(model)
            a = ce(ex["input_prefix"], ex["target_value"])
            inject(slora)
            b2 = ce(ex["input_prefix"], ex["target_value"])
            if a is not None and b2 is not None:
                bce.append(a)
                sce.append(b2)
            en += 1
            qwen_lora.clear_lora(model)
            eb += em(ex["input_prefix"], ex["target_value"])
            inject(slora)
            es += em(ex["input_prefix"], ex["target_value"])
    return np.mean(bce), np.mean(sce), 100 * eb / en, 100 * es / en


t0 = time.time()
step = 0
fwd = 0
running = []
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
        loss, g = nn.value_and_grad(slora, closure_for(ids))()
        acc = g if acc is None else tree_map(lambda a, b: a + b, acc, g)
        lsum += float(loss)
        nacc += 1
    if acc is None:
        continue
    acc = tree_map(lambda x: x / nacc, acc)
    acc = optim.clip_grad_norm(acc, CLIP)[0]
    opt.update(slora, acc)
    mx.eval(slora.parameters(), opt.state)
    step += 1
    running.append(lsum / nacc)
    if step % 50 == 0:
        print(f"step {step}/{opt_steps} fwd {fwd}/{total_fwd} loss {np.mean(running[-50:]):.3f} {(time.time()-t0)/60:.0f}min")
    if step % max(1, opt_steps // EPOCHS) == 0:
        bce, sce, beM, seM = evaluate(6)
        print(f"  [eval @ step {step}] CE base {bce:.3f} shared {sce:.3f} | EM base {beM:.1f}% shared {seM:.1f}%")

print(f"trained {step} opt-steps / {fwd} forwards in {(time.time()-t0)/60:.0f} min")
np.savez("shared_lora_proper.npz",
         **{f"A_{t}": np.array(slora.A[t]) for t in TYPES},
         **{f"B_{t}": np.array(slora.B[t]) for t in TYPES})
bce, sce, beM, seM = evaluate(12)
print("=" * 56)
print(f"PROPER shared LoRA (held-out, n=12/repo): CE base {bce:.3f} shared {sce:.3f} | EM base {beM:.1f}% shared {seM:.1f}%")
print("reference: official hypernetwork → CE 0.377 / EM 59.4% ; paper single-LoRA → 47.4%")
print("VERDICT: ~59% → hypernetwork adds nothing (baseline was weak); ~47% → hypernetwork trains a genuinely better generic adapter")
