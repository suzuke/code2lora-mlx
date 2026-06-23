"""Hardened ablation. Two upgrades over ablate_avg_adapter.py:
  (1) the shared adapter = centroid of TRAINING-repo adapters (not held-out) —
      removes the "centroid fit to the test set" objection.
  (2) add EXACT-MATCH (greedy generation + relaxed match) — the paper's metric,
      where Code2LoRA reportedly beats a single shared LoRA by +5.2pp.

Compares base / own(per-repo) / avg_train(shared) on held-out repos, on CE and EM.
"""

import json
import re

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download

from c2l import qwen_lora
from c2l.official_direct import OfficialDirectHead, TYPES

BASE = "Qwen/Qwen2.5-Coder-1.5B"
PER_CE = 12
PER_EM = 8
N_TRAIN = 150

ckpt = hf_hub_download("code2lora/code2lora-direct", "code2lora_direct.pt")
head = OfficialDirectHead(ckpt)

# held-out + per-repo own adapters
rows = [json.loads(l) for l in open("data/rpb25/heldout.jsonl")]
byrepo = {}
for r in rows:
    byrepo.setdefault(r["repo_id"], []).append(r)
own = {}
for rid, exs in byrepo.items():
    ad = head.forward(mx.array(np.array(exs[0]["repo_embedding"], dtype=np.float32)))
    mx.eval([t for ab in ad.values() for t in ab])
    own[rid] = {t: (ad[t][0], ad[t][1]) for t in TYPES}

# TRAINING-set centroid adapter (running mean over N_TRAIN unique repos)
print(f"building training centroid from {N_TRAIN} train repos ...")
acc = {t: [None, None] for t in TYPES}
seen = set()
for line in open("data/rpb25/train.jsonl"):
    r = json.loads(line)
    rid = r["repo_id"]
    if rid in seen:
        continue
    seen.add(rid)
    ad = head.forward(mx.array(np.array(r["repo_embedding"], dtype=np.float32)))
    for t in TYPES:
        acc[t][0] = ad[t][0] if acc[t][0] is None else acc[t][0] + ad[t][0]
        acc[t][1] = ad[t][1] if acc[t][1] is None else acc[t][1] + ad[t][1]
    mx.eval([x for ab in acc.values() for x in ab])
    if len(seen) >= N_TRAIN:
        break
ntr = len(seen)
avg_train = {t: (acc[t][0] / ntr, acc[t][1] / ntr) for t in TYPES}
mx.eval([x for ab in avg_train.values() for x in ab])
print(f"  centroid from {ntr} repos ready")

model, tok = qwen_lora.load_base_model(BASE)


def inject(ad):
    for layer in qwen_lora._layers(model):
        for _m, (s, a) in qwen_lora._PROJ.items():
            A, B = ad[a]
            getattr(getattr(layer, s), a).set_lora(A, 2.0 * B)


def ce(prefix, target):
    ip = tok.encode(prefix)
    ifu = tok.encode(prefix + target)
    b = len(ip)
    if len(ifu) <= b:
        return None
    lg = model(mx.array([ifu]))[0]
    return float(nn.losses.cross_entropy(lg[b - 1:len(ifu) - 1], mx.array(ifu[b:]), reduction="mean"))


def _norm(s):
    return re.sub(r"\s+", " ", s.strip()).rstrip(" \t;:,.)]}")


def em(prefix, target):
    g = _norm(qwen_lora.generate_text(model, tok, prefix, max_tokens=24))
    t = _norm(target)
    return bool(t) and g.startswith(t)


CONDS = [("base", None), ("own", "OWN"), ("avg_train", avg_train)]


def setcond(rid, c):
    qwen_lora.clear_lora(model)
    if c == "OWN":
        inject(own[rid])
    elif c is not None:
        inject(c)


# ---- CE ----
ce_acc = {k: [] for k, _ in CONDS}
for rid, exs in byrepo.items():
    for ex in exs[:PER_CE]:
        vals = {}
        ok = True
        for name, c in CONDS:
            setcond(rid, c)
            v = ce(ex["input_prefix"], ex["target_value"])
            if v is None:
                ok = False
                break
            vals[name] = v
        if ok:
            for name in vals:
                ce_acc[name].append(vals[name])

# ---- EM ----
em_acc = {k: 0 for k, _ in CONDS}
em_n = 0
for rid, exs in byrepo.items():
    for ex in exs[:PER_EM]:
        em_n += 1
        for name, c in CONDS:
            setcond(rid, c)
            if em(ex["input_prefix"], ex["target_value"]):
                em_acc[name] += 1

print("=" * 52)
print(f"CE  (n={len(ce_acc['base'])}):")
for name, _ in CONDS:
    print(f"    {name:10s} {np.mean(ce_acc[name]):.3f}")
print(f"EM  (n={em_n}, relaxed greedy match):")
for name, _ in CONDS:
    print(f"    {name:10s} {em_acc[name]}/{em_n} = {100*em_acc[name]/em_n:.1f}%")
bo = np.mean(ce_acc['base']) - np.mean(ce_acc['own'])
ba = np.mean(ce_acc['base']) - np.mean(ce_acc['avg_train'])
print(f"\nCE benefit: avg_train recovers {100*ba/bo:.0f}% of own")
print(f"EM gap own vs avg_train: {em_acc['own']-em_acc['avg_train']:+d} of {em_n} "
      f"({100*(em_acc['own']-em_acc['avg_train'])/em_n:+.1f}pp)")
