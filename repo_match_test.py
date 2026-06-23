"""Sensitive test for repo-customization signal (static): for each held-out repo's
examples, score CE under ALL repos' adapters and check whether the OWN-repo adapter
wins. If own-repo systematically ranks #1 → repo-specificity carries real signal.
If own-repo ranks ~random (mid) → the static adapters have no usable repo signal.
"""

import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download

from c2l import qwen_lora
from c2l.official_direct import OfficialDirectHead, TYPES

BASE = "Qwen/Qwen2.5-Coder-1.5B"
PER_REPO = 12

head = OfficialDirectHead(hf_hub_download("code2lora/code2lora-direct", "code2lora_direct.pt"))
rows = [json.loads(l) for l in open("data/rpb25/heldout.jsonl")]
byrepo = {}
for r in rows:
    byrepo.setdefault(r["repo_id"], []).append(r)
repos = list(byrepo)

adapters = {}
for rid in repos:
    ad = head.forward(mx.array(np.array(byrepo[rid][0]["repo_embedding"], dtype=np.float32)))
    mx.eval([t for ab in ad.values() for t in ab])
    adapters[rid] = {t: (ad[t][0], ad[t][1]) for t in TYPES}

model, tok = qwen_lora.load_base_model(BASE)


def inject(rid):
    for layer in qwen_lora._layers(model):
        for _m, (s, a) in qwen_lora._PROJ.items():
            A, B = adapters[rid][a]
            getattr(getattr(layer, s), a).set_lora(A, 2.0 * B)


def ce(prefix, target):
    ip = tok.encode(prefix)
    fu = tok.encode(prefix + target)
    b = len(ip)
    if len(fu) <= b:
        return None
    lg = model(mx.array([fu]))[0]
    return float(nn.losses.cross_entropy(lg[b - 1:len(fu) - 1], mx.array(fu[b:]), reduction="mean"))


own_ranks = []
own_wins = 0
own_ces, other_ces = [], []
n = 0
for R in repos:
    for ex in byrepo[R][:PER_REPO]:
        ces = {}
        ok = True
        for C in repos:
            inject(C)
            v = ce(ex["input_prefix"], ex["target_value"])
            if v is None:
                ok = False
                break
            ces[C] = v
        if not ok:
            continue
        n += 1
        order = sorted(repos, key=lambda c: ces[c])
        rank = order.index(R) + 1            # 1 = own-repo adapter gave lowest CE
        own_ranks.append(rank)
        own_wins += (rank == 1)
        own_ces.append(ces[R])
        other_ces.append(np.mean([ces[c] for c in repos if c != R]))

K = len(repos)
print("=" * 56)
print(f"REPO-MATCH ranking test (static, {n} examples, {K} candidate adapters each)")
print(f"  own-repo adapter mean rank : {np.mean(own_ranks):.2f} / {K}  (random≈{(K+1)/2:.1f}, best=1)")
print(f"  own-repo wins (rank #1)    : {own_wins}/{n} = {100*own_wins/n:.1f}%  (random≈{100/K:.1f}%)")
print(f"  CE own={np.mean(own_ces):.4f}  vs other-repo avg={np.mean(other_ces):.4f}  "
      f"(Δ={np.mean(own_ces)-np.mean(other_ces):+.4f})")
print("  → repo signal EXISTS if own rank ≪ random and Δ<0; NULL if rank≈random & Δ≈0")
