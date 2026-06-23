"""Ablation: how much of code2lora's benefit is repo-SPECIFIC vs repo-GENERIC?

For each held-out repo, compare base CE vs:
  - OWN  adapter (official hypernetwork, this repo's own)
  - AVG  adapter (mean of all repos' adapters — repo-independent, no conditioning)

If AVG ≈ OWN, the per-repo hypernetwork is largely redundant for this benefit
(a single shared LoRA would do). If OWN ≫ AVG, the repo-specific residual matters.
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

ckpt = hf_hub_download("code2lora/code2lora-direct", "code2lora_direct.pt")
head = OfficialDirectHead(ckpt)
rows = [json.loads(l) for l in open("data/rpb25/heldout.jsonl")]
byrepo = {}
for r in rows:
    byrepo.setdefault(r["repo_id"], []).append(r)

# per-repo adapters
adapters = {}
for rid, exs in byrepo.items():
    ad = head.forward(mx.array(np.array(exs[0]["repo_embedding"], dtype=np.float32)))
    mx.eval([t for ab in ad.values() for t in ab])
    adapters[rid] = {t: (ad[t][0], ad[t][1]) for t in TYPES}

# average adapter (repo-independent)
avg = {}
for t in TYPES:
    A = mx.mean(mx.stack([adapters[r][t][0] for r in adapters]), axis=0)
    B = mx.mean(mx.stack([adapters[r][t][1] for r in adapters]), axis=0)
    mx.eval(A, B)
    avg[t] = (A, B)

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


GB, GO, GA = [], [], []
ho = ha = 0
print(f"{'repo':34} base   own    avg")
for rid, exs in byrepo.items():
    bb, oo, aa = [], [], []
    for ex in exs[:PER_REPO]:
        qwen_lora.clear_lora(model)
        cb = ce(ex["input_prefix"], ex["target_value"])
        inject(adapters[rid])
        co = ce(ex["input_prefix"], ex["target_value"])
        inject(avg)
        ca = ce(ex["input_prefix"], ex["target_value"])
        if None in (cb, co, ca):
            continue
        bb.append(cb)
        oo.append(co)
        aa.append(ca)
    mb, mo, ma = np.mean(bb), np.mean(oo), np.mean(aa)
    GB += bb
    GO += oo
    GA += aa
    ho += mo < mb
    ha += ma < mb
    print(f"{rid:34} {mb:5.2f}  {mo:5.2f}  {ma:5.2f}")

print("-" * 56)
print(f"MEAN  base={np.mean(GB):.3f}  own={np.mean(GO):.3f}  avg={np.mean(GA):.3f}")
print(f"repos helped:  own={ho}/{len(byrepo)}   avg={ha}/{len(byrepo)}")
benefit_own = np.mean(GB) - np.mean(GO)
benefit_avg = np.mean(GB) - np.mean(GA)
print(f"benefit recovered by AVG (no conditioning): {100*benefit_avg/benefit_own:.0f}% of OWN")
