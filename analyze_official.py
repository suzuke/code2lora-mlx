"""Empirically probe the OFFICIAL code2lora-direct hypernetwork for optimization
headroom. No base-model forward needed — just the hypernetwork + linear algebra
on its outputs. Answers:
  1. Where do the ~680M params live? (which part is the bulk → compression target)
  2. How repo-distinct are the generated adapters? (cross-repo cosine; cf our
     from-scratch 0.998 near-constant failure)
  3. Is rank 16 actually used, or could a smaller rank do? (effective rank of A)
  4. Are the per-module log_scales redundant? (could collapse to 1 scalar)
"""

import itertools
import json
import math

import mlx.core as mx
import numpy as np
import numpy.linalg as la
import torch
from huggingface_hub import hf_hub_download

from c2l.official_direct import OfficialDirectHead, TYPES

ckpt = hf_hub_download("code2lora/code2lora-direct", "code2lora_direct.pt")
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
sd = ck["state_dict"]

print("=" * 64)
print("1) PARAMETER DISTRIBUTION — where do the params live?")
total = 0
bypart = {}
for k, v in sd.items():
    n = v.numel()
    total += n
    part = ("trunk" if k.startswith("trunk")
            else "heads_A" if k.startswith("heads_A")
            else "heads_B" if k.startswith("heads_B")
            else "log_scale")
    bypart[part] = bypart.get(part, 0) + n
print(f"  TOTAL: {total/1e6:.1f}M params")
for p, n in sorted(bypart.items(), key=lambda x: -x[1]):
    print(f"    {p:10s} {n/1e6:7.1f}M  ({100*n/total:4.1f}%)")
print("  per-module head sizes (the bulk):")
for t in TYPES:
    a = sd[f"heads_A.{t}.weight"].numel() + sd[f"heads_A.{t}.bias"].numel()
    b = sd[f"heads_B.{t}.weight"].numel() + sd[f"heads_B.{t}.bias"].numel()
    print(f"    {t:10s} A {a/1e6:5.1f}M  B {b/1e6:5.1f}M")

print("=" * 64)
print("2) log_scale REDUNDANCY")
ls = {k: math.exp(float(sd[k])) for k in sd if k.startswith("log_scale")}
vals = np.array(list(ls.values()))
print(f"  14 exp(log_scale) values: min={vals.min():.4f} max={vals.max():.4f} "
      f"mean={vals.mean():.4f} std={vals.std():.5f}")
print(f"  → spread {100*vals.std()/vals.mean():.1f}% of mean "
      f"({'≈ redundant, could be 1 scalar' if vals.std()/vals.mean() < 0.05 else 'meaningfully varied'})")

print("=" * 64)
print("3+4) ADAPTER ANALYSIS on held-out repos")
head = OfficialDirectHead(ckpt)
rows = [json.loads(l) for l in open("data/rpb25/heldout.jsonl")]
byrepo = {}
for r in rows:
    byrepo.setdefault(r["repo_id"], r)
adapters = {}
for rid, r in byrepo.items():
    ad = head.forward(mx.array(np.array(r["repo_embedding"], dtype=np.float32)))
    mx.eval([t for ab in ad.values() for t in ab])
    adapters[rid] = {t: (np.array(ad[t][0]), np.array(ad[t][1])) for t in TYPES}
repos = list(adapters)
print(f"  {len(repos)} held-out repos")


def cos(a, b):
    return float(a @ b / (la.norm(a) * la.norm(b) + 1e-12))


# (3) cross-repo distinctness: cosine of full flattened A-stack
vecs = {rid: np.concatenate([adapters[rid][t][0].ravel() for t in TYPES]) for rid in repos}
coss = [cos(vecs[i], vecs[j]) for i, j in itertools.combinations(repos, 2)]
print(f"  cross-repo adapter cosine (raw A): mean={np.mean(coss):.4f} "
      f"min={np.min(coss):.4f} max={np.max(coss):.4f}")
# also mean-centered (remove the shared 'average adapter' component)
allA = np.stack([vecs[r] for r in repos])
centered = allA - allA.mean(0, keepdims=True)
cc = [cos(centered[i], centered[j]) for i, j in itertools.combinations(range(len(repos)), 2)]
shared_norm = la.norm(allA.mean(0))
mean_indiv = np.mean([la.norm(v) for v in allA])
print(f"  → shared 'average adapter' is {100*shared_norm/mean_indiv:.0f}% of each adapter's norm")
print(f"  → after removing shared part, residual cross-repo cosine: mean={np.mean(cc):+.3f}")

# (4) effective rank of generated A per module (of 16)
print("  effective rank of generated A (participation ratio, max=16):")
for t in TYPES:
    ers = []
    for rid in repos:
        s = la.svd(adapters[rid][t][0], compute_uv=False)
        s2 = s ** 2
        ers.append((s2.sum() ** 2) / (s2 ** 2).sum())
    print(f"    {t:10s} eff_rank ≈ {np.mean(ers):4.1f} / 16")
