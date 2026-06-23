"""Scale up the cross-repo adapter-cosine measurement: generate the official
hypernetwork's adapter for EVERY available repo and report pairwise cosine.
(Pure head forward + numpy — no Qwen model.) Answers how robust the collapse is.
"""

import itertools
import json
import random

import mlx.core as mx
import numpy as np
import numpy.linalg as la
from huggingface_hub import hf_hub_download

from c2l.official_direct import OfficialDirectHead, TYPES

head = OfficialDirectHead(hf_hub_download("code2lora/code2lora-direct", "code2lora_direct.pt"))

repos = {}
for fn in ["data/rpb25/train.jsonl", "data/rpb25/heldout.jsonl"]:
    for l in open(fn):
        r = json.loads(l)
        repos.setdefault(r["repo_id"], r["repo_embedding"])
ids = list(repos)
print(f"unique repos available: {len(ids)}")

vecs = []
for k, rid in enumerate(ids):
    ad = head.forward(mx.array(np.array(repos[rid], dtype=np.float32)))
    mx.eval([x for ab in ad.values() for x in ab])
    vecs.append(np.concatenate([np.array(ad[t][0]).ravel() for t in TYPES]).astype(np.float32))
    if (k + 1) % 100 == 0:
        print(f"  generated {k+1}/{len(ids)} adapters")
vecs = np.stack(vecs)
vn = vecs / (la.norm(vecs, axis=1, keepdims=True) + 1e-12)

pairs = list(itertools.combinations(range(len(ids)), 2))
random.seed(0)
if len(pairs) > 50000:
    pairs = random.sample(pairs, 50000)
cos = np.array([float(vn[i] @ vn[j]) for i, j in pairs])
print(f"\ncross-repo adapter cosine over {len(ids)} repos ({len(pairs)} pairs):")
print(f"  mean {cos.mean():.5f}   min {cos.min():.5f}   max {cos.max():.5f}   std {cos.std():.6f}")
print(f"  fraction of pairs with cosine > 0.999: {100*(cos>0.999).mean():.1f}%")
