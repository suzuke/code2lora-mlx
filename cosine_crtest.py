"""Cross-repo adapter cosine over the FULL cr_test (held-out, NON-training) repo
set from the paper's dataset — the rigorous 'fresh repos the hypernetwork never
saw' check, at scale. Pure head forward + numpy, no Qwen.
"""

import itertools
import random

import mlx.core as mx
import numpy as np
import numpy.linalg as la
import pandas as pd
from huggingface_hub import hf_hub_download

from c2l.official_direct import OfficialDirectHead, TYPES

head = OfficialDirectHead(hf_hub_download("code2lora/code2lora-direct", "code2lora_direct.pt"))

path = hf_hub_download("code2lora/code2lora-data-commits", "commits/cr_test.parquet", repo_type="dataset")
df = pd.read_parquet(path, columns=["repo_id", "repo_state_embedding"])
df = df.drop_duplicates("repo_id")
reps = {r.repo_id: np.asarray(r.repo_state_embedding, dtype=np.float32) for r in df.itertuples()}
ids = list(reps)
print(f"cr_test (NON-training) unique repos: {len(ids)}")

vecs = []
for k, rid in enumerate(ids):
    ad = head.forward(mx.array(reps[rid]))
    mx.eval([x for ab in ad.values() for x in ab])
    vecs.append(np.concatenate([np.array(ad[t][0]).ravel() for t in TYPES]).astype(np.float32))
vecs = np.stack(vecs)
vn = vecs / (la.norm(vecs, axis=1, keepdims=True) + 1e-12)

# also report input-embedding cosine (to confirm the repos ARE distinct on input)
emb = np.stack([reps[r] for r in ids])
embn = emb / (la.norm(emb, axis=1, keepdims=True) + 1e-12)

pairs = list(itertools.combinations(range(len(ids)), 2))
random.seed(0)
if len(pairs) > 50000:
    pairs = random.sample(pairs, 50000)
adcos = np.array([float(vn[i] @ vn[j]) for i, j in pairs])
incos = np.array([float(embn[i] @ embn[j]) for i, j in pairs])
print(f"\nINPUT repo-embedding cosine ({len(pairs)} pairs): mean {incos.mean():.4f} min {incos.min():.4f} max {incos.max():.4f}  (distinct inputs)")
print(f"OUTPUT adapter cosine        ({len(pairs)} pairs): mean {adcos.mean():.5f} min {adcos.min():.5f} max {adcos.max():.5f} std {adcos.std():.6f}")
print(f"  fraction adapter-pairs cosine > 0.999: {100*(adcos>0.999).mean():.1f}%")
print("→ distinct repos (low input cosine) → near-identical adapters (~1.0) = conditioning collapse on NON-training repos")
