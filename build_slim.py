"""Build the slim code2lora artifact: a SINGLE fixed shared LoRA = centroid of the
official hypernetwork's outputs over N training repos. One-time build (uses the
official hypernetwork once); the result is a standalone ~659k-param adapter that
needs NO encoder and NO hypernetwork at inference.
"""

import json

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download

from c2l.official_direct import OfficialDirectHead, TYPES

N_TRAIN = 150
OUT = "slim_code2lora.npz"

head = OfficialDirectHead(hf_hub_download("code2lora/code2lora-direct", "code2lora_direct.pt"))
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

n = len(seen)
out = {}
for t in TYPES:
    out[f"A_{t}"] = np.array(acc[t][0] / n)
    out[f"B_{t}"] = np.array(acc[t][1] / n)
np.savez(OUT, **out)
nparams = sum(v.size for v in out.values())
print(f"saved {OUT} from {n} train repos — {nparams/1e3:.0f}k params, "
      f"{sum(v.nbytes for v in out.values())/1e6:.1f} MB")
