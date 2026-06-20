"""Validate our Qwen3 repo encoder against the OFFICIAL stored embedding.

Clones a held-out repo, encodes it with our encoder, and checks:
  (a) cosine(my_emb, official_stored_emb)  [confounded by commit drift, so
      "high-ish" is success, not necessarily ~1.0]
  (b) the REAL test: does an adapter generated from MY embedding also HELP
      on that repo's held-out examples (vs base, and vs the stored-emb adapter)?
"""

import json
import os
import subprocess

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download

from c2l import qwen_lora
from c2l.official_direct import OfficialDirectHead
from c2l.qwen3_repo_encoder import encode_repo

REPO_ID = "MrPowers/chispa"
REPO_GIT = "https://github.com/MrPowers/chispa"
CLONE = "/tmp/c2l_val_chispa"
BASE = "Qwen/Qwen2.5-Coder-1.5B"
N = 12

if not os.path.exists(CLONE):
    subprocess.run(["git", "clone", "--depth", "1", REPO_GIT, CLONE], check=True)

print("encoding repo with OUR Qwen3 encoder ...")
my_emb = encode_repo(CLONE)

rows = [json.loads(l) for l in open("data/rpb25/heldout.jsonl")]
rows = [r for r in rows if r["repo_id"] == REPO_ID]
stored = np.array(rows[0]["repo_embedding"], dtype=np.float32)

cos = float(np.dot(my_emb, stored) / (np.linalg.norm(my_emb) * np.linalg.norm(stored) + 1e-12))
print(f"\ncosine(my_emb, official_stored) = {cos:.4f}   (drift-confounded; >0.8 is strong)")

ckpt = hf_hub_download("code2lora/code2lora-direct", "code2lora_direct.pt")
head = OfficialDirectHead(ckpt)
model, tok = qwen_lora.load_base_model(BASE)


def adapter(emb):
    ad = head.forward(mx.array(emb))
    mx.eval([t for ab in ad.values() for t in ab])
    return ad


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


ad_my = adapter(my_emb)
ad_st = adapter(stored)
bb, mm, ss = [], [], []
for ex in rows[:N]:
    p, t = ex["input_prefix"], ex["target_value"]
    qwen_lora.clear_lora(model)
    cb = ce(p, t)
    inject(ad_my)
    cm = ce(p, t)
    inject(ad_st)
    cs = ce(p, t)
    if None in (cb, cm, cs):
        continue
    bb.append(cb)
    mm.append(cm)
    ss.append(cs)


def mean(x):
    return sum(x) / len(x)


print(f"\n[{REPO_ID}] n={len(bb)}")
print(f"  base CE                 = {mean(bb):.3f}")
print(f"  adapted (MY encoder)    = {mean(mm):.3f}   delta={mean(mm)-mean(bb):+.3f}")
print(f"  adapted (OFFICIAL store)= {mean(ss):.3f}   delta={mean(ss)-mean(bb):+.3f}")
print("\nencoder OK if MY-encoder adapter also helps (delta < 0), close to official.")
