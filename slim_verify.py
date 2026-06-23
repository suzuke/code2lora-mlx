"""STANDALONE slim code2lora — loads only slim_code2lora.npz (a fixed shared LoRA).
NO hypernetwork, NO encoder, NO repo embedding. Proves the slim artifact alone
reproduces the official's held-out CE/EM, and shows real completions.
"""

import json
import re

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from c2l import qwen_lora  # only the base-model LoRA wrapper — no head/encoder

BASE = "Qwen/Qwen2.5-Coder-1.5B"
TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

d = np.load("slim_code2lora.npz")
adapter = {t: (mx.array(d[f"A_{t}"]), mx.array(d[f"B_{t}"])) for t in TYPES}
nparams = sum(d[k].size for k in d.files)
print(f"loaded slim_code2lora.npz: {nparams/1e3:.0f}k params (no encoder, no hypernetwork)")

model, tok = qwen_lora.load_base_model(BASE)


def inject():
    for layer in qwen_lora._layers(model):
        for _m, (s, a) in qwen_lora._PROJ.items():
            A, B = adapter[a]
            getattr(getattr(layer, s), a).set_lora(A, 2.0 * B)


def ce(prefix, target):
    ip = tok.encode(prefix)
    fu = tok.encode(prefix + target)
    b = len(ip)
    if len(fu) <= b:
        return None
    lg = model(mx.array([fu]))[0]
    return float(nn.losses.cross_entropy(lg[b - 1:len(fu) - 1], mx.array(fu[b:]), reduction="mean"))


def _norm(s):
    return re.sub(r"\s+", " ", s.strip()).rstrip(" \t;:,.)]}")


def em(prefix, target):
    g = _norm(qwen_lora.generate_text(model, tok, prefix, max_tokens=24))
    return bool(_norm(target)) and g.startswith(_norm(target))


rows = [json.loads(l) for l in open("data/rpb25/heldout.jsonl")]
byrepo = {}
for r in rows:
    byrepo.setdefault(r["repo_id"], []).append(r)

cb, cs = [], []
for rid, exs in byrepo.items():
    for ex in exs[:12]:
        qwen_lora.clear_lora(model)
        a = ce(ex["input_prefix"], ex["target_value"])
        inject()
        b = ce(ex["input_prefix"], ex["target_value"])
        if a is not None and b is not None:
            cb.append(a)
            cs.append(b)
eb = es = en = 0
for rid, exs in byrepo.items():
    for ex in exs[:8]:
        en += 1
        qwen_lora.clear_lora(model)
        eb += em(ex["input_prefix"], ex["target_value"])
        inject()
        es += em(ex["input_prefix"], ex["target_value"])

print("=" * 56)
print(f"STANDALONE slim verify (held-out, loaded from disk):")
print(f"  CE  base={np.mean(cb):.3f}  slim={np.mean(cs):.3f}")
print(f"  EM  base={100*eb/en:.1f}%  slim={100*es/en:.1f}%")
print(f"  target (official 680M): CE 0.377 / EM 59.4%  →  "
      f"{'REPRODUCED ✓' if np.mean(cs) < 0.42 and es/en > 0.5 else 'MISMATCH'}")

# 2 real completions from the standalone artifact
print("-" * 56)
for rid in list(byrepo)[:2]:
    ex = byrepo[rid][0]
    qwen_lora.clear_lora(model)
    base = qwen_lora.generate_text(model, tok, ex["input_prefix"], max_tokens=16)
    inject()
    slim = qwen_lora.generate_text(model, tok, ex["input_prefix"], max_tokens=16)
    print(f"[{rid}] gold={ex['target_value'][:40]!r}")
    print(f"   BASE: {base[:44]!r}")
    print(f"   SLIM: {slim[:44]!r}")
