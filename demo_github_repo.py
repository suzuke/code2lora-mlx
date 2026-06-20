"""Pick a suitable PUBLIC GitHub Python repo (assertion-heavy, distinctive API,
NOT in the code2lora training/held-out set), clone it, and run the OFFICIAL
adapter on ITS OWN test assertions — the task the adapter was actually trained
for (predict the asserted value). Reports base-vs-adapted CE.
"""

import json
import os
import re
import subprocess
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download
from pathlib import Path

from c2l import qwen_lora
from c2l.official_direct import OfficialDirectHead
from c2l.qwen3_repo_encoder import _iter_py, encode_repo

BASE = "Qwen/Qwen2.5-Coder-1.5B"
CANDIDATES = [
    ("un33k/python-slugify", "https://github.com/un33k/python-slugify"),
    ("seatgeek/thefuzz", "https://github.com/seatgeek/thefuzz"),
    ("jd/tenacity", "https://github.com/jd/tenacity"),
    ("mahmoud/boltons", "https://github.com/mahmoud/boltons"),
    ("keleshev/schema", "https://github.com/keleshev/schema"),
]

train_ids = {json.loads(l)["repo_id"] for l in open("data/rpb25/train.jsonl")}
held_ids = {json.loads(l)["repo_id"] for l in open("data/rpb25/heldout.jsonl")}
SEEN = train_ids | held_ids


def extract_assertions(repo_dir, max_ex=40):
    exs = []
    for p in _iter_py(Path(repo_dir)):
        nm = p.name.lower()
        if "test" not in nm and "test" not in str(p.parent).lower():
            continue
        try:
            lines = p.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        for i, ln in enumerate(lines):
            if i < 5:
                continue
            s = ln.strip()
            if not s.startswith("assert ") or "==" not in ln:
                continue
            idx = ln.find("==")
            rhs = ln[idx + 2:].strip()
            if not (2 <= len(rhs) <= 60) or rhs.startswith("="):
                continue
            prefix = "\n".join(lines[max(0, i - 30):i]) + "\n" + ln[:idx + 2] + " "
            if len(prefix) > 40:
                exs.append((p.name, prefix, rhs))
            if len(exs) >= max_ex:
                return exs
    return exs


def pick_repo():
    for rid, url in CANDIDATES:
        if rid in SEEN:
            print(f"skip {rid} (in training/held-out set)")
            continue
        d = f"/tmp/c2l_demo_{rid.replace('/', '_')}"
        if not os.path.exists(d):
            r = subprocess.run(["git", "clone", "--depth", "1", url, d], capture_output=True)
            if r.returncode != 0:
                print(f"clone failed {rid}")
                continue
        exs = extract_assertions(d)
        print(f"candidate {rid}: {len(exs)} assertion examples")
        if len(exs) >= 12:
            return rid, d, exs
    return None, None, None


def ce(model, tok, prefix, target):
    ip = tok.encode(prefix)
    ifu = tok.encode(prefix + target)
    b = len(ip)
    if len(ifu) <= b:
        return None
    lg = model(mx.array([ifu]))[0]
    return float(nn.losses.cross_entropy(lg[b - 1:len(ifu) - 1], mx.array(ifu[b:]), reduction="mean"))


def main():
    rid, d, exs = pick_repo()
    if rid is None:
        print("no suitable repo found")
        sys.exit(1)
    print(f"\n=== chosen: {rid} ({len(exs)} assertion examples) — NOT in training set ===")

    emb = encode_repo(d)
    ckpt = hf_hub_download("code2lora/code2lora-direct", "code2lora_direct.pt")
    head = OfficialDirectHead(ckpt)
    ad = head.forward(mx.array(emb))
    mx.eval([t for ab in ad.values() for t in ab])

    model, tok = qwen_lora.load_base_model(BASE)

    def inject():
        for layer in qwen_lora._layers(model):
            for _m, (s, a) in qwen_lora._PROJ.items():
                A, B = ad[a]
                getattr(getattr(layer, s), a).set_lora(A, 2.0 * B)

    bb, aa = [], []
    shown = 0
    for fn, prefix, target in exs[:30]:
        qwen_lora.clear_lora(model)
        cb = ce(model, tok, prefix, target)
        inject()
        ca = ce(model, tok, prefix, target)
        if cb is None or ca is None:
            continue
        bb.append(cb)
        aa.append(ca)
        if shown < 4:
            print(f"  [{fn}] assert ... == {target!r:30} base={cb:.2f} adapt={ca:.2f}")
            shown += 1
    mb, ma = sum(bb) / len(bb), sum(aa) / len(aa)
    print(f"\n[{rid}] n={len(bb)} test-assertion completions")
    print(f"  base CE    = {mb:.3f}")
    print(f"  adapted CE = {ma:.3f}   delta={ma - mb:+.3f}  ({'HELPS' if ma < mb else 'hurts'})")
    helped = sum(1 for x, y in zip(aa, bb) if x < y)
    print(f"  examples improved: {helped}/{len(bb)}")

    # one greedy completion
    fn, prefix, target = exs[0]
    qwen_lora.clear_lora(model)
    base = qwen_lora.generate_text(model, tok, prefix, max_tokens=16)
    inject()
    adpt = qwen_lora.generate_text(model, tok, prefix, max_tokens=16)
    print(f"\n  gold value : {target!r}")
    print(f"  BASE  -> {base[:60]!r}")
    print(f"  ADAPT -> {adpt[:60]!r}")


if __name__ == "__main__":
    main()
