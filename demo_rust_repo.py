"""Empirical test: does the OFFICIAL (Python-trained) code2lora-direct checkpoint
help on a RUST repo? Encodes agend-terminal's .rs files, generates the adapter,
and measures base-vs-adapted CE on the repo's own `assert_eq!` completions.

Expectation: little-to-no benefit (or harm) — the hypernetwork only ever saw
Python repo embeddings + a Python assertion task. This quantifies that.
"""

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download

from c2l import qwen_lora
from c2l.official_direct import OfficialDirectHead
from c2l.qwen3_repo_encoder import _iter_py, encode_repo

REPO = sys.argv[1] if len(sys.argv) > 1 else "/path/to/a/rust/repo"
BASE = "Qwen/Qwen2.5-Coder-1.5B"
MAX_ENCODE_FILES = 200
N = 30


def extract_asserts(repo, max_ex=N):
    exs = []
    for p in _iter_py(Path(repo), exts=(".rs",)):
        try:
            lines = p.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        for i, ln in enumerate(lines):
            if i < 5 or "assert_eq!(" not in ln or ln.lstrip().startswith("//"):
                continue
            head, _, rhs = ln.partition("assert_eq!(")
            rhs = rhs.strip()
            if not (3 <= len(rhs) <= 80):
                continue
            prefix = "\n".join(lines[max(0, i - 25):i]) + "\n" + head + "assert_eq!("
            if len(prefix) > 40:
                exs.append((p.name, prefix, rhs))
            if len(exs) >= max_ex:
                return exs
    return exs


def ce(model, tok, prefix, target):
    ip = tok.encode(prefix)
    ifu = tok.encode(prefix + target)
    b = len(ip)
    if len(ifu) <= b:
        return None
    lg = model(mx.array([ifu]))[0]
    return float(nn.losses.cross_entropy(lg[b - 1:len(ifu) - 1], mx.array(ifu[b:]), reduction="mean"))


def main():
    print(f"repo: {REPO}")
    exs = extract_asserts(REPO)
    print(f"extracted {len(exs)} assert_eq! completions")

    print(f"encoding .rs (<= {MAX_ENCODE_FILES} files) with the official Qwen3 recipe ...")
    emb = encode_repo(REPO, exts=(".rs",), max_files=MAX_ENCODE_FILES)

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

    bb, aa, shown = [], [], 0
    for fn, prefix, target in exs:
        qwen_lora.clear_lora(model)
        cb = ce(model, tok, prefix, target)
        inject()
        ca = ce(model, tok, prefix, target)
        if cb is None or ca is None:
            continue
        bb.append(cb)
        aa.append(ca)
        if shown < 4:
            print(f"  [{fn}] assert_eq!({target[:34]:34} base={cb:.2f} adapt={ca:.2f}")
            shown += 1
    mb, ma = sum(bb) / len(bb), sum(aa) / len(aa)
    helped = sum(1 for x, y in zip(aa, bb) if x < y)
    print(f"\n[agend-terminal / RUST] n={len(bb)} assert_eq! completions")
    print(f"  base CE    = {mb:.3f}")
    print(f"  adapted CE = {ma:.3f}   delta={ma - mb:+.3f}  ({'HELPS' if ma < mb else 'HURTS'})")
    print(f"  examples improved: {helped}/{len(bb)}")

    fn, prefix, target = exs[0]
    qwen_lora.clear_lora(model)
    base = qwen_lora.generate_text(model, tok, prefix, max_tokens=16)
    inject()
    adpt = qwen_lora.generate_text(model, tok, prefix, max_tokens=16)
    print(f"\n  gold: assert_eq!({target[:50]!r}")
    print(f"  BASE  -> {base[:60]!r}")
    print(f"  ADAPT -> {adpt[:60]!r}")


if __name__ == "__main__":
    main()
