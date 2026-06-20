"""code2lora CLI — run the OFFICIAL released hypernetwork on YOUR OWN repo.

  encode  <repo>                 -> compute+cache the 2048-d repo embedding
  complete <repo> "<prompt>"     -> base vs repo-adapted completion (side by side)
  selftest <repo> [N]            -> base vs adapted CE on N held-out lines of the
                                    repo's own .py files (quantitative "does it help")

Flow: encode_repo (Qwen3-Embedding-0.6B, official recipe) -> OfficialDirectHead
(official code2lora-direct checkpoint) -> rank-16 LoRA -> inject into frozen
Qwen2.5-Coder-1.5B (all 28 layers) -> generate. Zero inference-time token overhead.
"""

import hashlib
import random
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download

from c2l import qwen_lora
from c2l.official_direct import OfficialDirectHead
from c2l.qwen3_repo_encoder import _iter_py, encode_repo

BASE = "Qwen/Qwen2.5-Coder-1.5B"
CACHE = Path(__file__).parent / ".emb_cache"


def repo_embedding(repo: str) -> np.ndarray:
    CACHE.mkdir(exist_ok=True)
    key = hashlib.sha1(str(Path(repo).resolve()).encode()).hexdigest()[:16]
    f = CACHE / f"{key}.npy"
    if f.exists():
        print(f"[cache] {f.name}")
        return np.load(f)
    emb = encode_repo(repo)
    np.save(f, emb)
    return emb


def make_adapter(repo: str):
    emb = repo_embedding(repo)
    ckpt = hf_hub_download("code2lora/code2lora-direct", "code2lora_direct.pt")
    head = OfficialDirectHead(ckpt)
    ad = head.forward(mx.array(emb))
    mx.eval([t for ab in ad.values() for t in ab])
    return ad


def inject(model, ad):
    for layer in qwen_lora._layers(model):
        for _m, (s, a) in qwen_lora._PROJ.items():
            A, B = ad[a]
            getattr(getattr(layer, s), a).set_lora(A, 2.0 * B)  # alpha/rank = 2


def _ce(model, tok, prefix, target):
    ip = tok.encode(prefix)
    ifu = tok.encode(prefix + target)
    b = len(ip)
    if len(ifu) <= b:
        return None
    lg = model(mx.array([ifu]))[0]
    return float(nn.losses.cross_entropy(lg[b - 1:len(ifu) - 1], mx.array(ifu[b:]), reduction="mean"))


def cmd_complete(repo, prompt, max_tokens=60):
    ad = make_adapter(repo)
    model, tok = qwen_lora.load_base_model(BASE)
    qwen_lora.clear_lora(model)
    base = qwen_lora.generate_text(model, tok, prompt, max_tokens=max_tokens)
    inject(model, ad)
    adpt = qwen_lora.generate_text(model, tok, prompt, max_tokens=max_tokens)
    print("\n=== PROMPT ===\n" + prompt)
    print("\n=== [BASE] (frozen Qwen2.5-Coder-1.5B) ===\n" + base)
    print("\n=== [ADAPTED] (+ repo LoRA) ===\n" + adpt)


def cmd_selftest(repo, n=12):
    ad = make_adapter(repo)
    model, tok = qwen_lora.load_base_model(BASE)
    rng = random.Random(0)
    files = [p for p in _iter_py(Path(repo))]
    rng.shuffle(files)
    samples = []
    for p in files:
        try:
            lines = p.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        idx = [i for i in range(len(lines)) if len(lines[i].strip()) > 12 and i > 8]
        if not idx:
            continue
        i = rng.choice(idx)
        prefix = "\n".join(lines[max(0, i - 40):i]) + "\n"
        target = lines[i]
        if len(prefix) > 30:
            samples.append((str(p.relative_to(repo)), prefix, target))
        if len(samples) >= n:
            break
    bb, aa = [], []
    for _, prefix, target in samples:
        qwen_lora.clear_lora(model)
        cb = _ce(model, tok, prefix, target)
        inject(model, ad)
        ca = _ce(model, tok, prefix, target)
        if cb is not None and ca is not None:
            bb.append(cb)
            aa.append(ca)
    mb, ma = sum(bb) / len(bb), sum(aa) / len(aa)
    print(f"\n[selftest {repo}] n={len(bb)} held-out lines of the repo's own .py")
    print(f"  base CE    = {mb:.3f}")
    print(f"  adapted CE = {ma:.3f}   delta={ma-mb:+.3f}  ({'HELPS' if ma<mb else 'hurts'})")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    cmd, repo = sys.argv[1], sys.argv[2]
    if cmd == "encode":
        emb = repo_embedding(repo)
        print(f"repo embedding: shape={emb.shape} norm={np.linalg.norm(emb):.3f}")
    elif cmd == "complete":
        cmd_complete(repo, sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 60)
    elif cmd == "selftest":
        cmd_selftest(repo, int(sys.argv[3]) if len(sys.argv) > 3 else 12)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
