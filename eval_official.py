"""Run the OFFICIAL released `code2lora/code2lora-direct` checkpoint and measure
base-vs-adapted on our held-out repos. This is the definitive "does the method
work" test — using the paper's own trained hypernetwork (Qwen2.5-Coder-1.5B,
rank 16), not our from-scratch 0.5B run.
"""

import json
import sys

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import hf_hub_download

from c2l import qwen_lora
from c2l.official_direct import OfficialDirectHead

BASE = "Qwen/Qwen2.5-Coder-1.5B"
HELDOUT = "data/rpb25/heldout.jsonl"
PER_REPO = int(sys.argv[1]) if len(sys.argv) > 1 else 15


def inject_official(model, adapter):
    # same per-type A/B into EVERY layer; fold alpha/rank=2 into B (linear)
    for layer in qwen_lora._layers(model):
        for _mt, (sub, attr) in qwen_lora._PROJ.items():
            a, b = adapter[attr]
            getattr(getattr(layer, sub), attr).set_lora(a, 2.0 * b)


def target_ce(model, tok, prefix, target):
    ids_p = tok.encode(prefix)
    ids_f = tok.encode(prefix + target)
    b = len(ids_p)
    if len(ids_f) <= b:          # tokenizer merged the boundary away; skip
        return None
    logits = model(mx.array([ids_f]))[0]          # (L, V)
    pred = logits[b - 1:len(ids_f) - 1]
    tgt = mx.array(ids_f[b:])
    return float(nn.losses.cross_entropy(pred, tgt, reduction="mean"))


def main():
    ckpt = hf_hub_download("code2lora/code2lora-direct", "code2lora_direct.pt")
    head = OfficialDirectHead(ckpt)
    print(f"official head loaded: rank={head.rank} hidden={head.hidden}")

    model, tok = qwen_lora.load_base_model(BASE)
    n_layers = len(qwen_lora._layers(model))
    print(f"loaded {BASE}: {n_layers} layers wrapped")

    rows = [json.loads(l) for l in open(HELDOUT)]
    by_repo = {}
    for r in rows:
        by_repo.setdefault(r["repo_id"], []).append(r)

    grand_base, grand_adapt, helped = [], [], 0
    for repo, exs in by_repo.items():
        emb = mx.array(exs[0]["repo_embedding"])
        adapter = head.forward(emb)
        mx.eval([t for ab in adapter.values() for t in ab])

        bce, ace = [], []
        for ex in exs[:PER_REPO]:
            qwen_lora.clear_lora(model)
            cb = target_ce(model, tok, ex["input_prefix"], ex["target_value"])
            inject_official(model, adapter)
            ca = target_ce(model, tok, ex["input_prefix"], ex["target_value"])
            if cb is not None and ca is not None:
                bce.append(cb)
                ace.append(ca)
        mb, ma = sum(bce) / len(bce), sum(ace) / len(ace)
        grand_base += bce
        grand_adapt += ace
        if ma < mb:
            helped += 1
        print(f"{repo:34s} base={mb:6.3f}  adapt={ma:6.3f}  delta={ma-mb:+.3f}  ({'HELP' if ma<mb else 'hurt'})  n={len(bce)}")

    GB = sum(grand_base) / len(grand_base)
    GA = sum(grand_adapt) / len(grand_adapt)
    print("-" * 70)
    print(f"MEAN  base={GB:.3f}  adapt={GA:.3f}  delta={GA-GB:+.3f}  repos_helped={helped}/{len(by_repo)}")

    # one qualitative completion (first repo)
    repo0 = next(iter(by_repo))
    ex = by_repo[repo0][0]
    adapter = head.forward(mx.array(ex["repo_embedding"]))
    mx.eval([t for ab in adapter.values() for t in ab])
    prompt = ex["input_prefix"][-300:]
    qwen_lora.clear_lora(model)
    base_out = qwen_lora.generate_text(model, tok, prompt, max_tokens=40)
    inject_official(model, adapter)
    adpt_out = qwen_lora.generate_text(model, tok, prompt, max_tokens=40)
    print(f"\n[{repo0}] gold target: {ex['target_value'][:80]!r}")
    print(f"  BASE : {base_out[:120]!r}")
    print(f"  ADAPT: {adpt_out[:120]!r}")


if __name__ == "__main__":
    main()
