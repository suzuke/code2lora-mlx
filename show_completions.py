"""Show the REAL completion effect: for held-out examples, print the prompt tail,
the gold value, and what BASE vs ADAPTED (official per-repo code2lora) actually
generate. No cherry-picking — first few per repo.
"""

import json
import re

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download

from c2l import qwen_lora
from c2l.official_direct import OfficialDirectHead, TYPES

BASE = "Qwen/Qwen2.5-Coder-1.5B"
PER_REPO = 2
GEN_TOK = 18

head = OfficialDirectHead(hf_hub_download("code2lora/code2lora-direct", "code2lora_direct.pt"))
rows = [json.loads(l) for l in open("data/rpb25/heldout.jsonl")]
byrepo = {}
for r in rows:
    byrepo.setdefault(r["repo_id"], []).append(r)

own = {}
for rid, exs in byrepo.items():
    ad = head.forward(mx.array(np.array(exs[0]["repo_embedding"], dtype=np.float32)))
    mx.eval([t for ab in ad.values() for t in ab])
    own[rid] = {t: (ad[t][0], ad[t][1]) for t in TYPES}

model, tok = qwen_lora.load_base_model(BASE)


def inject(ad):
    for layer in qwen_lora._layers(model):
        for _m, (s, a) in qwen_lora._PROJ.items():
            A, B = ad[a]
            getattr(getattr(layer, s), a).set_lora(A, 2.0 * B)


def _norm(s):
    return re.sub(r"\s+", " ", s.strip()).rstrip(" \t;:,.)]}")


def hit(gen, target):
    return _norm(gen).startswith(_norm(target)) and bool(_norm(target))


def short(s, n=46):
    return repr(s[:n])


hb = ha = n = 0
for rid, exs in byrepo.items():
    print("\n" + "═" * 70)
    print(f"📦 {rid}")
    for ex in exs[:PER_REPO]:
        prefix, gold = ex["input_prefix"], ex["target_value"]
        qwen_lora.clear_lora(model)
        base = qwen_lora.generate_text(model, tok, prefix, max_tokens=GEN_TOK)
        inject(own[rid])
        adpt = qwen_lora.generate_text(model, tok, prefix, max_tokens=GEN_TOK)
        n += 1
        bok, aok = hit(base, gold), hit(adpt, gold)
        hb += bok
        ha += aok
        tail = prefix[-90:].replace("\n", "⏎")
        print(f"\n  …{tail}")
        print(f"     GOLD : {short(gold)}")
        print(f"     BASE : {short(base)}   {'✓' if bok else '✗'}")
        print(f"     ADAPT: {short(adpt)}   {'✓' if aok else '✗'}")

print("\n" + "═" * 70)
print(f"TOTAL: BASE {hb}/{n} correct   ADAPTED {ha}/{n} correct")
