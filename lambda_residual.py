"""λ-residual functional test — does the official hypernetwork's per-repo residual
`(own - avg)` carry a REAL, functionally-useful conditioning direction, or is it dead?

Builds, per held-out repo & module, a blended adapter
    A_b = avgA + λ(ownA - avgA),  B_b = avgB + λ(ownB - avgB)
for λ ∈ {0, 0.5, 1, 2, 4, 8}, injects it (B×2 for α/r, exactly like everywhere),
and scores target-span token NLL (mean -log p over the target tokens) + gold
first-token logit rank. λ=0 = pure population (train-centroid) adapter; λ=1 = own.

CONTROL: same sweep with a *different* repo's own adapter as the residual source
(`other`) — if `other` helps as often as `own`, any gain is generic residual norm,
not repo signal.

Decisive reading:
  - λ=1 & λ>1 help NO stratum            -> residual functionally DEAD (collapse⇒no-cond STRENGTHENED)
  - λ>1 helps but λ=1 flat               -> conditioning dir EXISTS, head under-amplifies (the FLIP)
  - own helps, other doesn't             -> the residual direction is repo-specific (real signal)
  - own and other help equally           -> gain is generic residual magnitude, not repo signal

Run: cd code2lora-mlx && PYTHONPATH=. uv run --with torch python lambda_residual.py
Smoke (1 repo, few ex): C2L_SMOKE=1 PYTHONPATH=. uv run --with torch python lambda_residual.py
"""

import json
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download

from c2l import qwen_lora
from c2l.official_direct import OfficialDirectHead, TYPES

BASE = "Qwen/Qwen2.5-Coder-1.5B"
LAMBDAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
N_TRAIN = 150                      # population centroid = avg (no test leakage, matches ablate_harder)
SMOKE = os.environ.get("C2L_SMOKE") == "1"
PER_REPO = 8 if SMOKE else 60
N_BOOT = 2000
RANK_TOPK = None                   # full-vocab rank

ckpt = hf_hub_download("code2lora/code2lora-direct", "code2lora_direct.pt")
head = OfficialDirectHead(ckpt)


def adapter_for(emb):
    ad = head.forward(mx.array(np.array(emb, dtype=np.float32)))
    mx.eval([t for ab in ad.values() for t in ab])
    return {t: (ad[t][0], ad[t][1]) for t in TYPES}


# ---- held-out repos + their own adapters ----
rows = [json.loads(l) for l in open("data/rpb25/heldout.jsonl")]
byrepo = {}
for r in rows:
    byrepo.setdefault(r["repo_id"], []).append(r)
repo_ids = list(byrepo)
if SMOKE:
    repo_ids = repo_ids[:1]
own = {rid: adapter_for(byrepo[rid][0]["repo_embedding"]) for rid in repo_ids}
# deterministic control: each repo's "other" = the next repo's own adapter
other_src = {rid: repo_ids[(i + 1) % len(repo_ids)] for i, rid in enumerate(repo_ids)}

# ---- avg = running-mean adapter over N_TRAIN training repos ----
print(f"building population centroid from {N_TRAIN} train repos ...")
acc = {t: [None, None] for t in TYPES}
seen = set()
for line in open("data/rpb25/train.jsonl"):
    rid = json.loads(line)["repo_id"]
    if rid in seen:
        continue
    seen.add(rid)
    ad = head.forward(mx.array(np.array(json.loads(line)["repo_embedding"], dtype=np.float32)))
    for t in TYPES:
        acc[t][0] = ad[t][0] if acc[t][0] is None else acc[t][0] + ad[t][0]
        acc[t][1] = ad[t][1] if acc[t][1] is None else acc[t][1] + ad[t][1]
    mx.eval([x for ab in acc.values() for x in ab])
    if len(seen) >= N_TRAIN:
        break
ntr = len(seen)
avg = {t: (acc[t][0] / ntr, acc[t][1] / ntr) for t in TYPES}
mx.eval([x for ab in avg.values() for x in ab])
print(f"  centroid from {ntr} repos ready")

# residual norm sanity (how much signal is there to amplify, per codex's ~5%)
res_frac = []
for rid in repo_ids:
    num = den = 0.0
    for t in TYPES:
        for j in (0, 1):
            o, a = own[rid][t][j], avg[t][j]
            num += float(mx.sum((o - a) ** 2)); den += float(mx.sum(o ** 2))
    res_frac.append((num / den) ** 0.5)
print(f"  mean ||own-avg||/||own|| over {len(repo_ids)} held-out = {np.mean(res_frac):.3f}")


def blend(src, lam):
    return {t: (avg[t][0] + lam * (src[t][0] - avg[t][0]),
                avg[t][1] + lam * (src[t][1] - avg[t][1])) for t in TYPES}


def inject(ad):
    for layer in qwen_lora._layers(model):
        for _m, (s, a) in qwen_lora._PROJ.items():
            A, B = ad[a]
            getattr(getattr(layer, s), a).set_lora(A, 2.0 * B)


model, tok = qwen_lora.load_base_model(BASE)


def score(prefix, target):
    """Returns (nll, gold_rank, n_target_tokens) for the current injected adapter."""
    ip = tok.encode(prefix)
    ifu = tok.encode(prefix + target)
    b = len(ip)
    if len(ifu) <= b:
        return None
    lg = model(mx.array([ifu]))[0]                       # (seq, vocab)
    tgt = mx.array(ifu[b:])
    pred = lg[b - 1:len(ifu) - 1]                        # logits predicting each target tok
    nll = float(nn.losses.cross_entropy(pred, tgt, reduction="mean"))
    first = pred[0]                                      # logits for the first target token
    gold = ifu[b]
    rank = int(mx.sum(first > first[gold]))              # # tokens beating gold (0 = top-1)
    return nll, rank, len(ifu) - b


# ---- collect: for every (repo, example) score base + each (source,λ) config ----
# config key: "base", "own@λ", "other@λ". λ=0 is the same adapter for own/other (=avg) -> store once as "avg".
CONFIGS = ["base", "avg"] + [f"{s}@{l}" for s in ("own", "other") for l in LAMBDAS if l != 0.0]

records = []   # one dict per scored example, with nll/rank per config + strata
for ri, rid in enumerate(repo_ids):
    print(f"[{ri+1}/{len(repo_ids)}] {rid} ...", flush=True)
    exs = byrepo[rid][:PER_REPO]
    # pre-build this repo's adapters keyed by config
    ads = {"base": None, "avg": blend(own[rid], 0.0)}
    for l in LAMBDAS:
        if l == 0.0:
            continue
        ads[f"own@{l}"] = blend(own[rid], l)
        ads[f"other@{l}"] = blend(own[other_src[rid]], l)
    for cfg in CONFIGS:
        qwen_lora.clear_lora(model)
        if ads[cfg] is not None:
            inject(ads[cfg])
        for ei, ex in enumerate(exs):
            pre, tgt = ex["input_prefix"], ex["target_value"]
            r = score(pre, tgt)
            if r is None:
                continue
            nll, rank, ntok = r
            # find/create the record for this example
            if cfg == CONFIGS[0]:
                records.append({
                    "rid": rid, "ei": ei, "nll": {}, "rank": {},
                    "tlen": ntok, "tin": tgt.strip() in pre,
                })
            rec = next((x for x in records if x["rid"] == rid and x["ei"] == ei), None)
            if rec is None:
                continue
            rec["nll"][cfg] = nll
            rec["rank"][cfg] = rank

# keep only fully-scored examples (present in every config)
recs = [r for r in records if all(c in r["nll"] for c in CONFIGS)]
print(f"\nfully-scored examples: {len(recs)} / {len(records)}")
if not recs:
    raise SystemExit("no examples scored")

tlens = sorted(r["tlen"] for r in recs)
median_len = tlens[len(tlens) // 2]


def boot_ci(deltas):
    d = np.array(deltas, dtype=np.float64)
    if len(d) == 0:
        return (float("nan"), float("nan"), float("nan"))
    idx = np.random.default_rng(0).integers(0, len(d), size=(N_BOOT, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report(subset, label):
    if not subset:
        return
    print(f"\n### {label}  (n={len(subset)})")
    base_nll = np.mean([r["nll"]["base"] for r in subset])
    avg_nll = np.mean([r["nll"]["avg"] for r in subset])
    print(f"  base NLL={base_nll:.3f}   avg(λ=0) NLL={avg_nll:.3f}   "
          f"avg vs base Δ={avg_nll-base_nll:+.3f}")
    print(f"  {'λ':>4} | {'OWN ΔNLL vs λ=0 [95% CI]':<34} | {'OTHER ΔNLL vs λ=0 [95% CI]':<34}")
    for l in LAMBDAS:
        cells = []
        for s in ("own", "other"):
            if l == 0.0:
                cells.append(f"{'0.000 (=avg ref)':<34}")
                continue
            cfg = f"{s}@{l}"
            deltas = [r["nll"][cfg] - r["nll"]["avg"] for r in subset]
            m, lo, hi = boot_ci(deltas)
            sig = "*" if hi < 0 else ("+" if lo > 0 else " ")
            cells.append(f"{m:+.3f} [{lo:+.3f},{hi:+.3f}] {sig:<11}")
        print(f"  {l:>4} | {cells[0]} | {cells[1]}")
    # first-token rank @ λ=1 (own) vs avg
    r_avg = np.mean([r["rank"]["avg"] for r in subset])
    r_own1 = np.mean([r["rank"].get("own@1.0", r["rank"]["avg"]) for r in subset])
    print(f"  gold first-tok mean rank: avg={r_avg:.1f}  own@1={r_own1:.1f}  (lower=better)")


print("\n" + "=" * 70)
print("λ-RESIDUAL FUNCTIONAL TEST  —  ΔNLL < 0 means the residual HELPS")
print("  '*' = 95% CI entirely below 0 (sig. help); '+' = entirely above 0 (sig. hurt)")
print("=" * 70)
report(recs, "ALL held-out examples")
report([r for r in recs if r["tin"]], "STRATUM: target appears in prefix")
report([r for r in recs if not r["tin"]], "STRATUM: target NOT in prefix")
report([r for r in recs if r["tlen"] <= median_len], f"STRATUM: short target (≤{median_len} tok)")
report([r for r in recs if r["tlen"] > median_len], f"STRATUM: long target (>{median_len} tok)")

# dump raw for re-analysis
out = "lambda_residual_results.json"
with open(out, "w") as f:
    json.dump({"lambdas": LAMBDAS, "n_train_centroid": ntr,
               "mean_residual_frac": float(np.mean(res_frac)),
               "records": [{"rid": r["rid"], "tlen": r["tlen"], "tin": r["tin"],
                            "nll": r["nll"], "rank": r["rank"]} for r in recs]}, f)
print(f"\nraw results -> {out}")
