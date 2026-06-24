"""codex-challenger follow-ups #1/#2/#3 to the λ-residual functional test.

#1  embedding provenance (the only possible false-negative): confirm the repo_state
    embedding we fed IS the static/direct checkpoint's conditioning input, and that the
    only OTHER 2048-d embedding (diff_embedding, the Evo variant's input) does not
    resurrect a per-repo residual. (a) byte-identity of repo_state across the commits &
    snapshots dataset copies; (b) rerun own/other at λ∈{0,1,4} feeding L2-normed
    per-repo diff_embedding — if own stays flat/hurts, no alt embedding wakes it up.
#2  ΔW-space (not factor-space) λ blend: D=2·B@A (effective delta), Dλ=D_avg+λ(D_own-D_avg),
    SVD back to rank-16, inject raw. Kills the bilinear-cross-term escape hatch for the
    λ>1 under-amplification inference. λ=0,1 must MATCH the factor run (built-in sanity).
#3  all-wrong-repo control: replace the single deterministic next-repo null with ALL 7
    wrong repos per source repo. Report where own falls in the wrong-repo ΔNLL distribution
    (own should NOT beat the wrong-repo median).

Run: PYTHONPATH=. uv run --with torch python lambda_followups.py
Smoke: C2L_SMOKE=1 PYTHONPATH=. uv run --with torch python lambda_followups.py
"""
import json
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from c2l import qwen_lora
from c2l.official_direct import OfficialDirectHead, TYPES

BASE = "Qwen/Qwen2.5-Coder-1.5B"
N_TRAIN = 150
SMOKE = os.environ.get("C2L_SMOKE") == "1"
PER_REPO = 8 if SMOKE else 60
N_BOOT = 2000
RESULTS = {}

ckpt = hf_hub_download("code2lora/code2lora-direct", "code2lora_direct.pt")
head = OfficialDirectHead(ckpt)


def adapter_for(emb):
    ad = head.forward(mx.array(np.array(emb, dtype=np.float32)))
    mx.eval([t for ab in ad.values() for t in ab])
    return {t: (ad[t][0], ad[t][1]) for t in TYPES}


def centroid(emb_list):
    acc = {t: [None, None] for t in TYPES}
    for emb in emb_list:
        ad = head.forward(mx.array(np.array(emb, dtype=np.float32)))
        for t in TYPES:
            acc[t][0] = ad[t][0] if acc[t][0] is None else acc[t][0] + ad[t][0]
            acc[t][1] = ad[t][1] if acc[t][1] is None else acc[t][1] + ad[t][1]
        mx.eval([x for ab in acc.values() for x in ab])
    n = len(emb_list)
    out = {t: (acc[t][0] / n, acc[t][1] / n) for t in TYPES}
    mx.eval([x for ab in out.values() for x in ab])
    return out


rows = [json.loads(l) for l in open("data/rpb25/heldout.jsonl")]
byrepo = {}
for r in rows:
    byrepo.setdefault(r["repo_id"], []).append(r)
repo_ids = list(byrepo)
if SMOKE:
    repo_ids = repo_ids[:2]
own = {rid: adapter_for(byrepo[rid][0]["repo_embedding"]) for rid in repo_ids}

print(f"building repo_state population centroid from {N_TRAIN} train repos ...")
tr_seen, tr_embs = set(), []
for line in open("data/rpb25/train.jsonl"):
    r = json.loads(line)
    if r["repo_id"] in tr_seen:
        continue
    tr_seen.add(r["repo_id"]); tr_embs.append(r["repo_embedding"])
    if len(tr_seen) >= N_TRAIN:
        break
avg = centroid(tr_embs)
print(f"  centroid from {len(tr_embs)} repos ready")

model, tok = qwen_lora.load_base_model(BASE)


def inject_factor(ad):
    for layer in qwen_lora._layers(model):
        for _m, (s, a) in qwen_lora._PROJ.items():
            A, B = ad[a]
            getattr(getattr(layer, s), a).set_lora(A, 2.0 * B)


def inject_delta(deltas):
    """deltas[t] = (A',B') already carrying the FULL effective ΔW (no ×2)."""
    for layer in qwen_lora._layers(model):
        for _m, (s, a) in qwen_lora._PROJ.items():
            A, B = deltas[a]
            getattr(getattr(layer, s), a).set_lora(A, B)


def score(prefix, target):
    ip = tok.encode(prefix); ifu = tok.encode(prefix + target); b = len(ip)
    if len(ifu) <= b:
        return None
    lg = model(mx.array([ifu]))[0]
    tgt = mx.array(ifu[b:]); pred = lg[b - 1:len(ifu) - 1]
    nll = float(nn.losses.cross_entropy(pred, tgt, reduction="mean"))
    rank = int(mx.sum(pred[0] > pred[0][ifu[b]]))
    return nll, rank


def blend_factor(src, lam):
    return {t: (avg[t][0] + lam * (src[t][0] - avg[t][0]),
                avg[t][1] + lam * (src[t][1] - avg[t][1])) for t in TYPES}


def boot_ci(d):
    d = np.array(d, dtype=np.float64)
    if len(d) == 0:
        return (float("nan"),) * 3
    idx = np.random.default_rng(0).integers(0, len(d), size=(N_BOOT, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def run_configs(config_adapters, label):
    """config_adapters: {name: adapters-or-None or ('delta',deltas)}. Returns records."""
    recs = []
    for ri, rid in enumerate(repo_ids):
        exs = byrepo[rid][:PER_REPO]
        for cname, spec in config_adapters[rid].items():
            qwen_lora.clear_lora(model)
            if spec is not None:
                kind, ad = spec
                (inject_delta if kind == "delta" else inject_factor)(ad)
            for ei, ex in enumerate(exs):
                r = score(ex["input_prefix"], ex["target_value"])
                if r is None:
                    continue
                rec = next((x for x in recs if x["rid"] == rid and x["ei"] == ei), None)
                if rec is None:
                    rec = {"rid": rid, "ei": ei, "nll": {}, "rank": {}}
                    recs.append(rec)
                rec["nll"][cname] = r[0]; rec["rank"][cname] = r[1]
        print(f"  [{label}] {ri+1}/{len(repo_ids)} {rid}", flush=True)
    return recs


# ============================================================ #1 provenance
print("\n" + "=" * 64 + "\n#1  EMBEDDING PROVENANCE\n" + "=" * 64)
# (a) identity of repo_state across the two dataset copies, for our repos
pc = hf_hub_download("code2lora/code2lora-data-commits", "commits/cr_test.parquet", repo_type="dataset")
ps = hf_hub_download("code2lora/code2lora-data-snapshots", "commits/cr_test.parquet", repo_type="dataset")
def load_rs(path):
    d = pq.ParquetFile(path).read(columns=["repo_id", "commit_index", "repo_state_embedding"]).to_pydict()
    return {(r, c): np.asarray(e, dtype=np.float32) for r, c, e in
            zip(d["repo_id"], d["commit_index"], d["repo_state_embedding"])}
rs_comm, rs_snap = load_rs(pc), load_rs(ps)
keys = [k for k in rs_comm if k[0] in repo_ids][:200]
maxdiff = max(float(np.max(np.abs(rs_comm[k] - rs_snap[k]))) for k in keys) if keys else float("nan")
print(f"(a) repo_state_embedding identity across data-commits vs data-snapshots:")
print(f"    checked {len(keys)} (repo,commit) keys, max |Δ| = {maxdiff:.2e}  "
      f"-> {'IDENTICAL' if maxdiff < 1e-3 else 'DIFFER!'}")
RESULTS["provenance_identity_maxdiff"] = maxdiff

# (b) diff_embedding probe — L2-normed per-repo mean diff emb; does own wake up?
dd = pq.ParquetFile(pc).read(columns=["repo_id", "diff_embedding"]).to_pydict()
diff_by_repo = {}
for r, e in zip(dd["repo_id"], dd["diff_embedding"]):
    diff_by_repo.setdefault(r, []).append(np.asarray(e, dtype=np.float32))
def l2(v):
    return v / (np.linalg.norm(v) + 1e-9)
own_diff = {rid: adapter_for(l2(np.mean(diff_by_repo[rid], axis=0))) for rid in repo_ids if rid in diff_by_repo}
# diff-centroid over train repos
dtr = pq.ParquetFile(hf_hub_download("code2lora/code2lora-data-commits", "commits/cr_train.parquet",
                                     repo_type="dataset")).read(columns=["repo_id", "diff_embedding"]).to_pydict()
dtr_by_repo = {}
for r, e in zip(dtr["repo_id"], dtr["diff_embedding"]):
    if r in tr_seen:
        dtr_by_repo.setdefault(r, []).append(np.asarray(e, dtype=np.float32))
avg_diff = centroid([l2(np.mean(v, axis=0)) for v in list(dtr_by_repo.values())[:N_TRAIN]])
def blend_diff(src, lam):
    return {t: (avg_diff[t][0] + lam * (src[t][0] - avg_diff[t][0]),
                avg_diff[t][1] + lam * (src[t][1] - avg_diff[t][1])) for t in TYPES}
cfg1 = {}
for i, rid in enumerate(repo_ids):
    oth = repo_ids[(i + 1) % len(repo_ids)]
    cfg1[rid] = {"base": None,
                 "avg_state": ("factor", blend_factor(own[rid], 0.0)),
                 "own_state@1": ("factor", blend_factor(own[rid], 1.0)),
                 "avg_diff": ("factor", blend_diff(own_diff[rid], 0.0)),
                 "own_diff@1": ("factor", blend_diff(own_diff[rid], 1.0)),
                 "own_diff@4": ("factor", blend_diff(own_diff[rid], 4.0)),
                 "other_diff@1": ("factor", blend_diff(own_diff[oth], 1.0))}
rec1 = run_configs(cfg1, "#1diff")
def mnll(recs, c):
    return float(np.mean([r["nll"][c] for r in recs if c in r["nll"]]))
print("\n#1(b) diff-embedding probe (mean NLL over held-out):")
for c in ["base", "avg_state", "own_state@1", "avg_diff", "own_diff@1", "own_diff@4", "other_diff@1"]:
    print(f"    {c:14s} {mnll(rec1, c):.4f}")
RESULTS["diff_probe"] = {c: mnll(rec1, c) for c in
                         ["base", "avg_state", "own_state@1", "avg_diff", "own_diff@1", "own_diff@4", "other_diff@1"]}

# ============================================================ #2 ΔW-space SVD blend
print("\n" + "=" * 64 + "\n#2  ΔW-SPACE (SVD) λ BLEND\n" + "=" * 64)
LAM2 = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
RANK = head.rank
def eff_delta(ad):
    """effective applied ΔW per module = (2·B) @ A."""
    return {t: np.array((2.0 * ad[t][1]) @ ad[t][0]) for t in TYPES}
def svd_factors(D, rank):
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    s = np.sqrt(S[:rank])
    A = (s[:, None] * Vt[:rank])            # (rank, in)
    B = (U[:, :rank] * s[None, :])          # (out, rank)
    return mx.array(A.astype(np.float32)), mx.array(B.astype(np.float32))
D_avg = eff_delta(avg)
cfg2 = {}
for i, rid in enumerate(repo_ids):
    oth = repo_ids[(i + 1) % len(repo_ids)]
    D_own = eff_delta(own[rid]); D_oth = eff_delta(own[oth])
    d = {"own_factor@1": ("factor", blend_factor(own[rid], 1.0))}  # same-set anchor for sanity
    for lam in LAM2:
        d[f"own@{lam}"] = ("delta", {t: svd_factors(D_avg[t] + lam * (D_own[t] - D_avg[t]), RANK) for t in TYPES})
        if lam != 0.0:
            d[f"other@{lam}"] = ("delta", {t: svd_factors(D_avg[t] + lam * (D_oth[t] - D_avg[t]), RANK) for t in TYPES})
    cfg2[rid] = d
rec2 = run_configs(cfg2, "#2svd")
print("\n#2 ΔW-SVD blend — ΔNLL vs λ=0 (own@0.0 = ΔW-space avg):")
ref2 = "own@0.0"
print(f"    sanity (same set): ΔW-SVD own@1.0 NLL={mnll(rec2,'own@1.0'):.4f} vs factor own@1 "
      f"NLL={mnll(rec2,'own_factor@1'):.4f}  (≈ up to bf16 factor-redistribution)")
RESULTS["svd_sanity"] = {"svd_own1": mnll(rec2, "own@1.0"), "factor_own1": mnll(rec2, "own_factor@1")}
print(f"  {'λ':>4} | {'OWN ΔNLL [95%CI]':<30} | {'OTHER ΔNLL [95%CI]':<30}")
res2 = {}
for lam in LAM2:
    cells = []
    for s in ("own", "other"):
        if lam == 0.0:
            cells.append(f"{'0 (ref)':<30}"); continue
        d = [r["nll"][f"{s}@{lam}"] - r["nll"][ref2] for r in rec2 if f"{s}@{lam}" in r["nll"]]
        m, lo, hi = boot_ci(d); sig = "*HELP" if hi < 0 else ("*HURT" if lo > 0 else "")
        cells.append(f"{m:+.4f} [{lo:+.4f},{hi:+.4f}] {sig}")
        res2[f"{s}@{lam}"] = [m, lo, hi]
    print(f"  {lam:>4} | {cells[0]:<30} | {cells[1]:<30}")
RESULTS["svd_blend"] = res2

# ============================================================ #3 all-wrong-repo control
print("\n" + "=" * 64 + "\n#3  ALL-WRONG-REPO CONTROL (λ=1 and λ=4)\n" + "=" * 64)
cfg3 = {}
for rid in repo_ids:
    d = {"avg": ("factor", blend_factor(own[rid], 0.0)),
         "own@1": ("factor", blend_factor(own[rid], 1.0)),
         "own@4": ("factor", blend_factor(own[rid], 4.0))}
    for oth in repo_ids:
        if oth == rid:
            continue
        d[f"w@1::{oth}"] = ("factor", blend_factor(own[oth], 1.0))
        d[f"w@4::{oth}"] = ("factor", blend_factor(own[oth], 4.0))
    cfg3[rid] = d
rec3 = run_configs(cfg3, "#3wrong")
print("\n#3 own vs the distribution of all wrong repos (paired ΔNLL vs avg, mean over examples):")
res3 = {}
for lam in (1, 4):
    own_delta = float(np.mean([r["nll"][f"own@{lam}"] - r["nll"]["avg"] for r in rec3]))
    wrong_means = []
    for oth in repo_ids:
        ds = [r["nll"][f"w@{lam}::{oth}"] - r["nll"]["avg"] for r in rec3
              if f"w@{lam}::{oth}" in r["nll"] and r["rid"] != oth]
        if ds:
            wrong_means.append(float(np.mean(ds)))
    wm = np.array(wrong_means)
    pct = float(np.mean(wm <= own_delta) * 100)  # % of wrong repos at least as good as own
    print(f"  λ={lam}: own ΔNLL={own_delta:+.4f} | wrong-repo ΔNLL median={np.median(wm):+.4f} "
          f"[min {wm.min():+.4f}, max {wm.max():+.4f}] | own better than {100-pct:.0f}% of wrong repos")
    res3[f"lam{lam}"] = {"own": own_delta, "wrong_median": float(np.median(wm)),
                         "wrong_min": float(wm.min()), "wrong_max": float(wm.max()),
                         "own_pctile_among_wrong": pct}
RESULTS["all_wrong_repo"] = res3

with open("lambda_followups_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)
print("\nraw -> lambda_followups_results.json")
