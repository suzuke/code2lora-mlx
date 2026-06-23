"""Code2LoRA-Evo (GRU) evolution validation, faithful to the official
evaluation/run_code2lora_gru_v2_eval.py protocol:
  per repo: h=init_hidden(repo_state_emb[0]); for each commit (in commit_index
  order): h=gru.step(diff_emb,h); at commits that have QnAs: ctx=output_norm(h);
  adapter=head(ctx); inject; score EM/CE on that commit's QnAs.

THE TEST: does per-commit Evo beat a single fixed shared LoRA? (+ does the Evo
adapter actually change across commits, or collapse like static did?)
"""

import json
import re

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pds
import torch
from huggingface_hub import hf_hub_download

from c2l import qwen_lora
from c2l.commit_gru import EvoGRU
from c2l.official_direct import OfficialDirectHead, TYPES

BASE = "Qwen/Qwen2.5-Coder-1.5B"
MAX_PREFIX = 2000
MAX_SCORED_COMMITS = 4      # scored commits per repo (LLM eval bound)
QNAS_PER_COMMIT = 5
GEN_TOK = 24

# chosen held-out repos = same 8 we used before
chosen = sorted({json.loads(l)["repo_id"] for l in open("data/rpb25/heldout.jsonl")})
print("repos:", chosen)

# ---- load gru ckpt: head + gru ----
gp = hf_hub_download("code2lora/code2lora-gru", "code2lora_gru.pt")
gck = torch.load(gp, map_location="cpu", weights_only=False)
torch.save({"state_dict": gck["head_state"], "config": gck["head_config"]}, "/tmp/gru_head_only.pt")
head = OfficialDirectHead("/tmp/gru_head_only.pt")
gs = gck["gru_state"]
def _m(k):
    return mx.array(gs[k].float().numpy())
P = {
    "diff_w": _m("diff_proj.0.weight"), "diff_b": _m("diff_proj.0.bias"),
    "diff_ln_w": _m("diff_proj.1.weight"), "diff_ln_b": _m("diff_proj.1.bias"),
    "repo_w": _m("repo_init_proj.0.weight"), "repo_b": _m("repo_init_proj.0.bias"),
    "repo_ln_w": _m("repo_init_proj.2.weight"), "repo_ln_b": _m("repo_init_proj.2.bias"),
    "gru_Wih": _m("gru.weight_ih_l0"), "gru_bih": _m("gru.bias_ih_l0"),
    "gru_Whh": _m("gru.weight_hh_l0"), "gru_bhh": _m("gru.bias_hh_l0"),
    "out_ln_w": _m("output_norm.weight"), "out_ln_b": _m("output_norm.bias"),
}
evo = EvoGRU(P, hidden=2048)
print("gru + head loaded")

# ---- data ----
def emb(v):
    return np.asarray(v, dtype=np.float32)

cpath = hf_hub_download("code2lora/code2lora-data-commits", "commits/cr_test.parquet", repo_type="dataset")
ct = pds.dataset(cpath, format="parquet").to_table(
    columns=["repo_id", "commit_sha", "commit_index", "diff_embedding", "repo_state_embedding"],
    filter=pc.is_in(pds.field("repo_id"), value_set=pa.array(chosen)))
rows_by_repo = {}
for r in ct.to_pylist():
    rows_by_repo.setdefault(r["repo_id"], []).append(r)
for rid in rows_by_repo:
    rows_by_repo[rid].sort(key=lambda x: x["commit_index"])

qpath = hf_hub_download("code2lora/code2lora-data-snapshots", "qna/cr_test.parquet", repo_type="dataset")
qt = pds.dataset(qpath, format="parquet").to_table(
    columns=["repo_id", "commit_sha", "prefix", "target"],
    filter=pc.is_in(pds.field("repo_id"), value_set=pa.array(chosen)))
qnas_by_key = {}
for q in qt.to_pylist():
    qnas_by_key.setdefault((q["repo_id"], q["commit_sha"]), []).append(
        ((q["prefix"] or "")[-MAX_PREFIX:], q["target"] or ""))
print(f"commits rows: {ct.num_rows}, qna (repo,commit) keys: {len(qnas_by_key)}")

# ---- pass 1: roll GRU, collect adapters at scored commits ----
def flatA(ad):
    return np.concatenate([np.asarray(ad[t][0]).ravel() for t in TYPES])

scored = []          # list of (repo, sha, qnas, adapter_dict)
per_repo_vecs = {}
shared_acc = None
for rid in chosen:
    rows = rows_by_repo.get(rid, [])
    if not rows:
        continue
    h = evo.init_hidden(mx.array(emb(rows[0]["repo_state_embedding"])))
    nsc = 0
    for row in rows:
        h = evo.step(mx.array(emb(row["diff_embedding"])), h)
        key = (rid, row["commit_sha"])
        if key not in qnas_by_key or nsc >= MAX_SCORED_COMMITS:
            continue
        ctx = evo.ctx(h)
        ad = head.forward(ctx)
        mx.eval([t for ab in ad.values() for t in ab])
        scored.append((rid, row["commit_sha"], qnas_by_key[key][:QNAS_PER_COMMIT], ad))
        per_repo_vecs.setdefault(rid, []).append(flatA(ad))
        # running sum of A/B tensors for the shared (repo/commit-independent) adapter
        if shared_acc is None:
            shared_acc = {t: [np.asarray(ad[t][0]).copy(), np.asarray(ad[t][1]).copy()] for t in TYPES}
        else:
            for t in TYPES:
                shared_acc[t][0] += np.asarray(ad[t][0]); shared_acc[t][1] += np.asarray(ad[t][1])
        nsc += 1

N = len(scored)
shared = {t: (mx.array(shared_acc[t][0] / N), mx.array(shared_acc[t][1] / N)) for t in TYPES}
mx.eval([x for ab in shared.values() for x in ab])
print(f"scored commits: {N}")

# cross-commit adapter cosine within repos (does Evo vary across commits?)
def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
xc = []
for rid, vs in per_repo_vecs.items():
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            xc.append(cos(vs[i], vs[j]))
print(f"cross-COMMIT adapter cosine within repo: mean={np.mean(xc):.4f} (n={len(xc)})" if xc else "single commit/repo")

# ---- pass 2: eval base / shared / evo ----
model, tok = qwen_lora.load_base_model(BASE)
def inject(ad):
    for layer in qwen_lora._layers(model):
        for _m2, (s, a) in qwen_lora._PROJ.items():
            A, B = ad[a]
            getattr(getattr(layer, s), a).set_lora(A, 2.0 * B)
def ce(prefix, target):
    ip = tok.encode(prefix); fu = tok.encode(prefix + target); b = len(ip)
    if len(fu) <= b: return None
    lg = model(mx.array([fu]))[0]
    return float(nn.losses.cross_entropy(lg[b - 1:len(fu) - 1], mx.array(fu[b:]), reduction="mean"))
def _norm(s):
    return re.sub(r"\s+", " ", s.strip()).rstrip(" \t;:,.)]}")
def em(prefix, target):
    g = _norm(qwen_lora.generate_text(model, tok, prefix, max_tokens=GEN_TOK))
    return bool(_norm(target)) and g.startswith(_norm(target))

CE = {"base": [], "shared": [], "evo": []}
EM = {"base": 0, "shared": 0, "evo": 0}
n = 0
for rid, sha, qnas, ad in scored:
    for prefix, target in qnas:
        if not prefix or not target: continue
        n += 1
        qwen_lora.clear_lora(model)
        cb = ce(prefix, target); eb = em(prefix, target)
        inject(shared)
        cs = ce(prefix, target); es = em(prefix, target)
        inject(ad)
        cv = ce(prefix, target); ev = em(prefix, target)
        for k, v in [("base", cb), ("shared", cs), ("evo", cv)]:
            if v is not None: CE[k].append(v)
        EM["base"] += eb; EM["shared"] += es; EM["evo"] += ev

print("=" * 56)
print(f"EVO evolution eval — {n} QnAs over {N} scored commits, {len(per_repo_vecs)} repos")
for k in ["base", "shared", "evo"]:
    print(f"  {k:7s} CE={np.mean(CE[k]):.3f}  EM={100*EM[k]/n:.1f}%")
print(f"\n  Evo vs shared:  EM {100*(EM['evo']-EM['shared'])/n:+.1f}pp   CE {np.mean(CE['evo'])-np.mean(CE['shared']):+.3f}")
print(f"  cross-commit adapter cosine: {np.mean(xc):.4f}" if xc else "")
print("  → repo/commit customization WORKS if Evo > shared (EM) & cross-commit cosine < ~0.99")
