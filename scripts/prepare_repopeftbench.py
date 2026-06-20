"""Prepare REAL RepoPeftBench training data (Python port of the Windows
scripts/prepare_repopeftbench.ps1, adapted to use the paper's OWN precomputed
repo embeddings instead of re-encoding every repo).

DATA SOURCES (HuggingFace `code2lora` org):
  * code2lora/code2lora-data-snapshots  qna/{train,cr_test}.parquet
      -> the code-completion examples: `prefix` (test code up to the assertion)
         + `target` (the asserted value to predict). 400 train repos / 51 cr_test
         repos (disjoint = true cross-repo held-out).
  * code2lora/code2lora-data-commits    commits/{cr_train,cr_test}.parquet
      -> the paper's OWN `repo_state_embedding` per (repo_id, commit_index):
         Qwen3-Embedding-0.6B, dim=2048, fp16, concat(mean_files,max_files)+L2
         (per EMBEDDINGS_README.json). We join QnA -> embedding by
         (repo_id, commit_index); coverage is 100% (verified).

WHY the paper's 2048-d embeddings (not our 768-d MiniLM encoder):
  - Most faithful to the paper (their exact repo-state representation).
  - Avoids cloning + encoding 400 repos (hours). The hypernetwork's only
    dependence on the encoder is its INPUT dim, which we set to 2048 via
    HypernetworkConfig(repo_embed_dim=2048). Architecture is otherwise identical.

OUTPUT: data/rpb/{train,heldout}.jsonl in the schema c2l/dataset.py reads:
    {repo_id, repo_embedding:[2048], input_prefix, target_value,
     cross_repo_split, language, file_path}

Run:
  cd code2lora-mlx && PYTHONPATH=. uv run python scripts/prepare_repopeftbench.py \
      --max-per-repo 80 --heldout-repos 8 --max-per-heldout 60
"""

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

SNAP = "code2lora/code2lora-data-snapshots"
COMM = "code2lora/code2lora-data-commits"
EMBED_DIM = 2048

# Drop trivial / noise targets (smartcap README: bare ')' , leading-comma, len<min).
MIN_TARGET_CHARS = 2
# Keep prefixes bounded: very long prefixes (up to 113k chars) waste compute and
# get token-truncated anyway. Keep the tail (nearest the assertion) up to this many
# chars, which is where the repo-specific signal that predicts the target lives.
MAX_PREFIX_CHARS = 2800
MIN_PREFIX_CHARS = 40


def build_embedding_map(commits_file: str) -> dict:
    """(repo_id, commit_index) -> np.float32[2048] from a commits parquet."""
    path = hf_hub_download(COMM, commits_file, repo_type="dataset")
    pf = pq.ParquetFile(path)
    emap = {}
    for b in pf.iter_batches(
        batch_size=4096, columns=["repo_id", "commit_index", "repo_state_embedding"]
    ):
        d = b.to_pydict()
        for r, c, e in zip(
            d["repo_id"], d["commit_index"], d["repo_state_embedding"]
        ):
            emap[(r, c)] = np.asarray(e, dtype=np.float32)
    return emap


def _trim_prefix(prefix: str) -> str:
    if len(prefix) <= MAX_PREFIX_CHARS:
        return prefix
    # keep the tail (the lines just before the assertion carry the predictive context)
    return prefix[-MAX_PREFIX_CHARS:]


def collect_examples(qna_file: str, emap: dict, max_per_repo: int, seed: int):
    """Read a qna parquet, join the paper embedding, cap per repo. Returns
    {repo_id: [rows]} where each row is the final JSONL dict (minus embedding)."""
    path = hf_hub_download(SNAP, qna_file, repo_type="dataset")
    pf = pq.ParquetFile(path)
    by_repo = defaultdict(list)
    cols = ["repo_id", "commit_index", "prefix", "target", "test_file"]
    for b in pf.iter_batches(batch_size=8192, columns=cols):
        d = b.to_pydict()
        for r, c, p, t, tf in zip(
            d["repo_id"], d["commit_index"], d["prefix"], d["target"], d["test_file"]
        ):
            if (r, c) not in emap:
                continue
            if t is None or len(t.strip()) < MIN_TARGET_CHARS:
                continue
            if p is None or len(p.strip()) < MIN_PREFIX_CHARS:
                continue
            by_repo[r].append(
                {
                    "repo_id": r,
                    "commit_index": int(c),
                    "input_prefix": _trim_prefix(p),
                    "target_value": t,
                    "file_path": tf or "",
                }
            )
    # cap per repo (shuffle deterministically so we sample across commits/files)
    rng = random.Random(seed)
    capped = {}
    for r, rows in by_repo.items():
        rng.shuffle(rows)
        capped[r] = rows[:max_per_repo]
    return capped


def write_jsonl(path, capped, emap, split_label):
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r, rows in capped.items():
            for row in rows:
                emb = emap[(r, row["commit_index"])]
                rec = {
                    "repo_id": r,
                    "repo_embedding": [float(x) for x in emb.tolist()],
                    "input_prefix": row["input_prefix"],
                    "target_value": row["target_value"],
                    "cross_repo_split": split_label,
                    "language": "python",
                    "file_path": row["file_path"],
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="data/rpb")
    ap.add_argument("--max-per-repo", type=int, default=80)
    ap.add_argument("--heldout-repos", type=int, default=8)
    ap.add_argument("--max-per-heldout", type=int, default=60)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("=== Building paper-embedding maps (repo,commit) -> 2048-d ===")
    train_emap = build_embedding_map("commits/cr_train.parquet")
    print(f"  train embeddings: {len(train_emap)} (repo,commit) keys")
    held_emap = build_embedding_map("commits/cr_test.parquet")
    print(f"  cr_test embeddings: {len(held_emap)} keys")

    print("=== TRAIN: qna/train.parquet (400 repos) ===")
    train_capped = collect_examples(
        "qna/train.parquet", train_emap, args.max_per_repo, args.seed
    )
    train_path = os.path.join(args.out, "train.jsonl")
    n_train = write_jsonl(train_path, train_capped, train_emap, "train")
    print(f"  TRAIN: {len(train_capped)} repos, {n_train} examples -> {train_path}")

    print("=== HELD-OUT: qna/cr_test.parquet (51 repos, pick a subset) ===")
    held_capped_all = collect_examples(
        "qna/cr_test.parquet", held_emap, args.max_per_heldout, args.seed
    )
    # Pick the held-out repos with the most examples for a meaningful eval.
    ranked = sorted(held_capped_all.items(), key=lambda kv: -len(kv[1]))
    chosen = dict(ranked[: args.heldout_repos])
    held_path = os.path.join(args.out, "heldout.jsonl")
    n_held = write_jsonl(held_path, chosen, held_emap, "cr_test")
    print(f"  HELD-OUT: {len(chosen)} repos, {n_held} examples -> {held_path}")
    for r, rows in chosen.items():
        print(f"     {r}: {len(rows)}")

    # Sanity: train/heldout repo disjointness
    inter = set(train_capped) & set(chosen)
    print(f"\n  train/heldout repo overlap (must be 0): {len(inter)}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
