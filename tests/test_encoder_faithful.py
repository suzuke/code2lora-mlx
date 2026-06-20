"""Faithfulness check: MLX/sentence-transformers repo embedding vs the Candle
reference (/tmp/candle_ref.embed, produced by the Rust `encode`). Same MiniLM +
same mean/max aggregation should give near-identical 768-d vectors."""

import os

import numpy as np

from c2l.repo_encoder import RepoEncoder, load_embedding

REPO = os.environ.get("C2L_SAMPLE_DIR", "sample-proj")
REF = os.environ.get("C2L_CANDLE_REF", "/tmp/candle_ref.embed")


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


ref = load_embedding(REF)
enc = RepoEncoder()
mlx_emb = enc.embed_repo(REPO)

print("dims  ref:", ref.shape, " mlx:", mlx_emb.shape)
print(f"cosine full      : {cos(ref, mlx_emb):.5f}")
print(f"cosine mean-half : {cos(ref[:384], mlx_emb[:384]):.5f}")
print(f"cosine max-half  : {cos(ref[384:], mlx_emb[384:]):.5f}")
print(f"L2 diff full     : {np.linalg.norm(ref - mlx_emb):.4f}")
print(f"max |Δ|          : {np.max(np.abs(ref - mlx_emb)):.4f}")
