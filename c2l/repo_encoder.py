"""Repo encoder — faithful port of code2lora-lite/src/repo_encoder.rs.

Per chunk: all-MiniLM-L6-v2 mean-pool + L2-normalize (384-d) — identical to the
Candle BERT forward_pool. Per file: mean over chunks + elementwise max. Per repo:
weighted mean across files (weight = 0.3*size + 0.5*path + 0.2*name) + global max.
Output: concat(mean, max) = 768-d.  Same .embed binary format as the Rust version.
"""

import math
import os
import struct

import numpy as np

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
_MAX_CHARS = 512 * 8       # ~512 tokens of code
_OVERLAP_CHARS = 256 * 8
_HEADER_PREFIX = "CODE2LORA_EMBED_V1:"


class RepoEncoder:
    def __init__(self, device: str | None = None):
        # Imported lazily so `import c2l` stays cheap.
        from sentence_transformers import SentenceTransformer

        # device=None lets sentence-transformers pick mps/cpu.
        self.model = SentenceTransformer(MODEL_ID, device=device)
        self.embed_dim = EMBED_DIM

    # ── per-chunk embedding: mean pool + L2 normalize (matches forward_pool) ──
    def _embed_text(self, text: str) -> np.ndarray:
        v = self.model.encode(
            text,
            normalize_embeddings=True,   # L2 normalize, as the Rust version does
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return v.astype(np.float32)

    def _chunk_text(self, text: str) -> list[str]:
        chunks, start, n = [], 0, len(text)
        while start < n:
            end = min(start + _MAX_CHARS, n)
            chunks.append(text[start:end])
            if end >= n:
                break
            start += _MAX_CHARS - _OVERLAP_CHARS
        if not chunks and text:
            chunks.append(text)
        return chunks

    @staticmethod
    def _file_weight(path: str, content: str) -> float:
        p = path.lower()
        size_w = min(math.log1p(len(content)), 10.0) / 10.0
        if "test" in p:
            path_w = 0.8
        elif "__init__" in p:
            path_w = 0.6
        elif "src/" in p or "/lib/" in p:
            path_w = 1.0
        else:
            path_w = 0.5
        name_w = 1.2 if (p.endswith("main.py") or p.endswith("core.py")) else 1.0
        return 0.3 * size_w + 0.5 * path_w + 0.2 * name_w

    @staticmethod
    def _collect_py(repo_path: str) -> list[str]:
        out = []
        for root, _dirs, files in os.walk(repo_path):
            for f in files:
                if not f.endswith(".py"):
                    continue
                fp = os.path.join(root, f)
                # same hidden-path filter as the Rust collect_py_files
                if "/." in fp or "\\." in fp:
                    continue
                out.append(fp)
        return out

    def embed_repo(self, repo_path: str) -> np.ndarray:
        files = self._collect_py(repo_path)
        if not files:
            raise ValueError(f"No .py files found in {repo_path!r}")
        dim = self.embed_dim
        sum_mean = np.zeros(dim, np.float32)
        sum_max = np.full(dim, -np.inf, np.float32)
        total_w = 0.0

        for fp in files:
            with open(fp, encoding="utf-8", errors="replace") as fh:
                content = fh.read()
            if not content.strip():
                continue
            w = self._file_weight(fp, content)
            file_mean = np.zeros(dim, np.float32)
            file_max = np.full(dim, -np.inf, np.float32)
            cnt = 0
            for ch in self._chunk_text(content):
                emb = self._embed_text(ch)
                file_mean += emb
                file_max = np.maximum(file_max, emb)
                cnt += 1
            if cnt > 0:
                file_mean /= cnt
                sum_mean += w * file_mean
                sum_max = np.maximum(sum_max, file_max)
                total_w += w

        if total_w <= 0.0:
            raise ValueError("No embeddable content found in repo")
        sum_mean /= total_w
        return np.concatenate([sum_mean, sum_max]).astype(np.float32)  # 768-d

    def embed_repo_cached(self, repo_path: str, cache_dir: str) -> np.ndarray:
        os.makedirs(cache_dir, exist_ok=True)
        name = os.path.basename(os.path.normpath(repo_path))
        cache_path = os.path.join(cache_dir, f"{name}.embed")
        if os.path.exists(cache_path):
            return load_embedding(cache_path)
        emb = self.embed_repo(repo_path)
        save_embedding(emb, cache_path)
        return emb


# ── .embed binary format, byte-compatible with the Rust RepoEmbedding ──
def save_embedding(emb: np.ndarray, path: str) -> None:
    emb = np.asarray(emb, np.float32).ravel()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = f"{_HEADER_PREFIX}{emb.size}\n".encode()
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(struct.pack(f"<{emb.size}f", *emb.tolist()))


def load_embedding(path: str) -> np.ndarray:
    with open(path, "rb") as fh:
        raw = fh.read()
    nl = raw.index(b"\n")
    header = raw[:nl].decode()
    if not header.startswith(_HEADER_PREFIX):
        raise ValueError("Bad embedding header")
    dim = int(header[len(_HEADER_PREFIX):])
    data = np.frombuffer(raw[nl + 1:], dtype="<f4")
    if data.size != dim:
        raise ValueError(f"Dim mismatch: header {dim} vs {data.size}")
    return data.astype(np.float32)
