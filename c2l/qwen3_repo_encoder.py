"""Faithful re-implementation of the OFFICIAL `repo_state_embedding` recipe
(`create_dataset/build_rse.py` in anonymous.4open.science/r/code2lora-6857) —
the exact embedding the released `code2lora/code2lora-direct` head was trained
on (and what lives in the HF datasets / our held-out set).

Recipe (verified against source):
  * model  : Qwen/Qwen3-Embedding-0.6B, mean-pool over last_hidden_state (NOT
             last-token), no instruction prefix, fp32.
  * files  : .py only, <= 2,000,000 bytes.
  * chunk  : 2048 tokens, overlap 256 (step 1792); keep windows >= 8 tokens.
  * file   : mean over its chunk vectors                              -> [1024]
  * repo   : L2_normalize( concat(mean_over_files, max_over_files) )  -> [2048]
"""

import os
from pathlib import Path

import numpy as np
import torch

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
CHUNK_TOKENS = 2048
CHUNK_OVERLAP = 256
MAX_FILE_BYTES = 2_000_000
MIN_WINDOW_TOKENS = 8
EMBED_DIM = 1024
_SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".tox",
              "build", "dist", ".mypy_cache", ".pytest_cache", "target",
              # datasets / vendored clones / checkpoints — not the repo's own source
              "data", "datasets", "vendor", "third_party", "site-packages"}

_model = None
_tok = None


def _load():
    global _model, _tok
    if _model is None:
        from transformers import AutoModel, AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(dev).eval()
    return _model, _tok


def _iter_py(repo: Path, exts=(".py",)):
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        for fn in files:
            if fn.endswith(tuple(exts)):
                p = Path(root) / fn
                try:
                    if p.stat().st_size <= MAX_FILE_BYTES:
                        yield p
                except OSError:
                    pass


@torch.inference_mode()
def _embed_file(text, model, tok, device, batch_size=8):
    ids = tok.encode(text, add_special_tokens=False)
    if not ids:
        return None
    step = CHUNK_TOKENS - CHUNK_OVERLAP
    n = len(ids)
    windows = []
    for start in range(0, n, step):
        end = min(start + CHUNK_TOKENS, n)
        w = ids[start:end]
        if len(w) >= MIN_WINDOW_TOKENS:
            windows.append(w)
        if end >= n:
            break
    if not windows:
        return None
    chunk_vecs = []
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size]
        decoded = [tok.decode(w, skip_special_tokens=True) for w in batch]
        enc = tok(decoded, padding=True, truncation=True,
                  max_length=CHUNK_TOKENS, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        last = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(last.dtype)
        mean = ((last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).float().cpu()
        chunk_vecs.append(mean)
    return torch.cat(chunk_vecs, dim=0).mean(dim=0).numpy()


def encode_repo(repo_dir, verbose=True, exts=(".py",), max_files=None) -> np.ndarray:
    """Return the 2048-d repo_state_embedding (fp32, L2-normalized)."""
    repo = Path(repo_dir)
    model, tok = _load()
    dev = next(model.parameters()).device
    files = list(_iter_py(repo, exts))
    if max_files and len(files) > max_files:
        files = files[:max_files]
    fvecs = []
    for i, p in enumerate(files):
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        if not text.strip():
            continue
        v = _embed_file(text, model, tok, dev)
        if v is not None:
            fvecs.append(v)
        if verbose and (i + 1) % 25 == 0:
            print(f"  embedded {i + 1}/{len(files)} files")
    if not fvecs:
        raise RuntimeError(f"no .py files embedded under {repo}")
    fe = np.stack(fvecs, axis=0).astype(np.float32)          # [F, 1024]
    repo_vec = np.concatenate([fe.mean(axis=0), fe.max(axis=0)], axis=0)  # [2048]
    repo_vec /= (np.linalg.norm(repo_vec) + 1e-12)
    if verbose:
        print(f"  repo encoded from {len(fvecs)} .py files -> {repo_vec.shape}")
    return repo_vec.astype(np.float32)
