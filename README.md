# code2lora-mlx

An **MLX (Apple Silicon)** implementation of **Code2LoRA** — a hypernetwork that
reads a whole code repository and, in a single forward pass, generates a
**repository-specific LoRA adapter** for a frozen `Qwen2.5-Coder`. The adapter is
injected into the base model so completions become repo-aware **with zero
inference-time token overhead** (the repo knowledge lives in the weights, not the
prompt).

> Independent reimplementation for Apple Silicon. The method, the released
> hypernetwork checkpoints, and the RepoPeftBench datasets are from the paper
> **"Code2LoRA: Hypernetwork-Generated Adapters for Code Language Models under
> Software Evolution"** (Hotsko, Li, Deng, Nie — [arXiv:2606.06492](https://arxiv.org/abs/2606.06492))
> and the [`code2lora`](https://huggingface.co/code2lora) HuggingFace org. This
> repo is not affiliated with the authors.

## What works here

- **Faithful MLX inference of the official checkpoint.** Loads the released
  `code2lora/code2lora-direct` hypernetwork (Qwen2.5-Coder-1.5B, rank 16) and
  reproduces its behavior: on held-out repos the generated adapter **helps 8/8**
  (mean cross-entropy 1.44 → 0.42). On a fresh cross-repo (`jd/tenacity`,
  not in training) the test-assertion completion CE drops **0.84 → 0.31** (24/29
  improved) — e.g. `assert find_ordinal(1) == ?` → base guesses wrong, adapted
  emits the repo's actual `"st"`.
- **Faithful training math.** The MLX hypernetwork's forward **and backward** match
  the official PyTorch head to fp32 precision on a single step (gradient cosine
  `1.0000000`, loss rel-diff `3.6e-7`) — so given equal data/compute it converges
  to the same hypernetwork.
- **Official repo encoder, reproduced.** `c2l/qwen3_repo_encoder.py` mirrors the
  paper's `repo_state_embedding` recipe (Qwen3-Embedding-0.6B, `.py` files, chunk
  2048/overlap 256, masked mean-pool, `L2norm(concat(mean_files, max_files))` →
  2048-d). Validated against the official stored embeddings.

## Install

Requires [`uv`](https://docs.astral.sh/uv/) and Apple Silicon (Metal).

```bash
git clone <this repo> && cd code2lora-mlx
uv sync          # mlx, mlx-lm, transformers, torch, sentence-transformers, ...
```

## Use it on your own repo

The official checkpoint is downloaded automatically from HuggingFace on first run.

```bash
# repo-aware completion: base vs adapted, side by side
PYTHONPATH=. uv run python cli.py complete /path/to/python/repo \
  $'def test_thing():\n    assert my_api(1) == '

# quantitative: base-vs-adapted cross-entropy on held-out lines of the repo
PYTHONPATH=. uv run python cli.py selftest /path/to/python/repo 15

# just compute (and cache) the 2048-d repo embedding
PYTHONPATH=. uv run python cli.py encode /path/to/python/repo
```

## How it works

```
repo (.py files)
   │  Qwen3-Embedding-0.6B, mean-pool, L2norm(concat(mean,max))
   ▼
repo embedding ∈ R^2048
   │  hypernetwork: trunk(2048->1024->1024) -> per-module heads -> tanh·exp(log_scale)
   ▼
LoRA adapter  (rank 16, for q/k/v/o/gate/up/down; shared across all 28 layers)
   │  y = base(x) + (alpha/r)·(x·A^T)·B^T      alpha/r = 32/16 = 2
   ▼
frozen Qwen2.5-Coder-1.5B  ->  repo-aware completion (zero extra prompt tokens)
```

## Scope & honest notes

- **Python only** as shipped: the encoder reads `.py`, and the released
  hypernetwork was trained on Python repos for **test-assertion completion**
  (`assert expr == value`). It shines on that task/domain; on arbitrary code
  lines or out-of-domain repos the benefit is small. The *method* is
  language-agnostic, but another language needs a retrained hypernetwork.
- **Training from scratch** is supported and numerically faithful, but reproducing
  the official-scale run (604 repos x 3 epochs, seq 4096) on an M3 Max is a
  multi-day job (~9.8 s/step measured -> ~7.5 days). The official H100 run is ~6 h.
  For a working adapter you don't need to train — just use the released checkpoint.

## License

MIT — see [LICENSE](LICENSE).
