"""Held-out adapter-quality eval (Step 3 — the deliverable).

Loads a trained hypernetwork checkpoint, then on a repo NOT in training:
  1. adapt: hypernetwork(real 768-d embedding) -> per-layer LoRA.
  2. held-out CE loss BASE (no LoRA) vs ADAPTED (with LoRA) over the repo's
     own (prefix+target) examples — quantitative.
  3. complete: a few realistic prompts from that repo, base vs adapted — the
     actual generated text (greedy, deterministic), quoted in the report.

Run:
  cd code2lora-mlx && PYTHONPATH=. uv run python eval_heldout.py \
      --ckpt checkpoints_diy/final.safetensors \
      --heldout data/diy/heldout.jsonl --repo pallets/itsdangerous \
      --max-eval 40 --gen-tokens 48
"""

import argparse
import json
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_unflatten

from c2l import qwen_lora
from c2l.config import HypernetworkConfig
from c2l.hypernetwork import Code2LoRAHead
from c2l.trainer import MAX_SEQ_LEN

EOS = 151643


def load_hn(ckpt_path, cfg):
    hn = Code2LoRAHead(cfg)
    flat = mx.load(ckpt_path)
    hn.update(tree_unflatten(list(flat.items())))
    mx.eval(hn.parameters())
    return hn


def load_heldout(path, repo):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if repo is None or r.get("repo_id") == repo:
                rows.append(r)
    return rows


def ce_loss_over(model, tokenizer, rows, max_eval):
    """Mean next-token CE over rows. model must already have LoRA set/cleared."""
    total = 0.0
    n = 0
    for r in rows[:max_eval]:
        text = r["input_prefix"]
        if not text.endswith("\n"):
            text += "\n"
        text += r["target_value"]
        ids = tokenizer.encode(text)
        if len(ids) > MAX_SEQ_LEN:
            ids = ids[:MAX_SEQ_LEN]
        if len(ids) < 2:
            continue
        input_ids = mx.array([ids])
        logits = model(input_ids)
        seq_len = len(ids)
        shift_logits = logits[:, : seq_len - 1, :].reshape(seq_len - 1, -1)
        shift_labels = input_ids[:, 1:].reshape(seq_len - 1)
        loss = nn.losses.cross_entropy(shift_logits, shift_labels, reduction="mean")
        total += float(loss)
        n += 1
    return total / max(n, 1), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--repo", default=None, help="repo_id to eval (else all rows)")
    ap.add_argument("--max-eval", type=int, default=40)
    ap.add_argument("--gen-tokens", type=int, default=48)
    ap.add_argument("--gen-prompts", type=int, default=3)
    ap.add_argument("--repo-embed-dim", type=int, default=2048)
    args = ap.parse_args()

    print(f"Device: {mx.default_device()}")
    cfg = HypernetworkConfig(repo_embed_dim=args.repo_embed_dim)

    rows = load_heldout(args.heldout, args.repo)
    assert rows, f"no rows for repo={args.repo} in {args.heldout}"
    repo_id = rows[0]["repo_id"]
    emb = np.asarray(rows[0]["repo_embedding"], np.float32)
    assert emb.shape[0] == cfg.repo_embed_dim, (
        f"embedding dim {emb.shape[0]} != cfg.repo_embed_dim {cfg.repo_embed_dim}"
    )
    print(f"Held-out repo: {repo_id} | {len(rows)} examples | emb[:3]={emb[:3].tolist()}")
    print(f"  embedding L2-norm={float(np.linalg.norm(emb)):.4f} (non-zero => real)")

    t = time.time()
    model, tokenizer = qwen_lora.load_base_model()
    model.set_dtype(mx.float32)
    model.freeze()
    print(f"Base model loaded fp32 ({time.time() - t:.1f}s)")

    hn = load_hn(args.ckpt, cfg)
    print(f"Hypernetwork loaded from {args.ckpt}")

    repo_emb = mx.array(emb[None, :])  # (1, 768)
    t = time.time()
    all_lora = hn.forward_all(repo_emb)
    mx.eval([all_lora[0]["q"][0]])
    print(f"Adapt (hn -> LoRA for {len(all_lora)} layers) in {time.time() - t:.3f}s")

    # ── 1) Held-out CE loss: base vs adapted ──
    qwen_lora.clear_lora(model)
    base_loss, n_base = ce_loss_over(model, tokenizer, rows, args.max_eval)
    qwen_lora.clear_lora(model)
    qwen_lora.inject_lora(model, all_lora)
    adapt_loss, n_adapt = ce_loss_over(model, tokenizer, rows, args.max_eval)
    qwen_lora.clear_lora(model)

    print("\n=== HELD-OUT CE LOSS (lower = better) ===")
    print(f"  BASE  (no LoRA): {base_loss:.4f}   (n={n_base})")
    print(f"  ADAPT (w/ LoRA): {adapt_loss:.4f}   (n={n_adapt})")
    delta = base_loss - adapt_loss
    pct = 100.0 * delta / base_loss if base_loss else 0.0
    verdict = "ADAPTER HELPS" if delta > 0 else "adapter HURTS / no help"
    print(f"  delta = {delta:+.4f}  ({pct:+.2f}%)  => {verdict}")

    # ── 2) Generated completions: base vs adapted ──
    print("\n=== COMPLETIONS (greedy, base vs adapted) ===")
    prompts = []
    for r in rows:
        p = r["input_prefix"].rstrip("\n")
        if 20 <= len(p) <= 400:
            prompts.append(p)
        if len(prompts) >= args.gen_prompts:
            break

    for i, prompt in enumerate(prompts):
        qwen_lora.clear_lora(model)
        base_out = qwen_lora.generate_text(model, tokenizer, prompt, args.gen_tokens)
        qwen_lora.clear_lora(model)
        qwen_lora.inject_lora(model, all_lora)
        adapt_out = qwen_lora.generate_text(model, tokenizer, prompt, args.gen_tokens)
        qwen_lora.clear_lora(model)
        print(f"\n--- Prompt {i + 1} ---")
        print(f"PROMPT:\n{prompt}")
        print(f"\n[BASE]:\n{base_out}")
        print(f"\n[ADAPTED]:\n{adapt_out}")
        print("-" * 60)


if __name__ == "__main__":
    main()
