"""Aggregate held-out eval across ALL held-out repos (Step 5).

For every held-out repo: base vs adapted next-token CE on its own (prefix+target)
examples. Then the cross-repo adapter cosine across the held-out repos (the KEY
signal: did the hypernetwork learn to emit DIFFERENT adapters per repo? want the
mean pairwise cosine well below 0.98). Also reports the effective LoRA scale.

Run:
  cd code2lora-mlx && PYTHONPATH=. uv run python eval_heldout_all.py \
      --ckpt checkpoints_rpb/best.safetensors --heldout data/rpb/heldout.jsonl \
      --repo-embed-dim 2048 --max-eval 40
"""

import argparse
import json
import time
from collections import OrderedDict

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_unflatten

from c2l import qwen_lora
from c2l.config import HypernetworkConfig
from c2l.hypernetwork import Code2LoRAHead
from c2l.trainer import MAX_SEQ_LEN


def load_hn(ckpt_path, cfg):
    hn = Code2LoRAHead(cfg)
    flat = mx.load(ckpt_path)
    hn.update(tree_unflatten(list(flat.items())))
    mx.eval(hn.parameters())
    return hn


def load_by_repo(path):
    by = OrderedDict()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by.setdefault(r["repo_id"], []).append(r)
    return by


def ce_loss_over(model, tokenizer, rows, max_eval):
    total, n = 0.0, 0
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
        sl = len(ids)
        shift_logits = logits[:, : sl - 1, :].reshape(sl - 1, -1)
        shift_labels = input_ids[:, 1:].reshape(sl - 1)
        loss = nn.losses.cross_entropy(shift_logits, shift_labels, reduction="mean")
        total += float(loss)
        n += 1
    return total / max(n, 1), n


def adapter_flat(hn, emb):
    all_lora = hn.forward_all(mx.array(np.asarray(emb, np.float32)[None, :]))
    parts = [
        all_lora[li][m][0].reshape(-1)
        for li in range(0, hn.cfg.num_layers, 4)
        for m in ("q", "k", "v", "o")
    ]
    v = mx.concatenate(parts)
    mx.eval(v)
    return np.asarray(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--max-eval", type=int, default=40)
    ap.add_argument("--repo-embed-dim", type=int, default=768)
    ap.add_argument("--max-lora-scale", type=float, default=1.0)
    args = ap.parse_args()

    print(f"Device: {mx.default_device()}")
    cfg = HypernetworkConfig(
        repo_embed_dim=args.repo_embed_dim, max_lora_scale=args.max_lora_scale
    )

    by_repo = load_by_repo(args.heldout)
    print(f"Held-out repos: {len(by_repo)}")

    t = time.time()
    model, tokenizer = qwen_lora.load_base_model()
    model.set_dtype(mx.float32)
    model.freeze()
    print(f"Base model loaded fp32 ({time.time() - t:.1f}s)")

    hn = load_hn(args.ckpt, cfg)
    print(f"Hypernetwork loaded from {args.ckpt}")

    # effective scale
    scales = []
    for m in hn.log_scale_a:
        scales.append(float(hn._scale(hn.log_scale_a[m])))
        scales.append(float(hn._scale(hn.log_scale_b[m])))
    print(f"Effective LoRA scale: max={max(scales):.4f} mean={np.mean(scales):.4f}")

    rows_summary = []
    flats = []
    helped = 0
    for repo, rows in by_repo.items():
        emb = np.asarray(rows[0]["repo_embedding"], np.float32)
        all_lora = hn.forward_all(mx.array(emb[None, :]))
        qwen_lora.clear_lora(model)
        base_ce, nb = ce_loss_over(model, tokenizer, rows, args.max_eval)
        qwen_lora.clear_lora(model)
        qwen_lora.inject_lora(model, all_lora)
        adapt_ce, na = ce_loss_over(model, tokenizer, rows, args.max_eval)
        qwen_lora.clear_lora(model)
        delta = base_ce - adapt_ce
        if delta > 0:
            helped += 1
        rows_summary.append((repo, base_ce, adapt_ce, delta, nb))
        flats.append(adapter_flat(hn, emb))
        print(
            f"  {repo[:34]:36s} base={base_ce:.4f} adapt={adapt_ce:.4f} "
            f"delta={delta:+.4f} ({'HELP' if delta>0 else 'hurt'}) n={nb}"
        )

    # cross-repo adapter cosine
    cos_sum, cos_cnt = 0.0, 0
    for i in range(len(flats)):
        for j in range(i + 1, len(flats)):
            a, b = flats[i], flats[j]
            d = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
            cos_sum += float(a @ b) / d
            cos_cnt += 1
    xcos = cos_sum / max(cos_cnt, 1)

    base_mean = np.mean([r[1] for r in rows_summary])
    adapt_mean = np.mean([r[2] for r in rows_summary])
    print("\n=== AGGREGATE (held-out repos) ===")
    print(f"  mean BASE  CE: {base_mean:.4f}")
    print(f"  mean ADAPT CE: {adapt_mean:.4f}")
    print(f"  mean delta   : {base_mean - adapt_mean:+.4f}  "
          f"({'ADAPTER HELPS' if base_mean>adapt_mean else 'adapter HURTS'})")
    print(f"  repos helped : {helped}/{len(rows_summary)}")
    print(f"  cross-repo adapter cosine (held-out): {xcos:.4f}  "
          f"(<0.98 => repo-distinct adapters)")


if __name__ == "__main__":
    main()
