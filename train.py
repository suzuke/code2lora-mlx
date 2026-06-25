"""MLX `train` entry — mirrors code2lora-lite main.rs Commands::Train, plus the
diagnosed TUNING FIXES for the scaled-up run (lr=1e-5, grad clip, log_scale
regularization + clamp, repo-mixing batches, validation early-stop/best-ckpt).

Usage (scaled RepoPeftBench run, paper's 2048-d embeddings):
    PYTHONPATH=. uv run python train.py \
        --data-file data/rpb/train.jsonl --output checkpoints_rpb \
        --repo-embed-dim 2048 --epochs 12 --lr 1e-5 --batch-size 12 \
        --grad-clip 1.0 --log-scale-l2 1e-3 --val-frac 0.04

--data-file loads ONE jsonl (so held-out repos never leak in). --data-dir keeps
the old dir-walk behaviour. A small in-repo validation slice is carved from train
for early stopping / best-checkpoint selection.
"""

import argparse
import random
import time

import mlx.core as mx

from c2l import qwen_lora
from c2l.config import HypernetworkConfig
from c2l.dataset import CodeDataset
from c2l.hypernetwork import Code2LoRAHead
from c2l.trainer import Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data-dir", default=None)
    ap.add_argument("-f", "--data-file", default=None, help="single .jsonl (no leak)")
    ap.add_argument("-o", "--output", default="checkpoints")
    ap.add_argument("-e", "--epochs", type=int, default=12)
    ap.add_argument("-l", "--lr", type=float, default=1e-5)         # FIX: was 1e-4 (diverged)
    ap.add_argument("-b", "--batch-size", type=int, default=12)     # FIX: bigger for CR signal
    ap.add_argument("--repo-embed-dim", type=int, default=2048)     # official Qwen3-Embedding
    ap.add_argument("--grad-clip", type=float, default=1.0)         # global-norm clip
    # Official head bounds scale via init -3.5 + clamp(exp,1e-5,0.3); no log_scale L2 by default.
    ap.add_argument("--log-scale-l2", type=float, default=0.0)
    ap.add_argument("--cr-holdout", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.04, help="in-repo val slice")
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--limit", type=int, default=0, help="cap examples (smoke runs)")
    args = ap.parse_args()

    print(f"Using device: {mx.default_device()}")

    assert args.data_dir or args.data_file, "need --data-dir or --data-file"
    if args.data_file:
        examples = CodeDataset.load_jsonl(args.data_file)
        dataset = CodeDataset.from_examples(examples)
    else:
        dataset = CodeDataset.load_from_dir(args.data_dir)
    assert not dataset.is_empty(), "No training examples found"

    if args.limit and len(dataset.examples) > args.limit:
        rng = random.Random(args.seed)
        rng.shuffle(dataset.examples)
        dataset.examples = dataset.examples[: args.limit]
    print(f"Loaded {len(dataset)} examples; repos={dataset.summary()['repo_count']}")

    # Carve a small IN-REPO validation slice for early stopping (a random subset
    # of train; same repos, unseen examples). Held-out REPOS stay in heldout.jsonl.
    rng = random.Random(args.seed + 1)
    rng.shuffle(dataset.examples)
    n_val = max(8, int(len(dataset.examples) * args.val_frac))
    val_examples = dataset.examples[:n_val]
    dataset.examples = dataset.examples[n_val:]
    print(f"Validation slice: {len(val_examples)} examples; train: {len(dataset)}")

    cfg = HypernetworkConfig(repo_embed_dim=args.repo_embed_dim)

    t = time.time()
    model, tokenizer = qwen_lora.load_base_model()
    print(f"Base model loaded ({time.time() - t:.1f}s)")

    hn = Code2LoRAHead(cfg)
    mx.eval(hn.parameters())
    print(f"Hypernetwork created (repo_embed_dim={cfg.repo_embed_dim})")

    train_config = {
        "rank": cfg.rank,
        "base_model": "Qwen/Qwen2.5-Coder-0.5B",
        "output": args.output,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seq_len": 2048,
        "cr_holdout": args.cr_holdout,
        "grad_clip": args.grad_clip,
        "log_scale_l2": args.log_scale_l2,
        "val_every": args.val_every,
        "patience": args.patience,
        "seed": args.seed,
    }

    trainer = Trainer(hn, model, tokenizer, train_config, train_fp32=True)

    t = time.time()
    step_times = trainer.train(dataset, val_examples=val_examples)
    total = time.time() - t
    print(f"Training completed in {total:.1f}s over {len(step_times)} steps")

    if len(step_times) > 1:
        warm = step_times[1:]
        print(
            f"MLX s/step (steady-state, warmup dropped): "
            f"mean={sum(warm) / len(warm):.4f}s  min={min(warm):.4f}s  "
            f"n={len(warm)}  (warmup step0={step_times[0]:.4f}s)"
        )


if __name__ == "__main__":
    main()
