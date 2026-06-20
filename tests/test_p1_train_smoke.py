"""P1 smoke: confirm the MLX training loop produces finite, DECREASING loss and
that gradients flow into the hypernetwork ONLY (Qwen frozen).

1. Grad-flow check: nn.value_and_grad(hn, ...) returns nonzero grads for hn
   params; the frozen Qwen has zero trainable params.
2. Loss-decrease check: a few IR steps on one example drive the loss down.
3. Full-loop smoke on sample-proj (3 .py files).
"""

import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from c2l import qwen_lora
from c2l.config import HypernetworkConfig
from c2l.dataset import CodeDataset
from c2l.hypernetwork import Code2LoRAHead
from c2l.trainer import Trainer, _tokenize

SAMPLE_DIR = os.environ.get("C2L_SAMPLE_DIR", "sample-proj")


def main():
    print("=== P1 training smoke ===")
    cfg = HypernetworkConfig()
    hn = Code2LoRAHead(cfg)
    mx.eval(hn.parameters())

    t = time.time()
    model, tok = qwen_lora.load_base_model()
    print(f"loaded Qwen (mlx-lm): {time.time() - t:.1f}s")

    trainer = Trainer(hn, model, tok, {}, train_fp32=True)

    # ── Grad-flow: hn trainable, Qwen frozen ──
    qwen_trainable = len(tree_flatten(model.trainable_parameters()))
    hn_trainable = len(tree_flatten(hn.trainable_parameters()))
    print(f"Qwen trainable param arrays (must be 0): {qwen_trainable}")
    print(f"hn   trainable param arrays: {hn_trainable}")
    assert qwen_trainable == 0, "Qwen must be frozen (no trainable params)"
    assert hn_trainable > 0

    emb = mx.zeros((1, 768))  # zero embedding (faithful to dir-train path)
    code = "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n"
    ids = _tokenize(tok, code)
    print(f"example token length: {len(ids)}")

    batch = [(emb[0], ids)]

    def loss_fn():
        return trainer._ir_batch_loss(mx.array([b[0] for b in batch]), [b[1] for b in batch])

    lvg = nn.value_and_grad(hn, loss_fn)
    loss0, grads = lvg()
    mx.eval(loss0, grads)
    flat = tree_flatten(grads)
    nonzero = sum(1 for _k, v in flat if float(mx.sum(mx.abs(v))) > 0)
    print(f"grad arrays: {len(flat)}, nonzero: {nonzero}")
    assert mx.isfinite(loss0).item(), "loss must be finite"
    assert nonzero > 0, "hypernetwork grads must be nonzero"

    # ── Loss decreases over a few steps on the same example ──
    optimizer = optim.AdamW(learning_rate=1e-3, betas=[0.9, 0.999], eps=1e-8, weight_decay=0.01)
    losses = [float(loss0)]
    for step in range(6):
        lvg = nn.value_and_grad(hn, loss_fn)
        loss, grads = lvg()
        optimizer.update(hn, grads)
        mx.eval(hn.parameters(), optimizer.state)
        losses.append(float(loss))
    print("IR loss trajectory:", [f"{x:.4f}" for x in losses])
    assert losses[-1] < losses[0], f"loss must decrease ({losses[0]:.4f} -> {losses[-1]:.4f})"
    print(f"LOSS DECREASED: {losses[0]:.4f} -> {losses[-1]:.4f}  ✓")

    # ── Full-loop smoke on sample-proj ──
    print("\n=== full Trainer.train on sample-proj (3 epochs) ===")
    hn2 = Code2LoRAHead(cfg)
    mx.eval(hn2.parameters())
    trainer2 = Trainer(hn2, model, tok, {
        "epochs": 3, "lr": 1e-4, "batch_size": 2, "output": "/tmp/mlx_p1_ckpt", "cr_holdout": 0.2,
    }, train_fp32=True)
    dataset = CodeDataset.load_from_dir(SAMPLE_DIR)
    print(f"dataset: {len(dataset)} examples")
    step_times = trainer2.train(dataset)
    warm = step_times[1:] if len(step_times) > 1 else step_times
    print(f"\nMLX s/step (steady-state): mean={sum(warm) / len(warm):.4f}s "
          f"min={min(warm):.4f}s n={len(warm)} (warmup={step_times[0]:.4f}s)")
    print("default device:", mx.default_device())
    print("=== P1 SMOKE PASSED ===")


if __name__ == "__main__":
    main()
