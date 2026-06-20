"""Training loop — faithful MLX port of code2lora-lite/src/trainer.rs + the
IR/CR loss math in code2lora-lite/src/base_llm.rs.

Loop (trainer.rs:38-168):
  * AdamW(lr=cfg.lr, beta1=.9, beta2=.999, eps=1e-8, weight_decay=0.01) over the
    HYPERNETWORK vars only (trainer.rs:74-82). Qwen stays frozen.
  * split(cr_holdout) -> (CR, IR) examples (trainer.rs:64-71).
  * per epoch: run IR batches THEN CR batches (trainer.rs:84-127).
  * checkpoint every 5 epochs + final.safetensors (trainer.rs:154-165).

IR loss (base_llm.rs:138-210, compute_ir_loss -> compute_single_ir_loss ->
compute_ir_loss_with_lora): for each example, generate per-layer LoRA from the
hypernetwork, inject into the frozen Qwen, forward -> logits, next-token
cross-entropy over the shifted sequence; average over the batch.

CR loss (base_llm.rs:214-258, compute_cr_loss): for each example forward the
LoRA-adapted Qwen, mean-pool hidden states over the sequence -> a per-example
representation; L2-normalize the stacked (B, hidden) matrix; InfoNCE over the
similarity matrix / temperature 0.07 with identity targets.

DTYPE: the base Qwen is cast to fp32 for training (reviewer note: bf16 LoRA-delta
application loses precision for gradient accumulation). See Trainer.__init__.

GRADIENT FLOW: nn.value_and_grad(hn, fn) differentiates fn w.r.t. the
hypernetwork's trainable params only. The hn-generated A/B arrays are threaded
into the (frozen, constant) Qwen forward, so autograd traces hn -> A/B -> Qwen
-> loss while Qwen params remain constants. Qwen is never in the optimizer.
"""

import math
import os
import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from . import qwen_lora
from .dataset import BatchIterator, CodeDataset
from .hypernetwork import Code2LoRAHead

# base_llm.rs:112  max_len = max_position_embeddings.min(2048)
MAX_SEQ_LEN = 2048
# base_llm.rs:273  Qwen2.5 EOS / <|endoftext|>
EOS_TOKEN = 151643
# base_llm.rs:252  contrastive temperature
CR_TEMPERATURE = 0.07


def _tokenize(tokenizer, text: str):
    """Mirror base_llm.rs::tokenize — encode, truncate to <=2048 (with BOS/special
    tokens via the tokenizer's default add_special_tokens, matching Rust encode(.., true))."""
    ids = tokenizer.encode(text)
    if len(ids) > MAX_SEQ_LEN:
        ids = ids[:MAX_SEQ_LEN]
    return ids


def _l2_normalize(x: mx.array) -> mx.array:
    # base_llm.rs:322 l2_normalize — norm over axis=1, clamp(1e-12, inf).
    norm = mx.sqrt(mx.sum(x * x, axis=1, keepdims=True))
    norm = mx.maximum(norm, 1e-12)
    return x / norm


class Trainer:
    def __init__(
        self,
        hypernetwork: Code2LoRAHead,
        model,
        tokenizer,
        config: dict,
        train_fp32: bool = True,
    ):
        self.hn = hypernetwork
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        # Freeze + (optionally) cast Qwen to fp32 for precise gradient accumulation.
        if train_fp32:
            self.model.set_dtype(mx.float32)
        # Qwen is frozen: it is never passed to the optimizer / value_and_grad
        # target, so its params receive no updates. freeze() also detaches it from
        # any trainable-parameter traversal.
        self.model.freeze()

    # ── IR (generative) per-batch loss — base_llm.rs:138-210 ──
    def _ir_batch_loss(self, repo_embs: mx.array, token_lists: list):
        """Average next-token CE over the batch (compute_ir_loss)."""
        total = mx.zeros(())
        counted = 0
        for i, ids in enumerate(token_lists):
            seq_len = len(ids)
            if seq_len < 2:
                # base_llm.rs:157 — examples shorter than 2 contribute zero loss,
                # but still count toward the batch average (the Rust sum adds 0
                # and divides by batch_size).
                counted += 1
                continue
            repo_emb = repo_embs[i : i + 1]  # (1, 768)
            all_lora = self.hn.forward_all(repo_emb)
            qwen_lora.clear_lora(self.model)
            qwen_lora.inject_lora(self.model, all_lora)
            input_ids = mx.array([ids])  # (1, seq)
            logits = self.model(input_ids)  # (1, seq, vocab) — tied lm_head
            qwen_lora.clear_lora(self.model)

            shift_logits = logits[:, : seq_len - 1, :].reshape(seq_len - 1, -1)
            shift_labels = input_ids[:, 1:].reshape(seq_len - 1)
            ex_loss = nn.losses.cross_entropy(
                shift_logits, shift_labels, reduction="mean"
            )
            total = total + ex_loss
            counted += 1
        # base_llm.rs:208 — divide the summed loss by batch_size.
        return total / max(counted, 1)

    def _log_scale_l2(self) -> mx.array:
        """L2 penalty on the hypernetwork log_scale params (TUNING FIX). The
        clamp in hypernetwork._scale already caps the effective scale; this
        penalty additionally pulls log_scale toward 0 (scale->1) so the LoRA
        delta stays small unless the data really benefits from a larger one."""
        s = mx.zeros(())
        for m in self.hn.log_scale_a:
            s = s + mx.sum(self.hn.log_scale_a[m] ** 2)
            s = s + mx.sum(self.hn.log_scale_b[m] ** 2)
        return s

    # ── CR (contrastive) per-batch loss — base_llm.rs:214-258 ──
    def _cr_batch_loss(self, repo_embs: mx.array, token_lists: list):
        batch_size = len(token_lists)
        if batch_size < 2:
            return mx.zeros(())
        inner = getattr(self.model, "model", self.model)
        hidden_dim = self.hn.cfg.llm_hidden_dim
        reprs = []
        for i, ids in enumerate(token_lists):
            seq_len = len(ids)
            if seq_len < 2:
                reprs.append(mx.zeros((1, hidden_dim)))
                continue
            repo_emb = repo_embs[i : i + 1]
            all_lora = self.hn.forward_all(repo_emb)
            qwen_lora.clear_lora(self.model)
            qwen_lora.inject_lora(self.model, all_lora)
            input_ids = mx.array([ids])
            hidden = inner(input_ids)  # (1, seq, hidden)
            qwen_lora.clear_lora(self.model)
            reprs.append(mx.mean(hidden, axis=1))  # (1, hidden) mean-pool

        stacked = mx.concatenate(reprs, axis=0)  # (B, hidden)
        normalized = _l2_normalize(stacked)
        sim = normalized @ normalized.T  # (B, B)
        logits = sim / CR_TEMPERATURE
        labels = mx.arange(batch_size)
        return nn.losses.cross_entropy(logits, labels, reduction="mean")

    # ── TUNING-FIX diagnostics (cheap, run once per epoch) ──
    def _effective_scale(self) -> float:
        """Max effective LoRA scale = max_m exp(clamped log_scale). Should stay
        bounded (the 10-repo run hit ~2432)."""
        vals = []
        for m in self.hn.log_scale_a:
            vals.append(float(self.hn._scale(self.hn.log_scale_a[m])))
            vals.append(float(self.hn._scale(self.hn.log_scale_b[m])))
        return max(vals) if vals else 0.0

    def _cross_repo_cosine(self, emb_list) -> float:
        """Mean pairwise cosine of the FLAT LoRA-A vectors the hn emits for a few
        distinct repos. ~1.0 == the adapter is the same regardless of repo (the
        10-repo failure mode); WANT this well below 0.98."""
        flats = []
        for emb in emb_list:
            all_lora = self.hn.forward_all(mx.array(emb[None, :]))
            parts = [all_lora[li][m][0].reshape(-1) for li in range(0, self.hn.cfg.num_layers, 6) for m in ("q", "v")]
            flats.append(mx.concatenate(parts))
        mx.eval(flats)
        flats = [np.asarray(f) for f in flats]
        c,s = 0,0.0
        for i in range(len(flats)):
            for j in range(i + 1, len(flats)):
                a, b = flats[i], flats[j]
                d = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
                s += float(a @ b) / d
                c += 1
        return s / max(c, 1)

    def train(self, dataset: CodeDataset, val_examples=None):
        n = len(dataset)
        assert n > 0, "training dataset is empty"
        summary = dataset.summary()
        print(
            f"Dataset summary: repos={summary['repo_count']}, "
            f"languages={summary['language_count']}, commit_rows={summary['commit_rows']}"
        )

        n_epochs = int(self.config["epochs"])
        lr = float(self.config["lr"])
        batch_size = max(1, int(self.config["batch_size"]))
        out_dir = self.config["output"]
        os.makedirs(out_dir, exist_ok=True)
        cr_ratio = float(self.config.get("cr_holdout", 0.2))
        # TUNING-FIX knobs (with safe defaults matching the diagnosed fixes).
        grad_clip = float(self.config.get("grad_clip", 1.0))            # global-norm clip
        log_scale_l2 = float(self.config.get("log_scale_l2", 1e-3))      # log_scale L2 lambda
        val_every = int(self.config.get("val_every", 1))                 # epochs between vals
        patience = int(self.config.get("patience", 3))                   # early-stop patience
        seed = int(self.config.get("seed", 3407))
        rng = random.Random(seed)

        cr_examples, ir_examples = dataset.split(cr_ratio)
        # If split() routed everything to IR (explicit cross_repo_split="train"),
        # carve a CR holdout ourselves so the contrastive term has signal.
        if len(cr_examples) == 0 and len(ir_examples) > 4:
            shuf = list(ir_examples)
            rng.shuffle(shuf)
            k = max(2, int(len(shuf) * cr_ratio))
            cr_examples, ir_examples = shuf[:k], shuf[k:]
        print(f"CR examples: {len(cr_examples)}, IR examples: {len(ir_examples)}")

        # AdamW over hypernetwork params ONLY (trainer.rs:74-82).
        optimizer = optim.AdamW(
            learning_rate=lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=0.01
        )

        # Pre-tokenize so timing measures the train step, not tokenization.
        ir_tok = [
            (e.repo_embedding, _tokenize(self.tokenizer, e.code_content))
            for e in ir_examples
        ]
        cr_tok = [
            (e.repo_embedding, _tokenize(self.tokenizer, e.code_content))
            for e in cr_examples
        ]
        val_tok = None
        if val_examples:
            val_tok = [
                (e.repo_embedding, _tokenize(self.tokenizer, e.code_content))
                for e in val_examples
            ]

        # A few DISTINCT repo embeddings for the cross-repo-cosine diagnostic.
        seen, diag_embs = set(), []
        for e in ir_examples:
            if e.repo_id not in seen:
                seen.add(e.repo_id)
                diag_embs.append(np.asarray(e.repo_embedding, np.float32))
            if len(diag_embs) >= 6:
                break

        step_times = []

        def ir_loss_closure(batch):
            embs = mx.array(np.stack([b[0] for b in batch], axis=0))
            toks = [b[1] for b in batch]
            ce = self._ir_batch_loss(embs, toks)
            # log_scale L2 regularization (TUNING FIX) added to the IR objective.
            return ce + log_scale_l2 * self._log_scale_l2()

        def cr_loss_closure(batch):
            embs = mx.array(np.stack([b[0] for b in batch], axis=0))
            toks = [b[1] for b in batch]
            return self._cr_batch_loss(embs, toks)

        def clipped_update(grads):
            # Global-norm gradient clipping before the AdamW step (TUNING FIX:
            # lr=1e-4 + unclipped grads diverged at epoch 9 in the 10-repo run).
            if grad_clip and grad_clip > 0:
                grads, _gnorm = optim.clip_grad_norm(grads, grad_clip)
            else:
                _gnorm = mx.zeros(())
            optimizer.update(self.hn, grads)
            return _gnorm

        best_val = float("inf")
        best_epoch = -1
        no_improve = 0
        best_path = os.path.join(out_dir, "best.safetensors")

        for epoch in range(n_epochs):
            epoch_ir_loss = 0.0
            epoch_cr_loss = 0.0
            ir_steps = cr_steps = 0
            last_gnorm = 0.0

            # Repo-MIXING (TUNING FIX): shuffle examples each epoch so every batch
            # contrasts MULTIPLE repos (the 10-repo run had ~no CR signal because
            # examples were repo-contiguous).
            rng.shuffle(ir_tok)
            rng.shuffle(cr_tok)

            # ── IR (generative) ──
            for bidx in range(0, len(ir_tok), batch_size):
                batch = ir_tok[bidx : bidx + batch_size]
                t0 = time.perf_counter()
                lvg = nn.value_and_grad(self.hn, lambda: ir_loss_closure(batch))
                loss, grads = lvg()
                gn = clipped_update(grads)
                mx.eval(self.hn.parameters(), optimizer.state, gn)
                step_times.append(time.perf_counter() - t0)
                last_gnorm = float(gn)
                epoch_ir_loss += float(loss)
                ir_steps += 1
                if (bidx // batch_size) % 50 == 0:
                    print(
                        f"  IR batch {bidx // batch_size}/{len(ir_tok)//batch_size}: "
                        f"loss={float(loss):.4f} gnorm={last_gnorm:.3f}"
                    )

            # ── CR (contrastive) ──
            for bidx in range(0, len(cr_tok), batch_size):
                batch = cr_tok[bidx : bidx + batch_size]
                if len(batch) < 2:
                    continue
                t0 = time.perf_counter()
                lvg = nn.value_and_grad(self.hn, lambda: cr_loss_closure(batch))
                loss, grads = lvg()
                gn = clipped_update(grads)
                mx.eval(self.hn.parameters(), optimizer.state, gn)
                step_times.append(time.perf_counter() - t0)
                epoch_cr_loss += float(loss)
                cr_steps += 1

            avg_ir = epoch_ir_loss / ir_steps if ir_steps else 0.0
            avg_cr = epoch_cr_loss / cr_steps if cr_steps else 0.0
            eff_scale = self._effective_scale()
            xcos = self._cross_repo_cosine(diag_embs) if len(diag_embs) >= 2 else float("nan")
            print(
                f"Epoch {epoch + 1}/{n_epochs} — ir={avg_ir:.4f} cr={avg_cr:.4f} "
                f"| eff_scale={eff_scale:.3f} xrepo_cos={xcos:.4f} gnorm={last_gnorm:.3f}"
            )

            # ── Validation + best-checkpoint + early stop (TUNING FIX) ──
            if val_tok and (epoch + 1) % val_every == 0:
                vloss = self._val_ce(val_tok)
                improved = vloss < best_val - 1e-4
                print(
                    f"   VAL ce={vloss:.4f}  best={best_val:.4f}  "
                    f"{'IMPROVED' if improved else f'no-improve {no_improve+1}/{patience}'}"
                )
                if improved:
                    best_val, best_epoch, no_improve = vloss, epoch + 1, 0
                    self.save_hn(best_path)
                    print(f"   * new best -> {best_path}")
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"   EARLY STOP at epoch {epoch + 1} (val plateaued).")
                        break

            if (epoch + 1) % 5 == 0:
                ckpt = os.path.join(out_dir, f"epoch_{epoch + 1:04d}.safetensors")
                self.save_hn(ckpt)
                print(f"Checkpoint saved to {ckpt}")

        final = os.path.join(out_dir, "final.safetensors")
        self.save_hn(final)
        print(f"Final checkpoint saved to {final}")
        if best_epoch > 0:
            print(f"BEST checkpoint: epoch {best_epoch}, val_ce={best_val:.4f} -> {best_path}")
        return step_times

    def _val_ce(self, val_tok) -> float:
        """Mean ADAPTED next-token CE over a held-out-from-train validation set.
        No grad; the LoRA is injected/cleared per example like in IR."""
        total, cnt = 0.0, 0
        for emb, ids in val_tok:
            if len(ids) < 2:
                continue
            repo_emb = mx.array(np.asarray(emb, np.float32)[None, :])
            all_lora = self.hn.forward_all(repo_emb)
            qwen_lora.clear_lora(self.model)
            qwen_lora.inject_lora(self.model, all_lora)
            input_ids = mx.array([ids])
            logits = self.model(input_ids)
            qwen_lora.clear_lora(self.model)
            sl = len(ids)
            shift_logits = logits[:, : sl - 1, :].reshape(sl - 1, -1)
            shift_labels = input_ids[:, 1:].reshape(sl - 1)
            loss = nn.losses.cross_entropy(shift_logits, shift_labels, reduction="mean")
            total += float(loss)
            cnt += 1
        return total / max(cnt, 1)

    def save_hn(self, path: str):
        """Save hypernetwork params to safetensors (mirrors hypernetwork.rs::save)."""
        flat = dict(tree_flatten(self.hn.parameters()))
        mx.save_safetensors(path, flat)
