# Findings — Does Code2LoRA's repository conditioning actually do anything?

An independent MLX re-implementation and audit of the released **Code2LoRA** checkpoints
(Hotsko et al., [arXiv:2606.06492](https://arxiv.org/abs/2606.06492); HF [`code2lora`](https://huggingface.co/code2lora)).

## TL;DR
The released Code2LoRA hypernetwork — **both** the Static and the Evo (GRU) variant — exhibits
**conditioning collapse**: it generates a **near-identical LoRA regardless of which repository
(or commit) it is given** (cross-repo adapter cosine **0.9996**, cross-commit **0.9999**).
The method's accuracy gains over baselines are **real**, but our measurements indicate they come from
a **better *generic* task adapter** rather than the per-repository specialization the paper
credits — at least on the evaluations we could run. The paper reports only indirect
(downstream-accuracy) evidence and never measures adapter diversity. (Conclusions narrowed after an
adversarial review — see *Status after adversarial review*.)

## Setup
- Faithful MLX port of the official Static head (`OfficialDirectHead`) and the Evo GRU
  (`CommitGRU`), numerically verified against the PyTorch checkpoints: max abs error
  **≤2e-8** on generated A/B; single training-step gradient cosine **= 1.0** vs the official
  PyTorch head. Independently code-reviewed (verdict **VERIFIED** — no bug that fakes the result).
- Base model Qwen2.5-Coder-1.5B, rank 16. Eval: 8 held-out `cr_test` repos, the paper's
  test-assertion-completion task (cross-entropy on the target span + relaxed greedy exact-match).

## Finding 1 — The adapters barely depend on the repo (conditioning collapse)
- Measured over the full **cr_test set — 51 repos the hypernetwork never trained on**. Their input
  repo embeddings are genuinely diverse (pairwise cosine **0.30–0.95**, mean 0.74), yet the
  generated adapters **collapse to mean cosine 0.998** (min 0.979; 66% of pairs > 0.999). Distinct,
  unseen repos → near-identical adapters. (Holds even wider — over all 408 train+test repos, mean
  0.9998 — but the 51 non-training repos are the rigorous check.) Not a small-sample artifact.
  The fairer functional metric, the effective update **ΔW = B·A**, also collapses but a touch less
  strongly: mean cosine **0.996** (min 0.957), with a repo-specific residual ~**5%** of the
  adapter's norm — small, but *not literally zero*.
- Replacing every repo's adapter with the **average of the hypernetwork's outputs**
  (`avg_train`, one fixed adapter) reproduces the full model **exactly**: CE 0.377 → **0.378**,
  EM 59.4% → **59.4%** (held-out; loaded from a standalone 2.6 MB file, no encoder/hypernetwork).
- "Right-adapter-wins" test: a repo's **own** adapter is **no better than a random other
  repo's** adapter — own-repo mean rank **4.60 / 8** (random 4.5), wins **8.2%** (random 12.5%),
  ΔCE −0.0008.
- Evo: the GRU hidden state genuinely evolves across commits (step-to-step cosine 0.977), but the
  **head squashes evolved states into a near-constant adapter** (cross-commit cosine **0.9999**);
  per-commit Evo = a fixed shared adapter, **+0.0pp EM**.

→ On the audited eval, per-repo / per-commit specialization shows **no measurable benefit** (both
variants), and the collapse sits in the shared **head** (the front-end works — the Evo GRU state
does evolve). **Honest scope:** the ΔW residual is ~5%, not literally zero, and the functional
tests here (avg==own, rank) are coarse — 8 repos, relaxed greedy EM, short targets. A small residual
could still matter on slices we didn't probe (rare repo-specific identifiers, longer generations,
in-support repos). See *Status after adversarial review*.

## Finding 2 — Yet the method still beats a *from-scratch* shared LoRA (via a better *generic* adapter)
This reconciles Finding 1 with the paper (and corrects a naive "code2lora = a shared LoRA" reading):

| (paper Tables 2–3, 52 cr_test repos) | Code2LoRA | from-scratch single shared LoRA | gap |
|---|---|---|---|
| Static, cross-repo EM | 63.8% | **47.4%** | **+16.4pp** |
| Evo, cross-repo EM | 60.3% | 55.1% | +5.2pp |

- Code2LoRA **does** substantially beat an independently-trained shared LoRA. We reproduce the
  shape: a shared LoRA we trained from scratch reaches only **EM ~41%** (CE 0.37) — far below the
  hypernetwork's ~59%.
- But `avg_train` (the hypernetwork's **own averaged, repo-agnostic** output) already hits the
  full ~59%.

→ The +16.4pp is **not** explained by repo-specific adaptation (the adapters don't vary by repo);
it is explained by the hypernetwork training producing a **higher-quality generic adapter** than
direct shared-LoRA training.

**Confirmed — not a baseline-tuning artifact.** We then trained a shared LoRA *properly* (full
train set, 3 epochs, grad-accum effective batch 8, lr 1e-4 + warmup/cosine, full-sequence CE,
135 min). It plateaus at **EM ~44–48%** (CE ~1.27 target-span) — i.e. it lands right at the
paper's reported single-LoRA baseline (**47.4%**), **not** the hypernetwork's 59%. So the
hypernetwork's generic-adapter advantage (+~12pp) is **real**: directly training a shared LoRA
cannot reach it, even with a fair, well-tuned run. So the hypernetwork's value is producing a
better *generic* adapter — not the repo conditioning. (What in the training causes this is
isolated in Finding 3.)

## Finding 3 — The advantage comes from *training on many varied repos*, not the architecture
To isolate whether the big over-parameterized architecture is itself the source, we trained that
exact architecture (~679M: trunk + giant per-module heads, random init) end-to-end on the task,
but fed a **FIXED input** (mean train-repo embedding — no repo variety, no conditioning). It
reached only **EM ~40%** (flat across all 3 epochs) — *worse* than a directly-trained shared LoRA
(~47%), barely above base. So the over-parameterized architecture, fed a fixed input, does not on
its own produce a better adapter. **Leading hypothesis (NOT proven):** the +12pp comes from training
on ~600 varied repos (diversity as regularization). **Caveat — this is not a clean isolation:** the
fixed-input test confounds "no diversity" with "hard to optimize a 680M generator from scratch under
this recipe," and changes several factors at once. The true cause — output constraints (tanh +
clamped scale), input diversity, the contrastive (CR) loss, or over-parameterized optimization —
needs a *factorial* ablation, not done here.

→ **Working picture:** Code2LoRA's measured benefit is largely reproduced by one robust generic
adapter; its headline *conditioning* shows little measurable effect on this eval, and the
*architecture* alone is not the source of the generic-adapter quality (cause still open).

## Reconciliation with the paper
- The paper's accuracy gains are real; we do not dispute them.
- It attributes them to **repository-specific** adaptation but gives only **indirect** evidence
  (downstream EM) and **never measures adapter diversity**; the limitations section discusses
  parameter count, not conditioning.
- Our direct measurements (adapter cosine, average-adapter ablation, own-vs-other rank, Evo
  cross-commit) show the per-repo/commit signal is **weak on this evaluation** → the credited
  mechanism is most likely **mis-attributed** (a better generic adapter, not repo conditioning) —
  modulo the ~5% residual a finer functional test still needs to rule out (*Status*).

## Caveats (scope, stated honestly)
- The **geometric** evidence (adapter cosine) is broad: 51 non-training repos. But the **functional**
  evidence (avg==own, rank test) is narrow & coarse: 8 repos, relaxed greedy EM, short targets,
  PER_EM≈8. So "geometric collapse" is solid; "no functional per-repo benefit" is only established
  on this coarse slice.
- `avg_train` / the Evo "shared" baseline are centroids of the hypernetwork's **own** outputs
  (transductive) — they show "Code2LoRA ≈ its own mean," not a comparison to an independent baseline.
- Cosine on the A-stack; B and the effective ΔW=B·A independently checked (also collapse; ΔW residual ~5%).
- Our "proper" shared LoRA used seq-len 640 and our recipe; it *matches* the paper's 47.4% baseline,
  which is suggestive of saturation but is **not** an exact-paper (seq 2048, official harness) run.
- The benchmark (test-assertion completion) may be **weakly conditional** — prefixes/idioms may make
  the repo embedding mostly nuisance — so collapse here need not imply the mechanism is useless on
  tasks that genuinely require repo memory.
- A **Bayesian-shrinkage** reading is possible: maybe conditioning is real in-support and correctly
  shrinks to the population mean for far-from-support held-out repos. Counter-evidence: the collapse
  also holds over 400 **training** repos (cosine 0.9998), but that's still geometric — the functional
  in-support own-vs-other test is not done.

## Status after adversarial review
An independent skeptical agent (codex) attacked these conclusions and ran its own ΔW check. After
that pass, the **defensible claim** is the narrower one:

> *For the released checkpoints, on the audited cr_test assertion-completion evaluation, most of the
> measured benefit is reproduced by the hypernetwork's population-mean adapter, and evidence for
> useful held-out repo-specific conditioning is weak. The paper credits per-repo specialization but
> never measures adapter diversity.*

Demoted to **open hypotheses** (not yet proven): "per-repo customization contributes ≈0" (needs
full-distribution functional testing) and "the generic-adapter advantage comes from varied-repo
training, by elimination" (needs a factorial cause ablation).

**The single experiment most likely to overturn the central claim** (per the adversarial review):
a **λ-residual functional test** — build `avg + λ·(own − avg)` for λ∈{0,0.5,1,2,4,8} (with
`random-other` as control), score target-token NLL / logit-rank (not just EM), stratified by rare
identifiers / target length / in-support-vs-OOD / embedding distance. If λ>1 helps any pre-declared
stratum while λ=1 is flat, the released checkpoint *does* contain a repo-conditioned direction the
head under-amplifies — which would break "collapse ⇒ no conditioning."

## Open question
The interesting problem is not Code2LoRA as shipped, but **why the head collapses the
conditioning and whether it can be made to actually use the repo signal** — e.g. multiplicative/
FiLM conditioning, stronger contrastive pressure, or a benchmark whose answers genuinely require
repo-specific knowledge rather than generic Python. Code2LoRA is then a clean case study in
conditional-generation (hypernetwork) collapse.

## Reproduce
```bash
cd code2lora-mlx
PYTHONPATH=. uv run --with torch python analyze_official.py     # cross-repo adapter cosine 0.9996
PYTHONPATH=. uv run --with torch python repo_match_test.py      # own-repo rank 4.60/8 (≈random)
PYTHONPATH=. uv run --with torch python ablate_harder.py        # avg_train == official (CE/EM)
PYTHONPATH=. uv run --with torch python eval_evo.py             # Evo == shared, cross-commit cosine 0.9999
```

*Verification: code reviewed by an independent codex agent (VERIFIED); MLX↔PyTorch parity ≤2e-8.*
