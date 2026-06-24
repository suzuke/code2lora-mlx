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
  mechanism is most likely **mis-attributed** (a better generic adapter, not repo conditioning).
  The small residual that the geometry left open has now been **functionally ruled out**: amplifying
  each repo's `own−avg` residual (λ up to 8×) helps no stratum, and a wrong-repo residual helps just as
  much — so the residual carries no repo-specific function (*Status → λ-residual functional test*).

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

Of the two hypotheses previously demoted to **open**: "per-repo customization contributes ≈0" is now
**supported on the audited eval** by the functional λ-residual test below (was: "needs full-distribution
functional testing"); "the generic-adapter advantage comes from varied-repo training, by elimination"
remains **open** (still needs a factorial cause ablation).

**The single experiment most likely to overturn the central claim** (per the adversarial review)
was a **λ-residual functional test** — build `avg + λ·(own − avg)` for λ∈{0,0.5,1,2,4,8} (with
`random-other` as control), score target-token NLL / first-token logit-rank, stratified. If λ>1
helped any pre-declared stratum while λ=1 was flat, the released checkpoint would contain a
repo-conditioned direction the head under-amplifies — breaking "collapse ⇒ no conditioning."

**RESULT (`lambda_residual.py`, 8 held-out cr_test repos, 410 examples, official `code2lora-direct`):
the residual is functionally dead — the claim is *strengthened*, not overturned.**
- The population-mean (λ=0) adapter delivers essentially all the benefit: base NLL **1.411 →
  avg 0.376 (−1.035)**. The per-repo residual is tiny to begin with: mean `‖own−avg‖/‖own‖` = **0.016**.
- **`own` residual helps NO stratum at any λ** — every 95% bootstrap CI straddles 0 (ALL, target-in/out-of-prefix,
  short/long target); the largest effect anywhere is **≤0.007 NLL ≈ <1 %** of the generic adapter's −1.035,
  and λ=8 (8× amplification) trends *positive* (slightly hurts). The "under-amplified real direction" flip did **not** occur.
- **Control clinches it:** the wrong-repo (`random-other`) residual helps *as often as or more than* `own`
  (a few λ=1–2 CIs reach −0.003…−0.007). So the trace gain is **generic residual magnitude, not repo signal** —
  exactly the falsification the review predicted. Gold first-token rank is identical (avg 14.7 vs own@1 14.5).

This is a *functional* in-support test (NLL + logit-rank, not just geometry), with a proper wrong-repo control —
closing the gap the adversarial review flagged. It promotes the former open hypothesis
*"per-repo customization contributes ≈0"* from "open" to **supported on the audited eval**: on the released
checkpoint, the per-repo conditioning direction is functionally negligible and not repo-specific.

**Robustness follow-ups (`lambda_residual_hardness.py`, `lambda_followups.py`) — the verdict survives every
attack the reviewer raised; `own` stays flat under all four.**
1. **Saturation/hardness.** 41 % of examples already have avg NLL≈0 (a floor). Re-stratifying on hardness
   (avg NLL>0.5, >1.0) does **not** wake `own` up — every CI still straddles 0 — while wrong-repo `other`
   *significantly helps* on the hard slice (NLL>1.0: λ=1,2,4,8 all CI<0, −0.014…−0.052). Removing the floor
   makes the clincher stronger, not weaker.
2. **Embedding provenance** (the one thing that could make it a false negative). The `repo_state_embedding`
   we fed is **byte-identical** (max|Δ|=0 over 200 keys) to the static/snapshots dataset copy — i.e. exactly the
   conditioning input the *direct* checkpoint's own dataset ships (the only other 2048-d embedding,
   `diff_embedding`, belongs to the Evo/GRU variant). Feeding `diff_embedding` instead yields a *worse* generic
   adapter (0.408 vs 0.376) and `own_diff` ≈ `avg_diff` (residual still dead) — no embedding variant resurrects it.
3. **ΔW-space (SVD) λ blend** — removes the factor-space bilinear-cross-term confound (`Dλ=D_avg+λ(D_own−D_avg)`,
   SVD back to rank-16; sanity: ΔW-SVD own@1=0.3753 ≈ factor own@1=0.3755). `own` is **flat at every λ** in clean
   ΔW space too; `other` again helps more (λ=1,2,4 CI<0). The "under-amplified direction" escape hatch is closed.
4. **Robust wrong-repo null** — replacing the single deterministic next-repo control with **all 7 wrong repos**:
   `own` beats only **38 % (λ=1) / 25 % (λ=4)** of wrong repos, i.e. it sits *below* the wrong-repo median ΔNLL.
   The residual is not merely dead — it is no more repo-specific than a randomly chosen wrong repo's residual.

Net: across factor- and ΔW-space, hardness slices, the canonical vs the alternative embedding, and a robust
7-way wrong-repo null, the released static checkpoint's held-out per-repo residual is **functionally dead on
this benchmark** — meeting the reviewer's pre-stated bar. The independent skeptical agent (codex) re-ran the
artifacts and returned a final **VERIFIED** on exactly this scoped claim (it also judged the `N_TRAIN=150` vs
full-400 centroid choice immaterial here, since `own@1` is the released adapter itself and is flat, and the
all-wrong-repo test shares the same baseline). **Scope boundaries that must stay attached** (do not broaden
without separately running them): (1) the audited set is **8 repos / 410 examples**, not the full 51 cr_test;
(2) the benchmark is short **assertion-completion** and may be weakly conditional; (3) **train / in-support**
functionality is untested; (4) the **Evo (GRU)** variant must not inherit this static verdict automatically.

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
PYTHONPATH=. uv run --with torch python lambda_residual.py      # λ-residual functional test: own residual dead, control ≥ own
PYTHONPATH=. uv run python lambda_residual_hardness.py          # hardness re-analysis: own flat even on hard slices
PYTHONPATH=. uv run --with torch python lambda_followups.py     # provenance + ΔW-SVD + all-wrong-repo: own flat under all
```

*Verification: code reviewed by an independent codex agent (VERIFIED); MLX↔PyTorch parity ≤2e-8.*
