"""Re-analysis of lambda_residual_results.json on HARDNESS strata (codex-challenger
attack #3: ~41% of examples have avg NLL≈0 — a floor where no residual CAN help, so a
real repo-specific residual could be invisible in the aggregate). Strata predeclared
from λ=0 (avg) ONLY, so they don't peek at the residual effect. No model forward — pure
re-analysis of the dumped per-example NLL/rank.
"""
import json
import numpy as np

LAMBDAS = [0.5, 1.0, 2.0, 4.0, 8.0]
N_BOOT = 2000
recs = json.load(open("lambda_residual_results.json"))["records"]
print(f"loaded {len(recs)} records")


def boot_ci(d):
    d = np.array(d, dtype=np.float64)
    if len(d) == 0:
        return (float("nan"), float("nan"), float("nan"))
    idx = np.random.default_rng(0).integers(0, len(d), size=(N_BOOT, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def report(sub, label):
    if not sub:
        print(f"\n### {label}  (n=0 — empty)")
        return
    base = np.mean([r["nll"]["base"] for r in sub])
    avg = np.mean([r["nll"]["avg"] for r in sub])
    print(f"\n### {label}  (n={len(sub)})   base NLL={base:.3f}  avg NLL={avg:.3f}  (avg−base={avg-base:+.3f})")
    print(f"  {'λ':>4} | {'OWN ΔNLL vs avg [95% CI]':<32} | {'OTHER ΔNLL vs avg [95% CI]':<32}")
    for l in LAMBDAS:
        cells = []
        for s in ("own", "other"):
            cfg = f"{s}@{l}"
            d = [r["nll"][cfg] - r["nll"]["avg"] for r in sub if cfg in r["nll"]]
            m, lo, hi = boot_ci(d)
            sig = "*HELP" if hi < 0 else ("*HURT" if lo > 0 else "")
            cells.append(f"{m:+.4f} [{lo:+.4f},{hi:+.4f}] {sig:<5}")
        print(f"  {l:>4} | {cells[0]} | {cells[1]}")


# saturation fraction
n0 = sum(1 for r in recs if r["nll"]["avg"] <= 1e-9)
print(f"avg NLL≈0 (saturated): {n0}/{len(recs)} = {100*n0/len(recs):.0f}%")

report(recs, "ALL (reference)")
report([r for r in recs if r["nll"]["avg"] > 0.5], "HARD: avg NLL > 0.5")
report([r for r in recs if r["nll"]["avg"] > 1.0], "HARD: avg NLL > 1.0")
report([r for r in recs if r["rank"]["avg"] > 10], "HARD: avg gold-rank > 10")
report([r for r in recs if r["nll"]["avg"] > 0.5 and not r["tin"]],
       "HARD: avg NLL>0.5 AND target-not-in-prefix")
