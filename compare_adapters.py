"""Characterize HOW code2lora's adapter differs from a from-scratch shared LoRA.
Both are single fixed 659k adapters (same shape). Compares the effective update
ΔW = scale·(B@A) per module: direction (cosine), magnitude (Frobenius norm),
effective rank (participation ratio of singular values). Pure numpy, no model.
"""

import numpy as np
import numpy.linalg as la

TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
C = np.load("slim_code2lora.npz")    # code2lora centroid (avg_train); applied with alpha/rank=2
F = np.load("slim_shared_lora.npz")  # our from-scratch shared LoRA; applied with scale 1
S_C, S_F = 2.0, 1.0


def effrank(M):
    s = la.svd(M.astype(np.float64), compute_uv=False)
    s2 = s ** 2
    return float((s2.sum() ** 2) / (s2 ** 2).sum())


print(f"{'module':10} {'cos(ΔW)':>9} {'‖ΔW‖ c2l':>9} {'‖ΔW‖ fs':>9} {'erank c2l':>10} {'erank fs':>9}")
allc, allf = [], []
for t in TYPES:
    dW1 = S_C * (C[f"B_{t}"] @ C[f"A_{t}"])
    dW2 = S_F * (F[f"B_{t}"] @ F[f"A_{t}"])
    cos = float(dW1.ravel() @ dW2.ravel() / (la.norm(dW1) * la.norm(dW2) + 1e-12))
    print(f"{t:10} {cos:9.4f} {la.norm(dW1):9.3f} {la.norm(dW2):9.3f} {effrank(dW1):10.1f} {effrank(dW2):9.1f}")
    allc.append(dW1.ravel())
    allf.append(dW2.ravel())

uc, uf = np.concatenate(allc), np.concatenate(allf)
print("-" * 64)
print(f"overall cos(ΔW_code2lora, ΔW_fromscratch) = {float(uc@uf/(la.norm(uc)*la.norm(uf))):.4f}")
print(f"overall ‖ΔW‖ : code2lora={la.norm(uc):.2f}   from-scratch={la.norm(uf):.2f}   "
      f"(ratio {la.norm(uc)/la.norm(uf):.2f}×)")
