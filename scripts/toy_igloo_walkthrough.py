#!/usr/bin/env python3
"""
End-to-end IGLOO pipeline on a 5-residue toy protein, small enough to check by hand.

Every ground-truth number (importance weights, BLOSUM62 penalties, fitness values,
leave-one-out deltas) comes from the real codebase functions in
``src.igloo.synthetic_data`` -- only the protein is toy-sized. The embeddings and
predictor are deliberate stand-ins: 2 dimensions per residue instead of ESM-C's 960,
fixed attention weights instead of learned ones, and a single nonlinear head
(F = pooled_1 * pooled_2) instead of the real 4-layer MLP. The head is nonlinear on
purpose so that integrated-gradients path integration differs measurably from
gradient x input -- with a linear head the two would be identical by construction.

Every variant perturbs *all five* positions (large shift at the mutated site, small
shifts elsewhere), mimicking the context-sensitivity real ESM-C shows: a single point
mutation nudges neighbouring positions' embeddings too, so real IG rows are never
exactly zero away from the mutation.

Run:
    python scripts/toy_igloo_walkthrough.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import ndcg_score, roc_auc_score

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.igloo.synthetic_data import (  # noqa: E402
    ProteinDefinition,
    _importance_weight,
    blosum_penalty,
    fitness_additive,
)

np.set_printoptions(precision=5, suppress=True)

# --------------------------------------------------------------------------
# 1. Toy protein + ground truth (real codebase functions, toy-sized protein)
# --------------------------------------------------------------------------
WT = "MKTAY"
TOY = ProteinDefinition(
    name="MINI", uniprot_id="TOY00001", sequence=WT,
    annotations={1: "binding_site", 2: "structural", 3: "folding"},
    category_weights={"binding_site": 1.5, "structural": 1.0, "folding": 0.5},
)

VARIANTS = {
    "M1L":     [(0, "M", "L")],
    "K2R":     [(1, "K", "R")],
    "T3S":     [(2, "T", "S")],
    "A4G":     [(3, "A", "G")],
    "K2R_A4G": [(1, "K", "R"), (3, "A", "G")],
}

# --------------------------------------------------------------------------
# 2. Toy embeddings / predictor (illustrative stand-ins, NOT the real model)
# --------------------------------------------------------------------------
E_WT = np.array([
    [0.20, 0.10],   # pos 1  M  background
    [0.50, 0.60],   # pos 2  K  binding site
    [0.30, 0.40],   # pos 3  T  structural
    [0.15, 0.20],   # pos 4  A  folding
    [0.25, 0.05],   # pos 5  Y  background
])
ATTN = np.array([0.10, 0.35, 0.30, 0.15, 0.10])  # fixed; a learned layer in the real model

# Embedding the mutated residue moves to (large, primary shift).
PRIMARY = {
    (0, "L"): np.array([0.10, 0.30]),
    (1, "R"): np.array([0.10, 0.90]),
    (2, "S"): np.array([0.20, 0.30]),
    (3, "G"): np.array([0.35, 0.45]),
}
# Small context shifts at the *unmutated* positions (ESM-C-like context sensitivity).
BG_DELTA = {
    "M1L": {1: [0.015, -0.010], 2: [-0.008, 0.012], 3: [0.010, 0.006], 4: [-0.005, -0.008]},
    "K2R": {0: [0.008, -0.006], 2: [-0.012, 0.018], 3: [0.006, -0.010], 4: [0.014, 0.004]},
    "T3S": {0: [-0.006, 0.009], 1: [0.011, -0.007], 3: [-0.009, 0.013], 4: [0.007, -0.005]},
    "A4G": {0: [0.009, 0.006], 1: [-0.013, -0.008], 2: [0.010, 0.011], 4: [-0.007, 0.009]},
}
# Double mutant: context shift at each untouched position = sum of the two singles'.
BG_DELTA["K2R_A4G"] = {
    i: [a + b for a, b in zip(BG_DELTA["K2R"][i], BG_DELTA["A4G"][i])] for i in (0, 2, 4)
}

STEPS = 4  # trapezoidal interpolation steps (5 evaluation points, alpha = 0..1)


def pool(e):
    """Attention pooling: collapses the 5 positions, keeps the 2 embedding dims."""
    return ATTN @ e


def F(pooled):
    """Toy predictor head. Nonlinear on purpose (see module docstring)."""
    return pooled[0] * pooled[1]


def variant_embedding(name, mutations):
    e = E_WT.copy()
    for pos, _, mut_aa in mutations:
        e[pos] = PRIMARY[(pos, mut_aa)]
    for pos, delta in BG_DELTA[name].items():
        e[pos] = E_WT[pos] + np.array(delta)
    return e


def integrated_gradients(e0, e1, steps=STEPS):
    """Linear-path IG with the trapezoidal rule, per position, summed over dims."""
    alphas = np.linspace(0.0, 1.0, steps + 1)
    weights = np.ones(steps + 1)
    weights[0] = weights[-1] = 0.5

    grads = np.zeros((steps + 1, len(E_WT), 2))
    preds = np.zeros(steps + 1)
    for k, alpha in enumerate(alphas):
        p = pool(e0 + alpha * (e1 - e0))
        preds[k] = F(p)
        # dF/dpooled = [pooled_2, pooled_1]; dpooled_d/de_{i,d} = attn_i
        dF_dpooled = np.array([p[1], p[0]])
        for i in range(len(E_WT)):
            grads[k, i, :] = dF_dpooled * ATTN[i]

    avg_grad = (weights[:, None, None] * grads).sum(axis=0) / steps
    ig = ((e1 - e0) * avg_grad).sum(axis=1)
    completeness_gap = ig.sum() - (preds[-1] - preds[0])
    return ig, avg_grad, preds, completeness_gap


def main() -> None:
    iv = np.array([_importance_weight(TOY, i) for i in range(len(WT))])
    ann_mask = iv > 0.05
    relevance = np.where(ann_mask, iv, 0.0)  # SI Eq. 11: background zeroed for ranking metrics

    print("=" * 72)
    print("1. GROUND TRUTH")
    print("=" * 72)
    print(f"WT sequence            : {WT}")
    print(f"importance vector iv   : {iv}")
    print(f"relevance (bg zeroed)  : {relevance}   annotated: {int(ann_mask.sum())}/{len(WT)}")

    print()
    print("=" * 72)
    print("2. VARIANTS AND FITNESS  (real blosum_penalty + fitness_additive)")
    print("=" * 72)
    fitness = {}
    for name, muts in VARIANTS.items():
        seq = list(WT)
        for pos, _, mut_aa in muts:
            seq[pos] = mut_aa
        fitness[name] = fitness_additive("".join(seq), TOY, muts)
        pens = [f"{blosum_penalty(a, b):.4f}" for _, a, b in muts]
        print(f"  {name:10s} {''.join(seq)}  fitness={fitness[name]:.4f}  blosum={pens}")

    loo = {"K2R": fitness["A4G"] - fitness["K2R_A4G"],
           "A4G": fitness["K2R"] - fitness["K2R_A4G"]}
    print(f"  leave-one-out deltas for K2R_A4G: {loo}")

    print()
    print("=" * 72)
    print("3. POOLING AND PREDICTION  (toy stand-ins)")
    print("=" * 72)
    pooled_wt = pool(E_WT)
    print(f"  attention weights : {ATTN}   (sum={ATTN.sum():.2f})")
    print(f"  pooled WT vector  : {pooled_wt}   <- still a vector, not a scalar")
    print(f"  F(WT)             : {F(pooled_wt):.5f}   <- head collapses it to a scalar")

    print()
    print("=" * 72)
    print("4. INTEGRATED GRADIENTS, PER VARIANT")
    print("=" * 72)
    names = list(VARIANTS)
    IG = np.zeros((len(names), len(WT)))
    for i, name in enumerate(names):
        e1 = variant_embedding(name, VARIANTS[name])
        ig, avg_grad, preds, gap = integrated_gradients(E_WT, e1)
        IG[i] = ig
        print(f"  {name:10s} IG={ig}  F(wt)={preds[0]:.5f} F(var)={preds[-1]:.5f} "
              f"completeness_gap={gap:.2e}")
        if name == "K2R":  # detailed walkthrough for the headline example
            print("      per-position detail for K2R:")
            for pos in range(len(WT)):
                de = (e1 - E_WT)[pos]
                print(f"        pos{pos + 1}: delta_e={de}  avg_grad={avg_grad[pos]}  "
                      f"IG={ig[pos]:+.5f}")
            gxi = ((e1 - E_WT) * (np.array([pool(e1)[1], pool(e1)[0]]) * ATTN[:, None])).sum(axis=1)
            print(f"      gradient x input (endpoint only), pos2 = {gxi[1]:+.5f} "
                  f"vs full IG {ig[1]:+.5f}")

    abs_IG = np.abs(IG)
    ig_vec = abs_IG.mean(axis=0)
    ig_vec_signed = IG.mean(axis=0)

    print()
    print("=" * 72)
    print("5. THE SEVEN METRICS")
    print("=" * 72)
    print(f"  pooled mean |IG| per position : {ig_vec}")

    print(f"  Global NDCG      = {ndcg_score(relevance.reshape(1, -1), ig_vec.reshape(1, -1)):.4f}")
    print(f"  AUROC            = {roc_auc_score(ann_mask.astype(int), ig_vec):.4f}")

    order = np.argsort(-ig_vec)
    rel_sorted = relevance[order]
    ranks = np.arange(1, len(WT) + 1)
    apr = float(np.dot(rel_sorted, np.cumsum(rel_sorted) / ranks) / relevance.sum())
    print(f"  AP@R (graded)    = {apr:.4f}")

    pv = [ndcg_score(relevance.reshape(1, -1), abs_IG[i].reshape(1, -1)) for i in range(len(names))]
    print(f"  Per-variant NDCG = {np.mean(pv):.4f}   (per variant: "
          + ", ".join(f"{n}={v:.4f}" for n, v in zip(names, pv)) + ")")

    correct = total = 0
    for i, name in enumerate(names):
        thr = np.median(np.abs(IG[i]))
        for pos, _, _ in VARIANTS[name]:
            if ann_mask[pos]:
                total += 1
                correct += int(IG[i, pos] < -thr)
    print(f"  Sign accuracy    = {correct}/{total} = {correct / total:.4f}")

    cats = {"binding_site": [1], "structural": [2], "folding": [3], "background": [0, 4]}
    cat_means = {c: ig_vec_signed[idx].mean() for c, idx in cats.items()}
    top = max((c for c in cat_means if c != "background"), key=lambda c: cat_means[c])
    print(f"  Category means   = { {k: round(v, 5) for k, v in cat_means.items()} }")
    print(f"  Category sep     = {top}/background = "
          f"{cat_means[top] / cat_means['background']:.4f}")

    muts = VARIANTS["K2R_A4G"]
    loo_clipped = np.clip(np.array([loo["K2R"], loo["A4G"]]), 0, None)
    ig_at_muts = np.array([abs_IG[names.index("K2R_A4G")][pos] for pos, _, _ in muts])
    print(f"  Mutation NDCG    = "
          f"{ndcg_score(loo_clipped.reshape(1, -1), ig_at_muts.reshape(1, -1)):.4f}"
          f"   (LOO={loo_clipped}, |IG|={ig_at_muts.round(5)})")

    print()
    print("Caveat: five variants is far too small a sample to trust these values --")
    print("category separation in particular is a ratio with a near-zero denominator.")
    print("The point of this script is the mechanism, not the numbers.")


if __name__ == "__main__":
    main()
