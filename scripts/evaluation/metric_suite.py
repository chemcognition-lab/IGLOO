#!/usr/bin/env python3
"""
Interpretable attribution-quality metrics, computed for BOTH IG and ISM, scored
by how well each one separates low-performing (noisy-label) models from
high-performing ones.

Metrics per (position vector `a`, ground-truth importance `w`):
  spearman        Spearman ρ(|a|, w)                       rank agreement
  auroc           AUROC(annotated vs background, score=|a|) P(important > background); chance 0.5
  precision_at_k  of the top-k |a| positions, frac annotated (k = n_annotated); chance = prevalence
  recall_topcat   recall of the highest-weight category's positions within top-k
  enrichment      mean|a|(annotated) / mean|a|(background)  "important sites get X× more attr"; chance 1.0
  category_mono   Spearman(per-category mean |a|, category weight)  does |a| grow with importance tier
  skill_*         chance-normalised: (m - m_random)/(m_ideal - m_random)  0 = random, 1 = perfect

"Discrimination" per metric = Pearson r( metric , model_test_spearman ) across the
noise sweep — how strongly the metric tracks model quality.

Usage:
    python scripts/evaluation/metric_suite.py [--root .../results/synthetic] [--out .../metric_suite]
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.stats import spearmanr, pearsonr, rankdata
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    from scipy.stats import ConstantInputWarning
    warnings.filterwarnings("ignore", category=ConstantInputWarning)
except ImportError:
    pass

PROTEINS = ["GB1", "T4Lysozyme", "p53"]
LANDSCAPES = ["additive", "epistatic", "nonlinear"]
BACKBONE_ARCHS = ["PreMLP", "MeanPool", "AttnPool"]
BACKBONES = ["esmc_300m", "esmc_600m", "ankh_base", "ankh_large",
             "e1_150m", "e1_300m", "e1_600m"]
# metrics judged worth a full noise/model-performance breakdown for the
# backbone comparison (see metric-selection discussion): skill_spearman is
# the highest-discrimination, richest-signal metric; precision_at_k and
# recall_topcat are the next most consistently informative pair. auroc,
# enrichment, category_mono, epi_ratio were judged too weak/noisy or too
# landscape-specific to warrant the extra figures.
BACKBONE_METRICS = ["skill_spearman", "skill_precision_at_k", "skill_recall_topcat", "auroc", "sign_acc"]
_GT_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic"
_SP_RE = re.compile(r"Spearman\s*[ρr]\s*=\s*(-?\d+\.\d+)")
_GT: dict = {}


def gt(protein: str) -> dict:
    if protein not in _GT:
        j = json.loads((_GT_DIR / f"{protein}_ground_truth_weights.json").read_text())
        w = np.asarray(j["importance_vector"], dtype=float)
        ann_pos = j["annotated_positions"]                 # {cat: [1-indexed pos]}
        catw = j["category_weights"]
        bg = float(w.min())
        annotated = w > bg + 1e-9
        # "top category" mask: the highest-weight annotation tier(s).  Merge tiers
        # from the top down until we have >= 5 positions, so a tiny category
        # (e.g. T4 active_site has only 2 residues) doesn't quantise the metric.
        top_cat_mask = np.zeros(len(w), dtype=bool)
        cats_desc = sorted(catw, key=catw.get, reverse=True)
        top_cats = []
        for c in cats_desc:
            top_cats.append(c)
            for p in ann_pos.get(c, []):
                if 1 <= p <= len(w):
                    top_cat_mask[p - 1] = True
            if top_cat_mask.sum() >= 5:
                break
        top_cat = "+".join(top_cats)
        # positions that participate in >=1 contact pair with another annotated
        # position (for the position-level epistatic-recovery metric)
        in_contact = np.zeros(len(w), dtype=bool)
        try:
            import sys as _s
            root = str(Path(__file__).resolve().parents[2])
            if root not in _s.path:
                _s.path.insert(0, root)
            from src.igloo.synthetic_data import PROTEINS as _SP, _contact_weight as _cw
            pdef = _SP.get(protein)
            ann_idx = np.where(annotated)[0]
            for ii in range(len(ann_idx)):
                for jj in range(ii + 1, len(ann_idx)):
                    a_, b_ = int(ann_idx[ii]), int(ann_idx[jj])
                    if _cw(pdef, a_, b_) > 0:
                        in_contact[a_] = in_contact[b_] = True
        except Exception:
            in_contact = None
        _GT[protein] = dict(w=w, bg=bg, annotated=annotated, catw=catw,
                            ann_pos=ann_pos, top_cat=top_cat, top_cat_mask=top_cat_mask,
                            in_contact=in_contact)
    return _GT[protein]


def _model_perf(run_dir: Path) -> float:
    logs = sorted(run_dir.glob("*_run_log_*.txt"))
    if not logs:
        return np.nan
    m = _SP_RE.findall(logs[-1].read_text(errors="ignore"))
    return float(m[-1]) if m else np.nan


# ---- metric primitives -----------------------------------------------------
def _spearman(a, w):
    return spearmanr(np.abs(a), w)[0]


def _auroc(a, g):
    y = g["annotated"].astype(int)
    return roc_auc_score(y, np.abs(a)) if 0 < y.sum() < len(y) else np.nan


def _precision_at_k(a, g):
    k = int(g["annotated"].sum())
    top = np.argsort(-np.abs(a))[:k]
    return float(g["annotated"][top].mean())


def _recall_topcat(a, g):
    """Fraction of the top-category (most critical) positions that land in the
    top-k by |a|  (k = # annotated).  Denominator is >= 5 (categories merged in
    gt()), so this is no longer quantised to {0, 0.5, 1}."""
    m = g["top_cat_mask"]
    if m.sum() == 0:
        return np.nan
    k = int(g["annotated"].sum())
    top = set(np.argsort(-np.abs(a))[:k].tolist())
    return float(np.mean([i in top for i in np.where(m)[0]]))


def _topcat_rank(a, g):
    """Mean quantile of the top-category positions in the |a| ranking
    (1 = highest attribution, 0 = lowest).  Smooth, no quantisation.
    Random -> 0.5."""
    m = g["top_cat_mask"]
    if m.sum() == 0:
        return np.nan
    q = (rankdata(np.abs(a)) - 1) / (len(a) - 1)   # 0..1, 1 = biggest |a|
    return float(q[m].mean())


def _enrichment(a, g):
    aa = np.abs(a)
    imp = aa[g["annotated"]].mean()
    bgm = aa[~g["annotated"]].mean()
    return float(imp / bgm) if bgm > 0 else np.nan


def _category_mono(a, g):
    aa = np.abs(a)
    tiers, vals = [], []
    # background tier
    tiers.append(0.0); vals.append(aa[~g["annotated"]].mean())
    for cat, ws in g["catw"].items():
        idx = [p - 1 for p in g["ann_pos"].get(cat, []) if 1 <= p <= len(aa)]
        if idx:
            tiers.append(float(ws)); vals.append(aa[idx].mean())
    if len(tiers) < 3:
        return np.nan
    return spearmanr(tiers, vals)[0]


def _epi_ratio(a, g):
    """mean|a| at annotated positions in a contact pair  ÷  at annotated positions
    NOT in any contact pair.  >1 => attribution concentrates on interacting sites.
    Position-level analogue of evaluate_synthetic_ig's epi_delta_absig."""
    ic = g.get("in_contact")
    if ic is None:
        return np.nan
    aa = np.abs(a)
    inter = aa[g["annotated"] & ic]
    other = aa[g["annotated"] & ~ic]
    if len(inter) == 0 or len(other) == 0 or other.mean() == 0:
        return np.nan
    return float(inter.mean() / other.mean())


def _sign_acc(s, g):
    """fraction of annotated positions whose *signed* aggregate attribution is
    negative (i.e. the attribution says 'mutating this important site is
    deleterious').  Random signed vector -> 0.5."""
    m = g["annotated"]
    if m.sum() == 0:
        return np.nan
    return float((s[m] < 0).mean())


_METRICS = {
    "spearman": (_spearman, "rand->0", "ideal->1"),
    "auroc": (_auroc, 0.5, 1.0),
    "precision_at_k": (_precision_at_k, "prevalence", 1.0),
    "recall_topcat": (_recall_topcat, "prevalence", 1.0),
    "topcat_rank": (_topcat_rank, 0.5, 1.0),
    "enrichment": (_enrichment, 1.0, None),
    "category_mono": (_category_mono, 0.0, 1.0),
    "epi_ratio": (_epi_ratio, 1.0, None),
}
# ratio metrics — cap the plot y-axis so the (~1.0) null line stays visible
_RATIO_METRICS = {"enrichment", "epi_ratio", "skill_enrichment", "skill_epi_ratio"}
# metrics that operate on the SIGNED vector, handled separately (fixed random baseline)
_SIGNED_METRICS = {"sign_acc": (_sign_acc, 0.5, 1.0)}


_RAND: dict = {}


def _rand_means(protein: str, n_rand: int = 400) -> dict:
    if protein in _RAND:
        return _RAND[protein]
    g = gt(protein); w = g["w"]
    rng = np.random.default_rng(0)
    acc = {n: [] for n in _METRICS}
    for _ in range(n_rand):
        r = rng.random(len(w))
        for n2, (f2, _, _) in _METRICS.items():
            try:
                acc[n2].append(f2(r, w) if f2 is _spearman else f2(r, g))
            except Exception:
                acc[n2].append(np.nan)
    _RAND[protein] = {n: float(np.nanmean(v)) for n, v in acc.items()}
    return _RAND[protein]


def _null_for(protein: str, metric: str):
    """Chance level for any metric name, for the dotted baseline line."""
    if metric.startswith("skill_"):
        return 0.0                       # skill scores are chance-normalised
    if metric == "sign_acc":
        return 0.5                       # random signed vector
    m = _rand_means(protein).get(metric)
    return m if (m is not None and np.isfinite(m)) else None


def all_metrics(a: np.ndarray, protein: str, signed: np.ndarray | None = None) -> dict:
    g = gt(protein); w = g["w"]
    a = a[: len(w)]
    rand = _rand_means(protein)
    out = {}
    for name, (fn, lo, hi) in _METRICS.items():
        try:
            v = fn(a, w) if fn is _spearman else fn(a, g)
        except Exception:
            v = np.nan
        out[name] = v
        m_rand = rand[name]
        if name == "enrichment":
            m_ideal = _enrichment(w, g)
        elif name == "spearman":
            m_ideal = 1.0
        else:
            m_ideal = hi if isinstance(hi, (int, float)) else 1.0
        denom = (m_ideal - m_rand)
        out[f"skill_{name}"] = float((v - m_rand) / denom) if denom and np.isfinite(v) else np.nan

    # signed-vector metrics (fixed analytic random baseline)
    s = (signed[: len(w)] if signed is not None else a)
    for name, (fn, m_rand, m_ideal) in _SIGNED_METRICS.items():
        try:
            v = fn(s, g)
        except Exception:
            v = np.nan
        out[name] = v
        out[f"skill_{name}"] = (float((v - m_rand) / (m_ideal - m_rand))
                                if np.isfinite(v) else np.nan)
    return out


# ---- run discovery -------------------------------------------------------
# each yield: (protein, landscape, noise, seed, method, kind, mag, sgn, model_perf, backbone)
def ig_runs(root: Path):
    def _y(protein, landscape, noise, seed, kind, leaf, backbone="esmc_300m"):
        f = list(leaf.glob("*_position_importance_summary.csv"))
        if not f:
            return None
        d = pd.read_csv(f[0]).sort_values("position")
        mag = d["importance_score"].to_numpy()
        sgn = d["mean_gradient"].to_numpy() if "mean_gradient" in d else mag
        return (protein, landscape, noise, seed, "IG", kind, mag, sgn, _model_perf(leaf), backbone)

    for protein in PROTEINS:
        for landscape in LANDSCAPES:
            for nd in sorted((root / f"{protein}_{landscape}").glob("noise_*")):
                r = _y(protein, landscape, float(nd.name.split("_", 1)[1]), 0, "sweep", nd / "linear")
                if r:
                    yield r
            for sd in sorted((root / "seedsweep" / f"{protein}_{landscape}").glob("noise_*/seed_*/linear")):
                noise = float(sd.parents[1].name.split("_", 1)[1])
                seed = int(sd.parents[0].name.split("_", 1)[1])
                r = _y(protein, landscape, noise, seed, "sweep", sd)
                if r:
                    yield r
            for sd in sorted((root / "seedsweep" / f"{protein}_{landscape}").glob("shuffled/seed_*/linear")):
                seed = int(sd.parents[0].name.split("_", 1)[1])
                r = _y(protein, landscape, np.nan, seed, "shuffled", sd)
                if r:
                    yield r
            # architecture test: per-residue MLP before the pool
            for sd in sorted((root / "premlp" / f"{protein}_{landscape}").glob("noise_*/seed_*/linear")):
                noise = float(sd.parents[1].name.split("_", 1)[1])
                seed = int(sd.parents[0].name.split("_", 1)[1])
                r = _y(protein, landscape, noise, seed, "sweep", sd)
                if r:
                    yield (r[0], r[1], r[2], r[3], "PreMLP", r[5], r[6], r[7], r[8], r[9])
            # item 1: one-hot amino-acid input instead of ESM-C embedding (classic
            #   esm_attention head; tests how much the pLM geometry itself matters)
            # items 2/3: alternative pooling/attention architectures, pilot scale
            #   (noise in {0,1} x 3 seeds — see jobs/architectures/run_newarch_train_array.sh)
            _newarch_dirs = {
                "onehot": "OneHot",
                "newarch_perres_mean": "MeanPool",
                "newarch_perres_attn": "AttnPool",
                "newarch_crossattn": "CrossAttn",
                "newarch_perres_flatten": "FlattenPool",
                "mutation_pool": "MutationPool",
            }
            for dirname, method_name in _newarch_dirs.items():
                for sd in sorted((root / dirname / f"{protein}_{landscape}").glob("noise_*/seed_*/linear")):
                    noise = float(sd.parents[1].name.split("_", 1)[1])
                    seed = int(sd.parents[0].name.split("_", 1)[1])
                    r = _y(protein, landscape, noise, seed, "sweep", sd)
                    if r:
                        yield (r[0], r[1], r[2], r[3], method_name, r[5], r[6], r[7], r[8], r[9])
            # pLM backbone comparison: same 3 architectures re-run across 6
            # alternative protein language model backbones (see
            # jobs/backbones/run_plm_backbone_train_array.sh). Directory convention:
            # results/synthetic/<raw_arch>_<backbone>/<protein>_<landscape>/
            #   noise_*/seed_*/linear -- raw_arch is the model.pooling_method
            # config value, mapped to the same display names used above so
            # these rows line up with the esmc_300m-backbone baseline runs.
            _backbone_arch_dirs = {
                "esm_perres_attn": "AttnPool",
                "esm_attention_premlp": "PreMLP",
                "esm_perres_mean": "MeanPool",
            }
            for raw_arch, method_name in _backbone_arch_dirs.items():
                for backbone in BACKBONES:
                    if backbone == "esmc_300m":
                        continue  # that's the baseline run already yielded above
                    for sd in sorted((root / f"{raw_arch}_{backbone}" / f"{protein}_{landscape}")
                                     .glob("noise_*/seed_*/linear")):
                        noise = float(sd.parents[1].name.split("_", 1)[1])
                        seed = int(sd.parents[0].name.split("_", 1)[1])
                        r = _y(protein, landscape, noise, seed, "sweep", sd, backbone=backbone)
                        if r:
                            yield (r[0], r[1], r[2], r[3], method_name, r[5], r[6], r[7], r[8], r[9])
            # factor-ablation baselines (isolated attribution 'factors'):
            #   displacement  -> input-difference baseline (||x1 - x0||, NO model)
            #   gradient      -> gradient-magnitude baseline (||mean_k grad_k||)
            #   grad_input    -> gradient x input   (grad(F,x1) . x1, endpoint)
            _fa_map = {"displacement": "InputDiff", "gradient": "GradMag",
                       "grad_input": "GradInput"}
            for meth_dir, meth_name in _fa_map.items():
                for leaf in sorted((root / "factor_ablation" / f"{protein}_{landscape}")
                                   .glob(f"noise_*/{meth_dir}")):
                    noise = float(leaf.parents[0].name.split("_", 1)[1])
                    r = _y(protein, landscape, noise, 0, "sweep", leaf)
                    if r:
                        yield (r[0], r[1], r[2], r[3], meth_name, r[5], r[6], r[7], r[8], r[9])


def ism_runs(root: Path):
    for f in sorted((root / "ism_attribution").glob("ism_position_*.csv")):
        d = pd.read_csv(f).sort_values("position")
        sfile = f.parent / f.name.replace("ism_position_", "ism_summary_")
        mp = np.nan
        if sfile.exists():
            mp = pd.read_csv(sfile)["model_test_spearman"].iloc[0]
        sgn = d["ism_signed_mean"].to_numpy() if "ism_signed_mean" in d else d["ism_mean_abs"].to_numpy()
        mag = d["ism_mean_abs"].to_numpy()
        ms = re.match(r"ism_position_([A-Za-z0-9]+)_([a-z]+)__shuffled__seed_(\d+)", f.stem)
        if ms:
            yield (ms.group(1), ms.group(2), np.nan, int(ms.group(3)), "ISM", "shuffled", mag, sgn, mp, "esmc_300m")
            continue
        m = re.match(r"ism_position_([A-Za-z0-9]+)_([a-z]+)__noise_([0-9.]+)(?:__seed_(\d+))?", f.stem)
        if not m:
            continue
        yield (m.group(1), m.group(2), float(m.group(3)),
               int(m.group(4)) if m.group(4) else 0, "ISM", "sweep", mag, sgn, mp, "esmc_300m")


def plot_metric_vs_perf_band(df, m, methods_present, mk, cc, cmap, norm,
                             output_path, dpi=150):
    """The same per-noise-level mean points as plot_metric_vs_perf_summary,
    but connected directly into a line (in x order) instead of shown as
    discrete error-bar crosses -- the line passes exactly through every
    mean point (no separate least-squares fit), with a continuous
    uncertainty band built into that line itself:

      - the y-band is a ribbon (fill_between) hugging the mean-point line,
        with half-width at each node equal to that noise level's y std
        across seeds.
      - x spread is a thin horizontal whisker at each node (sitting ON the
        line, since the line passes through the exact mean point there)
        spanning +-1 std of model_spearman.

    Pearson r is still reported in the legend as a summary statistic, but
    the drawn line is the real mean-point trajectory, not that fit.
    """
    combos = [(p, l) for p in PROTEINS for l in LANDSCAPES
              if not df[(df.protein == p) & (df.landscape == l)].empty]
    ncol = 3
    nrow = int(np.ceil(len(combos) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.8 * ncol, 4.6 * nrow),
                             squeeze=False, layout="constrained")

    for ax, (p, l) in zip(axes.flat, combos):
        g_pl = df[(df.protein == p) & (df.landscape == l)]

        rm = _null_for(p, m)
        if rm is not None:
            lab = "chance" if (m.startswith("skill_") or m == "sign_acc") else "random ranking"
            ax.axhline(rm, color="#888", ls=":", lw=1.2, zorder=0)
            ax.text(0.99, rm, f" {lab}", color="#666", fontsize=7,
                    ha="right", va="bottom", transform=ax.get_yaxis_transform())

        if rm is None and g_pl[m].dropna().empty:
            ax.text(0.5, 0.5, "metric undefined for this landscape\n(see docs/attribution_methods_and_metrics.md)",
                    ha="center", va="center", fontsize=8, color="#999", style="italic",
                    transform=ax.transAxes, wrap=True)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{p} — {l}", fontsize=10, fontweight="bold")
            continue

        star_lo = None  # leftmost extent of any shuffled-label star's x error bar
        for method in methods_present:
            sw = g_pl[(g_pl.method == method) & (g_pl.kind == "sweep")]
            sw = sw.dropna(subset=["model_spearman", m])
            color = cc.get(method, "#7f7f7f")
            marker = mk.get(method, "o")
            nodes = []  # (noise, xm, ym, xstd, ystd)
            if not sw.empty:
                for noise, grp in sw.groupby("noise"):
                    if grp.empty:
                        continue
                    xm, ym = grp["model_spearman"].mean(), grp[m].mean()
                    xs_, ys_ = grp["model_spearman"].std(), grp[m].std()
                    nodes.append((noise, xm, ym, xs_ if len(grp) > 1 else 0.0,
                                 ys_ if len(grp) > 1 else 0.0))
                if len(nodes) >= 2:
                    mx = np.array([n[1] for n in nodes])
                    my = np.array([n[2] for n in nodes])
                    order = np.argsort(mx)
                    xs_sorted = mx[order]
                    line_y = my[order]                      # tracks the real means exactly
                    ystd_sorted = np.array([n[4] for n in nodes])[order]
                    xstd_sorted = np.array([n[3] for n in nodes])[order]

                    # y-band: continuous ribbon built into the mean-point line
                    ax.fill_between(xs_sorted, line_y - ystd_sorted, line_y + ystd_sorted,
                                    color=color, alpha=0.18, zorder=1, linewidth=0)
                    # x-spread: thin whisker at each node (on the line, since
                    # the line passes through the exact mean point here)
                    for xi, yi, xsi in zip(xs_sorted, line_y, xstd_sorted):
                        if xsi > 0:
                            ax.plot([xi - xsi, xi + xsi], [yi, yi], color=color,
                                   lw=1.1, alpha=0.55, zorder=2)
                    r_lab = (f"r={pearsonr(mx, my)[0]:+.2f}"
                            if mx.std() > 0 and my.std() > 0 else "flat")
                    ax.plot(xs_sorted, line_y, c=color, lw=2.0, zorder=3,
                            marker=marker, ms=5, mec="black", mew=0.4,
                            label=f"{method}  {r_lab}")
            sh = g_pl[(g_pl.method == method) & (g_pl.kind == "shuffled")].dropna(subset=[m])
            if not sh.empty:
                sx, sxstd = sh["model_spearman"].mean(), sh["model_spearman"].std()
                if pd.notna(sx):
                    left_edge = sx - (sxstd if pd.notna(sxstd) else 0.0)
                    star_lo = left_edge if star_lo is None else min(star_lo, left_edge)
                ax.errorbar(sx, sh[m].mean(),
                            xerr=sxstd, yerr=sh[m].std(),
                            fmt="*", ms=16, color=color, mec="black", mew=0.6,
                            capsize=3, zorder=4,
                            label=f"{method} shuffled-label model")

        if m in _RATIO_METRICS:
            vals = g_pl[m].replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals):
                hi = float(np.nanpercentile(vals, 95))
                lo = min(0.0, float(vals.min()))
                ax.set_ylim(lo, max(hi * 1.15, (rm or 1.0) * 1.3))
        if m == "topsis_score":
            xvals = g_pl["model_spearman"].replace([np.inf, -np.inf], np.nan).dropna()
            xhi = float(xvals.max()) if len(xvals) else 1.0
            xlo_candidates = [0.0]
            if len(xvals):
                xlo_candidates.append(float(xvals.min()))
            if star_lo is not None:
                xlo_candidates.append(float(star_lo))
            xlo = min(xlo_candidates)
            pad = 0.03
            ax.set_xlim(xlo - pad, min(1.02, max(xhi * 1.08, 0.1)))
            ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{p} — {l}", fontsize=12, fontweight="bold")
        ax.set_xlabel("model test Spearman ρ (predicted vs true fitness)", fontsize=10)
        ax.set_ylabel("TOPSIS" if m == "topsis_score" else m, fontsize=10)
        ax.tick_params(labelsize=10)
    for ax in axes.flat[len(combos):]:
        ax.set_visible(False)
    # one shared legend (method -> color/marker is identical across every
    # panel) instead of repeating it in each subplot; the per-panel Pearson r
    # that used to ride along in each panel's legend label is dropped since
    # it isn't constant across panels -- the discrimination table elsewhere
    # already reports it quantitatively. The noise colorbar is also dropped:
    # nothing in this plot is actually colored by noise (lines are colored by
    # method), it was inherited from the scatter-plot version and never used.
    method_handles = [plt.Line2D([0], [0], color=cc.get(meth, "#7f7f7f"), lw=2,
                                 marker=mk.get(meth, "o"), ms=6, mec="black", mew=0.4, label=meth)
                      for meth in methods_present]
    if method_handles:
        fig.legend(handles=method_handles, loc="lower center", ncol=len(method_handles),
                  fontsize=10, bbox_to_anchor=(0.5, -0.04))
    if m == "topsis_score":
        fig.suptitle("TOPSIS Score vs Model Performance", fontsize=13, fontweight="bold")
    else:
        fig.suptitle(f"{m}  vs model performance — line tracks the noise-level means exactly "
                     f"(y-ribbon + x-whiskers = ±1 std per noise level)",
                     fontsize=13, fontweight="bold")
    for ext in ("png", "svg"):
        fig.savefig(f"{output_path}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_metric_by_noise(df, m, methods_present, mk, cc, output_path, dpi=150):
    """Companion to the scatter-vs-model-performance panel for metric *m*:
    one box+strip panel per (protein, landscape), x=noise level, boxes
    grouped by method.

    A true violin plot (KDE) is misleading with only 3 seeds per
    (noise, method) cell -- it would draw a smooth density from 3 points.
    Box+strip (quartile box, individual seed points jittered on top) shows
    the same at-a-glance spread honestly, and is what this renders even
    though the CLI/README call it "the noise violin plots" -- decluttering
    the now-crowded scatter view was the actual goal, not the KDE shape.
    """
    combos = [(p, l) for p in PROTEINS for l in LANDSCAPES
              if not df[(df.protein == p) & (df.landscape == l)].empty]
    ncol = 3
    nrow = int(np.ceil(len(combos) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.5 * ncol, 4.8 * nrow),
                             squeeze=False, layout="constrained")

    noise_levels = sorted(df["noise"].dropna().unique())
    n_methods = len(methods_present)
    group_width = 0.82
    box_width = group_width / max(n_methods, 1)
    rng = np.random.default_rng(0)

    for ax, (p, l) in zip(axes.flat, combos):
        g_pl = df[(df.protein == p) & (df.landscape == l) & (df.kind == "sweep")]
        for mi, method in enumerate(methods_present):
            gm = g_pl[g_pl.method == method]
            positions, data = [], []
            for ni, noise in enumerate(noise_levels):
                vals = gm[gm.noise == noise][m].dropna().values
                if len(vals) == 0:
                    continue
                positions.append(ni + (mi - (n_methods - 1) / 2) * box_width)
                data.append(vals)
            if not data:
                continue
            color = cc.get(method, "#7f7f7f")
            ax.boxplot(
                data, positions=positions, widths=box_width * 0.85,
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor=color, alpha=0.45, edgecolor=color, linewidth=0.9),
                medianprops=dict(color="black", linewidth=1.1),
                whiskerprops=dict(color=color, linewidth=0.9),
                capprops=dict(color=color, linewidth=0.9),
            )
            for pos, vals in zip(positions, data):
                jitter = rng.uniform(-box_width * 0.18, box_width * 0.18, size=len(vals))
                ax.scatter(np.full(len(vals), pos) + jitter, vals, s=9, color=color,
                          edgecolors="black", linewidths=0.3, zorder=5, alpha=0.9)
        ax.set_xticks(range(len(noise_levels)))
        ax.set_xticklabels([f"{n:g}" for n in noise_levels], fontsize=8)
        ax.set_xlabel("label noise (std)")
        ax.set_ylabel(m)
        ax.set_title(f"{p} — {l}", fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        if m in _RATIO_METRICS:
            vals = g_pl[m].replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals):
                hi = float(np.nanpercentile(vals, 95))
                lo = min(0.0, float(vals.min()))
                ax.set_ylim(lo, hi * 1.15)
    for ax in axes.flat[len(combos):]:
        ax.set_visible(False)

    handles = [plt.Line2D([0], [0], marker="s", color="w",
                          markerfacecolor=cc.get(meth, "#7f7f7f"), markersize=9, label=meth)
               for meth in methods_present]
    fig.legend(handles=handles, loc="outside right upper", fontsize=7.5, ncol=1)
    fig.suptitle(f"{m}  by noise level  (box = quartiles across 3 seeds, dots = individual seeds)",
                 fontsize=11, fontweight="bold")
    for ext in ("png", "svg"):
        fig.savefig(f"{output_path}.{ext}", dpi=dpi)
    plt.close(fig)


def plot_metric_vs_perf_summary(df, m, methods_present, mk, cc, cmap, norm,
                                output_path, dpi=150):
    """Decluttered companion to the raw scatter-vs-model-performance panel:
    one mean point per (method, noise level) instead of one point per seed,
    with error bars in BOTH x (model_spearman) and y (the metric) showing
    the spread across that noise level's seeds. Same visual idiom already
    used for the shuffled-label-model star marker, just applied to every
    noise-level point instead of only the shuffled baseline.

    The trend line is fit through these noise-level means (fewer, less
    seed-noisy anchor points) rather than through every raw seed.
    """
    combos = [(p, l) for p in PROTEINS for l in LANDSCAPES
              if not df[(df.protein == p) & (df.landscape == l)].empty]
    ncol = 3
    nrow = int(np.ceil(len(combos) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.8 * ncol, 4.6 * nrow),
                             squeeze=False, layout="constrained")

    for ax, (p, l) in zip(axes.flat, combos):
        g_pl = df[(df.protein == p) & (df.landscape == l)]

        rm = _null_for(p, m)
        if rm is not None:
            lab = "chance" if (m.startswith("skill_") or m == "sign_acc") else "random ranking"
            ax.axhline(rm, color="#888", ls=":", lw=1.2, zorder=0)
            ax.text(0.99, rm, f" {lab}", color="#666", fontsize=7,
                    ha="right", va="bottom", transform=ax.get_yaxis_transform())

        if rm is None and g_pl[m].dropna().empty:
            ax.text(0.5, 0.5, "metric undefined for this landscape\n(see docs/attribution_methods_and_metrics.md)",
                    ha="center", va="center", fontsize=8, color="#999", style="italic",
                    transform=ax.transAxes, wrap=True)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{p} — {l}", fontsize=10, fontweight="bold")
            continue

        for method in methods_present:
            sw = g_pl[(g_pl.method == method) & (g_pl.kind == "sweep")]
            sw = sw.dropna(subset=["model_spearman", m])
            color = cc.get(method, "#7f7f7f")
            marker = mk.get(method, "o")
            means = []
            if not sw.empty:
                for noise, grp in sw.groupby("noise"):
                    if grp.empty:
                        continue
                    xm, ym = grp["model_spearman"].mean(), grp[m].mean()
                    xs_, ys_ = grp["model_spearman"].std(), grp[m].std()
                    ax.errorbar(xm, ym, xerr=xs_ if len(grp) > 1 else None,
                               yerr=ys_ if len(grp) > 1 else None,
                               fmt=marker, ms=7, color=cmap(norm(noise)),
                               mec="black", mew=0.4, ecolor=color, elinewidth=0.9,
                               capsize=2.5, zorder=3, alpha=0.9)
                    means.append((xm, ym))
                if len(means) >= 3:
                    mx = np.array([v[0] for v in means])
                    my = np.array([v[1] for v in means])
                    if mx.std() > 0 and my.std() > 0:
                        z = np.polyfit(mx, my, 1)
                        xs = np.linspace(mx.min(), mx.max(), 30)
                        ax.plot(xs, np.polyval(z, xs), c=color, lw=1.8, zorder=2,
                                label=f"{method}  r={pearsonr(mx, my)[0]:+.2f}")
                    else:
                        ax.plot([], [], c=color, lw=1.8, label=f"{method} (flat)")
            sh = g_pl[(g_pl.method == method) & (g_pl.kind == "shuffled")].dropna(subset=[m])
            if not sh.empty:
                ax.errorbar(sh["model_spearman"].mean(), sh[m].mean(),
                            xerr=sh["model_spearman"].std(), yerr=sh[m].std(),
                            fmt="*", ms=16, color=color, mec="black", mew=0.6,
                            capsize=3, zorder=4,
                            label=f"{method} shuffled-label model")

        if m in _RATIO_METRICS:
            vals = g_pl[m].replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals):
                hi = float(np.nanpercentile(vals, 95))
                lo = min(0.0, float(vals.min()))
                ax.set_ylim(lo, max(hi * 1.15, (rm or 1.0) * 1.3))
        ax.set_title(f"{p} — {l}", fontsize=10, fontweight="bold")
        ax.set_xlabel("model test Spearman ρ (predicted vs true fitness)")
        ax.set_ylabel(m)
        ax.legend(fontsize=7, loc="best")
    for ax in axes.flat[len(combos):]:
        ax.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.5, label="label noise (std)")
    fig.suptitle(f"{m}  vs model performance — mean ± spread across seeds per noise level",
                 fontsize=11, fontweight="bold")
    for ext in ("png", "svg"):
        fig.savefig(f"{output_path}.{ext}", dpi=dpi)
    plt.close(fig)


_BACKBONE_COLORS = {
    "esmc_300m": "#7f7f7f", "esmc_600m": "#d62728",
    "ankh_base": "#1f77b4", "ankh_large": "#2ca02c",
    "e1_150m": "#ff7f0e", "e1_300m": "#9467bd", "e1_600m": "#8c564b",
}
_BACKBONE_MARKERS = {
    "esmc_300m": "o", "esmc_600m": "D",
    "ankh_base": "s", "ankh_large": "^",
    "e1_150m": "v", "e1_300m": "P", "e1_600m": "X",
}


def plot_backbone_comparison(df_all, m, output_path, dpi=150):
    """Dedicated pLM-backbone comparison: one panel per architecture
    (perres_attn / IG_premlp / perres_mean), x = protein x landscape combo,
    y = metric *m* (mean ± std across noise levels and seeds for that
    combo), one line per backbone connecting the 6 combo points.

    This answers "which backbone wins, and is it consistent across
    datasets/landscapes" at a glance -- a backbone whose line stays high
    and flat across all 6 x-positions is the consistent winner; one that
    swings wildly is dataset-dependent. Unlike the other plots in this
    file (which are scoped to the esmc_300m baseline only), this is the
    one place df_all's full backbone dimension is used.
    """
    combos = [(p, l) for p in PROTEINS for l in LANDSCAPES]
    xlabels = [f"{p}\n{l}" for p, l in combos]
    xpos = np.arange(len(combos))

    fig, axes = plt.subplots(1, len(BACKBONE_ARCHS), figsize=(6.2 * len(BACKBONE_ARCHS), 5.2),
                             squeeze=False, layout="constrained", sharey=True)

    for ax, arch in zip(axes[0], BACKBONE_ARCHS):
        g_arch = df_all[(df_all.method == arch) & (df_all.kind == "sweep")]
        for backbone in BACKBONES:
            g_bb = g_arch[g_arch.backbone == backbone]
            ys, ystd, xs_ = [], [], []
            for xi, (p, l) in zip(xpos, combos):
                cell = g_bb[(g_bb.protein == p) & (g_bb.landscape == l)].dropna(subset=[m])
                if cell.empty:
                    continue
                xs_.append(xi)
                ys.append(cell[m].mean())
                ystd.append(cell[m].std() if len(cell) > 1 else 0.0)
            if not xs_:
                continue
            color = _BACKBONE_COLORS.get(backbone, "#333333")
            marker = _BACKBONE_MARKERS.get(backbone, "o")
            ls = "--" if backbone == "esmc_300m" else "-"
            ax.errorbar(xs_, ys, yerr=ystd, color=color, marker=marker, ls=ls,
                        lw=2.0 if backbone == "esmc_300m" else 1.6,
                        ms=7, mec="black", mew=0.4, capsize=3, alpha=0.9,
                        label=backbone, zorder=3 if backbone == "esmc_300m" else 2)
        ax.set_xticks(xpos); ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_title(arch, fontsize=11, fontweight="bold")
        ax.set_ylabel(m)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(f"{m}  by pLM backbone  (dashed grey = esmc_300m baseline; "
                 f"mean ± std across noise levels & seeds per combo)",
                 fontsize=11, fontweight="bold")
    for ext in ("png", "svg"):
        fig.savefig(f"{output_path}.{ext}", dpi=dpi)
    plt.close(fig)


def _arch_backbone_combo_table(df_all, metric_names):
    """Long-format table of one row per (arch, backbone, protein, landscape,
    metric) -- the metric value averaged over noise levels & seeds within
    that combo. This is the shared "cell" table behind backbone_ranking(),
    the arch x backbone heatmap, and the architecture-vs-backbone variance
    decomposition, so all three stay numerically consistent with each other.
    """
    rows = []
    sweep = df_all[df_all.kind == "sweep"]
    for (arch, backbone), g in sweep.groupby(["method", "backbone"]):
        if arch not in BACKBONE_ARCHS:
            continue
        for m in metric_names:
            per_combo = g.groupby(["protein", "landscape"])[m].mean().dropna()
            for (protein, landscape), val in per_combo.items():
                rows.append({"arch": arch, "backbone": backbone, "protein": protein,
                            "landscape": landscape, "metric": m, "value": val})
    return pd.DataFrame(rows)


def backbone_ranking(cell_table):
    """Rank (arch, backbone) pairs by mean and worst-case (min) score across
    all 6 protein x landscape combos, per metric -- the worst-case column is
    what "consistent across datasets/landscapes" means in practice: a
    backbone that's great on GB1 but poor on T4Lysozyme should not rank
    above one that's merely good everywhere.
    """
    g = cell_table.groupby(["arch", "backbone", "metric"])["value"]
    out = g.agg(mean="mean", worst_case="min", n_combos="count").reset_index()
    return out


def variance_decomposition(cell_table):
    """For each metric, decompose the variance of the per-combo cell values
    (one value per arch x backbone x protein x landscape) into how much is
    explained by architecture choice vs. backbone choice vs. everything else
    (interaction + protein/landscape-driven variation) -- a standard
    unreplicated two-way ANOVA sum-of-squares split, reported as eta-squared
    (share of total sum of squares), which directly answers "does the
    architecture or the pLM backbone matter more for attribution quality".

    Uses cell means (averaged over the 6 protein x landscape combos) for the
    arch/backbone main effects, same convention as backbone_ranking's "mean"
    column, so the two tables agree.
    """
    rows = []
    for m, g in cell_table.groupby("metric"):
        y = g["value"].to_numpy()
        if len(y) < 3:
            continue
        grand_mean = y.mean()
        ss_total = float(((y - grand_mean) ** 2).sum())
        if ss_total == 0:
            continue
        arch_means = g.groupby("arch")["value"].transform("mean")
        backbone_means = g.groupby("backbone")["value"].transform("mean")
        ss_arch = float(((arch_means - grand_mean) ** 2).sum())
        ss_backbone = float(((backbone_means - grand_mean) ** 2).sum())
        # unreplicated design: residual absorbs interaction + combo variation,
        # can't be split further without more replicates per cell
        ss_residual = ss_total - ss_arch - ss_backbone
        rows.append({"metric": m, "eta2_architecture": ss_arch / ss_total,
                    "eta2_backbone": ss_backbone / ss_total,
                    "eta2_residual_interaction": max(ss_residual, 0.0) / ss_total,
                    "n_cells": len(y)})
    return pd.DataFrame(rows)


_ARCH_DIRNAME = {"AttnPool": "esm_perres_attn", "PreMLP": "esm_attention_premlp",
                 "MeanPool": "esm_perres_mean"}
# Backbones under the clean {raw_arch}_{backbone}/... directory convention
# (excludes esmc_300m, whose baseline runs predate that convention and live
# under a different legacy layout -- newarch_perres_mean/newarch_perres_attn/
# premlp -- so it's left out of this attribution-agreement analysis rather
# than special-cased).
AGREEMENT_BACKBONES = [b for b in BACKBONES if b != "esmc_300m"]


def _position_vector(leaf_dir):
    """Load one run's per-position attribution vector (magnitude + signed
    gradient) from its position_importance_summary.csv, indexed by position.
    Returns None if the run has no such file (didn't complete / wrong path).
    """
    f = list(Path(leaf_dir).glob("*_position_importance_summary.csv"))
    if not f:
        return None
    d = pd.read_csv(f[0]).set_index("position")
    mag = d["importance_score"]
    sgn = d["mean_gradient"] if "mean_gradient" in d.columns else mag
    return mag, sgn


def pairwise_agreement(vectors: dict):
    """vectors: {label: (magnitude_series, signed_series)}, both indexed by
    position. Returns one row per unordered pair of labels with Pearson r and
    Spearman rho on magnitude, sign agreement on the signed gradient, and
    top-10 position overlap (fraction) -- the same four statistics used
    throughout this file's architecture-agreement checks, generalized into
    one reusable comparison so seed/architecture/backbone agreement are all
    computed identically and stay directly comparable.
    """
    labels = list(vectors.keys())
    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            l1, l2 = labels[i], labels[j]
            mag1, sgn1 = vectors[l1]
            mag2, sgn2 = vectors[l2]
            common = mag1.index.intersection(mag2.index)
            if len(common) < 4:
                continue
            m1, m2 = mag1.loc[common], mag2.loc[common]
            s1, s2 = sgn1.loc[common], sgn2.loc[common]
            pr = pearsonr(m1, m2)[0] if m1.std() > 0 and m2.std() > 0 else np.nan
            sr = spearmanr(m1, m2)[0] if m1.std() > 0 and m2.std() > 0 else np.nan
            sign_agree = float((np.sign(s1) == np.sign(s2)).mean())
            k = min(10, len(common))
            top1 = set(m1.nlargest(k).index)
            top2 = set(m2.nlargest(k).index)
            overlap = len(top1 & top2) / k
            rows.append({"pair": f"{l1}~{l2}", "pearson_r": pr, "spearman_rho": sr,
                        "sign_agreement": sign_agree, "top10_overlap": overlap,
                        "n_shared_positions": len(common)})
    return pd.DataFrame(rows)


def attribution_seed_agreement(root, best_cell_table, backbones=AGREEMENT_BACKBONES):
    """Within-model agreement (the noise floor): for each (arch, backbone,
    protein, landscape) cell, compares the raw per-position attribution
    vectors across the 3 random seeds at that cell's own best-noise
    operating point. Seeds differ only in random init / data order, nothing
    structural, so this is the best agreement architecture or backbone
    choice could ever be judged against.
    """
    rows = []
    cells = (best_cell_table[best_cell_table.backbone.isin(backbones)]
             [["arch", "backbone", "protein", "landscape", "best_noise"]].drop_duplicates())
    for _, row in cells.iterrows():
        dirname = _ARCH_DIRNAME[row.arch]
        base = root / f"{dirname}_{row.backbone}" / f"{row.protein}_{row.landscape}" / f"noise_{row.best_noise}"
        vectors = {}
        for seed in (1, 2, 3):
            v = _position_vector(base / f"seed_{seed}" / "linear")
            if v is not None:
                vectors[f"seed{seed}"] = v
        if len(vectors) < 2:
            continue
        agree = pairwise_agreement(vectors)
        if agree.empty:
            continue
        rows.append({"comparison": "seed", "arch": row.arch, "backbone": row.backbone,
                    "protein": row.protein, "landscape": row.landscape,
                    "mean_spearman": agree.spearman_rho.mean(),
                    "mean_pearson": agree.pearson_r.mean(),
                    "mean_sign_agreement": agree.sign_agreement.mean(),
                    "mean_top10_overlap": agree.top10_overlap.mean(),
                    "n_pairs": len(agree)})
    return pd.DataFrame(rows)


def attribution_cross_agreement(root, best_cell_table, axis, backbones=AGREEMENT_BACKBONES):
    """Across-model agreement: axis="arch" compares the 3 architectures
    (PreMLP/MeanPool/AttnPool) at a fixed backbone/protein/landscape;
    axis="backbone" compares backbones at a fixed architecture/protein/
    landscape. Each member is taken at ITS OWN best-noise operating point
    (which can differ from the others being compared -- that's the point:
    each is evaluated where it's genuinely best, not at a shared noise
    level). Averaged over the 3 available seeds for robustness, so this is
    directly comparable in spirit to attribution_seed_agreement's 3-seed
    average.
    """
    assert axis in ("arch", "backbone")
    fixed_cols = ["backbone", "protein", "landscape"] if axis == "arch" else ["arch", "protein", "landscape"]
    rows = []
    sub = best_cell_table[best_cell_table.backbone.isin(backbones)]
    for fixed_vals, g in sub.groupby(fixed_cols):
        members = g[[axis, "best_noise"]].drop_duplicates()
        if len(members) < 2:
            continue
        per_seed_agree = []
        for seed in (1, 2, 3):
            vectors = {}
            for _, m in members.iterrows():
                if axis == "arch":
                    arch, backbone, protein, landscape = m.arch, fixed_vals[0], fixed_vals[1], fixed_vals[2]
                else:
                    arch, backbone, protein, landscape = fixed_vals[0], m.backbone, fixed_vals[1], fixed_vals[2]
                dirname = _ARCH_DIRNAME[arch]
                leaf = (root / f"{dirname}_{backbone}" / f"{protein}_{landscape}"
                       / f"noise_{m.best_noise}" / f"seed_{seed}" / "linear")
                v = _position_vector(leaf)
                if v is not None:
                    vectors[str(m[axis])] = v
            if len(vectors) < 2:
                continue
            agree = pairwise_agreement(vectors)
            if not agree.empty:
                per_seed_agree.append(agree)
        if not per_seed_agree:
            continue
        allpairs = pd.concat(per_seed_agree, ignore_index=True)
        row = dict(zip(fixed_cols, fixed_vals))
        row.update({"comparison": axis,
                    "mean_spearman": allpairs.spearman_rho.mean(),
                    "mean_pearson": allpairs.pearson_r.mean(),
                    "mean_sign_agreement": allpairs.sign_agreement.mean(),
                    "mean_top10_overlap": allpairs.top10_overlap.mean(),
                    "n_pairs": len(allpairs)})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_attribution_agreement_box(agreement_df, output_path, dpi=150):
    """Box+strip plot (established idiom in this file) comparing the 3
    agreement levels -- seed (noise floor), architecture, backbone -- side
    by side per protein x landscape combo, on Spearman rho. Directly
    answers "is cross-model disagreement bigger than plain seed noise, or
    within it": if the arch/backbone boxes sit at or above the seed box,
    architecture/backbone choice barely matters beyond ordinary
    stochasticity; if they sit well below it, the choice genuinely changes
    what gets flagged as important.
    """
    if agreement_df.empty:
        return
    combos = [(p, l) for p in PROTEINS for l in LANDSCAPES
              if not agreement_df[(agreement_df.protein == p) & (agreement_df.landscape == l)].empty]
    ncol = 3
    nrow = int(np.ceil(len(combos) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 4.6 * nrow),
                             squeeze=False, layout="constrained")
    order = ["seed", "arch", "backbone"]
    labels = {"seed": "within-model\n(seed)", "arch": "across\narchitecture", "backbone": "across\nbackbone"}
    colors = {"seed": "#2ca02c", "arch": "#1f77b4", "backbone": "#d62728"}
    for ax, (p, l) in zip(axes.flat, combos):
        g_pl = agreement_df[(agreement_df.protein == p) & (agreement_df.landscape == l)]
        data, positions, box_colors = [], [], []
        for i, comp in enumerate(order):
            vals = g_pl[g_pl.comparison == comp]["mean_spearman"].dropna()
            if vals.empty:
                continue
            data.append(vals.to_numpy())
            positions.append(i)
            box_colors.append(_muted(colors[comp], factor=0.4))
        if not data:
            ax.set_visible(False)
            continue
        bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black", lw=1.5))
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c)
        rng = np.random.default_rng(0)
        for pos, vals, comp in zip(positions, data, [order[p] for p in positions]):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals, s=22, color=colors[comp],
                      edgecolor="black", linewidth=0.4, zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels([labels[order[p]] for p in positions], fontsize=9)
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color="#888", ls=":", lw=1)
        ax.set_ylabel("Spearman ρ (position-level attribution)", fontsize=10)
        ax.set_title(f"{p} — {l}", fontsize=12, fontweight="bold")
        ax.tick_params(labelsize=10)
    for ax in axes.flat[len(combos):]:
        ax.set_visible(False)
    fig.suptitle("Attribution agreement: within-model (seed) vs across-architecture vs across-backbone",
                 fontsize=14, fontweight="bold")
    for ext in ("png", "svg"):
        fig.savefig(f"{output_path}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _arch_backbone_best_cell_table_per_seed(df_all, metric_names):
    """Like _arch_backbone_best_cell_table, but keeps each seed's individual
    metric value instead of averaging over seeds -- needed to decompose
    variance into a seed component (see variance_decomposition_with_seed).
    Uses the same best-noise operating point (selected via the
    seed-averaged model_spearman, unchanged) as every other cell table in
    this file.
    """
    rows = []
    sweep = df_all[df_all.kind == "sweep"]
    for (arch, backbone, protein, landscape), g in sweep.groupby(
            ["method", "backbone", "protein", "landscape"]):
        if arch not in BACKBONE_ARCHS:
            continue
        by_noise = g.groupby("noise")["model_spearman"].mean().dropna()
        if by_noise.empty:
            continue
        best_noise = by_noise.idxmax()
        g_best = g[g.noise == best_noise]
        for m in metric_names:
            for seed, gs in g_best.groupby("seed"):
                val = gs[m].mean()  # mean over the (normally single) row for this seed
                if pd.notna(val):
                    rows.append({"arch": arch, "backbone": backbone, "protein": protein,
                                "landscape": landscape, "metric": m, "seed": seed, "value": val})
    return pd.DataFrame(rows)


def variance_decomposition_with_seed(per_seed_cell_table):
    """Same eta-squared variance decomposition as variance_decomposition,
    extended with a seed component: total variance (over every individual
    per-seed observation, not just cell means) splits into architecture +
    backbone + residual(protein/landscape/interaction) -- all computed on
    cell means, same as before -- plus seed (the within-cell spread across
    the 3 seeds, nested inside each arch x backbone x protein x landscape
    cell). This is the seed floor from attribution_seed_agreement, restated
    for the 5 shortlisted scalar metrics instead of the raw position
    vectors: how much of each metric's total variance is just seed noise,
    vs a real architecture or backbone effect.
    """
    rows = []
    for m, g in per_seed_cell_table.groupby("metric"):
        cell_means = g.groupby(["arch", "backbone", "protein", "landscape"])["value"].mean()
        n_cells = len(cell_means)
        n_seeds = g.groupby(["arch", "backbone", "protein", "landscape"])["value"].count()
        if n_cells < 3 or (n_seeds < 2).all():
            continue
        grand_mean = cell_means.mean()
        avg_n_seeds = n_seeds.mean()

        arch_means = cell_means.groupby("arch").transform("mean")
        backbone_means = cell_means.groupby("backbone").transform("mean")
        ss_arch = avg_n_seeds * float(((arch_means - grand_mean) ** 2).sum())
        ss_backbone = avg_n_seeds * float(((backbone_means - grand_mean) ** 2).sum())
        ss_between_cells = avg_n_seeds * float(((cell_means - grand_mean) ** 2).sum())
        ss_residual = max(ss_between_cells - ss_arch - ss_backbone, 0.0)

        cell_of = g.set_index(["arch", "backbone", "protein", "landscape"])
        ss_seed = 0.0
        for key, cm in cell_means.items():
            vals = cell_of.loc[[key], "value"] if key in cell_of.index else pd.Series(dtype=float)
            ss_seed += float(((vals - cm) ** 2).sum())

        ss_total = ss_arch + ss_backbone + ss_residual + ss_seed
        if ss_total == 0:
            continue
        rows.append({"metric": m, "eta2_architecture": ss_arch / ss_total,
                    "eta2_backbone": ss_backbone / ss_total,
                    "eta2_residual_interaction": ss_residual / ss_total,
                    "eta2_seed": ss_seed / ss_total, "n_cells": n_cells})
    return pd.DataFrame(rows)


def _arch_backbone_best_cell_table(df_all, metric_names):
    """Like _arch_backbone_combo_table, but instead of averaging over the
    whole noise sweep, picks -- per (arch, backbone, protein, landscape) --
    only the noise level with the highest MEAN model_spearman (averaged
    across seeds, to avoid picking a single lucky seed) and reports the
    metric's seed-mean value there. Answers "how good is attribution when
    the model is trained as well as it can be", for the heatmap.
    """
    rows = []
    sweep = df_all[df_all.kind == "sweep"]
    for (arch, backbone, protein, landscape), g in sweep.groupby(
            ["method", "backbone", "protein", "landscape"]):
        if arch not in BACKBONE_ARCHS:
            continue
        by_noise = g.groupby("noise")["model_spearman"].mean().dropna()
        if by_noise.empty:
            continue
        best_noise = by_noise.idxmax()
        g_best = g[g.noise == best_noise]
        for m in metric_names:
            val = g_best[m].mean()
            if pd.notna(val):
                rows.append({"arch": arch, "backbone": backbone, "protein": protein,
                            "landscape": landscape, "metric": m, "value": val,
                            "best_noise": best_noise,
                            "model_spearman": by_noise.loc[best_noise]})
    return pd.DataFrame(rows)


def plot_arch_backbone_heatmap(best_cell_table, m, output_path, dpi=150):
    """Heatmap of metric value across the arch x backbone grid (rows =
    backbone, cols = architecture), one panel per protein x landscape combo,
    using each cell's best-model-spearman operating point (see
    _arch_backbone_best_cell_table) rather than a noise-sweep average.

    Color scale spans the empirical data range (viridis, shared vmin/vmax
    across all 6 panels so the same color means the same performance level
    everywhere); the metric's chance/random baseline is marked as a
    reference line directly on the colorbar rather than forced to be the
    color floor, since it isn't always the worst value actually observed.
    Chance is protein-specific for non-skill metrics (skill_* metrics are
    chance=0 everywhere by construction) -- if GB1 and T4Lysozyme have
    different chance levels for this metric, both get their own labeled
    line.
    """
    g_all = best_cell_table[best_cell_table.metric == m]
    if g_all.empty:
        return
    combos = [(p, l) for p in PROTEINS for l in LANDSCAPES]
    backbones = [b for b in BACKBONES if b in g_all.backbone.values]
    archs = [a for a in BACKBONE_ARCHS if a in g_all.arch.values]
    vmin = float(g_all["value"].min())
    vmax = float(g_all["value"].max())

    ncol = 3
    nrow = int(np.ceil(len(combos) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=((1.35 * len(archs) + 1.5) * ncol, (0.48 * len(backbones) + 1.2) * nrow),
        squeeze=False, layout="constrained")

    im = None
    for ax, (p, l) in zip(axes.flat, combos):
        g = g_all[(g_all.protein == p) & (g_all.landscape == l)]
        grid = np.full((len(backbones), len(archs)), np.nan)
        for _, row in g.iterrows():
            i, j = backbones.index(row["backbone"]), archs.index(row["arch"])
            grid[i, j] = row["value"]
        im = ax.imshow(grid, cmap="inferno", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(archs))); ax.set_xticklabels(archs, fontsize=12, rotation=20, ha="right")
        ax.set_yticks(range(len(backbones))); ax.set_yticklabels(backbones, fontsize=12)
        for i in range(len(backbones)):
            for j in range(len(archs)):
                v = grid[i, j]
                if np.isnan(v):
                    continue
                frac = (v - vmin) / max(vmax - vmin, 1e-9)
                # viridis: dark (low) -> bright yellow (high), one contrast flip
                txt_color = "white" if frac < 0.6 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=12, color=txt_color)
        ax.set_title(f"{p} — {l}", fontsize=13, fontweight="bold")
    for ax in axes.flat[len(combos):]:
        ax.set_visible(False)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label=m)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(m, fontsize=13)
        chance_by_protein = {p: _null_for(p, m) for p in PROTEINS}
        chance_by_protein = {p: c for p, c in chance_by_protein.items()
                             if c is not None and vmin <= c <= vmax}
        # dedupe protein-invariant chance values (e.g. skill_*: 0, auroc: 0.5)
        # into one line instead of drawing overlapping duplicates
        seen = {}
        for p, c in chance_by_protein.items():
            seen.setdefault(round(c, 6), []).append(p)
        # Fold chance markers into the colorbar's own tick set (rather than
        # free-floating text past the axes bounds) so constrained_layout
        # actually reserves room for them instead of clipping them at the
        # figure edge.
        base_ticks = list(cbar.get_ticks())
        chance_ticks = sorted(seen.keys())
        all_ticks = sorted(set([t for t in base_ticks if vmin <= t <= vmax] + chance_ticks))
        labels = [(f"{t:.2f}\nchance ({'/'.join(seen[round(t,6)])})" if round(t, 6) in seen
                  else f"{t:.2f}") for t in all_ticks]
        cbar.set_ticks(all_ticks)
        cbar.set_ticklabels(labels)
        for tick_val, txt in zip(all_ticks, cbar.ax.get_yticklabels()):
            if round(tick_val, 6) in seen:
                txt.set_color("red"); txt.set_fontweight("bold"); txt.set_fontsize(11)
                cbar.ax.axhline(tick_val, color="red", lw=1.2, ls="--", zorder=5)
            else:
                txt.set_fontsize(12)
    fig.suptitle(f"{m}: architecture x pLM backbone at each combo's best-model "
                 f"operating point (highest mean model_spearman)",
                 fontsize=15, fontweight="bold")
    for ext in ("png", "svg"):
        fig.savefig(f"{output_path}.{ext}", dpi=dpi)
    plt.close(fig)


def plot_backbone_metric_bars(best_cell_table, metrics, output_path, dpi=150):
    """Bar chart with error bars, one panel per shortlisted metric (plus a
    "topsis" panel when the caller includes TOPSIS rows in best_cell_table):
    for a single fixed architecture (the caller filters best_cell_table to
    it, e.g. MeanPool), one bar per backbone -- height = mean value across
    the 6 protein/landscape combos, error bar = std across those combos, at
    each combo's best-model-spearman operating point (same convention as
    the arch x backbone heatmap and parallel-coordinates plot).
    Companion to the parallel-coordinates view of the same data: bars make
    "which backbone wins on metric X" and "how consistent is it" easy to
    read off directly, at the cost of not showing the per-metric tradeoffs
    a single thread in the parallel-coordinates plot shows.
    """
    if best_cell_table.empty:
        return
    ncol = 3
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.0 * ncol, 4.6 * nrow),
                             squeeze=False, layout="constrained")
    for ax, m in zip(axes.flat, metrics):
        g = best_cell_table[best_cell_table.metric == m]
        if g.empty:
            ax.set_visible(False)
            continue
        stats = g.groupby("backbone")["value"].agg(["mean", "std"]).reindex(
            [b for b in BACKBONES if b in g.backbone.values])
        colors = [_muted(_BACKBONE_COLORS.get(b, "#7f7f7f"), factor=0.45) for b in stats.index]
        ax.bar(stats.index, stats["mean"], yerr=stats["std"].fillna(0.0), color=colors,
              capsize=4, edgecolor="black", linewidth=0.6)
        label = "TOPSIS" if m == "topsis" else m
        if m != "topsis":
            # TOPSIS is a relative closeness score with no fixed chance
            # level, so the reference line only applies to the raw metrics.
            chance_vals = sorted({round(c, 6) for c in
                                  (_null_for(p, m) for p in PROTEINS) if c is not None})
            for c in chance_vals:
                ax.axhline(c, color="#888", ls=":", lw=1.3, zorder=0)
                ax.text(0.99, c, " chance", color="#666", fontsize=8,
                        ha="right", va="bottom", transform=ax.get_yaxis_transform())
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", labelsize=10, rotation=25)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(True, axis="y", alpha=0.25)
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    fig.suptitle("pLM backbone comparison: mean ± std across the 6 protein/landscape "
                 "combos, per metric, at each combo's best-model operating point",
                 fontsize=15, fontweight="bold")
    for ext in ("png", "svg"):
        fig.savefig(f"{output_path}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


_ARCH_LINESTYLES = {"PreMLP": "--", "MeanPool": ":", "AttnPool": "-"}


def plot_parallel_coordinates(best_cell_table, metrics, output_path, dpi=150,
                              per_panel_range=False, show_backbone=True):
    """Parallel-coordinates view, one subplot per protein x landscape combo:
    two categorical axes (architecture, backbone) followed by one
    continuous axis per shortlisted metric, using that combo's own
    best-model-operating-point value (see _arch_backbone_best_cell_table) --
    no averaging across combos, so real per-dataset tradeoffs stay visible
    instead of being smoothed away. Each (architecture, backbone)
    configuration draws one thread; color = backbone (consistent with every
    other backbone figure in this file), linestyle = architecture (forced
    solid when only one architecture is present, since style no longer
    distinguishes anything), with one shared legend for the whole figure.

    By default each metric axis uses ONE range shared across all 6 panels,
    so a given y-position means the same value everywhere. Set
    per_panel_range=True to instead scale each panel to its own local
    min/max (extended if needed so the chance/random reference line always
    stays visible, even when every thread happens to sit above or below
    it) -- this stretches out the separation between threads within a
    panel (useful with fewer threads, e.g. a single-architecture view) at
    the cost of y-position no longer being comparable across panels (the
    printed min/max labels on each axis still give the real numbers).

    show_backbone=False drops the backbone categorical column too (on top
    of architecture already being dropped whenever only one is present) --
    for a single-architecture view backbone identity is already fully
    conveyed by line color via the legend, so the column is redundant.
    """
    g_all = best_cell_table[best_cell_table.metric.isin(metrics)]
    if g_all.empty:
        return
    combos = [(p, l) for p in PROTEINS for l in LANDSCAPES]

    archs = [a for a in BACKBONE_ARCHS if a in g_all.arch.values]
    backbones = [b for b in BACKBONES if b in g_all.backbone.values]
    # when the caller has already filtered to a single architecture, that
    # column is constant (every thread at the same position) and adds
    # nothing -- drop it and start straight from backbone (or, with
    # show_backbone=False, straight into the metrics).
    show_arch = len(archs) > 1
    show_backbone = show_backbone or show_arch  # backbone can't be hidden while arch pairs with it
    cat_labels = (["architecture"] if show_arch else []) + (["backbone"] if show_backbone else [])
    axes_labels = cat_labels + metrics
    arch_pos = {a: (i / max(len(archs) - 1, 1)) for i, a in enumerate(archs)}
    bb_pos = {b: (i / max(len(backbones) - 1, 1)) for i, b in enumerate(backbones)}

    # global per-metric range, shared across all 6 panels unless
    # per_panel_range overrides it with each panel's own local min/max below.
    # Extended (widened, never narrowed) so the chance line is always within
    # range across BOTH proteins -- otherwise a metric where every observed
    # value sits well above chance would silently never show the reference.
    global_ranges = {}
    for m in metrics:
        lo = float(g_all[g_all.metric == m]["value"].min())
        hi = float(g_all[g_all.metric == m]["value"].max())
        for pp in PROTEINS:
            c = _null_for(pp, m)
            if c is not None:
                lo, hi = min(lo, c), max(hi, c)
        global_ranges[m] = (lo, hi)

    # x positions: architecture/backbone are categorical (not measurements),
    # so they sit on their own column(s) with a visible gap before the
    # continuous metric columns -- and, below, the connecting line is drawn
    # in two separate segments so it never bridges that gap, only tracing
    # (a) which architecture pairs with which backbone and (b) how that
    # config moves across the metrics. Color/linestyle (explained once in
    # the shared legend) is what actually identifies a thread across the
    # gap, not line continuity. With no categorical columns at all, the
    # metrics simply start at x=0 and there's nothing to divide.
    n_cat = len(cat_labels)
    if n_cat == 0:
        x_arch = x_bb = None
        x_metrics = list(range(len(metrics)))
        xs_cat = []
        divider_x = None
    else:
        x_arch = 0.0 if show_arch else None
        x_bb = float(n_cat - 1) if show_backbone else None
        gap = 0.9
        x_metrics = [x_bb + gap + i for i in range(len(metrics))]
        xs_cat = ([x_arch] if show_arch else []) + ([x_bb] if show_backbone else [])
        divider_x = (x_bb + x_metrics[0]) / 2
    xs_metric = x_metrics
    all_x = xs_cat + xs_metric

    ncol = 3
    nrow = int(np.ceil(len(combos) / ncol))
    n_axes = len(axes_labels)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * n_axes * ncol / 2.2, 4.8 * nrow),
                             squeeze=False, layout="constrained")
    label_bbox = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0)

    for col, (ax, (p, l)) in enumerate(zip(axes.flat, combos)):
        wide = (g_all[(g_all.protein == p) & (g_all.landscape == l)]
                .pivot(index=["arch", "backbone"], columns="metric", values="value")
                .reset_index())
        # a combo with zero data for this metric set (e.g. a protein whose
        # backbone-comparison run hasn't completed yet) pivots to a frame
        # missing some/all metric columns entirely, not just NaN values --
        # add them back as NaN so dropna() below empties the frame instead
        # of raising a KeyError, leaving this panel blank rather than
        # crashing the whole figure.
        for m in metrics:
            if m not in wide.columns:
                wide[m] = np.nan
        wide = wide.dropna(subset=metrics)
        if per_panel_range:
            ranges = {}
            for m in metrics:
                lo, hi = float(wide[m].min()), float(wide[m].max())
                c = _null_for(p, m)
                if c is not None:
                    lo, hi = min(lo, c), max(hi, c)
                ranges[m] = (lo, hi)
        else:
            ranges = global_ranges
        for _, row in wide.iterrows():
            color = _BACKBONE_COLORS.get(row["backbone"], "#333333")
            ls = _ARCH_LINESTYLES.get(row["arch"], "-") if show_arch else "-"
            ys_cat = ([arch_pos[row["arch"]]] if show_arch else []) + \
                     ([bb_pos[row["backbone"]]] if show_backbone else [])
            ys_metric = []
            for m in metrics:
                lo, hi = ranges[m]
                ys_metric.append((row[m] - lo) / max(hi - lo, 1e-9))
            if xs_cat:
                ax.plot(xs_cat, ys_cat, color=color, ls=ls, lw=1.6, alpha=0.8,
                        marker="o", ms=3.5, zorder=3)
            ax.plot(xs_metric, ys_metric, color=color, ls=ls, lw=1.6, alpha=0.8,
                    marker="o", ms=3.5, zorder=3)
        for xi in all_x:
            ax.axvline(xi, color="#cccccc", lw=1, zorder=0)
        if divider_x is not None:
            ax.axvline(divider_x, color="#888888", lw=1.4, ls=(0, (4, 2)), zorder=1)
        ax.set_xticks(all_x); ax.set_xticklabels(axes_labels, fontsize=11, rotation=20, ha="right")
        ax.set_yticks([])
        ax.set_xlim(all_x[0] - 0.35, x_metrics[-1] + 0.35)
        # categorical labels only on the leftmost column to cut clutter
        if col % ncol == 0:
            if show_arch:
                for a, yp in arch_pos.items():
                    ax.text(x_arch - 0.05, yp, a, fontsize=10, ha="right", va="center",
                            zorder=5, bbox=label_bbox)
            if show_backbone:
                for b, yp in bb_pos.items():
                    ax.text(x_bb + 0.05, yp, b, fontsize=10, ha="left", va="center",
                            zorder=5, bbox=label_bbox)
        for i, m in enumerate(metrics):
            lo, hi = ranges[m]
            xi = x_metrics[i]
            for frac, val in ((0.0, lo), (1.0, hi)):
                ax.text(xi + 0.06, frac, f"{val:.2f}", fontsize=9, va="center", color="#444444",
                        zorder=5, bbox=label_bbox)
            # this panel's own protein -> a single well-defined chance level
            chance = _null_for(p, m)
            if chance is not None and lo <= chance <= hi:
                cfrac = (chance - lo) / max(hi - lo, 1e-9)
                ax.plot([xi - 0.14, xi + 0.14], [cfrac, cfrac], color="red", lw=1.6,
                        ls="--", zorder=4)
                # nudge the label off the exact min/max tick when chance sits
                # right on the panel edge (common now that the range is
                # extended to always include it) so it doesn't collide with
                # that edge's numeric value label
                label_y = cfrac + 0.06 if cfrac < 0.08 else (cfrac - 0.06 if cfrac > 0.92 else cfrac)
                va = "bottom" if label_y >= cfrac else "top"
                ax.text(xi, label_y, "chance", fontsize=8, color="red", ha="center",
                        va=va, zorder=5, bbox=dict(facecolor="white", edgecolor="none",
                                                   alpha=0.85, pad=0.5))
        ax.set_title(f"{p} — {l}", fontsize=12, fontweight="bold")
    for ax in axes.flat[len(combos):]:
        ax.set_visible(False)

    backbone_handles = [plt.Line2D([0], [0], color=_BACKBONE_COLORS.get(b, "#333"), lw=2, label=b)
                        for b in backbones]
    fig.legend(handles=backbone_handles, title="backbone (color)", fontsize=10,
              title_fontsize=11, loc="lower center", ncol=len(backbones),
              bbox_to_anchor=(0.5, -0.03 if show_arch else -0.08))
    if show_arch:
        arch_handles = [plt.Line2D([0], [0], color="black", lw=1.8, ls=_ARCH_LINESTYLES.get(a, "-"), label=a)
                        for a in archs]
        fig.legend(handles=arch_handles, title="architecture (style)", fontsize=10,
                  title_fontsize=11, loc="lower center", ncol=len(archs),
                  bbox_to_anchor=(0.5, -0.09))
    title = ("Architecture x pLM backbone across metrics, per protein/landscape combo"
            if show_arch else f"pLM backbone across metrics ({archs[0]} only), per protein/landscape combo")
    range_note = "each panel scaled to its own min/max" if per_panel_range else "shared scale across panels"
    fig.suptitle(f"{title}\n(best-model operating point; each thread = one "
                 f"{'architecture/backbone' if show_arch else 'backbone'} configuration; {range_note})",
                 fontsize=14, fontweight="bold")
    for ext in ("png", "svg"):
        fig.savefig(f"{output_path}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _muted(hex_color, factor=0.45):
    """Soften a color by blending it toward white -- factor=0 is the
    original color, factor=1 is white. Used where a plot's own dedicated
    palette should be gentler than the saturated per-method colors (cc)
    used everywhere else in this file for cross-figure consistency."""
    r, g, b = matplotlib.colors.to_rgb(hex_color)
    return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)


def plot_model_perf_noise_variance_box(df, methods_present, mk, cc, output_path, dpi=150):
    """Box+strip plot (matching the idiom used in plot_metric_by_noise): for
    each architecture/method, the distribution -- across the 6 protein/
    landscape combos -- of std(mean model_spearman across the 9 noise
    levels). Shows the real spread (median, IQR, individual combo points)
    instead of collapsing it to a single mean ± error bar.
    """
    combos = [(p, l) for p in PROTEINS for l in LANDSCAPES
              if not df[(df.protein == p) & (df.landscape == l)].empty]
    rows = []
    for method in methods_present:
        for p, l in combos:
            sw = df[(df.protein == p) & (df.landscape == l) & (df.method == method)
                    & (df.kind == "sweep")].dropna(subset=["model_spearman"])
            if sw.empty:
                continue
            noise_means = sw.groupby("noise")["model_spearman"].mean()
            if len(noise_means) >= 2:
                rows.append({"method": method, "combo": f"{p} — {l}",
                            "combo_std": float(noise_means.std())})
    if not rows:
        return
    rdf = pd.DataFrame(rows)
    order = (rdf.groupby("method")["combo_std"].median()
            .sort_values(ascending=False).index.tolist())

    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(order)), 5.5), layout="constrained")
    rng = np.random.default_rng(0)
    for i, method in enumerate(order):
        vals = rdf[rdf.method == method]["combo_std"].values
        color = _muted(cc.get(method, "#7f7f7f"), factor=0.4)
        ax.boxplot(
            [vals], positions=[i], widths=0.6,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=color, edgecolor="#555555", linewidth=0.9),
            medianprops=dict(color="#333333", linewidth=1.3),
            whiskerprops=dict(color="#555555", linewidth=0.9),
            capprops=dict(color="#555555", linewidth=0.9),
        )
        jitter = rng.uniform(-0.14, 0.14, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=22, color=color,
                  edgecolors="#333333", linewidths=0.5, zorder=5, alpha=0.95)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=11, rotation=25)
    ax.set_ylabel("std(model_spearman) across the 9 noise levels\n"
                  "(one point per protein/landscape combo)", fontsize=11)
    ax.set_title("Model-performance sensitivity to label noise, by architecture",
                fontsize=13, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    for ext in ("png", "svg"):
        fig.savefig(f"{output_path}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


LAYER_SWEEP_LAYERS = [0, 12, 24, 36, 48]
_COMBO_COLORS = {
    ("GB1", "additive"): "#1f77b4", ("GB1", "epistatic"): "#ff7f0e",
    ("GB1", "nonlinear"): "#2ca02c", ("T4Lysozyme", "additive"): "#d62728",
    ("T4Lysozyme", "epistatic"): "#9467bd", ("T4Lysozyme", "nonlinear"): "#8c564b",
    ("p53", "additive"): "#17becf", ("p53", "epistatic"): "#bcbd22",
    ("p53", "nonlinear"): "#e377c2",
}


def layer_sweep_best_cell_table(root, metric_names):
    """Best-model-operating-point cell table for the ankh_base/MeanPool
    intermediate-layer sweep (results/synthetic/meanpool_ankh_base_layersweep/
    L<layer>/<protein>_<landscape>/noise_*/seed_*/linear) -- same convention
    as _arch_backbone_best_cell_table: per (layer, protein, landscape), picks
    the noise level with the highest mean model_spearman across seeds and
    reports each metric's seed-mean value there.
    """
    base = root / "meanpool_ankh_base_layersweep"
    rows = []
    for L in LAYER_SWEEP_LAYERS:
        for protein in PROTEINS:
            for landscape in LANDSCAPES:
                combo_dir = base / f"L{L}" / f"{protein}_{landscape}"
                if not combo_dir.is_dir():
                    continue
                recs = []
                for nd in sorted(combo_dir.glob("noise_*")):
                    noise = float(nd.name.split("_", 1)[1])
                    for sd in sorted(nd.glob("seed_*")):
                        seed = int(sd.name.split("_", 1)[1])
                        leaf = sd / "linear"
                        f = list(leaf.glob("*_position_importance_summary.csv"))
                        if not f:
                            continue
                        d = pd.read_csv(f[0]).sort_values("position")
                        mag = d["importance_score"].to_numpy()
                        sgn = d["mean_gradient"].to_numpy() if "mean_gradient" in d else mag
                        mp = _model_perf(leaf)
                        recs.append((noise, seed, mag, sgn, mp))
                if not recs:
                    continue
                by_noise = {}
                for noise, seed, mag, sgn, mp in recs:
                    by_noise.setdefault(noise, []).append(mp)
                noise_means = {n: np.nanmean(v) for n, v in by_noise.items()}
                noise_means = {n: v for n, v in noise_means.items() if np.isfinite(v)}
                if not noise_means:
                    continue
                best_noise = max(noise_means, key=noise_means.get)
                sel = [(seed, mag, sgn, mp) for noise, seed, mag, sgn, mp in recs if noise == best_noise]
                for m in metric_names:
                    vals = []
                    for seed, mag, sgn, mp in sel:
                        met = all_metrics(mag, protein, signed=sgn)
                        if m in met and np.isfinite(met[m]):
                            vals.append(met[m])
                    if vals:
                        rows.append({"layer": L, "protein": protein, "landscape": landscape,
                                    "metric": m, "value": float(np.mean(vals)),
                                    "std": float(np.std(vals)) if len(vals) > 1 else 0.0,
                                    "n_seeds": len(vals), "best_noise": best_noise,
                                    "model_spearman": float(noise_means[best_noise])})
                # model_spearman as its own pseudo-metric row too, so it can
                # be plotted alongside the attribution-quality metrics with
                # the same renderer -- this is the predictive-performance
                # companion panel: does the model's raw accuracy also dip in
                # the middle layers, or does only the attribution quality
                # break down while the model still predicts fine?
                mp_vals = [mp for _, _, _, mp in sel if np.isfinite(mp)]
                if mp_vals:
                    rows.append({"layer": L, "protein": protein, "landscape": landscape,
                                "metric": "model_spearman", "value": float(np.mean(mp_vals)),
                                "std": float(np.std(mp_vals)) if len(mp_vals) > 1 else 0.0,
                                "n_seeds": len(mp_vals), "best_noise": best_noise,
                                "model_spearman": float(noise_means[best_noise])})
    return pd.DataFrame(rows)


def _topsis_fill_worst(sub):
    """Impute a missing metric value with that metric's worst (min)
    observed value within the group, rather than dropping the whole
    alternative outright. An alternative where one criterion is genuinely
    undefined (e.g. a Spearman correlation that's undefined because
    attribution collapsed to a near-constant vector -- exactly what
    happens at the collapsed intermediate layers) is real, worst-case
    signal, not missing data that should silently disqualify the
    alternative from the ranking. Rows where every metric is NaN are
    still dropped, since there's nothing to rank there.
    """
    sub = sub.dropna(how="all")
    if sub.empty:
        return sub
    return sub.fillna(sub.min())


def topsis_score(df, metrics, group_cols=("protein", "landscape")):
    """TOPSIS closeness coefficient (Technique for Order Preference by
    Similarity to Ideal Solution), computed per group -- default: within
    each protein x landscape combo, not pooled globally, for the same
    reason skill_precision_at_k exists: pooling runs from GB1 and
    T4Lysozyme before normalizing would let one protein's scale dominate
    (the exact ceiling-effect distortion already found and fixed for raw
    precision_at_k).

    All shortlisted metrics are treated as benefit criteria (higher =
    better) with equal weights. For each row (run): vector-normalize each
    criterion within its group, find the ideal-best and ideal-worst point
    (max/min per criterion within the group), then score
    C = dist_to_worst / (dist_to_best + dist_to_worst) -- 0 = worst
    alternative in the group, 1 = best. Unlike a naive mean, this accounts
    for each criterion's actual spread rather than assuming they're all on
    comparable scales, and rewards being simultaneously far from the worst
    case and close to the best, not just numerically high on average.
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, idx in df.groupby(list(group_cols)).groups.items():
        sub = _topsis_fill_worst(df.loc[idx, list(metrics)])
        if len(sub) < 2:
            continue
        X = sub.to_numpy(dtype=float)
        norm = np.sqrt((X ** 2).sum(axis=0))
        norm[norm == 0] = 1.0
        R = X / norm  # vector-normalized decision matrix (equal weights)
        ideal_best = R.max(axis=0)
        ideal_worst = R.min(axis=0)
        s_plus = np.sqrt(((R - ideal_best) ** 2).sum(axis=1))
        s_minus = np.sqrt(((R - ideal_worst) ** 2).sum(axis=1))
        denom = s_plus + s_minus
        denom[denom == 0] = 1.0
        out.loc[sub.index] = s_minus / denom
    return out


def arch_backbone_topsis_cells(best_cell_table, metrics, group_cols=("protein", "landscape")):
    """Same TOPSIS procedure as topsis_score(), but for the long-format
    arch x backbone cell table (columns: arch, backbone, protein,
    landscape, metric, value) used by the heatmap/parallel-coordinates
    renderers, instead of the per-run df. Pivots each protein x landscape
    group to wide (one row per (arch, backbone) alternative, one column per
    metric) before applying the same normalize -> ideal-point -> distance
    procedure. Returns a long-format frame with metric="topsis", value=the
    closeness coefficient, so it can be fed into plot_arch_backbone_heatmap
    or plot_parallel_coordinates exactly like any other metric.
    """
    rows = []
    for (p, l), g in best_cell_table.groupby(list(group_cols)):
        wide = g.pivot_table(index=["arch", "backbone"], columns="metric", values="value")
        cols = [m for m in metrics if m in wide.columns]
        sub = _topsis_fill_worst(wide[cols])
        if len(sub) < 2:
            continue
        X = sub.to_numpy(dtype=float)
        norm = np.sqrt((X ** 2).sum(axis=0))
        norm[norm == 0] = 1.0
        R = X / norm
        ideal_best = R.max(axis=0)
        ideal_worst = R.min(axis=0)
        s_plus = np.sqrt(((R - ideal_best) ** 2).sum(axis=1))
        s_minus = np.sqrt(((R - ideal_worst) ** 2).sum(axis=1))
        denom = s_plus + s_minus
        denom[denom == 0] = 1.0
        c = s_minus / denom
        for (arch, backbone), score in zip(sub.index, c):
            rows.append({"arch": arch, "backbone": backbone, "protein": p,
                        "landscape": l, "metric": "topsis", "value": float(score)})
    return pd.DataFrame(rows)


def plot_layer_sweep_lines(cell_table, metrics, output_path, dpi=150):
    """One panel per shortlisted metric, x = layer depth (0=embedding output
    .. 48=final layer for ankh_base), y = mean value (across the 3 seeds at
    that combo's best-model operating point) with a ±1 seed-std ribbon band,
    one line per protein/landscape combo -- directly shows whether
    attribution quality concentrates at a particular depth or is roughly
    flat across the network, and how much of any apparent change is within
    seed noise.
    """
    if cell_table.empty:
        return
    combos = [(p, l) for p in PROTEINS for l in LANDSCAPES]
    ncol = 3
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.8 * nrow),
                             squeeze=False, layout="constrained")
    for ax, m in zip(axes.flat, metrics):
        g_all = cell_table[cell_table.metric == m]
        for (p, l) in combos:
            g = g_all[(g_all.protein == p) & (g_all.landscape == l)].sort_values("layer")
            if g.empty:
                continue
            color = _COMBO_COLORS.get((p, l), "#333333")
            std = g["std"] if "std" in g.columns else pd.Series(0.0, index=g.index)
            ax.fill_between(g["layer"], g["value"] - std, g["value"] + std,
                            color=color, alpha=0.15, zorder=1, linewidth=0)
            ax.plot(g["layer"], g["value"], color=color, marker="o", ms=6,
                    mec="black", mew=0.4, lw=2, zorder=3, label=f"{p} — {l}")
        chance_vals = sorted(set(round(c, 6) for c in
                                 (_null_for(p, m) for p in PROTEINS) if c is not None))
        for c in chance_vals:
            ax.axhline(c, color="red", ls="--", lw=1.2, alpha=0.7, zorder=0)
        ax.set_xticks(LAYER_SWEEP_LAYERS)
        ax.set_xlabel("ankh_base layer (0=embedding, 48=final)", fontsize=10)
        ax.set_ylabel(m, fontsize=11)
        ax.set_title(m, fontsize=13, fontweight="bold")
        ax.tick_params(labelsize=10)
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    handles = [plt.Line2D([0], [0], color=_COMBO_COLORS[(p, l)], marker="o", lw=2,
                          mec="black", mew=0.4, label=f"{p} — {l}")
              for p in PROTEINS for l in LANDSCAPES]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=10,
              bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("ankh_base / MeanPool: attribution quality across intermediate layers "
                 "(best-model operating point; red dashed = chance)",
                 fontsize=15, fontweight="bold")
    for ext in ("png", "svg"):
        fig.savefig(f"{output_path}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def layer_sweep_topsis_cells(cell_table, metrics, group_cols=("protein", "landscape")):
    """Same TOPSIS procedure as arch_backbone_topsis_cells(), but the
    alternatives being ranked are layers (within each protein x landscape
    group) instead of (arch, backbone) pairs. Returns a long-format frame
    with metric="topsis", one row per (layer, protein, landscape).
    """
    rows = []
    for (p, l), g in cell_table.groupby(list(group_cols)):
        wide = g.pivot_table(index="layer", columns="metric", values="value")
        cols = [m for m in metrics if m in wide.columns]
        sub = _topsis_fill_worst(wide[cols])
        if len(sub) < 2:
            continue
        X = sub.to_numpy(dtype=float)
        norm = np.sqrt((X ** 2).sum(axis=0))
        norm[norm == 0] = 1.0
        R = X / norm
        ideal_best = R.max(axis=0)
        ideal_worst = R.min(axis=0)
        s_plus = np.sqrt(((R - ideal_best) ** 2).sum(axis=1))
        s_minus = np.sqrt(((R - ideal_worst) ** 2).sum(axis=1))
        denom = s_plus + s_minus
        denom[denom == 0] = 1.0
        c = s_minus / denom
        for layer, score in zip(sub.index, c):
            rows.append({"layer": layer, "protein": p, "landscape": l,
                        "metric": "topsis", "value": float(score)})
    return pd.DataFrame(rows)


def plot_layer_sweep_topsis_vs_perf(combined, output_path, dpi=150):
    """TOPSIS closeness score (5 shortlisted metrics combined into one
    attribution-quality score per layer) plotted together with raw model
    test Spearman performance, on the same 0-1 axis, one panel per protein
    x landscape combo -- puts the U-shaped attribution-quality dip and the
    model's much gentler predictive-performance trend on one plot instead
    of two separate panels, since both series are already 0-1-bounded and
    directly comparable.
    """
    if combined.empty:
        return
    combos = [(p, l) for p in PROTEINS for l in LANDSCAPES]
    ncol = 3
    nrow = int(np.ceil(len(combos) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.8 * nrow),
                             squeeze=False, layout="constrained")
    line_style = {"topsis": ("#2ca02c", "TOPSIS score (attribution quality)"),
                 "model_spearman": ("#1f77b4", "model test Spearman ρ")}
    for ax, (p, l) in zip(axes.flat, combos):
        g_pl = combined[(combined.protein == p) & (combined.landscape == l)]
        if g_pl.empty:
            ax.set_visible(False)
            continue
        for metric_key, (color, label) in line_style.items():
            g = g_pl[g_pl.metric == metric_key].sort_values("layer")
            if g.empty:
                continue
            std = g["std"].fillna(0.0) if "std" in g.columns else pd.Series(0.0, index=g.index)
            ax.fill_between(g["layer"], g["value"] - std, g["value"] + std,
                            color=color, alpha=0.15, zorder=1, linewidth=0)
            ax.plot(g["layer"], g["value"], color=color, marker="o", ms=6,
                    mec="black", mew=0.4, lw=2, zorder=3, label=label)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(LAYER_SWEEP_LAYERS)
        ax.set_xlabel("ankh_base layer (0=embedding, 48=final)", fontsize=10)
        ax.set_ylabel("score", fontsize=11)
        ax.set_title(f"{p} — {l}", fontsize=13, fontweight="bold")
        ax.tick_params(labelsize=10)
    for ax in axes.flat[len(combos):]:
        ax.set_visible(False)
    handles = [plt.Line2D([0], [0], color=c, lw=2, marker="o", ms=6,
                          mec="black", mew=0.4, label=lab)
              for c, lab in line_style.values()]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=11,
              bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("TOPSIS Score vs Model Performance across ankh_base layers",
                 fontsize=15, fontweight="bold")
    for ext in ("png", "svg"):
        fig.savefig(f"{output_path}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def perf_attribution_correlation_table(df, metric, methods):
    """Pearson r between model test Spearman performance and an
    attribution-quality metric (e.g. topsis_score), one row per method --
    computed separately within each protein x landscape combo (same
    per-combo discrimination convention as metric_discrimination.csv, so
    one combo's sample size/scale can't dominate a pooled number) then
    summarized as the mean +/- std of r across the up-to-6 combos actually
    available for that method.
    """
    rows = []
    sweep = df[df.kind == "sweep"]
    for method in methods:
        g_m = sweep[sweep.method == method]
        rs = []
        for (p, l), g in g_m.groupby(["protein", "landscape"]):
            s = g.dropna(subset=["model_spearman", metric])
            if (len(s) >= 4 and s["model_spearman"].nunique() >= 3
                    and s[metric].nunique() > 2):
                r, _ = pearsonr(s["model_spearman"], s[metric])
                rs.append(r)
        if rs:
            rows.append({"method": method, "mean_r": float(np.mean(rs)),
                        "std_r": float(np.std(rs)), "n_combos": len(rs),
                        "min_r": float(np.min(rs)), "max_r": float(np.max(rs))})
        else:
            rows.append({"method": method, "mean_r": np.nan, "std_r": np.nan,
                        "n_combos": 0, "min_r": np.nan, "max_r": np.nan})
    return pd.DataFrame(rows)


def write_perf_correlation_latex(table, output_path, metric_label="TOPSIS score"):
    """Renders perf_attribution_correlation_table()'s output as a
    booktabs-style LaTeX table, sorted by mean r descending, one row per
    architecture/baseline, mean +/- std (across protein/landscape combos)
    per cell.
    """
    ordered = table.sort_values("mean_r", ascending=False, na_position="last")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Pearson $r$ (mean $\pm$ std) & Range & \# combos \\",
        r"\midrule",
    ]
    for _, row in ordered.iterrows():
        if pd.isna(row["mean_r"]):
            r_cell, range_cell = "--", "--"
        else:
            r_cell = f"{row['mean_r']:+.2f} $\\pm$ {row['std_r']:.2f}"
            range_cell = f"[{row['min_r']:+.2f}, {row['max_r']:+.2f}]"
        lines.append(f"{row['method']} & {r_cell} & {range_cell} & {int(row['n_combos'])} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Pearson correlation between model test Spearman $\rho$ and attribution "
        f"quality ({metric_label}), per architecture/baseline. Computed separately within "
        r"each protein $\times$ landscape combo across the noise sweep, then averaged "
        r"(mean $\pm$ std across combos); range gives the min/max per-combo $r$. InputDiff "
        r"is a model-independent baseline (embedding L2 distance to WT) included as a "
        r"negative control.}",
        r"\label{tab:perf_attribution_correlation}",
        r"\end{table}",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def main():
    project_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    default=Path("/project/aip-bensl/ranaab/IGLOO/results/synthetic"))
    ap.add_argument("--out", type=Path,
                    default=project_root / "results" / "synthetic" / "metric_suite")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for gen in (ig_runs, ism_runs):
        for protein, landscape, noise, seed, method, kind, vec, sgn, mperf, backbone in gen(args.root):
            rows.append({"protein": protein, "landscape": landscape, "noise": noise,
                         "seed": seed, "method": method, "kind": kind,
                         "backbone": backbone, "model_spearman": mperf,
                         **all_metrics(vec, protein, signed=sgn)})
    df_all = pd.DataFrame(rows)
    if df_all.empty:
        print("no runs found"); return
    df_all.to_csv(args.out / "metric_suite.csv", index=False)
    n_bb = (df_all.backbone != "esmc_300m").sum()
    print(f"{len(df_all)} runs total ({n_bb} from non-baseline pLM backbones); "
          f"esmc_300m-baseline breakdown: "
          f"{df_all[df_all.backbone == 'esmc_300m'].groupby(['method', 'kind']).size().to_dict()}")

    # The pLM-backbone-comparison rows (method in BACKBONE_ARCHS, backbone !=
    # esmc_300m) share method names with the esmc_300m baseline runs by
    # design (same architecture, different backbone). Every plot/table below
    # this point except the dedicated backbone-comparison figure is scoped
    # to backbone == esmc_300m so those baseline comparisons aren't silently
    # diluted by the new backbone sweep.
    df = df_all[df_all.backbone == "esmc_300m"]

    all_m = list(_METRICS) + list(_SIGNED_METRICS)
    metric_names = all_m + [f"skill_{m}" for m in all_m]
    sweep = df[df.kind == "sweep"]

    # discrimination table: |Pearson r(metric, model_spearman)| per method x landscape
    disc = []
    for (method, landscape), g in sweep.groupby(["method", "landscape"]):
        for m in metric_names:
            s = g.dropna(subset=["model_spearman", m])
            # need real variation in BOTH axes; a near-constant metric (e.g. the
            # model-independent input_diff baseline) is reported as flat, not
            # spuriously correlated with the tiny variant-selection jitter.
            if (len(s) >= 4 and s[m].nunique() > 2
                    and (s[m].max() - s[m].min()) > 0.01
                    and s["model_spearman"].nunique() >= 3):
                r, p = pearsonr(s["model_spearman"], s[m])
                disc.append({"method": method, "landscape": landscape, "metric": m,
                             "r_vs_modelperf": r, "p": p, "n": len(s)})
    disc = pd.DataFrame(disc)
    disc.to_csv(args.out / "metric_discrimination.csv", index=False)
    print("\n=== discrimination: mean |r(metric, model perf)| across landscapes ===")
    summ = (disc.assign(absr=disc["r_vs_modelperf"].abs())
                .groupby(["method", "metric"])["absr"].mean().unstack("method")
                .sort_values("ISM" if "ISM" in disc["method"].values else "IG", ascending=False))
    print(summ.round(3).to_string())

    # ---- plots: metric vs model performance, noise-coloured, with baselines ----
    combos = [(p, l) for p in PROTEINS for l in LANDSCAPES
              if not df[(df.protein == p) & (df.landscape == l)].empty]
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(0.0, 1.0)
    MK = {"IG": "o", "ISM": "s", "PreMLP": "D",
          "GradMag": "^", "GradInput": "P",
          "OneHot": "X", "MeanPool": "h",
          "AttnPool": "H", "CrossAttn": "8", "FlattenPool": "p",
          "MutationPool": "d"}
    CC = {"IG": "#d62728", "ISM": "#1f77b4", "PreMLP": "#2ca02c",
          "GradMag": "#8c564b", "GradInput": "#e377c2",
          "OneHot": "#7f7f7f", "MeanPool": "#17becf",
          "AttnPool": "#ff7f0e", "CrossAttn": "#393b79", "FlattenPool": "#8c9e2f",
          "MutationPool": "#000000"}
    # InputDiff (embedding L2 distance to WT) is model-independent -- it
    # can't track model performance by construction, so it's excluded from
    # the comparison plots (still computed/available in the raw CSVs).
    PLOT_METHODS = ["IG", "ISM", "PreMLP", "GradMag", "GradInput",
                     "OneHot", "MeanPool", "AttnPool", "CrossAttn",
                     "FlattenPool", "MutationPool"]

    for m in all_m:
        ncol = 3
        nrow = int(np.ceil(len(combos) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.8 * ncol, 4.6 * nrow),
                                 squeeze=False, layout="constrained")
        for ax, (p, l) in zip(axes.flat, combos):
            g_pl = df[(df.protein == p) & (df.landscape == l)]

            # random / chance baseline line
            rm = _null_for(p, m)
            if rm is not None:
                lab = "chance" if (m.startswith("skill_") or m == "sign_acc") else "random ranking"
                ax.axhline(rm, color="#888", ls=":", lw=1.2, zorder=0)
                ax.text(0.99, rm, f" {lab}", color="#666", fontsize=7,
                        ha="right", va="bottom", transform=ax.get_yaxis_transform())

            # metric is structurally undefined for this protein (e.g. epi_ratio
            # on GB1: every annotated position lies within 10 residues of another
            # annotated one, so there is no "non-contact annotated" reference
            # group — see docs/attribution_methods_and_metrics.md). Say so
            # instead of leaving a silent blank panel.
            if rm is None and g_pl[m].dropna().empty:
                ax.text(0.5, 0.5, "metric undefined for this landscape\n(see docs/attribution_methods_and_metrics.md)",
                        ha="center", va="center", fontsize=8, color="#999", style="italic",
                        transform=ax.transAxes, wrap=True)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{p} — {l}", fontsize=10, fontweight="bold")
                continue

            for method in [x for x in PLOT_METHODS if x in g_pl.method.values]:
                sw = g_pl[(g_pl.method == method) & (g_pl.kind == "sweep")]
                sw = sw.dropna(subset=["model_spearman", m])
                if not sw.empty:
                    ax.scatter(sw["model_spearman"], sw[m], c=sw["noise"], cmap=cmap,
                               norm=norm, marker=MK[method], s=42, alpha=0.85,
                               edgecolors="black", linewidths=0.4, zorder=3)
                    if (len(sw) >= 4 and sw["model_spearman"].nunique() >= 2
                            and sw[m].nunique() >= 2):
                        z = np.polyfit(sw["model_spearman"], sw[m], 1)
                        xs = np.linspace(sw["model_spearman"].min(), sw["model_spearman"].max(), 30)
                        ax.plot(xs, np.polyval(z, xs), c=CC[method], lw=1.8, zorder=2,
                                label=f"{method}  r={pearsonr(sw['model_spearman'], sw[m])[0]:+.2f}")
                    elif not sw.empty:
                        ax.plot([], [], c=CC[method], lw=1.8, label=f"{method} (flat)")
                # shuffled-label-model baseline (near x=0)
                sh = g_pl[(g_pl.method == method) & (g_pl.kind == "shuffled")].dropna(subset=[m])
                if not sh.empty:
                    cc = CC[method]
                    ax.errorbar(sh["model_spearman"].mean(), sh[m].mean(),
                                xerr=sh["model_spearman"].std(), yerr=sh[m].std(),
                                fmt="*", ms=16, color=cc, mec="black", mew=0.6,
                                capsize=3, zorder=4,
                                label=f"{method} shuffled-label model")

            # ratio metrics: cap y so a few big outliers don't hide the null line
            if m in _RATIO_METRICS:
                vals = g_pl[m].replace([np.inf, -np.inf], np.nan).dropna()
                if len(vals):
                    hi = float(np.nanpercentile(vals, 95))
                    lo = min(0.0, float(vals.min()))
                    ax.set_ylim(lo, max(hi * 1.15, (rm or 1.0) * 1.3))
            ax.set_title(f"{p} — {l}", fontsize=10, fontweight="bold")
            ax.set_xlabel("model test Spearman ρ (predicted vs true fitness)")
            ax.set_ylabel(m)
            ax.legend(fontsize=7, loc="best")
        for ax in axes.flat[len(combos):]:
            ax.set_visible(False)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
        fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.5,
                     label="label noise (std)")
        fig.suptitle(f"{m}  vs model performance   "
                     f"(●IG ■ISM ◆PreMLP ▲GradMag ✚GradInput;  "
                     f"★=shuffled-label model, ⋯=random-ranking null)",
                     fontsize=11, fontweight="bold")
        for ext in ("png", "svg"):
            fig.savefig(args.out / f"metric_{m}.{ext}", dpi=150)
        plt.close(fig)

        methods_present = [x for x in PLOT_METHODS if x in df.method.values]
        plot_metric_by_noise(df, m, methods_present, MK, CC, args.out / f"metric_{m}_by_noise")
        plot_metric_vs_perf_summary(df, m, methods_present, MK, CC, cmap, norm,
                                    args.out / f"metric_{m}_summary")
        plot_metric_vs_perf_band(df, m, methods_present, MK, CC, cmap, norm,
                                 args.out / f"metric_{m}_band")
        if not df_all[df_all.backbone != "esmc_300m"].empty:
            plot_backbone_comparison(df_all, m, args.out / f"metric_{m}_backbone")

    # model-performance sensitivity to noise, by architecture (companion to
    # the _band plots: how much does model_spearman itself move across the
    # noise sweep for each method, not the attribution quality)
    methods_present = [x for x in PLOT_METHODS if x in df.method.values]
    plot_model_perf_noise_variance_box(df, methods_present, MK, CC,
                                       args.out / "model_perf_noise_variance")

    # TOPSIS closeness score over the shortlisted metrics (skill_spearman,
    # skill_precision_at_k, skill_recall_topcat, auroc, sign_acc) vs model
    # performance, reusing the same band-plot renderer used for every
    # individual metric above (it only needs a metric column to exist in
    # df; topsis_score is just another one)
    composite_metrics = [mm for mm in BACKBONE_METRICS if mm in df.columns]
    if composite_metrics:
        df["topsis_score"] = topsis_score(df, composite_metrics)
        plot_metric_vs_perf_band(df, "topsis_score", methods_present, MK, CC,
                                 cmap, norm, args.out / "metric_topsis_score_band")

        # results table: model performance vs TOPSIS attribution quality,
        # per architecture/baseline (every method actually run, including
        # InputDiff -- excluded from the comparison plots since it's
        # model-independent and flat, but included here as a negative
        # control since it's supposed to show ~0 correlation)
        corr_methods = list(PLOT_METHODS)
        if "InputDiff" in df.method.values and "InputDiff" not in corr_methods:
            corr_methods.append("InputDiff")
        corr_table = perf_attribution_correlation_table(df, "topsis_score", corr_methods)
        corr_table.to_csv(args.out / "topsis_perf_correlation.csv", index=False)
        write_perf_correlation_latex(corr_table, args.out / "topsis_perf_correlation.tex")
        print("\n=== model test Spearman vs TOPSIS attribution quality: "
              "Pearson r per method (mean +/- std across protein/landscape combos) ===")
        print(corr_table.sort_values("mean_r", ascending=False).round(3).to_string(index=False))

    # by-noise / vs-model-performance breakdown, per architecture, for the
    # shortlisted backbone metrics (see metric-selection discussion:
    # skill_spearman, precision_at_k, recall_topcat). This is a SEPARATE
    # loop over BACKBONE_METRICS directly, not nested in `for m in all_m`
    # above, because skill_* columns exist in the dataframe but are never
    # themselves a value of `m` there (all_m holds only the raw metric
    # names -- skill_X is a derived column of the same row). Reuses the
    # exact same plot_metric_by_noise / _summary / _band renderers as the
    # rest of this file by relabeling the backbone column as "method" for
    # a single architecture at a time (each already groups/colors by
    # whatever's in df.method, so this needs no new plotting code), plus
    # the arch-faceted overview for skill_spearman specifically since that
    # never ran in the all_m loop either.
    if not df_all[df_all.backbone != "esmc_300m"].empty:
        for m in BACKBONE_METRICS:
            if m not in df_all.columns:
                continue
            if m not in all_m:  # skill_* etc. -- overview didn't run above
                plot_backbone_comparison(df_all, m, args.out / f"metric_{m}_backbone")
            for arch in BACKBONE_ARCHS:
                df_bb = df_all[df_all.method == arch].copy()
                df_bb["method"] = df_bb["backbone"]
                backbones_present = [b for b in BACKBONES if b in df_bb.method.values]
                if len(backbones_present) < 2:
                    continue
                tag = f"metric_{m}_backbone_{arch}"
                plot_metric_by_noise(df_bb, m, backbones_present, _BACKBONE_MARKERS,
                                     _BACKBONE_COLORS, args.out / f"{tag}_by_noise")
                plot_metric_vs_perf_summary(df_bb, m, backbones_present, _BACKBONE_MARKERS,
                                            _BACKBONE_COLORS, cmap, norm,
                                            args.out / f"{tag}_summary")
                plot_metric_vs_perf_band(df_bb, m, backbones_present, _BACKBONE_MARKERS,
                                         _BACKBONE_COLORS, cmap, norm,
                                         args.out / f"{tag}_band")

    if not df_all[df_all.backbone != "esmc_300m"].empty:
        cell_table = _arch_backbone_combo_table(df_all, metric_names)
        cell_table.to_csv(args.out / "backbone_combo_cells.csv", index=False)

        rank = backbone_ranking(cell_table)
        rank.to_csv(args.out / "backbone_ranking.csv", index=False)
        print("\n=== backbone ranking: worst-case (min) skill_spearman across the 6 "
              "protein x landscape combos, per architecture ===")
        if "skill_spearman" in rank["metric"].values:
            piv = (rank[rank.metric == "skill_spearman"]
                   .pivot(index="backbone", columns="arch", values="worst_case")
                   .sort_values(BACKBONE_ARCHS[0], ascending=False))
            print(piv.round(3).to_string())

        # architecture vs backbone: which one actually drives attribution
        # quality more? Eta-squared = share of total variance explained.
        vd = variance_decomposition(cell_table)
        vd.to_csv(args.out / "architecture_vs_backbone_variance.csv", index=False)
        print("\n=== architecture vs. pLM backbone: share of variance explained "
              "(eta-squared) ===")
        print(vd.set_index("metric").round(3).to_string())

        # same decomposition, but adding a seed component (needs individual
        # per-seed values, not the seed-averaged cell_table above) -- for
        # just the 5 shortlisted metrics, answers "how much of this metric's
        # variance is plain seed noise vs a real architecture/backbone effect"
        per_seed_cells = _arch_backbone_best_cell_table_per_seed(df_all, BACKBONE_METRICS)
        vd_seed = variance_decomposition_with_seed(per_seed_cells)
        if not vd_seed.empty:
            vd_seed.to_csv(args.out / "architecture_vs_backbone_vs_seed_variance.csv", index=False)
            print("\n=== architecture vs. backbone vs. seed: share of variance explained "
                  "(eta-squared, 5 shortlisted metrics) ===")
            print(vd_seed.set_index("metric").round(3).to_string())

        # arch x backbone heatmap, one per shortlisted metric, faceted by
        # protein x landscape, at each cell's best-model operating point
        best_cell_table = _arch_backbone_best_cell_table(df_all, metric_names)
        best_cell_table.to_csv(args.out / "backbone_best_cells.csv", index=False)

        # attribution agreement: does the RAW per-position attribution vector
        # agree within-model (across seeds -- the noise floor) as well as it
        # does across architecture or across backbone choice?
        seed_agree = attribution_seed_agreement(args.root, best_cell_table)
        arch_agree = attribution_cross_agreement(args.root, best_cell_table, axis="arch")
        backbone_agree = attribution_cross_agreement(args.root, best_cell_table, axis="backbone")
        agreement_all = pd.concat([seed_agree, arch_agree, backbone_agree], ignore_index=True)
        if not agreement_all.empty:
            agreement_all.to_csv(args.out / "attribution_agreement.csv", index=False)
            plot_attribution_agreement_box(agreement_all, args.out / "attribution_agreement_box")
            print("\n=== attribution agreement: mean Spearman rho on raw position vectors, "
                  "by comparison type ===")
            print(agreement_all.groupby("comparison")["mean_spearman"].agg(["mean", "std", "count"])
                  .round(3).to_string())
        for m in BACKBONE_METRICS:
            if m in best_cell_table.metric.values:
                plot_arch_backbone_heatmap(best_cell_table, m, args.out / f"metric_{m}_heatmap")

        # parallel coordinates: architecture, backbone, then every
        # shortlisted metric as its own axis, one thread per config,
        # faceted by protein x landscape combo
        plot_parallel_coordinates(best_cell_table, BACKBONE_METRICS,
                                  args.out / "parallel_coordinates")
        plot_parallel_coordinates(best_cell_table[best_cell_table.arch == "MeanPool"],
                                  BACKBONE_METRICS, args.out / "parallel_coordinates_meanpool",
                                  per_panel_range=True, show_backbone=False)

        # TOPSIS heatmap: architecture x backbone, same 5 shortlisted
        # metrics combined into one closeness score per cell, via the exact
        # same heatmap renderer used for every individual metric above
        topsis_cells = arch_backbone_topsis_cells(best_cell_table, BACKBONE_METRICS)
        if not topsis_cells.empty:
            topsis_cells.to_csv(args.out / "backbone_topsis_cells.csv", index=False)
            plot_arch_backbone_heatmap(topsis_cells, "topsis", args.out / "metric_topsis_heatmap")

        bars_metrics = list(BACKBONE_METRICS)
        bars_table = best_cell_table[best_cell_table.arch == "MeanPool"]
        if not topsis_cells.empty:
            bars_table = pd.concat(
                [bars_table, topsis_cells[topsis_cells.arch == "MeanPool"]], ignore_index=True)
            bars_metrics = bars_metrics + ["topsis"]
        plot_backbone_metric_bars(bars_table, bars_metrics,
                                  args.out / "backbone_metric_bars_meanpool")

    # ankh_base / MeanPool intermediate-layer sweep (0, 12, 24, 36, 48)
    if (args.root / "meanpool_ankh_base_layersweep").is_dir():
        layer_cells = layer_sweep_best_cell_table(args.root, metric_names)
        if not layer_cells.empty:
            layer_cells.to_csv(args.out / "layersweep_best_cells.csv", index=False)
            plot_layer_sweep_lines(layer_cells, BACKBONE_METRICS + ["model_spearman"],
                                   args.out / "layersweep_lines")
            print("\n=== ankh_base/MeanPool layer sweep: mean and worst-case (min) value "
                  "across the 6 protein/landscape combos, per layer ===")
            for m in BACKBONE_METRICS:
                sub = layer_cells[layer_cells.metric == m]
                if sub.empty:
                    continue
                piv = sub.groupby("layer")["value"].agg(["mean", "min"]).round(3)
                print(f"-- {m} --")
                print(piv.to_string())

            topsis_layer_cells = layer_sweep_topsis_cells(layer_cells, BACKBONE_METRICS)
            if not topsis_layer_cells.empty:
                topsis_layer_cells.to_csv(args.out / "layersweep_topsis_cells.csv", index=False)
                mp_rows = layer_cells[layer_cells.metric == "model_spearman"][
                    ["layer", "protein", "landscape", "metric", "value", "std"]].copy()
                combined = pd.concat(
                    [topsis_layer_cells.assign(std=0.0), mp_rows], ignore_index=True)
                plot_layer_sweep_topsis_vs_perf(combined, args.out / "layersweep_topsis_vs_perf")

    print(f"\nsaved plots + tables to {args.out}")


if __name__ == "__main__":
    main()
