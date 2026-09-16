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
    python scripts/metric_suite.py [--root .../results/synthetic] [--out .../metric_suite]
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
from scipy.stats import spearmanr, pearsonr, rankdata
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    from scipy.stats import ConstantInputWarning
    warnings.filterwarnings("ignore", category=ConstantInputWarning)
except ImportError:
    pass

PROTEINS = ["GB1", "T4Lysozyme"]
LANDSCAPES = ["additive", "epistatic", "nonlinear"]
_GT_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"
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
            root = str(Path(__file__).resolve().parent.parent)
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
# each yield: (protein, landscape, noise, seed, method, kind, mag, sgn, model_perf)
def ig_runs(root: Path):
    def _y(protein, landscape, noise, seed, kind, leaf):
        f = list(leaf.glob("*_position_importance_summary.csv"))
        if not f:
            return None
        d = pd.read_csv(f[0]).sort_values("position")
        mag = d["importance_score"].to_numpy()
        sgn = d["mean_gradient"].to_numpy() if "mean_gradient" in d else mag
        return (protein, landscape, noise, seed, "IG", kind, mag, sgn, _model_perf(leaf))

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
                    yield (r[0], r[1], r[2], r[3], "IG_premlp", r[5], r[6], r[7], r[8])
            # item 1: one-hot amino-acid input instead of ESM-C embedding (classic
            #   esm_attention head; tests how much the pLM geometry itself matters)
            # items 2/3: alternative pooling/attention architectures, pilot scale
            #   (noise in {0,1} x 3 seeds — see jobs/run_newarch_train_array.sh)
            _newarch_dirs = {
                "onehot": "onehot",
                "newarch_channel": "channel_attn",
                "newarch_perres_mean": "perres_mean",
                "newarch_perres_attn": "perres_attn",
                "newarch_crossattn": "crossattn",
                "newarch_perres_flatten": "perres_flatten",
            }
            for dirname, method_name in _newarch_dirs.items():
                for sd in sorted((root / dirname / f"{protein}_{landscape}").glob("noise_*/seed_*/linear")):
                    noise = float(sd.parents[1].name.split("_", 1)[1])
                    seed = int(sd.parents[0].name.split("_", 1)[1])
                    r = _y(protein, landscape, noise, seed, "sweep", sd)
                    if r:
                        yield (r[0], r[1], r[2], r[3], method_name, r[5], r[6], r[7], r[8])
            # factor-ablation baselines (isolated attribution 'factors'):
            #   displacement  -> input-difference baseline (||x1 - x0||, NO model)
            #   gradient      -> gradient-magnitude baseline (||mean_k grad_k||)
            #   grad_input    -> gradient x input   (grad(F,x1) . x1, endpoint)
            _fa_map = {"displacement": "input_diff", "gradient": "grad_mag",
                       "grad_input": "grad_input"}
            for meth_dir, meth_name in _fa_map.items():
                for leaf in sorted((root / "factor_ablation" / f"{protein}_{landscape}")
                                   .glob(f"noise_*/{meth_dir}")):
                    noise = float(leaf.parents[0].name.split("_", 1)[1])
                    r = _y(protein, landscape, noise, 0, "sweep", leaf)
                    if r:
                        yield (r[0], r[1], r[2], r[3], meth_name, r[5], r[6], r[7], r[8])


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
            yield (ms.group(1), ms.group(2), np.nan, int(ms.group(3)), "ISM", "shuffled", mag, sgn, mp)
            continue
        m = re.match(r"ism_position_([A-Za-z0-9]+)_([a-z]+)__noise_([0-9.]+)(?:__seed_(\d+))?", f.stem)
        if not m:
            continue
        yield (m.group(1), m.group(2), float(m.group(3)),
               int(m.group(4)) if m.group(4) else 0, "ISM", "sweep", mag, sgn, mp)


def main():
    project_root = Path(__file__).resolve().parent.parent
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
        for protein, landscape, noise, seed, method, kind, vec, sgn, mperf in gen(args.root):
            rows.append({"protein": protein, "landscape": landscape, "noise": noise,
                         "seed": seed, "method": method, "kind": kind,
                         "model_spearman": mperf,
                         **all_metrics(vec, protein, signed=sgn)})
    df = pd.DataFrame(rows)
    if df.empty:
        print("no runs found"); return
    df.to_csv(args.out / "metric_suite.csv", index=False)
    print(f"{len(df)} runs  "
          f"({df.groupby(['method', 'kind']).size().to_dict()})")

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
    MK = {"IG": "o", "ISM": "s", "IG_premlp": "D",
          "input_diff": "v", "grad_mag": "^", "grad_input": "P",
          "onehot": "X", "channel_attn": "*", "perres_mean": "h",
          "perres_attn": "H", "crossattn": "8", "perres_flatten": "p"}
    CC = {"IG": "#d62728", "ISM": "#1f77b4", "IG_premlp": "#2ca02c",
          "input_diff": "#9467bd", "grad_mag": "#8c564b", "grad_input": "#e377c2",
          "onehot": "#7f7f7f", "channel_attn": "#bcbd22", "perres_mean": "#17becf",
          "perres_attn": "#ff7f0e", "crossattn": "#393b79", "perres_flatten": "#8c9e2f"}
    PLOT_METHODS = ["IG", "ISM", "IG_premlp", "input_diff", "grad_mag", "grad_input",
                     "onehot", "channel_attn", "perres_mean", "perres_attn", "crossattn",
                     "perres_flatten"]

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
                     f"(●IG ■ISM ◆IG-preMLP ▼input-diff ▲grad-mag ✚grad×input;  "
                     f"★=shuffled-label model, ⋯=random-ranking null)",
                     fontsize=11, fontweight="bold")
        for ext in ("png", "svg"):
            fig.savefig(args.out / f"metric_{m}.{ext}", dpi=150)
        plt.close(fig)
    print(f"\nsaved plots + tables to {args.out}")


if __name__ == "__main__":
    main()
