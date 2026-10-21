# IGLOO attribution: methods & metrics

Reference for the Path-2 investigation — what each attribution method computes,
what each evaluation metric measures, and the baselines used for comparison.

Notation:

- `F` — the model's fitness prediction
- `x⁰` — ESM embedding of the wild-type sequence
- `x¹` — ESM embedding of the variant
- `Δx = x¹ − x⁰` — per-residue embedding displacement (960-dim per position)
- `∇F` — gradient of `F` w.r.t. the per-residue embedding
- `w` — ground-truth position-importance vector (from the synthetic data generator)
- "annotated" positions — the functionally important ones; "background" — the rest
- `k` — number of annotated positions (used for top-`k` metrics)

---

## 1. Attribution methods

Each produces **one importance score per residue position**.

| Method | Formula (position *i*) | What it isolates | Notes |
|---|---|---|---|
| **IG (signed)** | `aᵢ = ⟨ ∫₀¹ ∇F(x⁰+αΔx) dα , Δxᵢ ⟩` | model gradient **×** pLM displacement (the product) | The main IGLOO method. Completeness-exact: `Σᵢ aᵢ = F(x¹) − F(x⁰)`. |
| **grad_mag** (`gradient`) | `aᵢ = ‖ ∫ ∇F dα ‖₂` | **model gradient only** — magnitude, no displacement weighting | Factor ablation. |
| **displacement** (`input_diff`) | `aᵢ = ‖ Δxᵢ ‖₂` | **pLM embedding geometry only — no model at all** | Factor ablation. Identical across seeds / noise / model quality. |
| **grad×input** (`grad_input`) | `aᵢ = ⟨ ∇F(x¹)ᵢ , x¹ᵢ ⟩` | single-point gradient × input, no path integral | Classic Shrikumar-style baseline. |
| **ISM** (in-silico mutagenesis) | `aᵢ = mean₍19 AAs₎ \| F(WT: i→a) − F(WT) \|` | how much the **prediction** moves when position *i* is actually mutated | Perturbation-based, no gradient, runs the whole model. `~19·L` forward passes. Also keeps a signed mean. |
| **IG_premlp** | IG (signed), on the architecture with a shared per-residue MLP **before** the attention pool | whether the architecture fix makes IG "see" the learned function | Architecture test. |
| **onehot** | IG (signed), input = one-hot amino-acid encoding instead of the ESM-C embedding | how much the pLM embedding *geometry itself* matters, vs. bare sequence identity | Input-representation test. Same `esm_attention` head, `d=21`. |
| **channel_attn** | IG (signed) on `ESMAttentionChannel` — attention pools the **channel** axis (960→1, shared weights per position), keeping length `L` intact for a per-position MLP before a final length-attention pool | whether swapping which axis attention collapses changes what IG sees | Architecture test (item 2). |
| **perres_mean** | IG (signed) on `ESMPerResidueMLP` (mean pool) — the *entire* MLP runs per-residue on the raw embedding, aggregated by a plain mean | direct fix candidate: mean-pool has a uniform 1/L gradient, so `∂F/∂xᵢ` comes entirely from the trained per-position MLP | Architecture test (item 3a). |
| **perres_attn** | Same per-residue MLP, aggregated by a learned attention pool instead of mean | per-position depth **and** learned position weighting together | Architecture test (item 3b). |
| **crossattn** | Same per-residue MLP, but 1-2 real multi-head self-attention layers mix information **across positions** first | best chance at modeling true epistasis; attributions harder to interpret since positions now mix | Architecture test (item 3c). |
| **perres_flatten** | Same per-residue MLP, but **no pooling at all** — the per-position hidden vectors are flattened and fed directly into one `Linear(L·H → 1)` head | freely-learned, per-position coefficients instead of a mean or a softmax-normalised convex combination; most unconstrained way to let position matter individually | Architecture test (item 3d). Requires fixed-length input (true here: point substitutions only, no indels) — does not generalise to variable-length sequences. |

Items 2/3 are run at **pilot scale** (noise ∈ {0, 1} blend, ×3 seeds) before any full 9-level sweep — see `jobs/architectures/run_newarch_train_array.sh` / `jobs/smoketests/run_newarch_perres_flatten_pilot.sh`.

### Baselines (reference lines, not methods)

| Baseline | Definition | Purpose |
|---|---|---|
| **Shuffled-label model** | a model trained on permuted fitness labels (held-out test ρ ≈ 0) | "what any method scores from a model that learned nothing" — the floor a real model must beat |
| **Random-ranking null** | the metric's value under random position orderings (permutation test) | the chance level for each metric on this specific ground truth |

### The label-noise sweep (how models of varying quality are produced)

Models spanning "excellent" → "learned nothing" are produced by corrupting the
training labels by a **scale-free** amount `p ∈ [0, 1]`
(`training.noise.label_noise_mode = blend`, `label_noise_level = p`):

```
y_train  :=  renorm( (1 − p) · y_true  +  p · y_perm )
```

`y_perm` is a random permutation of the labels; `renorm` rescales back to the
original mean/std so the label *distribution* is identical at every `p`.

| `p` | `corr(y_noisy, y_true) = (1−p)/√((1−p)²+p²)` | meaning |
|---|---|---|
| 0.0 | 1.00 | exact labels |
| 0.2 | 0.97 | barely corrupted |
| 0.5 | 0.71 | half signal |
| 0.85 | 0.28 | mostly noise |
| 1.0 | 0.00 | full permutation — **identical to the shuffled-label baseline** |

Sweep grid: `0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0`, ×5 seeds for error
bands. This replaces the earlier `label_noise_std` (absolute Gaussian σ in raw
fitness units), whose effect depended on each landscape's label scale and which
never reached ρ ≈ 0.

---

## 2. Evaluation metrics

All compare an attribution vector `a` to the ground-truth importance vector `w`.

| Metric | Definition | Interpretation | Chance value |
|---|---|---|---|
| **spearman** | Spearman ρ( \|a\| , w ) | rank agreement — do high-attribution positions have high true importance | 0 |
| **auroc** | AUROC(annotated vs background, score = \|a\|) | P(a truly-important position outranks a background one) | 0.5 |
| **precision@k** | of the top-`k` positions by \|a\|, fraction truly annotated | **hit rate** — "of the sites you flag, how many are real" | prevalence (`k`/L) |
| **recall_topcat** | of the top-tier sites (highest-weight annotation categories, merged from the top down until ≥ 5 positions), fraction that land in the top-`k` | **do you catch the most critical sites** (hard hit-rate) | prevalence |
| **topcat_rank** | mean quantile of the top-tier sites in the \|a\| ranking (1 = highest attribution) | same question, **smooth** — no small-denominator quantisation | 0.5 |
| **enrichment** | mean\|a\|(annotated) ÷ mean\|a\|(background) | "important sites get X× more attribution than background" | 1.0 |
| **category_mono** | Spearman(per-category mean \|a\| , category weight) | does attribution grow monotonically across importance tiers (background → folding → structural → binding) | 0 |
| **epi_ratio** | mean\|a\|(annotated sites in a contact pair) ÷ mean\|a\|(annotated sites not paired) | does attribution concentrate on epistatically-**interacting** positions | 1.0 |¹
| **sign_acc** | fraction of annotated positions whose **signed** aggregate attribution is negative | does the attribution correctly say "mutating this important site is harmful" | 0.5 |

¹ **`epi_ratio` is undefined for GB1** (all landscapes): GB1 is a 56-aa domain
with 43/56 positions annotated and SS regions tiling the whole sequence, so
every annotated position lies within 10 residues of another annotated one —
the "annotated & not-in-any-contact-pair" reference group is empty, so the
ratio is `NaN` for every run (real and random-null alike). It's only meaningful
for **T4Lysozyme**, which has 7 sparse, widely-separated annotated positions.
The plots mark the empty GB1 panels accordingly instead of leaving them blank.

### Two derived views of every metric

| View | Definition | Reads as |
|---|---|---|
| **skill score** | `(metric − random) / (ideal − random)` | **0 = random, 1 = perfect** — puts all metrics on one scale |
| **discrimination** | `\| Pearson r( metric , model test Spearman ) \|` across the noise sweep | **how well the metric separates good models from noisy/bad ones** (the property we actually care about) |

### Findings from the (bf16) run — to be re-confirmed on the float32 rebuild

- `precision@k` and `recall_topcat` were the **best discriminators** (r ≈ 0.6–0.85 for both IG and ISM, across all landscapes) and are the most interpretable.
- `auroc` and `enrichment` had the **wrong sign** — attribution "looked better" as the model got worse.
- `NDCG` (earlier metric, now dropped) was saturated near its random floor and carried no signal.
- `sign_acc` works for IG (positive, r ≈ 0.5) but goes backwards for ISM.
- `epi_ratio` is meaningful only for the epistatic landscape and only for IG.

---

## 3. How methods and metrics connect to the Path-2 question

**Question:** does IGLOO's IG attribution reflect what the *model learned*, or something
model-independent (pLM embedding geometry + the shallow attention-pooling layer)?

- **Factor ablation** (`displacement` / `grad_mag` / `signed` / `grad_input`) — decomposes
  the IG product to see which factor carries the per-position signal.
- **ISM** — a model-*dependent* attribution with no gradient pathology; if ISM quality
  tracks model quality where IG's doesn't, the problem is IG-on-this-architecture, not
  "attribution is impossible here".
- **Mechanism verification** — directly checks `∂F/∂xᵢ ≈ attnᵢ · (∂F/∂pooled)` and that the
  attention weights are unchanged when the MLP is randomized.
- **Cascading randomization** — Adebayo sanity check: randomize predictor layers
  output→input, watch when the attribution changes.
- **k-predictor agreement** — do independently-trained seeds agree more on predictions
  than on attributions?
- **IG_premlp** — architecture fix: put a learnable per-residue transform *before* the
  pool so `∂F/∂xᵢ` depends on the trained weights per position.
