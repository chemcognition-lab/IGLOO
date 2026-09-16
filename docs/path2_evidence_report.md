# Does IGLOO's IG attribution reflect what the model learned?

**Question.** Does IGLOO's integrated-gradients attribution reflect the *learned*
fitness function, or something model-independent — the pLM embedding geometry
plus the shallow attention-pooling layer?

**Answer.** Mostly **model-independent**. The per-position attribution pattern is
set by (a) the shallow `Linear(960→1)` attention-pooling layer and (b) the
model-free pLM embedding displacement. The deep MLP — where the learned
epistatic / nonlinear fitness structure actually lives — is **structurally
invisible to IG**, because it sits entirely *after* the pooling bottleneck.

> All figures below are from the **bfloat16** embedding run (Sept 2026). The
> float32 rebuild is regenerating them; the *quantitative* numbers may shift by
> ~0.01–0.02, but findings 1–2 are about architecture and are precision-independent.
> Figure paths are relative to the repo root.

---

## The mechanism

`ESMAttention1d`:

```
x  ∈ ℝ^{L×960}   (per-residue ESM-C embeddings)
      │
      ▼  attnᵢ = softmax( Linear₉₆₀→₁(xᵢ) )          ← one weight per position
      ▼  pooled = Σᵢ attnᵢ · xᵢ  ∈ ℝ^{960}            ← collapses L → 1
      │
      ▼  layernorm → fc1 → fc2 → fc3 → fc4 → final    ← the MLP: acts on ONE vector
      ▼
      F  (predicted fitness)
```

Chain rule for the IG gradient at residue *i*:

```
∂F/∂xᵢ  =  (∂F/∂pooled) · (∂pooled/∂xᵢ)  ≈  attnᵢ · g
           └────┬────┘                        └┬┘  └┬┘
       one 960-vec g, identical for          attention   the MLP contributes
       every position (this is the MLP)       weight      only a global rescale
```

So the **per-position pattern** of `∂F/∂xᵢ` comes from `attnᵢ` (a shallow, low-
capacity, under-constrained layer). The MLP only scales/rotates `g` globally and
cannot say "position 34 matters more than position 12."

---

## Evidence

### 1. Mechanism verification — direct test of `∂F/∂xᵢ ≈ attnᵢ · g`

`scripts/attribution_mechanism.py` extracts the attention weights, the
path-averaged gradient, and `g = ∂F/∂pooled` for ~300 test variants per run.

![mechanism](../results/synthetic/attribution_mechanism/mechanism_GB1_additive__noise_0.0.png)

| Check | Result (all 18 GB1+T4 × noise conditions) |
|---|---|
| `pearson(attnᵢ, mean‖IGᵢ‖)` — gradient magnitude *is* the attention weight | **0.84 – 0.99** |
| `pearson(attnᵢ·‖Δxᵢ‖, ‖signed IGᵢ‖)` — full attribution *is* attention × displacement | 0.89 – 0.99 (GB1); looser for T4 at high noise |
| `spearman(attn before vs after randomizing the ENTIRE MLP)` | **1.000 — exactly, every run** |
| cosine dispersion of `g` across variants | **0.002 – 0.07** (≈ one fixed vector) |

**→ The MLP has no per-position effect; the attention layer sets the pattern.**

---

### 2. Cascading model-parameter randomization (Adebayo et al. sanity check)

`scripts/cascading_randomization.py` — randomize predictor layers output→input,
recompute IG, measure how much the attribution changes.

![cascading vs original](../results/synthetic/cascading_randomization/cascading_summary_original.png)
![cascading vs ground truth](../results/synthetic/cascading_randomization/cascading_summary_gt.png)

- Randomizing `final → fc4 → fc3 → fc2 → fc1 → layernorm` leaves
  `spearman(attr, original)` at **0.85 – 1.0**.
- The attribution only collapses (to ~0.1 – 0.3) once **`attention1d`** is also
  randomized.
- Same at every noise level, all three GB1 landscapes.
- The `_gt` panel: a model with the whole MLP randomized recovers ground truth
  about as well as the trained model — the (weak) ground-truth agreement doesn't
  come from training either.

**→ The entire MLP head is irrelevant to the attribution.**

---

### 3. Factor ablation — decomposing the IG product

The signed IG product is `⟨ ∫∇F dα , Δxᵢ ⟩`. `scripts/factor_ablation_analysis.py`
isolates each factor: `gradient` (`‖∇F‖`), `displacement` (`‖Δx‖`, no model),
`signed` (the product).

![factor corr to signed](../results/synthetic/factor_ablation_analysis/factor_corr_to_signed.png)
![factor ablation spearman](../results/synthetic/factor_ablation_analysis/factor_ablation_spearman.png)
![factor ablation ndcg](../results/synthetic/factor_ablation_analysis/factor_ablation_ndcg.png)

- `gradient` alone reproduces the signed product's per-position ranking at
  **ρ ≈ 0.85 – 0.94**; `displacement` alone only ρ ≈ 0.25 – 0.5.
- But that "gradient" ≈ `attnᵢ · const` (finding 1) — so it *is* the attention
  pattern, not learned knowledge.
- `displacement` alone recovers ground-truth `w` at a **flat ρ ≈ 0.12** —
  weak, and completely independent of seed / noise / model quality.
- On aggregate NDCG all factors score ~0.9 — but that's because NDCG is
  saturated here (random ranking ≈ 0.84). The signal shows only on
  Spearman / AUROC.

**→ The position-selecting factor is the attention pattern; the displacement is a
weak, fixed, model-free contribution.**

---

### 4. k-predictor agreement — do independently-trained seeds agree?

`scripts/k_predictor_agreement.py` — for 5 seeds per condition, mean pairwise
agreement on test predictions vs on attribution rankings.

![k-predictor agreement](../results/synthetic/k_predictor_agreement/k_predictor_agreement_by_noise.png)

- **Prediction agreement**: ρ ≈ 0.75 – 0.95 (degrades gently with noise, as expected).
- **Attribution-ranking agreement**: ρ ≈ **0.40 – 0.55**, roughly noise-independent.

Five models that learned nearly the same function produce quite different
position rankings — the attribution is far more **under-determined** than the
learned function. (For T4, ISM rankings agreed much more across seeds — 0.68–0.81
vs IG's 0.29–0.39 — so IG's disagreement is a gradient-underdetermination
problem, not inherent to the task.)

---

### 5. ISM comparison — is the problem fixable?

In-silico mutagenesis (`scripts/ism_attribution.py`): mutate every position to
all 19 AAs, measure the prediction change. Model-dependent by construction, no
gradient.

![ISM vs IG, Spearman vs ground truth](../results/synthetic/ism_attribution/ism_vs_ig_spearman_vs_w.png)

- **ISM quality tracks model performance where IG's doesn't** — for the
  epistatic and nonlinear landscapes: r(ISM metric, model test Spearman) =
  **+0.79 to +0.87** (significant); IG's within-condition r ≈ 0.
- ISM recovers ground truth better than IG, especially for T4 (2–3× higher
  `precision@k`), and shows a much larger gap between a good model's attribution
  and a random (shuffled-label) model's.

**→ The problem is IG-*on-this-architecture*, not "attribution is impossible here."**

---

### 6. Metric choice — an important nuance

`scripts/metric_suite.py`, discrimination = `|Pearson r(metric, model test Spearman)|`.

![precision@k vs model performance](../results/synthetic/metric_suite/metric_precision_at_k.png)
![Spearman vs model performance](../results/synthetic/metric_suite/metric_spearman.png)
![sign accuracy vs model performance](../results/synthetic/metric_suite/metric_sign_acc.png)

- With the *right* metric (`precision@k`, `recall_topcat`), even **IG's**
  discrimination r is 0.6 – 0.85 — IG *does* rank models.
- But that is because label noise degrades the whole model **including the
  `attention1d` layer**, which is exactly what IG sees. So IG `precision@k`
  measures "did the attention pooling get worse at focusing," not "did the
  learned fitness function get worse."
- `AUROC` / `enrichment` had the *wrong sign* (attribution "looked better" as the
  model degraded). `NDCG` was pinned near its random floor and carried no signal.

---

## Bottom line

| | Source of the per-position attribution | Reflects the learned MLP? |
|---|---|---|
| **attention-pooling layer** (`Linear 960→1`) | trained, low-capacity, under-constrained — degrades with label noise | **no** (not where fitness knowledge lives) |
| **pLM embedding displacement** (`Δx`) | fixed, model-free | **no** |
| **the deep MLP** | structurally post-pooling | **invisible to IG** |

Not "pLM geometry overwhelms the gradient" — more precisely: the architecture
funnels all per-position signal through one shallow layer *before* the
non-linearity, so IG can never see the MLP.

**Fix candidates (data landing now):**
- **ISM** — perturbation-based, no gradient pathology; does reflect the learned function.
- **IG_premlp** — architecture with a learnable per-residue transform *before* the
  pool, so `∂F/∂xᵢ` depends on trained weights per position. If cascading then
  shows the `pre` layer matters and IG discrimination rises, that's the fix.

## Scripts & outputs

| Experiment | Script | Output dir |
|---|---|---|
| Mechanism verification | `scripts/attribution_mechanism.py` | `results/synthetic/attribution_mechanism/` |
| Cascading randomization | `scripts/cascading_randomization.py` (+ `_summary.py`) | `results/synthetic/cascading_randomization/` |
| Factor ablation | `scripts/factor_ablation_analysis.py` | `results/synthetic/factor_ablation_analysis/` |
| k-predictor agreement | `scripts/k_predictor_agreement.py` | `results/synthetic/k_predictor_agreement/` |
| ISM vs IG | `scripts/ism_attribution.py` + `ism_attribution_analysis.py` | `results/synthetic/ism_attribution/` |
| ISM seed agreement | `scripts/ism_seed_agreement.py` | `results/synthetic/ism_seed_agreement/` |
| Metric suite | `scripts/metric_suite.py` | `results/synthetic/metric_suite/` |
| Metric / method definitions | — | `docs/attribution_methods_and_metrics.md` |
