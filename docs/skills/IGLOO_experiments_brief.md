# IGLOO — experiment brief for the coding agent

This document specifies a set of experiments for the IGLOO revision. It is written to be
handed to a coding agent working **inside the IGLOO repository**. Work through it in order;
P0 first. Do not refactor beyond what each task needs.

## Background (why these experiments exist)

The paper's current framing treats residue-level attribution quality as a single capability
measured by a composite score. Feedback from three sources converges on a sharper story:

- **Global, pooled magnitude metrics** (NDCG / AP@R / AUROC on `|IG|`) appear **invariant**
  to label noise, to model test performance, and to predictor capacity. They may be
  measuring a property of the **pLM embedding geometry**, not of the learned fitness function.
- **Per-variant, directional (sign) metrics** *do* respond to the trained predictor and are
  the signal worth benchmarking.

The revision reorganizes around two questions:

- **Q_A** — are the metrics sensitive to the axes we test at all (dynamic range above random)?
- **Q_B** — do attributions add anything over ranking residues by embedding displacement alone?

The experiments below produce the evidence for that reframe.

---

## 0. Repo orientation (do this first, report back)

Locate and note the paths / entry points for:

1. **Synthetic landscape generation** — the code that builds GB1 / T4 lysozyme / p53
   datasets, assigns per-position category weights `w`, and evaluates the additive /
   epistatic / nonlinear fitness functions. Note where the ground-truth `w` vector and the
   category annotations are serialized (the "ground-truth JSON" referenced in SI A.5).
2. **Predictor model + training** — the attention-pooled MLP, its training loop, checkpoint
   layout, and the per-model hyperparameters (SI Tables 6–7).
3. **Attribution runner** — where linear / slerp / nearest-neighbor IG, `gradient×input`,
   and `input difference` are implemented, and how per-position embeddings are extracted
   from the frozen pLM (layer index, pooling).
4. **Metrics module** — the seven metrics (SI A.5), the model-collapse filter (`|f| < δ`),
   the reliability mask, and the composite / trapezoidal-AUC / Friedman pipeline (SI A.6).
5. **Config system** — how a `(protein, landscape, noise, method, pLM)` run is specified and
   launched; where results tables/figures are written.

Produce a short `EXPERIMENTS_STATUS.md` at repo root listing these paths and a checkbox per
task below. Update it as you go.

## Conventions for every task

- Fix and log all random seeds (numpy, torch, CUDA). Each experiment gets its own output
  directory: `results/revision/E<N>_<shortname>/`.
- Write results as tidy long-format CSV (`one row per (protein, landscape, noise, method,
  metric, seed, value)`) plus a `config.yaml` snapshot. Figures are regenerated from CSV,
  never hand-edited.
- Do not change existing default behavior. Add new modes behind flags / new config keys.
- Where a task says "all 9 scenarios" that means `{GB1, T4 lysozyme, p53} × {additive,
  epistatic, nonlinear}` with ESM-C 300M unless stated otherwise.
- Prefer reusing cached pLM embeddings; these are the expensive artifact.

---

# P0 — core to the reframe

## E1 — Untrained-predictor baseline

**Goal.** Quantify how much of attribution quality comes from the *trained* predictor vs the
frozen embedding + a random readout. Central evidence for Q_B.

**Task.**
1. Add a predictor variant `--predictor-init {trained, random}`. For `random`: instantiate
   the same architecture, apply the standard weight init, **do not train**, keep the output
   de-normalization constants `(µ_y, σ_y)` at their data values (or set `µ_y=mean(f_train)`,
   `σ_y=std(f_train)` so output scale is comparable).
2. Run the full attribution + metric suite for `random` across all 9 scenarios, all 5
   methods, all 7 noise levels. Repeat over **≥5 random seeds** for the init.
3. Also run `random` for Ankh-base and E1 150M on GB1 (all landscapes) to check the finding
   is not ESM-C-specific.

**Outputs.**
- `E1_untrained/metrics_long.csv` (includes `predictor_init` column).
- A paired plot per metric: trained vs random composite across noise, one panel per
  scenario, seed spread shown as a band.
- A single summary table: mean Δ(trained − random) per metric family, per protein.

**Acceptance.**
- Random-init runs complete for all 9 scenarios × 5 seeds without training.
- The summary table cleanly separates metrics where trained ≈ random (expected: global
  ranking) from metrics where trained > random (expected: sign accuracy, per-variant NDCG).

## E2 — Gradient-free displacement ranking

**Goal.** Test whether `|IG|` position rankings are explained by embedding displacement
magnitude alone (no predictor gradient).

**Task.**
1. For every variant, compute the per-position displacement score
   `d_i = ||e1_i − e0_i||_2` from the frozen pLM embeddings (same layer/pooling as the
   attribution runner).
2. Aggregate `d_i` across non-collapsed variants exactly as `|IG|` is aggregated for the
   global metrics, producing a "displacement-only" attribution vector per scenario.
3. Feed it through the global ranking metrics (NDCG, AP@R, AUROC) and the mutation-level
   NDCG.
4. Compute rank correlation (Spearman + Kendall τ) between the displacement-only position
   ranking and each method's `|IG|` position ranking, per scenario and noise level.

**Outputs.**
- `E2_displacement/metrics_long.csv` with `method = "displacement_only"` as an added method.
- `E2_displacement/rank_correlation.csv` — τ and ρ between displacement-only and each IG
  method, per (scenario, noise).
- A heatmap: scenario × method, colored by τ to displacement-only ranking.

**Acceptance.**
- Displacement-only appears as a first-class method in the metrics CSV.
- Rank-correlation table is populated for all (scenario, method, noise) cells.

## E3 — Skill-score metric refactor

**Goal.** Replace per-block min–max normalization (which compresses dynamic range and
amplifies noise under annotation sparsity) with a skill score against explicit references.

**Task.**
1. For each metric `m`, scenario, and noise level, compute:
   - `m_rand` — mean metric value over **N_rand ≥ 200** random `|IG|` vectors (uniform or
     permutation of the observed attribution values), OR the E1 random-init value where
     available; make the choice a config key `skill.random_reference`.
   - `m_perfect` — metric evaluated with the ground-truth importance vector `w` as the
     attribution.
2. Define `skill(m) = (m − m_rand) / (m_perfect − m_rand)`, clipped to a sane range
   (report raw too). Handle `m_perfect ≈ m_rand` (metric has no headroom) by emitting NaN
   and flagging the metric as non-informative for that block.
3. Recompute the composite as the mean of `skill(m)` over informative metrics; regenerate
   the trapezoidal-AUC-over-noise summary and the Friedman pipeline on the new AUC matrix.
4. Keep the old composite available under `--composite {minmax, skill}` (default `skill`).

**Outputs.**
- `E3_skillscore/skill_long.csv` (`m_raw`, `m_rand`, `m_perfect`, `skill`, `informative`).
- Regenerated noise-vs-composite figure (main-text Figure 2 / SI Figure 9 equivalents).
- A table of which (metric, scenario) blocks are flagged non-informative.

**Acceptance.**
- Every metric block has `m_rand` and `m_perfect` recorded.
- New composite figures regenerate end-to-end from CSV.
- Friedman / Kendall's W rerun on the skill-based AUC matrix; results written alongside the
  originals for comparison.

## E4 — Per-variant-first reporting

**Goal.** Restructure result tables/figures so per-variant and sign metrics are primary and
global pooled metrics are presented as an embedding-geometry probe.

**Task.**
1. Add a `metric_role` field to the metric registry: `{per_variant, directional,
   embedding_probe, mutation_level}`. Map: per-variant NDCG → per_variant; sign accuracy →
   directional; NDCG/AP@R/AUROC (global) → embedding_probe; category separation →
   per_variant; mutation NDCG → mutation_level.
2. Provide a reporting function that emits the main results table grouped by `metric_role`,
   with `embedding_probe` metrics visually separated and captioned as measuring the pLM
   representation rather than the predictor.
3. No numerical changes — this is presentation only.

**Outputs.** Regenerated main results table(s); a short markdown note describing the grouping
for the paper's methods section.

**Acceptance.** Tables regenerate; `embedding_probe` metrics are clearly delineated.

---

# P1 — reviewer-critical (new compute; GB1-first except E15)

## E5 — Occlusion / in-silico mutagenesis baseline

**Goal.** Add a perturbation-based attribution as a reference point for the "first-order
gradients suffice" claim.

**Task.**
1. Implement per-position occlusion in embedding space: for position `i`, replace `e_var,i`
   with the WT embedding `e_wt,i` (single-position revert) and record
   `Δf_i = f(e_var) − f(e_var with i reverted)`. This is leave-one-position-out on the
   embedding.
2. Optionally also implement sequence-space ISM (mutate residue `i` to WT in the sequence,
   re-embed, re-predict) as `--ism-space {embedding, sequence}`; sequence-space is more
   expensive (re-embed per position) — gate behind a flag, run on GB1 only.
3. Use `Δf_i` (and `|Δf_i|`) as an attribution vector; run the full metric suite.

**Outputs.** `E5_occlusion/metrics_long.csv` with `method ∈ {occlusion_embed, ism_seq}`;
comparison plot vs `gradient×input` and IG-linear across scenarios.

**Acceptance.** Occlusion runs across all 9 scenarios (embedding-space); ISM on GB1.

## E6 — IG baseline sensitivity

**Goal.** Address "only the WT embedding was used as the IG baseline."

**Task.** Add `--ig-baseline {wt, mean, zero, random}`:
- `mean` — mean training-set per-position embedding (align by position; for length
  mismatch, use the mean-pooled vector broadcast).
- `zero` — zero vector.
- `random` — Gaussian matched to training embedding moments; average attributions over
  ≥5 draws.
Run all baselines for GB1 × {additive, epistatic, nonlinear} × all 5 methods × all noise
levels.

**Outputs.** `E6_baseline/metrics_long.csv` with `ig_baseline` column; a plot of composite
vs noise faceted by baseline; a note on completeness-gap per baseline.

**Acceptance.** All four baselines run on GB1; completeness gap reported.

## E7 — Layer sweep

**Goal.** Report sensitivity of attribution quality to the pLM layer.

**Task.** Parameterize the embedding extraction layer. For ESM-C 300M run layers
`{6, 12, 18, 24, 33}` on GB1 × 3 landscapes × {IG-linear, gradient×input} × all noise
levels. Re-fit the predictor per layer (embeddings differ) — reuse E1 infra.

**Outputs.** `E7_layers/metrics_long.csv` with `layer` column; composite-vs-layer plot,
one line per landscape; note whether the ranking of methods is layer-stable.

**Acceptance.** Five layers × 3 landscapes complete; predictor retrained per layer.

## E15 — Cross-pLM comparison, all proteins

**Goal.** The paper's "pLM identity dominates interpolation choice" finding currently rests
on GB1 alone (n = 3 scenarios). Compute is available; extend it to n = 9.

**Task.**
1. Run the full pipeline — embed, train predictor, attribute, score — for
   **{ESM-C 300M, Ankh-base, E1 150M} × {T4 lysozyme, p53} × {additive, epistatic,
   nonlinear} × all 5 methods × all 7 noise levels**. GB1 is already done; reuse it.
2. Keep per-model predictor hyperparameters consistent with the existing GB1 cross-pLM
   runs (do not re-tune per pLM — that would confound the comparison). Document the choice.
3. Rerun the Friedman / Kendall's W / Nemenyi pipeline on the full n = 9 pLM comparison
   (this replaces the n = 3 test in Table 3 / Section 3).
4. Also emit the E1 untrained-baseline and E3 skill scores for these new runs so the
   reframe analyses extend cleanly.

**Outputs.**
- `E15_crosspLM/metrics_long.csv` (all three proteins, `pLM` column).
- Updated Table 3 equivalent spanning all proteins; updated pLM-comparison statistics.
- Per-pLM composite-vs-noise figure, faceted by protein × landscape.

**Acceptance.**
- All 3 pLMs × 2 new proteins × 3 landscapes complete.
- Friedman/Kendall's W rerun on n = 9; χ², p, W reported next to the old n = 3 values.
- Embedding-identical to the GB1 cross-pLM runs in method/config except protein.

## E8 — IG convergence / saturation

**Goal.** Show the path integral is converged and quantify completeness violation.

**Task.**
1. For IG-linear and IG-slerp on GB1 × 3 landscapes, sweep interpolation steps
   `S ∈ {16, 32, 64, 128, 256, 500}`; record metric values and the mean absolute
   completeness gap `|Σ_i IG_i − (F(e1) − F(e0))|` per variant.
2. For `gradient×input` and `input difference`, record the completeness gap (no step sweep).
3. Flag any scenario where metrics have not stabilized by `S = 500`.

**Outputs.** `E8_convergence/convergence_long.csv`; a metric-vs-steps plot and a
completeness-gap bar chart per method.

**Acceptance.** Step sweep complete for both path methods; completeness gap tabulated for
all five methods.

---

# P2 — analysis, low / no compute

## E9 — Model-collapse filter audit

**Task.** Trace the `|f| < δ` (default `δ = 0.01`) filter and the `|Δf| ≥ 0.01` condition in
the metrics module. Confirm both use **ground-truth** fitness (from the synthetic fitness
function), not predictor output, in the synthetic setting. If predictor output is used
anywhere in the synthetic path, switch it to ground truth and re-run affected metrics;
document the semantics in a docstring and in `EXPERIMENTS_STATUS.md`. For the real-protein
datasets, document what the filter keys on (there is no ground truth there).

**Acceptance.** Written confirmation of the filter's input; fix applied if needed.

## E10 — α-amylase enrichment test

**Task.** Using the α-amylase attribution scores already computed (ESM-C, slerp) for each of
the four FLIP2 splits and the 1BAG structure:
1. Get catalytic / ligand-binding residue positions from the independent functional
   annotation (already referenced in the paper).
2. For each split, compute the relationship between `|IG|` (or signed IG) and Cα distance to
   the nearest catalytic residue — report Spearman ρ and a binned mean profile.
3. Permutation null: shuffle attribution-to-position assignment `n = 10,000` times, recompute
   the statistic, report an empirical p-value per split.
4. Report enrichment of annotated functional positions in the top-k `|IG|` (k ∈ {5, 10, 20})
   vs the hypergeometric expectation.

**Outputs.** `E10_amylase/enrichment.csv`; a distance-vs-attribution figure with the
permutation band; an enrichment table across splits.

**Acceptance.** All four splits have a p-value and an enrichment number.

## E11 — β-lactamase literature-annotation enrichment

**Task.** For TEM-1 β-lactamase, assemble a literature/curated set of functional and
stability-critical positions (UniProt features for P62593, the Ω-loop, the catalytic Ser70 /
Lys73 / Ser130 / Glu166 set, known global-suppressor and destabilizing hotspots from
Firnberg et al. 2014 and follow-ups). Compute enrichment of the top-10 IGLOO positions
(Table 9) against this set vs a background model, and report which of the 10 fall in vs
outside annotated regions. Present alongside the existing Boltz-2 / Rosetta results, not
instead of them.

**Outputs.** `E11_blactamase/annotation_enrichment.csv`; an updated version of Table 4 with
an "annotated?" column.

**Acceptance.** All 10 positions classified against the curated set; enrichment statistic
reported.

## E12 — p53 rank diagnostic

**p53 stays in the paper.** This task produces the in-text explanation, not a keep/drop
decision.

**Task.** Explain why p53 predictors reach Pearson r ≈ 0.77–0.88 but Spearman ρ ≈ 0.19
at zero noise (SI Table 8). Report: the fitness value distribution per landscape (histogram,
skew, fraction near WT), per-mutation-tier performance (singles vs doubles vs triples),
and whether a small number of large-effect variants dominate Pearson. Frame the result:
the multi-domain, distributed functional architecture of p53 is part of the failure-mode
story, not a modeling defect — connect it to the R2 / R3 narrative.

**Outputs.** `E12_p53/diagnostic.md` with figures and a drafted paragraph for the paper.

**Acceptance.** Distribution + per-tier breakdown produced; paragraph drafted.

## E13 — Mutation-level MAE metric

**Task.** Add a mutation-level MAE metric: for each mutation `k`, compare normalized `|IG_k|`
to the normalized ground-truth per-mutation cost `c_k` (LOO delta where available, BLOSUM
fallback otherwise); report mean absolute error, per scenario. Do **not** add a global-level
MAE (Shengqing's finding: it is not meaningful at the pooled global scale). Register it under
`metric_role = mutation_level`.

**Outputs.** `E13_mae/metrics_long.csv` with the new metric; a short note on how it compares
to mutation NDCG.

**Acceptance.** Metric computes for all 9 scenarios and integrates into the reporting suite.

---

## Suggested order

1. E0 orientation → `EXPERIMENTS_STATUS.md`. **(done)**
2. **E15 first** — it is the long pole (3 pLMs × 2 proteins × 3 landscapes × 5 methods ×
   7 noise). Queue it now; it runs while the rest proceeds.
3. E9 (filter audit — cheap, unblocks trust in everything else).
4. E1 → E2 → E3 → E4 (the reframe core; E3 depends on E1's random reference). Extend E1/E3
   to the E15 runs as they land.
5. E8, E6, E7, E5 (remaining P1, GB1-first).
6. E10, E11, E12, E13 (analysis, parallelizable).

## P3 (only if compute and time allow — not specced here)

- E14 Manifold IG (Riemannian path) on GB1.
- E16 DeepLIFT / DeepSHAP baseline.
