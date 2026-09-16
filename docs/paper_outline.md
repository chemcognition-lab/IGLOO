---
title: "Paper Outline Draft"
subtitle: "Working title candidates inside — IGLOO / Path-2 investigation"
date: "2026-09-15"
geometry: margin=1in
fontsize: 11pt
---

# Positioning: what's already published vs. what's ours

A literature pass turned up close cousins — good news, it means the niche is
active and citable; the gap they leave is exactly where this paper sits.

**Closest existing work:**

- **"Residue-Level Attributions in Protein Language Models Do Not Recover
  Allergen Epitopes"** (arXiv 2606.22181, 2026) — same *symptom* (pLM
  attribution doesn't recover known-important residues), but on real
  biological annotations (noisy, incomplete ground truth) and *without* a
  mechanistic diagnosis or a fix. This is the paper reviewers will ask "how is
  yours different from this."
- **"Towards the Explainability of Protein Language Models"** (*Nature Machine
  Intelligence*, arXiv 2506.19532) — broad review/framework; good
  context-setting citation, not a competitor.
- General finding echoed across several 2026 papers: *"attention weights are
  not necessarily faithful causal explanations,"* and a documented
  *"dissociation between predictive performance, model faithfulness, and
  [domain] faithfulness"* in pLM classifiers — this is the same phenomenon we
  found, described elsewhere without the mechanism. Good external corroboration
  that we're not looking at a fluke of our synthetic setup.
- **Adebayo et al., "Sanity Checks for Saliency Maps"** (NeurIPS 2018) — cascading
  / model-parameter randomization test, image-classification only as far as
  the literature shows. We are the ones extending it to a pLM+regression-head
  setting — cite explicitly, frame cascading randomization as "we apply
  Adebayo et al.'s test to a new domain," not as a novel test.
- **Genomics ISM/gradient-correction literature** ("Correcting gradient-based
  interpretations of deep neural networks for genomics," *Genome Biology*;
  fastISM; compressed-sensing ISM) — establishes ISM as a standard,
  respected comparator (not something we invented) and documents the same
  gradient-saturation family of concerns in a sibling domain (regulatory
  genomics DNNs). Strong parallel-domain citation.
- Sundararajan et al. 2017 (IG), Shrikumar et al. 2017 (DeepLIFT / grad×input)
  — foundational method citations, standard.

**What's actually new here, that the closest cousin (the epitope paper) doesn't have:**

1. A **synthetic benchmark with exact, generator-defined ground truth** — lets
   us measure faithfulness directly instead of against noisy real annotations.
2. A **mechanistic explanation of *why*** attribution fails — the chain-rule
   argument plus three independent verifications (mechanism check, cascading
   randomization, factor ablation) that converge on one cause.
3. A **validated architectural fix** that simultaneously improves accuracy and
   attribution quality — not just a diagnosis, a prescription.
4. A **metric-reliability audit** — which evaluation metrics are trustworthy
   for this kind of study, independently useful to anyone building a similar
   benchmark.

---

# Working titles (pick one, or riff)

1. *"Shallow Bottlenecks, Blind Gradients: Why Integrated Gradients Misattributes Protein Fitness Predictions — and How to Fix It"*
2. *"When the Pool is the Problem: Diagnosing and Repairing Attention-Pooling Bottlenecks in pLM-Based Attribution"*
3. *"Does Your Attention Head Even See the Mutation? A Mechanistic Audit of Integrated Gradients for Protein Fitness Models"*
4. *"Faithful by Design: An Architectural Fix for Gradient Attribution on Frozen Protein Language Model Embeddings"* ← leans into the "fix," probably the strongest for a methods-focused venue

---

# Abstract (draft)

> Integrated Gradients (IG) is the default tool for asking a trained protein
> fitness model *which residues matter*, but whether its answer reflects what
> the model actually learned — rather than the geometry of the frozen protein
> language model (pLM) embedding it was computed on — has not been
> systematically tested. Using synthetic fitness landscapes with exact,
> generator-defined ground-truth residue importance, we show that IG on the
> standard architecture (ESM-C embedding → shallow attention pool → deep MLP)
> correlates only weakly and inconsistently with model quality. We trace this
> to a structural cause: because every learnable nonlinearity sits *after* the
> attention pool, the chain rule forces `∂F/∂xᵢ ≈ attnᵢ · g` — a single,
> position-independent vector `g` scaled by a shallow, under-constrained
> attention weight — making the deep network's learned function invisible to
> the per-position gradient. We confirm this mechanism three independent ways
> (direct verification, Adebayo-style cascading parameter randomization, and
> factor ablation of the IG product), and show a simple architectural fix —
> moving the deep MLP to run per-residue, *before* pooling — simultaneously
> improves predictive accuracy (Spearman ρ 0.68→0.78) and restores IG's
> ability to track model quality (discrimination r +0.41→+0.84), corroborated
> independently by a gradient-free perturbation baseline (ISM). We further
> show which of nine common attribution-evaluation metrics are actually
> trustworthy for this kind of audit, and two architectural changes that
> *don't* work (a vanishing-gradient failure and a capacity/data mismatch),
> clarifying that the fix is specific, not incidental.

*(Numbers above are from the pilot-scale runs; update once the full 9-level
sweep — currently rebuilding — completes.)*

---

# Section-by-section outline

## 1. Introduction

- Hook: protein engineers use fitness models not just for prediction but to
  *learn which residues matter* — active-site redesign, stability engineering,
  directed evolution prioritization. IG is the default tool for this.
- The question nobody had ground truth to answer: does IG's answer reflect the
  model, or the frozen pLM embedding underneath it?
- One-sentence preview of the finding + the fix.
- **Generalization hook** (put this in the intro, not just the discussion):
  *frozen-encoder + shallow-pool + trainable head* is now the default recipe
  far beyond protein ML — CLIP linear probes, LoRA-adapted heads, DNA-LM
  regressors. The mechanism here is a property of that architecture shape, not
  a protein-specific quirk.

## 2. Background / Related Work

- IG (Sundararajan et al. 2017), axioms (Sensitivity, Completeness).
- DeepLIFT / grad×input (Shrikumar et al. 2017).
- Adebayo et al. 2018 sanity checks — cite explicitly as the source of the
  cascading-randomization test; note it has not (to our knowledge) been
  applied to a pLM+regression setting.
- pLM interpretability: the epitope-recovery paper, the NMI explainability
  review, "attention is not [not] explanation" debate (Jain & Wallace /
  Wiegreffe & Pinter) — note ours is a gradient-based, not raw-attention,
  version of that concern.
- Genomics parallel: ISM as an established comparator (fastISM,
  compressed-sensing ISM), gradient-correction-for-saturation literature
  (Genome Biology) — same family of concerns in a sibling domain.

## 3. Synthetic benchmark

- GB1 (56 aa) and T4 Lysozyme (164 aa); additive / epistatic / nonlinear
  fitness landscapes with BLOSUM-weighted, category-annotated ground-truth
  importance (`binding_site` / `structural` / `folding` / `active_site` /
  `metal_binding`, etc.).
- Why synthetic: exact ground truth, no confound between "biologically
  correct" and "recovers the generator's rule."
- Label-noise sweep as the tool for producing models of controlled, known
  quality: `y := renorm((1-p)·y_true + p·y_perm)`, p ∈ [0,1] scale-free
  (0 = exact labels, 1 = fully random) — cleaner than an absolute-σ Gaussian
  sweep (report the std-vs-blend comparison as a methods contribution;
  absolute σ conflates noise level with each landscape's own label scale and
  never actually reaches ρ≈0).

## 4. Baseline result: IG is a weak, inconsistent quality signal

- Report the original (bf16-era) finding, contrasted with the corrected
  finding once labels actually spanned the noise range (this progression is
  itself worth narrating briefly — it's a good methods lesson: too narrow a
  noise sweep manufactures a false negative).
- Figure: model test ρ (x) vs. IG spearman-to-ground-truth (y), colored by
  noise, baseline architecture only.

## 5. Mechanism

- Derive `∂F/∂xᵢ ≈ attnᵢ · g` from the architecture (`ESMAttention1d`).
- **Verification 1 — direct check**: pearson(attnᵢ, mean‖IGᵢ‖) = 0.84–0.99;
  cosine dispersion of `g` across variants ≈ 0.002–0.07 (≈ one fixed vector).
- **Verification 2 — cascading randomization** (Adebayo-style): randomizing
  `final→fc4→fc3→fc2→fc1→layernorm` leaves attribution correlation at
  0.85–1.0; only randomizing `attention1d` collapses it.
- **Verification 3 — factor ablation**: decompose the IG product into
  `gradient` (model-only) and `displacement` (pLM-geometry-only, ρ≈0.12 flat
  across every condition) factors; `gradient` alone ≈ the attention pattern.
- One figure per verification; these three converging independently is the
  strongest evidence in the paper — say so explicitly.

## 6. The fix: move depth before the pool

- `perres_mean` / `perres_attn` (entire MLP runs per-residue on the raw
  embedding, aggregated last by mean / learned attention) vs. `IG_premlp`
  (smaller residual per-residue layer before the existing pool).
- Result: accuracy **and** discrimination both improve
  (ρ 0.68→0.72–0.78; discrimination r +0.41→+0.84 for `perres_mean`, the
  strongest of any architecture tested).
- Mechanistic explanation for *why* it works: mean-pool has a uniform 1/L
  gradient, so 100% of the per-position IG signal is forced through the
  trained per-position MLP, not a shallow attention scalar.

## 7. It's not "any architecture change helps" — two negative results

- `channel_attn`: attention pools the channel axis instead of length — a
  **vanishing-gradient failure**, not a subtler version of the same problem.
  Show the raw-magnitude comparison (attributions ~10,000× smaller than
  baseline) and the exactly-identical noise-0-vs-noise-1 metric distributions.
  Root cause: two stacked softmax bottlenecks + collapsing every position to
  one scalar before any MLP runs.
- `crossattn`: adding real cross-position self-attention *hurts* accuracy at
  this data scale (0.48 vs. 0.68 baseline) — capacity/data mismatch, and its
  discrimination on hit-rate metrics is the weakest of the working
  architectures. Frame as: more expressive ≠ better, at this training-set
  size.
- Together these show the fix is *specific* (remove the post-pool bottleneck)
  rather than "architecture surgery helps in general."

## 8. Corroboration: a gradient-free method agrees

- ISM (in-silico mutagenesis) — no gradient, no embedding-space assumption,
  established in the genomics literature as a faithful (if expensive)
  comparator.
- ISM independently shows the same model-quality-tracking pattern
  (spearman r = 0.64–0.87 depending on landscape) — rules out "this is an
  IG-specific artifact" as an alternative explanation.
- Report the honest caveat found along the way: ISM's *sign*-based metric
  disagreed in direction from every gradient method (r = −0.44 on
  `sign_acc`) — plausibly a sign-convention mismatch (raw prediction delta
  vs. path-integrated gradient dot product), flagged as future work to
  resolve rather than papered over.

## 9. Which evaluation metrics can you trust?

- Nine metrics audited (spearman, auroc, precision@k, recall_topcat,
  topcat_rank, enrichment, category_mono, epi_ratio, sign_acc) against the
  full noise sweep, across every architecture.
- `spearman` and `sign_acc`: reliable, positive discrimination almost
  everywhere.
- `precision@k` / `recall_topcat`: real signal, but protein-prevalence-dependent
  chance floors (GB1's 77% annotation density vs. T4's 4%) make raw values
  incomparable across proteins — always skill-normalize.
- `category_mono`: indistinguishable from noise on average.
- `epi_ratio`: actively misleading (average discrimination flips sign once
  skill-normalized) *and* structurally undefined for densely-annotated
  proteins like GB1 (the "non-contact annotated" reference group is empty) —
  worth a explicit worked derivation of why, it's a nice self-contained
  methods point.
- Practical contribution: a reusable checklist for anyone building a similar
  ground-truth benchmark.

## 10. A striking random-baseline result

- Under full label noise (p=1), `sign_acc` doesn't fall to chance (0.5) — it
  drops to ≈0.19–0.32 (pooled), *below* chance: a model trained on pure noise
  doesn't just lose the correct sign, it systematically prefers the wrong one.
  Good, quotable result for the discussion of what "learned nothing" looks
  like at the attribution level.

## 11. Discussion

- Restate the generalization claim with the specific chain-rule condition
  spelled out: *any* architecture where 100% of the trainable nonlinear depth
  sits after a pooling/aggregation step will have this problem, regardless of
  domain.
- Practical recommendations for practitioners (ranked, from the "what would
  improve the attributions" analysis):
  1. Architecture: move depth before pooling.
  2. Cross-check gradient attribution against a perturbation method.
  3. Metric choice: spearman / sign_acc first; skill-normalize anything with a
     prevalence-dependent floor.
  4. Sanity-check any new architecture with cascading randomization before
     trusting its attributions.
- Limitations (see gaps below) stated plainly, not buried.

## 12. Limitations & Future Work

- **Everything is on synthetic data** — the single biggest limitation; state
  it early and directly.
- Token-level IG (interpolating through the pLM's own weights, not just the
  small head) is mathematically informative only for multi-mutants — derive
  briefly, point to it as a partially-explored future direction.
- Causal mediation / activation patching through the pLM as the correct tool
  for testing whether the model-independent "displacement" signal (ρ≈0.12) has
  exploitable structure — not yet built, name it as the clear next step.
- ISM sign-convention discrepancy unresolved.

---

# Figures checklist (mapped to what already exists)

| # | Figure | Source |
|---|---|---|
| 1 | Model quality vs. IG quality, baseline arch | `results/synthetic/metric_suite/metric_spearman.png` |
| 2 | Mechanism verification scatter | `results/synthetic/attribution_mechanism/` |
| 3 | Cascading randomization curves | `results/synthetic/cascading_randomization/` |
| 4 | Factor ablation decomposition | `results/synthetic/factor_ablation_analysis/` |
| 5 | Architecture leaderboard (accuracy) | "The Attribution Trial" artifact, Q1 |
| 6 | Discrimination heatmap, all architectures | "The Attribution Trial" artifact, Q2/Q3 |
| 7 | `channel_attn` degeneracy illustration | "The Attribution Trial" artifact, Q3 |
| 8 | ISM vs. IG agreement | `results/synthetic/ism_attribution/` |
| 9 | Metric usefulness ranking | "The Attribution Trial" artifact, Q5 |
| 10 | sign_acc collapse below chance at full noise | "The Attribution Trial" artifact, Q4 |

**Still needed:** one qualitative vignette (a specific mutation, baseline vs.
fixed architecture, tied to a named annotated site) — none of the current
figures show a single concrete example, only aggregates.

---

# Gaps before this is submission-ready

1. **Real (non-synthetic) validation case** — reviewers will not accept
   synthetic-only; even one small real dataset with literature-known
   catalytic/binding residues as a sanity check matters a lot for credibility.
2. **Finish the full 9-level noise sweep** on the winning architectures
   (currently rebuilding after the BOS/EOS embedding fix) — headline numbers
   should come from the full sweep, not the 2-point pilot.
3. **Dose-response ablation**: does one per-residue layer suffice, or does
   more depth help further? Currently one data point per architecture.
4. **The qualitative vignette** (see figures checklist).
5. Resolve or at least characterize the ISM sign-convention discrepancy before
   presenting it as "corroboration" — right now it's corroboration-with-an-asterisk.
6. Decide on venue (see below) — affects framing emphasis and length.

---

# Possible venues

- **ISMB / Bioinformatics / PLOS Computational Biology** — biology-forward
  framing (Sections 1–3, 6, 9 lead); mechanism section can be more compact.
- **NeurIPS / ICML interpretability workshop, or the main track as an
  application paper** — mechanism-forward framing (Section 5 leads); the
  generalization-beyond-protein argument needs to be earned with at least one
  non-protein toy example if aiming here.
- **A short technical note to EvolutionaryScale / an ESM-C GitHub issue**,
  separate from the paper — the BOS/EOS special-token bug in
  `ESMCBackend`-style high-level embedding APIs is a distinct, self-contained,
  practically useful finding (anyone using ESM-C's `.logits(..., return_embeddings=True)`
  for per-residue analysis silently gets an off-by-one). Worth reporting
  independently regardless of the paper's timeline — it's not really "our"
  finding to sit on if it affects other users right now.
