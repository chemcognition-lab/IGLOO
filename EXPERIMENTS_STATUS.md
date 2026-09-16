# IGLOO Experiments Status

Tracking progress on the IGLOO revision experiments as outlined in `docs/skills/IGLOO_experiments_brief.md`.

**Last Updated**: With revised brief (2025-09-03)  
**Key Changes from Original Brief**:
- ⭐ **E15 moved to P1** and marked as "long pole" — queue immediately for parallel execution
- ⭐ **E12 reframed**: p53 *stays in paper* with explanation (not a drop decision)
- Updated execution order prioritizes queuing E15 first

---

## E0 — Repo Orientation

### Key Paths Identified

#### 1. Synthetic Landscape Generation
- **Location**: `src/igloo/` (synthetic data generation & fitness functions)
- **Key files to locate**: 
  - Protein dataset definitions (GB1, T4 lysozyme, p53)
  - Fitness function implementations (additive, epistatic, nonlinear)
  - Ground-truth importance vector `w` and category annotations serialization
  - **Status**: ⏳ Need to locate exact fitness function module and ground-truth JSON format

#### 2. Predictor Model + Training
- **Location**: `src/igloo/train_and_eval.py` (training loop and evaluation)
- **Key components**:
  - `train_model_with_embeddings()` - Main training function with hyperparameter configs
  - `_create_epoch_plot()` - Checkpoint and visualization
  - Model architectures in `src/igloo/models.py` (MeanPoolingModel, AttentionConvPoolingModel, LinearRegressionModel, ESMAttention1d)
  - Checkpoint saving/loading with normalization params
  - **Status**: ✅ Identified; uses AdamW optimizer with cosine annealing, Huber/MSE/MAE loss

#### 3. Attribution Runner
- **Location**: `src/igloo/` (IG and gradient-based attribution methods)
- **Key components to identify**:
  - Linear / slerp / nearest-neighbor IG implementations
  - Gradient×input and input difference methods
  - Per-position embedding extraction from frozen pLM (layer index, pooling)
  - **Status**: ⏳ Need to locate exact attribution module

#### 4. Metrics Module
- **Location**: `src/igloo/` (metrics computation)
- **Key components to identify**:
  - Seven metrics (global NDCG, AP@R, AUROC; sign accuracy; per-variant NDCG; mutation NDCG; category separation)
  - Model-collapse filter (`|f| < δ`, default δ = 0.01)
  - Reliability mask
  - Composite, trapezoidal-AUC, and Friedman pipeline (SI A.6)
  - **Status**: ⏳ Need to locate exact metrics module

#### 5. Config System
- **Location**: `src/igloo/` or YAML config files
- **Key aspects**:
  - Runtime specification: `(protein, landscape, noise, method, pLM)`
  - Config loading and parameter passing
  - Results directory structure: `results/revision/E<N>_<shortname>/`
  - **Status**: ⏳ Need to identify exact config files and entry points

---

## Experiment Checklist

### P0 — Core to the Reframe

- [ ] **E1 — Untrained-predictor baseline**
  - Add `--predictor-init {trained, random}` flag
  - Run full attribution suite for random init (9 scenarios, 5 methods, 7 noise levels, ≥5 seeds)
  - Cross-pLM check (Ankh-base, ESM-C 150M on GB1)
  
- [ ] **E2 — Gradient-free displacement ranking**
  - Compute per-position displacement `d_i = ||e1_i − e0_i||_2`
  - Global metrics on displacement-only vector
  - Spearman/Kendall τ rank correlation

- [ ] **E3 — Skill-score metric refactor**
  - Compute `m_rand` and `m_perfect` for each metric
  - Define `skill(m) = (m − m_rand) / (m_perfect − m_rand)`
  - Recompute composite on skill scores
  
- [ ] **E4 — Per-variant-first reporting**
  - Add `metric_role` field to metric registry
  - Restructure results grouping by role
  - Update presentation (embedding_probe clearly delineated)

### P1 — Reviewer-Critical (GB1-first except E15)

- [ ] **E15 — Cross-pLM comparison, all proteins (LONG POLE — TWO PHASES)**
  - ⭐ **PHASE 1 (QUEUE NOW)**: T4 lysozyme × 3 pLMs × 3 landscapes = 9 combos (~36-54 GPU hrs)
    - All 5 methods × 7 noise levels per combo
    - Use consistent hyperparameters from existing GB1 cross-pLM runs
    - Output intermediate Table 3 (n=6: GB1 + T4)
  - ⏳ **PHASE 2 (QUEUE AFTER E12)**: p53 × 3 pLMs × 3 landscapes = 9 combos
    - Queue p53 runs only after E12 (p53 diagnostic) findings
    - Adjust settings if needed based on Pearson r / Spearman ρ findings
    - Output final Table 3 (n=9: GB1 + T4 + p53)
  - Include E1 untrained-baseline and E3 skill scores for all new runs as they complete
  
- [ ] **E5 — Occlusion / in-silico mutagenesis baseline**
  - Embedding-space occlusion: `Δf_i = f(e_var) − f(e_var with i reverted)`
  - Optional sequence-space ISM
  
- [ ] **E6 — IG baseline sensitivity**
  - Implement `--ig-baseline {wt, mean, zero, random}`
  - Run GB1 × 3 landscapes × 5 methods × all noise
  
- [ ] **E7 — Layer sweep**
  - ESM-C 300M layers `{6, 12, 18, 24, 33}`
  - GB1 × 3 landscapes × {IG-linear, gradient×input}
  - Retrain per layer
  
- [ ] **E8 — IG convergence / saturation**
  - Step sweep `S ∈ {16, 32, 64, 128, 256, 500}`
  - Completeness gap tracking for all 5 methods

### P2 — Analysis (Low/No Compute)

- [ ] **E9 — Model-collapse filter audit**
  - Trace `|f| < δ` filter usage
  - Confirm ground-truth vs predictor output
  
- [ ] **E10 — α-amylase enrichment test**
  - Distance to catalytic residues vs `|IG|`
  - Permutation p-value (n=10,000)
  - Top-k enrichment analysis
  
- [ ] **E11 — β-lactamase literature-annotation enrichment**
  - Curate functional positions set
  - Top-10 position enrichment vs background
  
- [ ] **E12 — p53 rank diagnostic (p53 STAYS; provide explanation)**
  - ⭐ **NOT a drop decision** — p53 remains in paper with framing
  - Explain Pearson r ≈ 0.77–0.88 vs Spearman ρ ≈ 0.19 at zero noise
  - Per-landscape fitness distribution (histogram, skew, fraction near WT)
  - Per-tier performance (singles vs doubles vs triples)
  - Connect to multi-domain architecture narrative (R2/R3 discussion)
  - Draft paragraph for paper methods/discussion
  
- [ ] **E13 — Mutation-level MAE metric**
  - Add MAE metric against per-mutation costs
  - Register under `metric_role = mutation_level`

---

## Suggested Execution Order (UPDATED)

1. ✅ **E0** — Repo orientation (marked done)
2. 🚀 **E15** — Cross-pLM comparison, all proteins **(QUEUE IMMEDIATELY — long pole)**
   - Run in parallel while other tasks proceed
   - Compute for {T4 lysozyme, p53} × {ESM-C, Ankh, E1} (GB1 already done)
3. ⏳ **E9** — Model-collapse filter audit (cheap, unblocks trust in everything else)
4. ⏳ **E1** → **E2** → **E3** → **E4** (reframe core; E3 depends on E1)
   - **Extend E1 and E3 to E15 runs** as they complete
5. ⏳ **E8**, **E6**, **E7**, **E5** (remaining P1, GB1-first)
6. ⏳ **E10**, **E11**, **E12**, **E13** (analysis, parallelizable)

---

## Disk Space Status (2026-09-03)

✅ **Available**: 1,337 GB in `/project/aip-bensl/`  
📊 **E15 Requirements**: 149 GB (43 GB PHASE 1 + 106 GB PHASE 2)  
✅ **Verdict**: ABUNDANT - Both phases can queue immediately

## E15 Setup Complete! ✅

All 6 queuing steps finished:

1. ✅ E15 job configuration created (`config/E15_crosspLM_config.yaml`)
2. ✅ Disk space estimated (149 GB needed, 1,337 GB available)
3. ✅ Disk space verified (abundant headroom)
4. ✅ Job submission script PHASE 1 created (`run_E15_phase1.py`)
5. ✅ Job submission script PHASE 2 created (`run_E15_phase2.py`)
6. ✅ Final execution plan document created (`E15_EXECUTION_PLAN.md`)

## Ready to Execute E15

**NEXT ACTION**: 
```bash
# Verify setup
python run_E15_phase1.py --dry-run

# Then queue PHASE 1
python run_E15_phase1.py --output-dir results/revision/E15_crosspLM
```

See `E15_EXECUTION_PLAN.md` for full instructions.

---

## Notes

- All experiments require reproducibility: fixed random seeds (numpy, torch, CUDA)
- Results format: Long-format CSV (one row per `(protein, landscape, noise, method, metric, seed, value)`)
- Reuse cached pLM embeddings (expensive artifact)
- New modes added behind flags/config keys (no breaking changes to defaults)

