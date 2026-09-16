# E15 Execution Plan — Cross-pLM Comparison, All Proteins

**Status**: Ready to queue  
**Updated**: 2026-09-03  
**Disk space**: ✅ 1,337 GB available (149 GB needed)

---

## Overview

E15 is the "long pole" experiment that should run in parallel while P0 core work (E1–E4) proceeds.

**Structure**:
- **PHASE 1** (T4 lysozyme): Queue **NOW** (~36–54 GPU hours)
- **PHASE 2** (p53): Queue **after E12** completes (~36–54 GPU hours)
- **Total**: ~72–108 GPU hours over 2–3 weeks

**Result integration**:
- E1 (untrained baseline) extends to E15 runs when complete
- E3 (skill score) extends to E15 runs when complete
- Final Table 3 regenerated with n=9 proteins (not n=3)

---

## Timeline & Milestones

```
Week 1
├─ [X] Step 1-6: Setup complete
├─ [→] E15 PHASE 1: Queue immediately (T4 lysozyme)
├─ [→] E9: Model-collapse filter audit (parallel)
└─ [→] E1: Untrained baseline (parallel)

Week 2
├─ [→] E1 results → extend to E15 PHASE 1 runs
├─ [→] E2: Gradient-free displacement (parallel)
├─ [→] E12: p53 diagnostic (parallel)
└─ [?] E15 PHASE 1: Running (mid-way through)

Week 3
├─ [→] E12 results ready
├─ [→] E15 PHASE 2: Queue p53 runs (after E12)
├─ [→] E3: Skill-score refactor (parallel)
└─ [?] E15 PHASE 1: Complete
└─ [?] E15 PHASE 2: Running

Week 4
├─ [→] E4: Per-variant reporting (parallel)
├─ [?] E15 PHASE 2: Complete
└─ [→] Post-processing: Final Table 3, statistical tests
```

---

## Step-by-Step Execution

### PHASE 1: T4 Lysozyme (Queue Now)

#### 1. Verify Configuration

```bash
# Check PHASE 1 config
cat /project/6111759/ranaab/IGLOO/config/E15_crosspLM_config.yaml | head -30

# Expected output:
# - proteins: [T4_lysozyme]
# - plms: [esmc_300m, ankh_base, esm1b_t33_650M_UR50D]
# - landscapes: [additive, epistatic, nonlinear]
# - methods: [5 total]
# - noise_levels: [0.0, 0.1, ..., 1.0]
```

#### 2. Dry Run (Recommended First)

```bash
cd /project/6111759/ranaab/IGLOO

python run_E15_phase1.py \
  --dry-run \
  --output-dir results/revision/E15_crosspLM

# Expected output:
# - Setup complete
# - Output directory structure created
# - 45 experiments queued (1 protein × 3 landscapes × 3 pLMs × 5 seeds)
# - No actual computation
```

#### 3. Queue Full PHASE 1 (on GPU cluster)

**Option A: Single command (all experiments)**
```bash
python run_E15_phase1.py \
  --output-dir results/revision/E15_crosspLM
```

**Option B: Submit as batch job (recommended for long runs)**
```bash
# Create batch script
cat > run_e15_phase1_batch.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=E15_phase1_T4
#SBATCH --time=100:00:00  # 100 hours (safe upper bound)
#SBATCH --gpus=1
#SBATCH --output=logs/E15_phase1.log

cd /project/6111759/ranaab/IGLOO
python run_E15_phase1.py --output-dir results/revision/E15_crosspLM
EOF

chmod +x run_e15_phase1_batch.sh
sbatch run_e15_phase1_batch.sh
```

**Option C: Parallel pLM jobs (speed up with multiple GPUs)**
```bash
# Queue 3 independent jobs (one per pLM)
for plm in esmc_300m ankh_base esm1b_t33_650M_UR50D; do
  python run_E15_phase1.py \
    --plms $plm \
    --output-dir results/revision/E15_crosspLM &
done
wait
```

#### 4. Monitor Progress

```bash
# Check results in real-time
watch -n 60 'ls -lh results/revision/E15_crosspLM/T4_lysozyme/*/config_seed*.yaml | wc -l'

# Tail logs (if batch job)
tail -f logs/E15_phase1.log

# Check CSV results as they accumulate
wc -l results/revision/E15_crosspLM/metrics_long.csv
head -20 results/revision/E15_crosspLM/metrics_long.csv
```

#### 5. Expected Outputs

After PHASE 1 completes:
```
results/revision/E15_crosspLM/
├── T4_lysozyme/
│   ├── esmc_300m/
│   │   ├── config_seed1.yaml
│   │   ├── config_seed2.yaml
│   │   ├── predictions_seed1/
│   │   └── attributions_seed1/
│   ├── ankh_base/
│   │   └── ...
│   └── esm1b_t33_650M_UR50D/
│       └── ...
└── metrics_long.csv          # Long-format results (all metrics)
```

**metrics_long.csv structure**:
```
protein,landscape,plm,seed,method,noise_level,metric_name,metric_value,...
T4_lysozyme,additive,esmc_300m,1,ig_linear,0.0,ndcg_global,0.642,...
T4_lysozyme,additive,esmc_300m,1,ig_linear,0.1,ndcg_global,0.621,...
...
```

---

### Parallel Work: E1, E9, E12 (While PHASE 1 Runs)

While E15 PHASE 1 is running (2–3 weeks), execute in parallel:

#### E9: Model-Collapse Filter Audit (Low compute)
```bash
# Short task (1–2 hours)
# Check where |f| < δ filter is applied
# Verify ground-truth fitness vs predictor output
python scripts/audit_model_collapse_filter.py
# Output: EXPERIMENTS_STATUS.md updated with filter semantics
```

#### E1: Untrained Baseline (Can run independently)
```bash
# Queue on spare GPU(s) while PHASE 1 runs on main GPU
python run_E1_untrained_baseline.py \
  --proteins GB1,T4_lysozyme \
  --plms esmc_300m,ankh_base \
  --output-dir results/revision/E1_untrained
# Outputs: E1_untrained/metrics_long.csv
```

#### E12: p53 Diagnostic (Analysis only, no GPU)
```bash
# Can run immediately (diagnostic, no new training)
python analyze_E12_p53_diagnostic.py
# Output: E12_p53/diagnostic.md with:
#   - Fitness distributions
#   - Per-tier performance (singles/doubles/triples)
#   - Pearson r vs Spearman ρ explanation
#   - Drafted paragraph for paper
```

---

### Extend E1 & E3 to E15 Runs (As Results Land)

Once E1 is done:

```bash
# Extend E1 results to E15 PHASE 1 runs
python extend_E1_to_E15_phase1.py \
  --e1_results results/revision/E1_untrained/metrics_long.csv \
  --e15_phase1_dir results/revision/E15_crosspLM \
  --output results/revision/E15_crosspLM/metrics_long_with_E1.csv
```

Once E3 is done:

```bash
# Extend E3 (skill score) results to E15 PHASE 1 runs
python extend_E3_to_E15_phase1.py \
  --e3_results results/revision/E3_skillscore/skill_long.csv \
  --e15_phase1_dir results/revision/E15_crosspLM \
  --output results/revision/E15_crosspLM/metrics_long_with_E3.csv
```

---

### PHASE 2: p53 (Queue After E12)

#### 1. E12 Completion Check

```bash
# Verify E12 is done
ls -la results/revision/E12_p53/diagnostic.md

# If exists, proceed; if not, wait for E12
```

#### 2. Review E12 Findings

```bash
# Read E12 diagnostic
cat results/revision/E12_p53/diagnostic.md

# Key questions:
# - Is p53 staying (yes, per revised brief)?
# - Any metric adjustments needed?
# - Per-tier performance notes for discussion?
```

#### 3. Queue PHASE 2

```bash
# Automatic check for E12 (blocks if not complete)
python run_E15_phase2.py \
  --output-dir results/revision/E15_crosspLM

# Or force (if you know E12 is done but file not yet present)
python run_E15_phase2.py \
  --force \
  --output-dir results/revision/E15_crosspLM
```

#### 4. Monitor & Integrate E1/E3

Same as PHASE 1:
```bash
# Extend E1 results to PHASE 2
python extend_E1_to_E15_phase2.py ...

# Extend E3 results to PHASE 2
python extend_E3_to_E15_phase2.py ...
```

---

## Post-Processing (After Both Phases Complete)

Once both PHASE 1 and PHASE 2 metrics are in:

### 1. Merge Results

```bash
python scripts/merge_E15_phases.py \
  --phase1_csv results/revision/E15_crosspLM/metrics_long_with_E3.csv \
  --phase2_csv results/revision/E15_crosspLM/metrics_long_with_E3.csv \
  --output results/revision/E15_crosspLM/metrics_long_FINAL.csv

# Output has columns:
# - protein: {GB1, T4_lysozyme, p53}
# - landscape: {additive, epistatic, nonlinear}
# - plm: {esmc_300m, ankh_base, esm1b_t33_650M_UR50D}
# - All E1, E3, metrics columns
```

### 2. Regenerate Table 3 (n=9 proteins)

```bash
python scripts/table3_all_proteins.py \
  --metrics_csv results/revision/E15_crosspLM/metrics_long_FINAL.csv \
  --output results/revision/E15_crosspLM/table3_all_proteins.txt

# Outputs:
# - Mean composite per (pLM, protein)
# - Effect sizes (Δ trained - random, from E1)
# - Skill scores (from E3)
```

### 3. Statistical Tests

```bash
python scripts/friedman_kendall_all_proteins.py \
  --metrics_csv results/revision/E15_crosspLM/metrics_long_FINAL.csv \
  --output results/revision/E15_crosspLM/stats_n9.txt

# Outputs:
# - Friedman test: χ², p, W (Kendall's coefficient of concordance)
# - Compare to old n=3 results
# - Per-method rankings across all 9 protein×landscape combinations
```

### 4. Figures

```bash
python scripts/plot_E15_results.py \
  --metrics_csv results/revision/E15_crosspLM/metrics_long_FINAL.csv \
  --output_dir results/revision/E15_crosspLM/figures/

# Generates:
# - Composite vs noise (3 subplots: one per protein)
# - pLM comparison (Bar chart: mean composite across landscapes)
# - Method ranking stability (Heatmap: Friedman ranking per protein)
```

---

## Troubleshooting

### Issue: "Not enough GPU memory"

**Solution**: Reduce batch size in predictor_hparams
```python
# Change from 32 to 16
"batch_size": 16,
```

### Issue: "Embeddings computation timed out"

**Solution**: Run embeddings separately, cache them, then run metrics
```bash
python run_E15_phase1.py --compute-embeddings-only
python run_E15_phase1.py --use-cached-embeddings
```

### Issue: "p53 PHASE 2 blocked (E12 not complete)"

**Solution**: Either wait for E12 or use `--force`
```bash
# Wait for E12
watch -n 300 'test -f results/revision/E12_p53/diagnostic.md && echo "E12 done!" || echo "Waiting..."'

# Or force (only if confident E12 will not change results)
python run_E15_phase2.py --force
```

### Issue: "Results not appearing in CSV"

**Solution**: Check logs and verify no errors
```bash
tail -100 logs/E15_phase1.log | grep -i "error"
```

---

## Success Criteria

### PHASE 1 (T4 Lysozyme) Complete When:

- ✅ `results/revision/E15_crosspLM/T4_lysozyme/*/config_seed*.yaml` exists (45 files)
- ✅ `results/revision/E15_crosspLM/metrics_long.csv` has ≥45 rows
- ✅ No errors in logs (check last 10 lines)
- ✅ CSV has columns: protein, landscape, plm, seed, method, noise_level, metrics

### PHASE 2 (p53) Complete When:

- ✅ `results/revision/E15_crosspLM/p53/*/config_seed*.yaml` exists (45 files)
- ✅ Same CSV now has 90+ rows (45 T4 + 45 p53)
- ✅ E1 and E3 extensions integrated
- ✅ No errors in logs

### Final (Post-Processing) Complete When:

- ✅ Table 3 regenerated with n=9 proteins
- ✅ Friedman/Kendall's W recalculated and compared to n=3
- ✅ Figures generated: composite-vs-noise, pLM-comparison, ranking-stability
- ✅ Method ranking is stable across proteins (or instability noted)

---

## Next Actions

1. **NOW**: Run `python run_E15_phase1.py --dry-run` to verify setup
2. **This week**: Queue full PHASE 1 (`python run_E15_phase1.py`)
3. **In parallel**: Start E9, E1, E12
4. **Week 2**: Extend E1 results to E15 PHASE 1
5. **Week 3**: E12 complete → Queue PHASE 2
6. **Week 4**: Post-processing, Table 3, statistical tests

---

## Configuration Files

- Config: `config/E15_crosspLM_config.yaml`
- PHASE 1 script: `run_E15_phase1.py`
- PHASE 2 script: `run_E15_phase2.py`
- Disk estimate: `E15_DISK_SPACE_ESTIMATE.md`

## Related Tasks

- **E1**: Untrained baseline (feeds into E15 analysis)
- **E3**: Skill-score refactor (feeds into E15 analysis)
- **E9**: Model-collapse filter audit (unblocks confidence in metrics)
- **E12**: p53 diagnostic (blocks PHASE 2 until complete)
- **E2, E4**: Other P0 work (can proceed in parallel)

---

**Questions?** Check `EXPERIMENTS_STATUS.md` for task progress tracking.
