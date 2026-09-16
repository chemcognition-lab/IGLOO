# E15 Disk Space Estimation

## Embedding File Size Calculation

### pLM Model Dimensions
- **ESM-C 300M**: embedding_dim = 960
- **Ankh-base**: embedding_dim = 768  
- **E1 150M (ESM2 t33)**: embedding_dim = 1280

### Sequence Information (Synthetic Landscapes)

Each synthetic landscape has variants of a single protein:

| Protein | Approx Seq Length | Num Variants (approx) | Notes |
|---------|-------------------|----------------------|-------|
| GB1 | 56 aa | 5,000+ | Already cached |
| T4 lysozyme | 164 aa | 5,000-10,000 | NEW — PHASE 1 |
| p53 | 393 aa | 5,000-10,000 | NEW — PHASE 2 (after E12) |

### Per-File Storage Calculation

**Batch size = 32 sequences (from training code)**

File structure: `{dataset_name}_embeddings_batch_{start}_{end-1}.safetensors`

#### Storage per batch (safetensors format):
- **ESM-C 300M**: 32 seqs × 164 aa × 960 dim × 4 bytes (float32) ≈ **19.5 MB** per batch
- **Ankh-base**: 32 seqs × 164 aa × 768 dim × 4 bytes ≈ **15.6 MB** per batch
- **E1 150M**: 32 seqs × 164 aa × 1280 dim × 4 bytes ≈ **26.0 MB** per batch

*Note: Add ~5-10% overhead for safetensors metadata + key encoding*

---

## PHASE 1: T4 Lysozyme (164 aa)

### Scope per pLM:
- 3 landscapes × 1 protein = 3 datasets
- ~7,500 variants per landscape (typical)
- Batch size = 32

**Batches per landscape**: ⌈7,500 / 32⌉ = 235 batches

**Per pLM breakdown** (ESM-C 300M example):
- 3 landscapes × 235 batches × 19.5 MB = **13.8 GB** per pLM

### Total PHASE 1 Estimate:

| pLM | ESM-C 300M | Ankh-base | E1 150M | Total |
|-----|------------|-----------|---------|-------|
| Storage | 13.8 GB | 11.0 GB | 18.4 GB | **43.2 GB** |

**PHASE 1 Total: ~43 GB** (with 5-10% metadata overhead)

---

## PHASE 2: p53 (393 aa) — *Queued After E12*

### Scope per pLM:
- 3 landscapes × 1 protein = 3 datasets
- ~7,500 variants per landscape (typical)
- Batch size = 32

**Batches per landscape**: ⌈7,500 / 32⌉ = 235 batches

**Per pLM breakdown** (ESM-C 300M example, proportional to sequence length):
- 3 landscapes × 235 batches × (19.5 × 393/164) MB ≈ **48 MB/batch**
- Total: 3 × 235 × 48 ≈ **33.8 GB** per pLM

### Total PHASE 2 Estimate:

| pLM | ESM-C 300M | Ankh-base | E1 150M | Total |
|-----|------------|-----------|---------|-------|
| Storage | 33.8 GB | 27.0 GB | 45.1 GB | **105.9 GB** |

**PHASE 2 Total: ~106 GB** (with 5-10% metadata overhead)

---

## Grand Total Storage Requirements

```
PHASE 1 (T4 lysozyme):        ~43 GB
PHASE 2 (p53):                ~106 GB
---
TOTAL NEW EMBEDDINGS:         ~149 GB

EXISTING (GB1):               ~43 GB (already cached)
---
FULL E15 CACHE (both phases): ~192 GB
```

### Breakdown by pLM (all proteins)

| pLM | PHASE 1 | PHASE 2 | Combined | Notes |
|-----|---------|---------|----------|-------|
| ESM-C 300M | 13.8 GB | 33.8 GB | 47.6 GB | Smallest model |
| Ankh-base | 11.0 GB | 27.0 GB | 38.0 GB | 768-dim embeddings |
| E1 150M | 18.4 GB | 45.1 GB | 63.5 GB | 1280-dim embeddings |
| **TOTAL** | **43.2 GB** | **105.9 GB** | **149.1 GB** | |

---

## Caching Strategy to Minimize Disk

✅ **Good news**: The code reuses cached embeddings (lines 54-75 in embedding.py)

1. **Embeddings cached by**: `data/embeddings/{subdir}/{dataset_name}_embeddings_batch_*.safetensors`
2. **Subdirectories**: Can organize by protein/landscape
   - `data/embeddings/synthetic/T4_lysozyme_additive_embeddings_batch_*.safetensors`
   - `data/embeddings/synthetic/T4_lysozyme_epistatic_embeddings_batch_*.safetensors`
   - etc.

3. **Once computed, embeddings are REUSED for**:
   - All 5 attribution methods
   - All 7 noise levels
   - Different seeds
   - **No re-computation if batch files exist**

---

## Available Disk Space Check Required

**To proceed, we need to check:**

```bash
df -h /project/6111759/ranaab/IGLOO/data/
```

**Minimum requirements**:
- PHASE 1 only: 50 GB available (43 GB + 15% buffer)
- Full E15: 160 GB available (149 GB + 15% buffer)

---

## Risk Mitigation

If disk space is limited:

### Option A: Staggered pLM approach
- Run PHASE 1 with 1 pLM at a time
- Delete embedding batch files after metrics computed
- Recompute if needed (ESM embeddings are deterministic)

### Option B: Compress embeddings
- Store in lower precision (float16 instead of float32)
- Saves ~50% space, minimal accuracy loss
- Requires code change to `embedding.py` (safetensors save format)

### Option C: Remote storage
- Stream embeddings to cloud storage
- Load on-demand (slower, but unlimited space)

---

## Recommendations

1. ✅ Check available disk space first (Step 3)
2. ✅ If ≥160 GB available → Proceed with full E15 as-is
3. ✅ If 50-160 GB available → Run PHASE 1 first, then PHASE 2
4. ✅ If <50 GB available → Use Option A (stagger by pLM)

