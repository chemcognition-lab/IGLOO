#!/bin/bash
# Option-B rebuild, step 1: purge the stale (bf16 / mixed-precision) GB1 + T4
# synthetic embedding caches, then warm the new content-addressed store with a
# float32 embedding of every unique sequence (variants + single mutants + WT).
#
# p53 (already float32, expensive) and revision/E15 caches are left untouched.

#SBATCH --account=aip-bensl
#SBATCH --job-name=rebuild_warm
#SBATCH --output=jobs/logs/rebuild_warm_%j.out
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=4:00:00

cd /project/aip-bensl/ranaab/IGLOO || exit 1
module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate

echo "=== purging stale GB1/T4 synthetic embedding caches ==="
rm -f data/embeddings/synthetic/GB1_*_integrated_gradients_*.safetensors
rm -f data/embeddings/synthetic/T4Lysozyme_*_integrated_gradients_*.safetensors
rm -f data/embeddings/synthetic/GB1_*_sample_*.safetensors
rm -f data/embeddings/synthetic/T4Lysozyme_*_sample_*.safetensors
rm -f data/embeddings/synthetic/sequence_ig_seq_*.safetensors
rm -f data/embeddings/synthetic/sequence_ig_baseline_*.safetensors
rm -f data/embeddings/sequence_ig_seq_*.safetensors
rm -f data/embeddings/sequence_ig_baseline_*.safetensors
rm -f data/embeddings/ism_GB1_*.safetensors data/embeddings/ism_T4Lysozyme_*.safetensors
echo "purge done"

python - <<'PY'
import pandas as pd
from src.igloo.embedding import get_embeddings

WT = {
 'GB1': 'MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE',
 'T4Lysozyme': ('MNIFEMLRIDEGLRLKIYKDTEGYYHIGIGHLLTKSPSLNAAKSELDKAIGRNCNGVITKDEAEKLFN'
                'QDVDAAVRGILRNAKLKPVYDSLDAVRRCALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKS'
                'RWYNQTPNRAKRVITTFRTGTWDAYKNL'),
}
AA = 'ACDEFGHIKLMNPQRSTVWY'
for P, wt in WT.items():
    seqs = set(pd.read_csv(f'data/synthetic/{P}_additive.csv')['SEQ'])
    seqs |= {wt[:i] + m + wt[i+1:] for i in range(len(wt)) for m in AA if m != wt[i]}
    seqs.add(wt)
    seqs = sorted(seqs)
    print(f'{P}: warming {len(seqs)} unique sequences (float32) ...', flush=True)
    get_embeddings(seqs, layer=33, batch_size=64, esm_model_name='esmc_300m')
    print(f'{P}: done', flush=True)
PY

echo "=== by_seq store ==="
find data/embeddings/by_seq -name '*.safetensors' | wc -l
