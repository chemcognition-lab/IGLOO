#!/bin/bash
# Warm the by_seq embedding cache for ankh_base at each of the 5 layer-sweep
# depths (0, 12, 24, 36, 48 -- deliberately excludes 33: the existing
# data/embeddings/by_seq/ankh_base/L33/ cache is MISLABELED, computed by the
# pre-fix AnkhBackend which ignored the requested layer and always returned
# the true final layer (48) while writing it under the "L33" path. Any
# genuine layer=33 request would silently read back that stale, wrong data
# until that directory is purged/renamed -- so this sweep just avoids 33
# entirely rather than risk it).
#
# Warms the same sequence set as the other cache-warm jobs: all GB1/
# T4Lysozyme dataset sequences (additive/epistatic/nonlinear) + every
# single-mutant variant + WT.
#
# Task list: 5 layers
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --array=0-4
#SBATCH --output=jobs/logs/warm_cache_ankh_layersweep_%A_%a.out

set -u
cd /project/6007171/ranaab/IGLOO || exit 1

LAYERS=(0 12 24 36 48)
LAYER="${LAYERS[$SLURM_ARRAY_TASK_ID]}"
echo ">>> warming ankh_base cache for layer: $LAYER"

module load python/3.12 gcc arrow cuda/12.2
source IGLOO_env/bin/activate

python - "$LAYER" <<'PY'
import sys
import pandas as pd
from src.igloo.embedding import get_embeddings

layer = int(sys.argv[1])

WT = {
 'GB1': 'MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE',
 'T4Lysozyme': ('MNIFEMLRIDEGLRLKIYKDTEGYYHIGIGHLLTKSPSLNAAKSELDKAIGRNCNGVITKDEAEKLFN'
                'QDVDAAVRGILRNAKLKPVYDSLDAVRRCALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKS'
                'RWYNQTPNRAKRVITTFRTGTWDAYKNL'),
}
AA = 'ACDEFGHIKLMNPQRSTVWY'
for P, wt in WT.items():
    seqs = set()
    for landscape in ('additive', 'epistatic', 'nonlinear'):
        seqs |= set(pd.read_csv(f'data/synthetic/{P}_{landscape}.csv')['SEQ'])
    seqs |= {wt[:i] + m + wt[i+1:] for i in range(len(wt)) for m in AA if m != wt[i]}
    seqs.add(wt)
    seqs = sorted(seqs)
    print(f'{P} / ankh_base L{layer}: warming {len(seqs)} unique sequences ...', flush=True)
    get_embeddings(seqs, layer=layer, batch_size=32, esm_model_name='ankh_base')
    print(f'{P} / ankh_base L{layer}: done', flush=True)
PY

echo "=== by_seq store for ankh_base L${LAYER} ==="
find "data/embeddings/by_seq/ankh_base/L${LAYER}" -name '*.safetensors' 2>/dev/null | wc -l
