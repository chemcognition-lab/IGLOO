#!/bin/bash
# Warm the by_seq embedding cache for the 6 new pLM backbones being added to
# the architecture comparison (esmc_300m's cache already exists from the
# earlier BOS/EOS-fix rebuild -- see rebuild_warm_cache_v3.sh).
#
# ankh/esmc backbones use IGLOO_env (py3.11); e1_* need IGLOO_env_e1 (py3.12,
# separate venv -- E1's pyproject.toml requires python>=3.12).
#
# Warms the same sequence set as rebuild_warm_cache_v3.sh: all GB1/T4Lysozyme
# dataset sequences (additive/epistatic/nonlinear) + every single-mutant
# variant + WT, so results are directly comparable across backbones.
#
# Task list: 6 backbones
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --array=0-5
#SBATCH --output=jobs/logs/warm_cache_newbb_%A_%a.out

set -u
cd /project/6007171/ranaab/IGLOO || exit 1

BACKBONES=(esmc_600m ankh_base ankh_large e1_150m e1_300m e1_600m)
BB="${BACKBONES[$SLURM_ARRAY_TASK_ID]}"
echo ">>> warming cache for backbone: $BB"

case "$BB" in
  e1_*)
    module load python/3.12 cuda/12.2
    source IGLOO_env_e1/bin/activate
    ;;
  *)
    module load python/3.12 gcc arrow cuda/12.2
    source IGLOO_env/bin/activate
    ;;
esac

python - "$BB" <<'PY'
import sys
import pandas as pd
from src.igloo.embedding import get_embeddings

model_name = sys.argv[1]

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
    print(f'{P} / {model_name}: warming {len(seqs)} unique sequences ...', flush=True)
    get_embeddings(seqs, layer=33, batch_size=32, esm_model_name=model_name)
    print(f'{P} / {model_name}: done', flush=True)
PY

echo "=== by_seq store for $BB ==="
find "data/embeddings/by_seq/${BB}" -name '*.safetensors' 2>/dev/null | wc -l
