#!/bin/bash
# Warm the by_seq embedding cache for p53 at ankh_large (the established
# consistent-winner backbone from the GB1/T4Lysozyme backbone comparison) --
# first step of a pilot to validate the new-backbone pipeline on p53 (393
# residues, notably longer than T4Lysozyme's 164) before committing to the
# full backbone-comparison + layer-sweep replication.
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=jobs/logs/warm_cache_p53_pilot_%j.out

set -u
cd /project/6007171/ranaab/IGLOO || exit 1

module load python/3.12 gcc arrow cuda/12.2
source IGLOO_env/bin/activate

python - <<'PY'
import pandas as pd
from src.igloo.embedding import get_embeddings

WT = ("MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQ"
      "KTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGN"
      "LRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELP"
      "PGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD")
AA = "ACDEFGHIKLMNPQRSTVWY"

seqs = set()
for landscape in ("additive", "epistatic", "nonlinear"):
    seqs |= set(pd.read_csv(f"data/synthetic/p53_{landscape}.csv")["SEQ"])
seqs |= {WT[:i] + m + WT[i+1:] for i in range(len(WT)) for m in AA if m != WT[i]}
seqs.add(WT)
seqs = sorted(seqs)
print(f"p53 / ankh_large: warming {len(seqs)} unique sequences (WT length={len(WT)}) ...", flush=True)
get_embeddings(seqs, layer=33, batch_size=16, esm_model_name="ankh_large")
print("p53 / ankh_large: done", flush=True)
PY

echo "=== by_seq store for ankh_large (all layers) ==="
find "data/embeddings/by_seq/ankh_large" -name '*.safetensors' 2>/dev/null | wc -l
