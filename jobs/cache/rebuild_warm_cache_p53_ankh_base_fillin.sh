#!/bin/bash
# Fill-in retry for jobs/cache/rebuild_warm_cache_p53_backbones.sh's ankh_base
# task (array index 0): that task warms 5 layers sequentially (0/12/24/36/48)
# for 10,410 sequences each and hit the 1:30:00 time limit mid-layer-24.
# The by_seq cache is content-addressed and written incrementally, so
# layers 0/12 (already confirmed "done" in the original log) are skipped
# automatically -- this just re-requests all 5 layers with a longer time
# limit; get_embeddings() will only actually compute what's missing.
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2:30:00
#SBATCH --output=jobs/logs/warm_cache_p53_ankh_base_fillin_%j.out

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

for layer in (0, 12, 24, 36, 48):
    print(f"p53 / ankh_base L{layer}: warming {len(seqs)} unique sequences ...", flush=True)
    get_embeddings(seqs, layer=layer, batch_size=16, esm_model_name="ankh_base")
    print(f"p53 / ankh_base L{layer}: done", flush=True)
PY

echo "=== by_seq store for ankh_base (per layer) ==="
for L in 0 12 24 36 48; do
  echo "L${L}: $(find "data/embeddings/by_seq/ankh_base/L${L}" -name '*.safetensors' 2>/dev/null | wc -l) files"
done
