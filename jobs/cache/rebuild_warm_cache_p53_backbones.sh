#!/bin/bash
# Warm the by_seq embedding cache for p53's top-3-backbone comparison
# (ankh_base, ankh_large, esmc_600m -- the top 3 by TOPSIS score from the
# GB1/T4Lysozyme comparison) plus the ankh_base layer sweep.
#
# IMPORTANT -- final-layer convention for Ankh backbones:
# AnkhBackend now genuinely honors `layer` (fixed earlier this session for
# the ankh_base layer-sweep experiment). The GB1/T4Lysozyme backbone
# comparison's ankh_base/ankh_large results were generated *before* that
# fix, when the backend silently ignored `layer` and always returned the
# true final layer (48) -- so those results are final-layer data despite
# being filed under a misleadingly-named "L33" cache path (33 was the
# shared config default, an ESM2-era value never meant for Ankh). Now that
# the backend is fixed, requesting layer=33 for Ankh genuinely extracts an
# INTERMEDIATE layer, not final -- so this script explicitly warms Ankh at
# layer=48 (the real final layer for both ankh_base and ankh_large, per
# their HF configs: num_layers=48) to preserve a true final-layer
# comparison with the existing GB1/T4L data.
#
# esmc_600m is UNCHANGED (still layer=33): ESMCBackend was never fixed --
# it still silently ignores `layer` and always returns the true final
# layer, so its existing "L33"-labeled cache is (like the old Ankh one)
# harmlessly mislabeled-but-correct. Keeping the same call here reuses that
# convention rather than creating a redundant new cache path.
#
# ankh_base also gets layers 0/12/24/36/48 (not just 48) to cover the
# ankh_base/MeanPool intermediate-layer sweep for p53, mirroring
# rebuild_warm_cache_ankh_layersweep.sh's GB1/T4L version (deliberately
# excludes 33 -- see that script's header for the separate ankh_base/L33
# mislabeled-cache landmine, sidestepped by never requesting 33 anywhere
# in new work).
#
# Task list: 3 (ankh_base 5-layer warm, ankh_large L48 warm, esmc_600m L33 warm)
#
#SBATCH --account=rrg-mahadeva_gpu
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1:30:00
#SBATCH --array=0-2
#SBATCH --output=jobs/logs/warm_cache_p53_backbones_%A_%a.out

set -u
cd /project/6007171/ranaab/IGLOO || exit 1

module load python/3.12 gcc arrow cuda/12.2
source IGLOO_env/bin/activate

python - "$SLURM_ARRAY_TASK_ID" <<'PY'
import sys
import pandas as pd
from src.igloo.embedding import get_embeddings

task_id = int(sys.argv[1])

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

if task_id == 0:
    for layer in (0, 12, 24, 36, 48):
        print(f"p53 / ankh_base L{layer}: warming {len(seqs)} unique sequences ...", flush=True)
        get_embeddings(seqs, layer=layer, batch_size=16, esm_model_name="ankh_base")
        print(f"p53 / ankh_base L{layer}: done", flush=True)
elif task_id == 1:
    print(f"p53 / ankh_large L48 (true final layer): warming {len(seqs)} unique sequences ...", flush=True)
    get_embeddings(seqs, layer=48, batch_size=16, esm_model_name="ankh_large")
    print("p53 / ankh_large L48: done", flush=True)
elif task_id == 2:
    print(f"p53 / esmc_600m L33 (final layer, backend bug still active): warming {len(seqs)} unique sequences ...", flush=True)
    get_embeddings(seqs, layer=33, batch_size=16, esm_model_name="esmc_600m")
    print("p53 / esmc_600m L33: done", flush=True)
PY

echo "=== by_seq store sizes ==="
for bb in ankh_base ankh_large esmc_600m; do
  echo "-- $bb --"
  find "data/embeddings/by_seq/${bb}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | while read -r d; do
    echo "  $(basename "$d"): $(find "$d" -name '*.safetensors' | wc -l) files"
  done
done
