#!/bin/bash
# Option-B: rerun the full Path-2 analysis suite on the float32 rebuild results.
# CPU only. Backs up the existing (bf16) summary CSVs first so a bf16-vs-fp32
# comparison is possible later.
#
#SBATCH --account=aip-bensl
#SBATCH --job-name=rb_analysis
#SBATCH --output=jobs/logs/rb_analysis_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:15:00

set -u
cd /project/aip-bensl/ranaab/IGLOO || exit 1
module load python/3.12 cuda/12.2
source IGLOO_env/bin/activate
PY=python
R=results/synthetic

echo "=== backing up bf16 summary CSVs ==="
for f in \
  "$R/metric_suite/metric_suite.csv" \
  "$R/metric_suite/metric_discrimination.csv" \
  "$R/factor_ablation_analysis/factor_ablation.csv" \
  "$R/k_predictor_agreement/k_predictor_agreement.csv" \
  "$R/ism_attribution/ism_summary_all.csv" \
  "$R/attrq_vs_modelperf_noise/attrq_vs_modelperf_noise.csv" \
  "$R/cascading_randomization/cascading_summary.csv" ; do
  [[ -f "$f" && ! -f "${f%.csv}.bf16.csv" ]] && cp "$f" "${f%.csv}.bf16.csv" && echo "  backed up $f"
done

run() { echo; echo "########## $* ##########"; "$@" 2>&1 | tail -40; echo "(exit $?)"; }

run $PY scripts/metric_suite.py
run $PY scripts/factor_ablation_analysis.py
run $PY scripts/k_predictor_agreement.py
run $PY scripts/ism_attribution_analysis.py
run $PY scripts/ism_seed_agreement.py
run $PY scripts/cascading_randomization_summary.py
run $PY scripts/analyze_attrq_vs_modelperf_noise.py

echo
echo "=== DONE.  figures in $R/{metric_suite,factor_ablation_analysis,k_predictor_agreement,ism_attribution,ism_seed_agreement,cascading_randomization,attrq_vs_modelperf_noise}/ ==="
