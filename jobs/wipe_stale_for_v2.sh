#!/bin/bash
# Wipe the synthetic rebuild outputs before the v2 (blend-noise) rebuild.
#
# WHY: the previous rebuild used gaussian label noise on the grid
# {0, .05, .1, .3, .5, .8, 1}.  v2 uses BLEND noise on {0, .05, .1, .2, .35,
# .5, .7, .85, 1}.  Every non-zero level now means something different, and the
# analysis scripts glob `noise_*` — so stale dirs (esp. noise_0.3 / noise_0.8,
# which v2 never overwrites) would be silently mixed into the new curves.
#
# Run this from a login node BEFORE submitting run_train_array_v2.sh.
# It only touches results/synthetic/. Models + attributions are all regenerated
# by the v2 arrays (force_retrain=true).

set -u
cd /project/aip-bensl/ranaab/IGLOO || exit 1
R=results/synthetic

echo "=== about to remove (counts) ==="
for d in \
  "$R"/GB1_additive/noise_* "$R"/GB1_epistatic/noise_* "$R"/GB1_nonlinear/noise_* \
  "$R"/T4Lysozyme_additive/noise_* "$R"/T4Lysozyme_epistatic/noise_* "$R"/T4Lysozyme_nonlinear/noise_* \
  "$R"/seedsweep "$R"/premlp "$R"/factor_ablation \
  "$R"/ism_attribution "$R"/attribution_mechanism "$R"/attribution_mechanism_premlp \
  "$R"/cascading_randomization "$R"/cascading_randomization_premlp ; do
  [[ -e "$d" ]] && printf '  %8s  %s\n' "$(find "$d" -type f 2>/dev/null | wc -l)" "$d"
done

read -rp $'\ntype  WIPE  to proceed: ' ans
[[ "$ans" == "WIPE" ]] || { echo "aborted"; exit 1; }

for P in GB1 T4Lysozyme; do for D in additive epistatic nonlinear; do
  rm -rf "$R/${P}_${D}"/noise_*
done; done
rm -rf "$R/seedsweep" "$R/premlp" "$R/factor_ablation" \
       "$R/ism_attribution" "$R/attribution_mechanism" "$R/attribution_mechanism_premlp" \
       "$R/cascading_randomization" "$R/cascading_randomization_premlp"

# NOTE: the summary dirs ($R/metric_suite, factor_ablation_analysis, ...) are left
# in place — their CSVs are fully overwritten by run_analysis.sh, and the
# *.bf16.csv backups there must be preserved for the bf16-vs-fp32 comparison.

echo "done — now: sbatch jobs/run_train_array_v2.sh"
