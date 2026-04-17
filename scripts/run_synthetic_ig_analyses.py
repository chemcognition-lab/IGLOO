#!/usr/bin/env python3
"""
Batch runner for integrated gradients analysis on synthetic protein fitness datasets.

Runs all combinations of:
  - Proteins:       GB1, T4Lysozyme, p53
  - Fitness models: additive, epistatic, nonlinear
  - IG methods:     embedding, token
  - Attribution:    magnitude, signed

Base configs live in configs/{protein}_{fitness_model}_config.yaml and default to
embedding-magnitude. This script overrides the IG settings on the fly for each
combination and saves results to results/synthetic/{protein}_{fitness_model}_{method}_{attribution}/.

Usage:
    # Run all 36 combinations
    python scripts/run_synthetic_ig_analyses.py

    # Run only GB1 analyses
    python scripts/run_synthetic_ig_analyses.py --protein GB1

    # Run only additive fitness model
    python scripts/run_synthetic_ig_analyses.py --fitness-model additive

    # Run only embedding-magnitude (use base configs directly)
    python scripts/run_synthetic_ig_analyses.py --method embedding --attribution magnitude

    # Dry run to preview
    python scripts/run_synthetic_ig_analyses.py --dry-run

    # Single combination
    python scripts/run_synthetic_ig_analyses.py --protein GB1 --fitness-model epistatic \\
        --method token --attribution signed
"""

import argparse
import os
import subprocess
import sys
import tempfile
from datetime import datetime

import yaml

PROTEINS = ["GB1", "T4Lysozyme", "p53"]
FITNESS_MODELS = ["additive", "epistatic", "nonlinear"]
IG_METHODS = ["embedding", "token"]
ATTRIBUTIONS = ["magnitude", "signed"]


def get_all_combinations(protein_filter=None, fitness_model_filter=None,
                         method_filter=None, attribution_filter=None):
    """Return all (protein, fitness_model, method, attribution) combos matching filters."""
    combos = []
    for protein in PROTEINS:
        if protein_filter and protein != protein_filter:
            continue
        for fitness_model in FITNESS_MODELS:
            if fitness_model_filter and fitness_model != fitness_model_filter:
                continue
            config_path = f"configs/synthetic/{protein}_{fitness_model}_config.yaml"
            if not os.path.exists(config_path):
                print(f"Warning: base config not found, skipping — {config_path}")
                continue
            for method in IG_METHODS:
                if method_filter and method != method_filter:
                    continue
                for attribution in ATTRIBUTIONS:
                    if attribution_filter and attribution != attribution_filter:
                        continue
                    combos.append({
                        "protein": protein,
                        "fitness_model": fitness_model,
                        "method": method,
                        "attribution": attribution,
                        "base_config": config_path,
                        "results_dir": f"results/synthetic/{protein}_{fitness_model}_{method}_{attribution}",
                        "label": f"{protein}_{fitness_model}_{method}_{attribution}",
                    })
    return combos


def build_temp_config(combo):
    """Load base config, override IG settings, write to a temp file, return path."""
    with open(combo["base_config"]) as f:
        cfg = yaml.safe_load(f)

    cfg["integrated_gradients"]["method"] = combo["method"]
    cfg["integrated_gradients"]["attribution_method"] = combo["attribution"]
    cfg["results_dir"] = combo["results_dir"]

    # Use a named temp file so subprocess can open it by path
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix=f"synthetic_{combo['label']}_",
        delete=False
    )
    yaml.dump(cfg, tmp, default_flow_style=False, sort_keys=False)
    tmp.close()
    return tmp.name


def run_analysis(combo, log_dir, dry_run=False):
    """Run integrated gradients for one combination. Returns True on success."""
    label = combo["label"]
    config_path = build_temp_config(combo)
    log_file = os.path.join(log_dir, f"{label}.log")

    cmd = [sys.executable, "scripts/run_integrated_gradients.py", config_path]

    if dry_run:
        print(f"  [dry-run] Would run: {' '.join(cmd)}")
        print(f"            Results → {combo['results_dir']}/")
        os.unlink(config_path)
        return True

    print(f"  Command : {' '.join(cmd)}")
    print(f"  Results → {combo['results_dir']}/")
    print(f"  Log     → {log_file}")

    try:
        os.makedirs(log_dir, exist_ok=True)
        with open(log_file, "w") as f:
            f.write(f"Starting {label} at {datetime.now()}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write("=" * 60 + "\n\n")
            f.flush()

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )
            for line in process.stdout:
                print(line.rstrip())
                f.write(line)
                f.flush()

            process.wait()
            f.write(f"\nFinished at {datetime.now()}\n")
            f.write(f"Exit code: {process.returncode}\n")

        success = process.returncode == 0
    except Exception as exc:
        print(f"  ERROR: {exc}")
        success = False
    finally:
        try:
            os.unlink(config_path)
        except OSError:
            pass

    if success:
        print(f"  ✓  {label} — done")
    else:
        print(f"  ✗  {label} — FAILED")
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Batch IGLOO analysis on synthetic protein fitness datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--protein", choices=PROTEINS,
        help="Restrict to one protein"
    )
    parser.add_argument(
        "--fitness-model", choices=FITNESS_MODELS, dest="fitness_model",
        help="Restrict to one fitness model"
    )
    parser.add_argument(
        "--method", choices=IG_METHODS,
        help="Restrict to one IG method (embedding / token)"
    )
    parser.add_argument(
        "--attribution", choices=ATTRIBUTIONS,
        help="Restrict to one attribution type (magnitude / signed)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be run without executing"
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Keep going even if a run fails"
    )
    parser.add_argument(
        "--log-dir", default="logs/ig_batch_run/synthetic",
        help="Directory for log files (default: logs/ig_batch_run/synthetic)"
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="After all IG runs complete, run evaluate_synthetic_ig.py on the "
             "results directories to compute Spearman/Kendall/AUROC/AUPRC metrics."
    )
    parser.add_argument(
        "--eval-output-dir", default="results/synthetic/evaluation",
        dest="eval_output_dir",
        help="Output directory for the evaluation summary (default: results/synthetic/evaluation)"
    )
    args = parser.parse_args()

    combos = get_all_combinations(
        protein_filter=args.protein,
        fitness_model_filter=args.fitness_model,
        method_filter=args.method,
        attribution_filter=args.attribution,
    )

    if not combos:
        print("No matching combinations found. Check filters or that base configs exist.")
        sys.exit(1)

    print(f"Synthetic IGLOO batch run — {len(combos)} combination(s) queued")
    print(f"{'Protein':<14} {'Fitness model':<12} {'Method':<10} {'Attribution':<10}")
    print("-" * 50)
    for c in combos:
        print(f"{c['protein']:<14} {c['fitness_model']:<12} {c['method']:<10} {c['attribution']:<10}")

    if args.dry_run:
        print("\n[dry-run] no analyses will be executed.\n")

    print(f"\nStarted at {datetime.now()}")
    print("=" * 60)

    results = []
    for i, combo in enumerate(combos, 1):
        print(f"\n[{i}/{len(combos)}] {combo['label']}")
        success = run_analysis(combo, args.log_dir, dry_run=args.dry_run)
        results.append((combo["label"], success))
        if not success and not args.continue_on_error and not args.dry_run:
            print("\nStopped after failure. Use --continue-on-error to proceed.")
            break
        print("-" * 60)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    ok = sum(1 for _, s in results if s)
    fail = len(results) - ok
    print(f"Total: {len(results)}  |  OK: {ok}  |  Failed: {fail}")
    if fail:
        print("\nFailed:")
        for label, ok in results:
            if not ok:
                print(f"  ✗ {label}")
    print(f"\nLogs: {args.log_dir}")
    print(f"Completed at {datetime.now()}")
    # ── Optional validation step ──────────────────────────────────────────
    if args.evaluate and not args.dry_run:
        succeeded_dirs = [
            combo["results_dir"]
            for combo, (label, ok) in zip(combos[: len(results)], results)
            if ok
        ]
        if not succeeded_dirs:
            print("\n[evaluate] No successful runs to evaluate.")
        else:
            print(f"\n{'='*60}")
            print(f"EVALUATION ({len(succeeded_dirs)} directories)")
            print(f"{'='*60}")
            eval_cmd = [
                sys.executable,
                "scripts/evaluate_synthetic_ig.py",
                "--output-dir", args.eval_output_dir,
                "--per-position-plots",
            ] + succeeded_dirs
            print(f"Running: {' '.join(eval_cmd[:5])} ...")
            subprocess.run(eval_cmd, check=False)

if __name__ == "__main__":
    main()
