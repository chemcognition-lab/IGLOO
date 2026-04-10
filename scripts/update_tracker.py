#!/usr/bin/env python3
"""
Experiment tracker for IGLOO runs.

Usage
-----
# Add a row from a completed run:
python scripts/update_tracker.py \\
    --results-dir results/synthetic/GB1_additive_LINEAR \\
    --command "python scripts/run_integrated_gradients.py configs/synthetic/GB1_additive_config.yaml" \\
    --notes "first run with new MLP hidden dims"

# Just initialise / re-create sheet headers (no row added):
python scripts/update_tracker.py --init

The tracker is written to experiment_tracker.xlsx in the project root.
"""

import argparse
import glob
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Locate project root (one level above this script)
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TRACKER_PATH = PROJECT_ROOT / "experiment_tracker.xlsx"

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
COLUMNS = [
    # --- identity ---
    "Run #",
    "Timestamp",
    "Dataset",
    "Command",
    # --- paths ---
    "Config (input)",
    "Config snapshot",
    "Results dir",
    "Run log",
    # --- model hyperparameters ---
    "pooling_method",
    "hidden_dims",
    "dropout_rate",
    "learning_rate",
    "batch_size",
    "epochs",
    "min_epochs",
    "patience",
    "loss_type",
    "normalization",
    # --- CV metrics ---
    "CV mean val R²",
    "CV std val R²",
    "CV mean val Spearman",
    "CV std val Spearman",
    "CV avg best epoch",
    # --- test metrics ---
    "Test R²",
    "Test Pearson r",
    "Test Spearman ρ",
    # --- figure links ---
    "CV loss curves",
    "Test pred vs actual",
    "IG heatmap",
    "IG analysis",
    # --- free text ---
    "Notes",
]

# Columns that store numeric metrics (used for conditional formatting)
METRIC_COLS = {
    "CV mean val R²", "Test R²", "Test Pearson r",
    "CV mean val Spearman", "Test Spearman ρ",
}

# Column widths (approximate)
COL_WIDTHS = {
    "Run #": 7,
    "Timestamp": 20,
    "Dataset": 22,
    "Command": 60,
    "Config (input)": 45,
    "Config snapshot": 45,
    "Results dir": 45,
    "Run log": 45,
    "pooling_method": 16,
    "hidden_dims": 20,
    "dropout_rate": 12,
    "learning_rate": 14,
    "batch_size": 12,
    "epochs": 8,
    "min_epochs": 12,
    "patience": 10,
    "loss_type": 12,
    "normalization": 14,
    "CV mean val R²": 16,
    "CV std val R²": 14,
    "CV mean val Spearman": 20,
    "CV std val Spearman": 18,
    "CV avg best epoch": 18,
    "Test R²": 10,
    "Test Pearson r": 15,
    "Test Spearman ρ": 16,
    "CV loss curves": 20,
    "Test pred vs actual": 22,
    "IG heatmap": 18,
    "IG analysis": 18,
    "Notes": 40,
}

# Known datasets — one sheet each
DATASETS = [
    "GB1_additive", "GB1_epistatic", "GB1_nonlinear",
    "p53_additive", "p53_epistatic", "p53_nonlinear",
    "T4Lysozyme_additive", "T4Lysozyme_epistatic", "T4Lysozyme_nonlinear",
    "betalactamase", "erbc", "PafA",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_wb():
    from openpyxl import load_workbook, Workbook
    if TRACKER_PATH.exists():
        return load_workbook(TRACKER_PATH)
    return Workbook()


def _ensure_sheets(wb):
    """Make sure every known dataset has a sheet; also ensure a Summary sheet."""
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    header_font  = Font(bold=True, color="FFFFFF")
    header_fill  = PatternFill("solid", fgColor="2F5496")
    header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

    needed = ["Summary"] + DATASETS

    # Remove default "Sheet" if it was auto-created and empty
    if "Sheet" in wb.sheetnames and len(wb["Sheet"].calculate_dimension()) < 5:
        del wb["Sheet"]

    for name in needed:
        if name not in wb.sheetnames:
            ws = wb.create_sheet(name)
            if name == "Summary":
                _write_summary_headers(ws, header_font, header_fill, header_align)
            else:
                # Write column headers
                for col_idx, col_name in enumerate(COLUMNS, start=1):
                    cell = ws.cell(row=1, column=col_idx, value=col_name)
                    cell.font  = header_font
                    cell.fill  = header_fill
                    cell.alignment = header_align
                # Set column widths
                for col_idx, col_name in enumerate(COLUMNS, start=1):
                    letter = get_column_letter(col_idx)
                    ws.column_dimensions[letter].width = COL_WIDTHS.get(col_name, 15)
                ws.row_dimensions[1].height = 30
                ws.freeze_panes = "A2"


def _write_summary_headers(ws, header_font, header_fill, header_align):
    from openpyxl.utils import get_column_letter
    cols = ["Dataset", "Best Test R²", "Best Test Spearman ρ", "# Runs",
            "Best Run Timestamp", "Best Results Dir", "Notes"]
    for col_idx, name in enumerate(cols, start=1):
        cell = ws.cell(row=1, column=col_idx, value=name)
        cell.font  = header_font
        cell.fill  = header_fill
        cell.alignment = header_align
        ws.column_dimensions[get_column_letter(col_idx)].width = 28
    ws.row_dimensions[1].height = 30
    ws.freeze_panes = "A2"


def _col_idx(name):
    return COLUMNS.index(name) + 1


def _latest_file(directory: Path, pattern: str):
    """Return the most recently modified file matching glob pattern, or None."""
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _rel(path: Path) -> str:
    """Return path relative to project root as a string."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _hyperlink(ws, row, col, path: Path, display: str = None):
    """Write a file hyperlink into a cell."""
    cell = ws.cell(row=row, column=col)
    if path and path.exists():
        uri = path.resolve().as_uri()
        cell.hyperlink = uri
        cell.value = display or path.name
        from openpyxl.styles import Font
        cell.font = Font(color="0563C1", underline="single")
    else:
        cell.value = display or (str(path) if path else "")


def _parse_cv_metrics(results_dir: Path, dataset_name: str):
    """Read the CV fold metrics CSV and return summary stats."""
    csv_path = results_dir / f"{dataset_name}_cv_fold_metrics.csv"
    if not csv_path.exists():
        return {}
    import csv
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return {}
    val_r2       = [float(r["val_r2"])           for r in rows]
    val_spearman = [float(r["val_spearman_rho"])  for r in rows]
    best_epochs  = [float(r["best_epoch"])        for r in rows]
    return {
        "cv_mean_r2":       sum(val_r2)       / len(val_r2),
        "cv_std_r2":        (sum((x - sum(val_r2)/len(val_r2))**2 for x in val_r2) / len(val_r2))**0.5,
        "cv_mean_spearman": sum(val_spearman) / len(val_spearman),
        "cv_std_spearman":  (sum((x - sum(val_spearman)/len(val_spearman))**2 for x in val_spearman) / len(val_spearman))**0.5,
        "cv_avg_best_epoch": sum(best_epochs) / len(best_epochs),
    }


def _parse_test_metrics(results_dir: Path, dataset_name: str):
    """Parse test R², Pearson r, Spearman ρ from the summary txt."""
    txt_path = results_dir / f"{dataset_name}_integrated_gradients_summary.txt"
    if not txt_path.exists():
        return {}
    text = txt_path.read_text()
    def _extract(pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None
    return {
        "test_r2":       _extract(r"R²\s*=\s*([-\d.]+)"),
        "test_pearson":  _extract(r"Pearson r\s*=\s*([-\d.]+)"),
        "test_spearman": _extract(r"Spearman\s*[ρp]\s*=\s*([-\d.]+)"),
    }


def _parse_config_snapshot(results_dir: Path, dataset_name: str):
    """Load the latest config snapshot YAML and extract key model params."""
    snap = _latest_file(results_dir, f"{dataset_name}_config_snapshot_*.yaml")
    if not snap:
        return {}, None
    with open(snap) as f:
        cfg = yaml.safe_load(f)
    model = cfg.get("model", {})
    dims  = "-".join(str(model.get(f"hidden_dim{i}", "")) for i in range(1, 5))
    params = {
        "pooling_method": model.get("pooling_method", ""),
        "hidden_dims":    dims,
        "dropout_rate":   model.get("dropout_rate", ""),
        "learning_rate":  model.get("learning_rate", ""),
        "batch_size":     model.get("batch_size", ""),
        "epochs":         model.get("epochs", ""),
        "min_epochs":     model.get("min_epochs", ""),
        "patience":       model.get("patience", ""),
        "loss_type":      model.get("loss_type", ""),
        "normalization":  cfg.get("data", {}).get("normalization", ""),
    }
    return params, snap


def _metric_cell_color(value):
    """Return fill color based on metric value."""
    from openpyxl.styles import PatternFill
    if value is None:
        return None
    if value >= 0.5:
        return PatternFill("solid", fgColor="C6EFCE")  # green
    if value >= 0.2:
        return PatternFill("solid", fgColor="FFEB9C")  # yellow
    return PatternFill("solid", fgColor="FFC7CE")       # red


def _update_summary(wb, dataset_name: str, row_data: dict):
    """Refresh the Summary sheet best-row for this dataset."""
    from openpyxl.styles import PatternFill
    ws = wb["Summary"]
    # Find or create row for this dataset
    target_row = None
    for row in ws.iter_rows(min_row=2):
        if row[0].value == dataset_name:
            target_row = row[0].row
            break
    if target_row is None:
        target_row = ws.max_row + 1

    # Only update if this run's test R² is better than current best
    current_best = ws.cell(row=target_row, column=2).value
    new_r2 = row_data.get("test_r2")
    if current_best is None or (new_r2 is not None and new_r2 > (current_best or -999)):
        ws.cell(row=target_row, column=1).value = dataset_name
        ws.cell(row=target_row, column=2).value = new_r2
        ws.cell(row=target_row, column=3).value = row_data.get("test_spearman")
        ws.cell(row=target_row, column=4).value = row_data.get("n_runs")
        ws.cell(row=target_row, column=5).value = row_data.get("timestamp")
        ws.cell(row=target_row, column=6).value = row_data.get("results_dir_str")
        for col in (2, 3):
            val = ws.cell(row=target_row, column=col).value
            fill = _metric_cell_color(val)
            if fill:
                ws.cell(row=target_row, column=col).fill = fill
    else:
        # Just increment run count
        cur = ws.cell(row=target_row, column=4).value or 0
        ws.cell(row=target_row, column=4).value = cur + 1


# ---------------------------------------------------------------------------
# Main add-row logic
# ---------------------------------------------------------------------------
def add_run(results_dir_str: str, command: str, config_input: str, notes: str):
    import openpyxl
    from openpyxl.styles import Alignment, PatternFill

    results_dir = Path(results_dir_str)
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir
    results_dir = results_dir.resolve()

    if not results_dir.exists():
        print(f"ERROR: results directory not found: {results_dir}")
        sys.exit(1)

    # Infer dataset name from directory contents
    dataset_name = None
    for name in DATASETS:
        if (results_dir / f"{name}_cv_fold_metrics.csv").exists():
            dataset_name = name
            break
    if dataset_name is None:
        # Fallback: use directory name
        dataset_name = results_dir.name
        print(f"Warning: could not identify dataset from files; using directory name '{dataset_name}'")

    print(f"Dataset: {dataset_name}")

    wb = _load_wb()
    _ensure_sheets(wb)

    # Make sure this dataset sheet exists (handles unknown datasets)
    if dataset_name not in wb.sheetnames:
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter
        ws = wb.create_sheet(dataset_name)
        header_font  = Font(bold=True, color="FFFFFF")
        header_fill  = PatternFill("solid", fgColor="2F5496")
        header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for col_idx, col_name in enumerate(COLUMNS, start=1):
            cell = ws.cell(row=1, column=col_idx, value=col_name)
            cell.font = header_font; cell.fill = header_fill; cell.alignment = header_align
            ws.column_dimensions[get_column_letter(col_idx)].width = COL_WIDTHS.get(col_name, 15)
        ws.row_dimensions[1].height = 30
        ws.freeze_panes = "A2"

    ws = wb[dataset_name]

    # Determine next run number
    next_row = ws.max_row + 1
    run_number = next_row - 1  # row 1 is header

    # Collect data
    cv      = _parse_cv_metrics(results_dir, dataset_name)
    test    = _parse_test_metrics(results_dir, dataset_name)
    params, snap_path = _parse_config_snapshot(results_dir, dataset_name)
    run_log  = _latest_file(results_dir, f"{dataset_name}_run_log_*.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # --- Write scalar cells ---
    def _write(col_name, value):
        ws.cell(row=next_row, column=_col_idx(col_name), value=value)

    _write("Run #",          run_number)
    _write("Timestamp",      timestamp)
    _write("Dataset",        dataset_name)
    _write("Command",        command or "")
    _write("Config (input)", config_input or "")
    _write("pooling_method", params.get("pooling_method", ""))
    _write("hidden_dims",    params.get("hidden_dims", ""))
    _write("dropout_rate",   params.get("dropout_rate", ""))
    _write("learning_rate",  params.get("learning_rate", ""))
    _write("batch_size",     params.get("batch_size", ""))
    _write("epochs",         params.get("epochs", ""))
    _write("min_epochs",     params.get("min_epochs", ""))
    _write("patience",       params.get("patience", ""))
    _write("loss_type",      params.get("loss_type", ""))
    _write("normalization",  params.get("normalization", ""))
    _write("CV mean val R²",       round(cv.get("cv_mean_r2",       0), 4) if cv else None)
    _write("CV std val R²",        round(cv.get("cv_std_r2",        0), 4) if cv else None)
    _write("CV mean val Spearman", round(cv.get("cv_mean_spearman", 0), 4) if cv else None)
    _write("CV std val Spearman",  round(cv.get("cv_std_spearman",  0), 4) if cv else None)
    _write("CV avg best epoch",    round(cv.get("cv_avg_best_epoch",0), 1) if cv else None)
    _write("Test R²",       round(test.get("test_r2",       0), 4) if test.get("test_r2")       is not None else None)
    _write("Test Pearson r", round(test.get("test_pearson", 0), 4) if test.get("test_pearson")  is not None else None)
    _write("Test Spearman ρ",round(test.get("test_spearman",0), 4) if test.get("test_spearman") is not None else None)
    _write("Notes",          notes or "")
    _write("Results dir",    _rel(results_dir))

    # --- Hyperlink cells ---
    _hyperlink(ws, next_row, _col_idx("Config snapshot"),
               snap_path, snap_path.name if snap_path else None)
    _hyperlink(ws, next_row, _col_idx("Run log"),
               run_log, run_log.name if run_log else None)
    _hyperlink(ws, next_row, _col_idx("CV loss curves"),
               results_dir / f"{dataset_name}_cv_loss_curves.png", "cv_loss_curves.png")
    _hyperlink(ws, next_row, _col_idx("Test pred vs actual"),
               results_dir / f"{dataset_name}_predicted_vs_actual_test.png", "pred_vs_actual_test.png")
    _hyperlink(ws, next_row, _col_idx("IG heatmap"),
               results_dir / f"{dataset_name}_integrated_gradients_heatmap.png", "ig_heatmap.png")
    _hyperlink(ws, next_row, _col_idx("IG analysis"),
               results_dir / f"{dataset_name}_integrated_gradients_analysis.png", "ig_analysis.png")

    # --- Conditional colour on metric cells ---
    for col_name in METRIC_COLS:
        val = ws.cell(row=next_row, column=_col_idx(col_name)).value
        fill = _metric_cell_color(val)
        if fill:
            ws.cell(row=next_row, column=_col_idx(col_name)).fill = fill

    # --- Update summary sheet ---
    _update_summary(wb, dataset_name, {
        "test_r2":        test.get("test_r2"),
        "test_spearman":  test.get("test_spearman"),
        "timestamp":      timestamp,
        "results_dir_str": _rel(results_dir),
        "n_runs":         run_number,
    })

    wb.save(TRACKER_PATH)
    print(f"Row {run_number} added for '{dataset_name}' → {TRACKER_PATH}")


def init_only():
    """Create or refresh the tracker file with all sheets and headers."""
    wb = _load_wb()
    _ensure_sheets(wb)
    wb.save(TRACKER_PATH)
    print(f"Tracker initialised → {TRACKER_PATH}")
    print(f"Sheets: {wb.sheetnames}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Add a completed IGLOO run to the experiment tracker spreadsheet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--results-dir", "-r",
                        help="Path to the results directory of the completed run.")
    parser.add_argument("--command", "-c",
                        help="The command used to launch the run (for the log).")
    parser.add_argument("--config", "-y",
                        help="Path to the input config YAML (optional; stored as reference).")
    parser.add_argument("--notes", "-n", default="",
                        help="Free-text notes about this run.")
    parser.add_argument("--init", action="store_true",
                        help="Initialise / re-create sheet headers only (no row added).")
    args = parser.parse_args()

    if args.init or args.results_dir is None:
        init_only()
        return

    add_run(
        results_dir_str=args.results_dir,
        command=args.command or "",
        config_input=args.config or "",
        notes=args.notes,
    )


if __name__ == "__main__":
    main()
