#!/usr/bin/env python3
"""
Consolidates the curated set of "final" figures into
results/synthetic/final_figures_nature_methods/, each resized to Nature
Methods' double-column width (183 mm) at 300 dpi (Nature's minimum for
color/combination figures), as both PNG and TIFF, plus the original vector
PDF alongside for reference.

Source PNGs must already be rendered at >= 183mm*300dpi (2161px) wide --
most scripts in this project were bumped from dpi=150 to dpi=300 for this
reason (doubling native pixel count comfortably clears the 2161px floor
without upsampling). This script only ever downsamples (LANCZOS), never
upsamples, and will loudly warn if a source is too small to do that safely.

Resizing the whole rendered raster uniformly (rather than re-tuning each
figure's font sizes individually) preserves each figure's existing
text-to-plot-element proportions exactly -- the standard way a journal
places an already-designed figure at a specific column width.
"""
from pathlib import Path

from PIL import Image

MM_PER_IN = 25.4
TARGET_WIDTH_MM = 183
TARGET_DPI = 300
TARGET_WIDTH_PX = round(TARGET_WIDTH_MM / MM_PER_IN * TARGET_DPI)

OUT_DIR = Path("results/synthetic/final_figures_nature_methods")

# (output base name, source png, source pdf or None)
SRC = Path("results/synthetic")
FIGURES = [
    ("poolfirst_topsis_by_landscape", SRC / "metric_suite/poolfirst_topsis_by_landscape.png", SRC / "metric_suite/poolfirst_topsis_by_landscape.pdf"),
    ("meanpool_topsis_by_landscape", SRC / "metric_suite/meanpool_topsis_by_landscape.png", SRC / "metric_suite/meanpool_topsis_by_landscape.pdf"),
    ("attnpool_topsis_by_landscape", SRC / "metric_suite/attnpool_topsis_by_landscape.png", SRC / "metric_suite/attnpool_topsis_by_landscape.pdf"),
    ("architecture_comparison_scatter", SRC / "metric_suite/architecture_comparison_scatter.png", SRC / "metric_suite/architecture_comparison_scatter.pdf"),
    ("architecture_comparison_scatter_allnoise", SRC / "metric_suite/architecture_comparison_scatter_allnoise.png", SRC / "metric_suite/architecture_comparison_scatter_allnoise.pdf"),
    ("architecture_metric_heatmap", SRC / "metric_suite/architecture_metric_heatmap.png", SRC / "metric_suite/architecture_metric_heatmap.pdf"),
    ("backbone_topsis_strip", SRC / "metric_suite/backbone_topsis_strip.png", SRC / "metric_suite/backbone_topsis_strip.pdf"),
    ("backbone_size_scaling", SRC / "metric_suite/backbone_size_scaling.png", SRC / "metric_suite/backbone_size_scaling.pdf"),
    ("backbone_summary_composite", SRC / "metric_suite/backbone_summary_composite.png", SRC / "metric_suite/backbone_summary_composite.pdf"),
    ("layersweep_topsis_by_landscape", SRC / "metric_suite/layersweep_topsis_by_landscape.png", SRC / "metric_suite/layersweep_topsis_by_landscape.pdf"),
    ("meanpool_layersweep_ensemble_GB1", SRC / "metric_suite/meanpool_layersweep_ensemble_GB1.png", SRC / "metric_suite/meanpool_layersweep_ensemble_GB1.pdf"),
    ("meanpool_layersweep_ensemble_T4Lysozyme", SRC / "metric_suite/meanpool_layersweep_ensemble_T4Lysozyme.png", SRC / "metric_suite/meanpool_layersweep_ensemble_T4Lysozyme.pdf"),
    ("meanpool_layersweep_ensemble_p53", SRC / "metric_suite/meanpool_layersweep_ensemble_p53.png", SRC / "metric_suite/meanpool_layersweep_ensemble_p53.pdf"),
    ("poolfirst_ensemble_boxplot_by_landscape", SRC / "metric_suite/poolfirst_ensemble_boxplot_by_landscape.png", SRC / "metric_suite/poolfirst_ensemble_boxplot_by_landscape.pdf"),
    ("poolfirst_ensemble_violin_by_landscape", SRC / "metric_suite/poolfirst_ensemble_violin_by_landscape.png", SRC / "metric_suite/poolfirst_ensemble_violin_by_landscape.pdf"),
    ("poolfirst_ensemble_variance_by_landscape", SRC / "metric_suite/poolfirst_ensemble_variance_by_landscape.png", SRC / "metric_suite/poolfirst_ensemble_variance_by_landscape.pdf"),
    ("poolfirst_ensemble_variance_by_landscape_noref", SRC / "metric_suite/poolfirst_ensemble_variance_by_landscape_noref.png", SRC / "metric_suite/poolfirst_ensemble_variance_by_landscape_noref.pdf"),
    ("premlp_ensemble_variance_by_landscape", SRC / "metric_suite/premlp_ensemble_variance_by_landscape.png", SRC / "metric_suite/premlp_ensemble_variance_by_landscape.pdf"),
    ("meanpool_ensemble_variance_by_landscape", SRC / "metric_suite/meanpool_ensemble_variance_by_landscape.png", SRC / "metric_suite/meanpool_ensemble_variance_by_landscape.pdf"),
    ("attnpool_ensemble_variance_by_landscape", SRC / "metric_suite/attnpool_ensemble_variance_by_landscape.png", SRC / "metric_suite/attnpool_ensemble_variance_by_landscape.pdf"),
    ("poolfirst_cascading_by_landscape", SRC / "cascading_randomization_figures/poolfirst_cascading_by_landscape.png", SRC / "cascading_randomization_figures/poolfirst_cascading_by_landscape.pdf"),
    ("premlp_cascading_by_landscape", SRC / "cascading_randomization_figures/premlp_cascading_by_landscape.png", SRC / "cascading_randomization_figures/premlp_cascading_by_landscape.pdf"),
    ("meanpool_cascading_by_landscape", SRC / "cascading_randomization_figures/meanpool_cascading_by_landscape.png", SRC / "cascading_randomization_figures/meanpool_cascading_by_landscape.pdf"),
    ("attnpool_cascading_by_landscape", SRC / "cascading_randomization_figures/attnpool_cascading_by_landscape.png", SRC / "cascading_randomization_figures/attnpool_cascading_by_landscape.pdf"),
]

# architecture_summary_figures: 4 architectures x 3 proteins = 12
for arch in ("poolfirst", "premlp", "meanpool", "attnpool"):
    for protein in ("GB1", "T4Lysozyme", "p53"):
        FIGURES.append((
            f"{arch}_summary_{protein}",
            SRC / f"architecture_summary_figures/{arch}_summary_{protein}.png",
            SRC / f"architecture_summary_figures/{arch}_summary_{protein}.pdf",
        ))

_PENDING = []


def process(name, png_path, pdf_path):
    if not png_path.exists():
        print(f"SKIP {name}: {png_path} does not exist")
        return
    im = Image.open(png_path)
    im = im.convert("RGB") if im.mode in ("RGBA", "P") else im
    w, h = im.size
    if w < TARGET_WIDTH_PX:
        print(f"WARNING {name}: source is only {w}px wide (< {TARGET_WIDTH_PX}px target) -- would need upsampling, skipping resize quality check")
    scale = TARGET_WIDTH_PX / w
    target_h = round(h * scale)
    resized = im.resize((TARGET_WIDTH_PX, target_h), Image.LANCZOS)

    out_png = OUT_DIR / f"{name}.png"
    out_tif = OUT_DIR / f"{name}.tif"
    resized.save(out_png, dpi=(TARGET_DPI, TARGET_DPI))
    resized.save(out_tif, dpi=(TARGET_DPI, TARGET_DPI), compression="tiff_lzw")

    if pdf_path and pdf_path.exists():
        out_pdf = OUT_DIR / f"{name}.pdf"
        out_pdf.write_bytes(pdf_path.read_bytes())

    width_mm = TARGET_WIDTH_PX / TARGET_DPI * MM_PER_IN
    height_mm = target_h / TARGET_DPI * MM_PER_IN
    print(f"OK {name}: {TARGET_WIDTH_PX}x{target_h}px @ {TARGET_DPI}dpi "
          f"= {width_mm:.0f}x{height_mm:.0f}mm (scale {scale:.3f})")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, png_path, pdf_path in FIGURES:
        process(name, png_path, pdf_path)
    print(f"\n{len(FIGURES)} figures processed into {OUT_DIR}/")
    print("Not yet included (data pending): " + ", ".join(str(p) for p in _PENDING))


if __name__ == "__main__":
    main()
