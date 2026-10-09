"""Shared setup for the experiment scripts: put src/ on the import path,
set a CJK-capable matplotlib font, and define where results are written."""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# Under outputs/, which is git-ignored (results derived from real data must
# stay local-only once real spectra replace the synthetic generator).
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, "outputs", "experiments")

# Independent of Config.seed (42), so the evaluation spectra are never ones
# a model was trained or early-selected on.
EVAL_SEED = 20261009

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # matplotlib's default font has no CJK glyphs (see app.py).
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_SECONDARY, "text.color": TEXT,
        "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "legend.frameon": False,
    })
    return plt
