"""How predictions degrade when the measurement differs from the training data.

    python experiments/robustness.py outputs/experiments/ablation_arch/base outputs/experiments/ablation_arch/mlp
    python experiments/robustness.py --plot-only      # redraw from robustness.json

Two perturbations, applied to fresh evaluation spectra only (the models are
the ones trained on unperturbed data):

  * wavelength calibration offset -- the instrument's wavelength axis is off
    by a constant, so each reported point actually samples lambda + offset.
  * noise level -- measurement noise smaller or larger than in training.

For each trained network: its own prediction, and the prediction after
refinement against the learned forward model (experiments/refine.py). The
refinement's leftover misfit doubles as an alarm: the fraction of spectra
whose misfit is above what noise alone explains is reported as "flagged".
"""
import json
import os
import sys

import numpy as np

from _common import AQUA, BLUE, EVAL_SEED, EXPERIMENT_DIR, ORANGE, setup_matplotlib

from data_generation import clean_spectrum
from refine import Refiner

N_EVAL = 5000
OFFSETS_NM = [-1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0]
NOISE_FACTORS = [0.25, 0.5, 1.0, 2.0, 4.0]


def stats(err, threshold):
    return {"bias": float(err.mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "max": float(np.abs(err).max()), "fraction_over_threshold": float((np.abs(err) > threshold).mean())}


def evaluate(ref, offset_nm, noise_factor, rng):
    cfg = ref.cfg
    wavelength = np.linspace(cfg.wavelength_min, cfg.wavelength_max, cfg.n_wavelength)
    cd = rng.uniform(cfg.cd_min, cfg.cd_max, size=N_EVAL)
    noise_std = cfg.noise_std * noise_factor
    X = clean_spectrum(cd, wavelength + offset_nm, cfg.n_channels, cfg.n_shared_bases, cfg.physics_seed)
    X = (X + rng.normal(0, noise_std, size=X.shape)).astype(np.float32)
    net_cd, refined_cd, misfit, _ = ref.predict(X)
    _, offset_cd, offset_misfit, fitted = ref.predict(X, fit_offset=True)
    # Mean squared residual over D points is noise_std^2 * (1 +- sqrt(2/D));
    # flag anything 4 sd above the training-noise expectation.
    limit = cfg.noise_std ** 2 * (1 + 4 * np.sqrt(2.0 / X[0].size))
    return {
        "offset_nm": offset_nm, "noise_factor": noise_factor,
        "network": stats(net_cd - cd, cfg.error_threshold_nm),
        "refined": stats(refined_cd - cd, cfg.error_threshold_nm),
        "flagged": float((misfit > limit).mean()),
        # Same refinement with the wavelength offset as an extra fit parameter.
        "refined_offset_fit": stats(offset_cd - cd, cfg.error_threshold_nm),
        "fitted_offset_mean_nm": float(fitted.mean()),
        "fitted_offset_sd_nm": float(fitted.std()),
        "flagged_offset_fit": float((offset_misfit > limit).mean()),
    }


LABELS = {"base": "CNN", "mlp": "MLP"}  # run folder name -> legend name


def main():
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    result_path = os.path.join(EXPERIMENT_DIR, "robustness.json")
    if sys.argv[1:] == ["--plot-only"]:
        with open(result_path) as f:
            plot(json.load(f))
        return
    run_dirs = sys.argv[1:]
    results = {}
    for run_dir in run_dirs:
        name = os.path.basename(os.path.normpath(run_dir))
        ref = Refiner(run_dir, log=lambda *_: None)
        if ref.cfg.n_nuisance:
            raise SystemExit("robustness.py handles the no-nuisance generator only")
        rng = np.random.default_rng(EVAL_SEED)
        rows = [evaluate(ref, o, 1.0, rng) for o in OFFSETS_NM]
        rows += [evaluate(ref, 0.0, f, rng) for f in NOISE_FACTORS if f != 1.0]
        results[name] = rows
        for r in rows:
            print(f"{name:8s} offset {r['offset_nm']:+.1f}nm noise x{r['noise_factor']:<4}  "
                  f"network bias {r['network']['bias']:+.3f} rmse {r['network']['rmse']:.3f}  |  "
                  f"refined bias {r['refined']['bias']:+.3f} rmse {r['refined']['rmse']:.3f}  "
                  f"flagged {r['flagged']:.1%}  |  +offset fit bias {r['refined_offset_fit']['bias']:+.3f} "
                  f"rmse {r['refined_offset_fit']['rmse']:.3f} (fitted {r['fitted_offset_mean_nm']:+.2f}"
                  f"+-{r['fitted_offset_sd_nm']:.2f}nm)", flush=True)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    plot(results)


def plot(results):
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    colors = [BLUE, ORANGE]
    for (name, rows), color in zip(results.items(), colors):
        off = [r for r in rows if r["noise_factor"] == 1.0]
        axes[0].plot([r["offset_nm"] for r in off], [r["network"]["rmse"] for r in off],
                     color=color, marker="o", markersize=6, label=f"{LABELS.get(name, name)} 直接預測")
    first = next(iter(results.values()))
    off = [r for r in first if r["noise_factor"] == 1.0]
    axes[0].plot([r["offset_nm"] for r in off], [r["refined"]["rmse"] for r in off],
                 color=AQUA, marker="o", markersize=6, label="擬合修正後")
    axes[0].plot([r["offset_nm"] for r in off], [r["refined_offset_fit"]["rmse"] for r in off],
                 color=AQUA, linestyle="--", marker="s", markersize=6, label="擬合修正，同時擬合波長偏移")
    axes[0].set_xlabel("波長校正偏移 (nm)")
    axes[0].set_ylabel("CD RMSE (nm)")
    axes[0].set_title("波長校正偏移的影響", loc="left")
    axes[0].set_ylim(bottom=0)
    axes[0].legend()

    for (name, rows), color in zip(results.items(), colors):
        nz = sorted((r for r in rows if r["offset_nm"] == 0.0), key=lambda r: r["noise_factor"])
        axes[1].plot([r["noise_factor"] for r in nz], [r["network"]["rmse"] for r in nz],
                     color=color, marker="o", markersize=6, label=f"{LABELS.get(name, name)} 直接預測")
    nz = sorted((r for r in first if r["offset_nm"] == 0.0), key=lambda r: r["noise_factor"])
    axes[1].plot([r["noise_factor"] for r in nz], [r["refined"]["rmse"] for r in nz],
                 color=AQUA, marker="o", markersize=6, label="擬合修正後")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    ticks = [r["noise_factor"] for r in nz]
    axes[1].set_xticks(ticks, [f"{t:g}×" for t in ticks], minor=False)
    axes[1].xaxis.set_minor_formatter(plt.NullFormatter())
    yt = [0.02, 0.05, 0.1, 0.2, 0.5]
    axes[1].set_yticks(yt, [f"{t:g}" for t in yt], minor=False)
    axes[1].yaxis.set_minor_formatter(plt.NullFormatter())
    axes[1].set_xlabel("量測雜訊（相對於訓練時）")
    axes[1].set_ylabel("CD RMSE (nm)")
    axes[1].set_title("雜訊大小的影響", loc="left")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(EXPERIMENT_DIR, "robustness.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
