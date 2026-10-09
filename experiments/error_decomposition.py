"""Split a trained model's CD error into systematic bias and noise response.

    python experiments/error_decomposition.py [checkpoint.pt]

At a fixed CD the model's error has two parts:

  * bias     -- what it predicts for the noise-free spectrum, minus the truth.
                Pure function-approximation error; more noise-free precision
                (capacity, optimisation) fixes it.
  * noise sd -- how much the prediction scatters across noise draws. The
                Cramer-Rao bound is the smallest this can be for an unbiased
                estimator; anything above it means the model is not weighting
                the spectrum points optimally.

Comparing the two against the bound says which one the gap is made of.
"""
import json
import os
import sys

import numpy as np

from _common import AQUA, BLUE, EXPERIMENT_DIR, ORANGE, PROJECT_ROOT, setup_matplotlib

from cramer_rao import cnn_predict, cramer_rao_bound
from data_generation import clean_spectrum
from inference import load_checkpoint

N_CD = 241
N_NOISE_DRAWS = 400


def decompose(model, cfg, norm, seed=0):
    wavelength = np.linspace(cfg.wavelength_min, cfg.wavelength_max, cfg.n_wavelength)
    cd = np.linspace(cfg.cd_min, cfg.cd_max, N_CD)
    clean = clean_spectrum(cd, wavelength, cfg.n_channels, cfg.n_shared_bases, cfg.physics_seed)

    bias = cnn_predict(model, norm, clean.astype(np.float32)) - cd

    rng = np.random.default_rng(seed)
    noise_sd = np.empty(N_CD)
    mean_shift = np.empty(N_CD)
    for i in range(N_CD):
        noisy = clean[i][None] + rng.normal(0, cfg.noise_std, size=(N_NOISE_DRAWS,) + clean[i].shape)
        pred = cnn_predict(model, norm, noisy.astype(np.float32))
        noise_sd[i] = pred.std(ddof=1)
        mean_shift[i] = pred.mean() - cd[i]

    bound = cramer_rao_bound(cd, wavelength, cfg)
    rms = lambda a: float(np.sqrt((a ** 2).mean()))
    return {
        "cd": cd, "bias": bias, "noise_sd": noise_sd, "bound": bound,
        "summary": {
            "bias_rms": rms(bias),
            "bias_max_abs": float(np.abs(bias).max()),
            "mean_error_under_noise_rms": rms(mean_shift),
            "noise_sd_rms": rms(noise_sd),
            "bound_rms": rms(bound),
            "noise_sd_over_bound": rms(noise_sd) / rms(bound),
            "total_rmse": float(np.sqrt(rms(mean_shift) ** 2 + rms(noise_sd) ** 2)),
        },
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT_ROOT, "outputs", "multichannel_cnn.pt")
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    model, cfg, norm = load_checkpoint(path)
    out = decompose(model, cfg, norm)
    print(json.dumps(out["summary"], indent=2))
    with open(os.path.join(EXPERIMENT_DIR, "error_decomposition.json"), "w") as f:
        json.dump({"checkpoint": path, **out["summary"]}, f, indent=2)

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(out["cd"], out["bound"], color=AQUA, label="Cramér–Rao 下限")
    ax.plot(out["cd"], out["noise_sd"], color=BLUE, label="雜訊造成的預測標準差")
    ax.plot(out["cd"], np.abs(out["bias"]), color=ORANGE, linewidth=1.2, label="無雜訊光譜的預測偏差（絕對值）")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("CD (nm)")
    ax.set_ylabel("CD 誤差 (nm)")
    ax.set_title("CNN 誤差拆解：系統偏差與雜訊響應", loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(EXPERIMENT_DIR, "error_decomposition.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
