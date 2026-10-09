"""Cramer-Rao lower bound on CD error vs. what the trained CNN achieves.

    python experiments/cramer_rao.py

For spectrum = f(CD) + Gaussian noise(noise_std), no unbiased estimator can
have a CD standard deviation below

    sigma_CD(CD) = noise_std / || d f / d CD ||

(norm over every channel and wavelength point). This script computes that
bound across the CD range and compares it with:

  * the trained checkpoint (outputs/multichannel_cnn.pt) on fresh spectra
  * a brute-force maximum-likelihood estimator (nearest noise-free template
    on a fine CD grid) -- a check that the bound is actually attainable for
    this generator, i.e. that the gap to the CNN is the model's, not the
    data's.

The bound only knows about measurement noise with CD as the single unknown.
With real spectra (other structure parameters varying, systematic errors)
the achievable floor is higher.
"""
import json
import os

import numpy as np
import torch

from _common import AQUA, BLUE, EVAL_SEED, EXPERIMENT_DIR, ORANGE, PROJECT_ROOT, setup_matplotlib

from data_generation import clean_spectrum, generate_dataset
from inference import load_checkpoint

N_EVAL = 20000
TEMPLATE_STEP_NM = 0.01
N_BINS = 12


def cramer_rao_bound(cd, wavelength, cfg, h=1e-3):
    """sigma_CD lower bound (nm) at each CD value."""
    kw = dict(wavelength=wavelength, n_channels=cfg.n_channels,
              n_shared_bases=cfg.n_shared_bases, physics_seed=cfg.physics_seed)
    dfdcd = (clean_spectrum(cd + h, **kw) - clean_spectrum(cd - h, **kw)) / (2 * h)
    return cfg.noise_std / np.sqrt((dfdcd ** 2).sum(axis=(1, 2)))


def cramer_rao_bound_joint(cd, nuisance, wavelength, cfg, h=1e-3, chunk=1000):
    """CD bounds (nm) when nuisance parameters also vary: (known, unknown).

    known   -- nuisances given exactly; only CD is estimated.
    unknown -- CD and every nuisance estimated together from the spectrum
               ([F^-1]_00 of the full Fisher matrix). The ratio is what
               the CD/nuisance confounding costs.

    Ignores the prior range of the nuisances, which is fine while the
    spectrum pins them down far more tightly than [-1, 1].
    """
    kw = dict(wavelength=wavelength, n_channels=cfg.n_channels, n_shared_bases=cfg.n_shared_bases,
              physics_seed=cfg.physics_seed, nuisance_strength=cfg.nuisance_strength)
    n, n_nuis = nuisance.shape
    known, unknown = np.empty(n), np.empty(n)
    for s in range(0, n, chunk):
        c, u = cd[s:s + chunk], nuisance[s:s + chunk].astype(np.float64)
        cols = [(clean_spectrum(c + h, nuisance=u, **kw) - clean_spectrum(c - h, nuisance=u, **kw)) / (2 * h)]
        for j in range(n_nuis):
            e = np.zeros_like(u)
            e[:, j] = h
            cols.append((clean_spectrum(c, nuisance=u + e, **kw) - clean_spectrum(c, nuisance=u - e, **kw)) / (2 * h))
        J = np.stack([col.reshape(len(c), -1) for col in cols], axis=1)  # (chunk, 1 + n_nuis, D)
        fisher = J @ J.transpose(0, 2, 1) / cfg.noise_std ** 2
        known[s:s + chunk] = 1.0 / np.sqrt(fisher[:, 0, 0])
        unknown[s:s + chunk] = np.sqrt(np.linalg.inv(fisher)[:, 0, 0])
    return known, unknown


def mle_predict(X, wavelength, cfg):
    """Nearest noise-free template on a fine CD grid, refined by fitting a
    parabola through the squared distance at the best grid point and its two
    neighbours."""
    grid = np.arange(cfg.cd_min, cfg.cd_max + TEMPLATE_STEP_NM / 2, TEMPLATE_STEP_NM)
    T = clean_spectrum(grid, wavelength, cfg.n_channels, cfg.n_shared_bases, cfg.physics_seed)
    T = T.reshape(len(grid), -1)
    t_sq = (T ** 2).sum(axis=1)

    Xf = X.reshape(len(X), -1).astype(np.float64)
    pred = np.empty(len(Xf))
    for start in range(0, len(Xf), 500):
        x = Xf[start:start + 500]
        d = t_sq[None, :] - 2.0 * x @ T.T  # squared distance up to a per-sample constant
        i = d.argmin(axis=1)
        j = np.clip(i, 1, len(grid) - 2)
        rows = np.arange(len(x))
        d0, d1, d2 = d[rows, j - 1], d[rows, j], d[rows, j + 1]
        curv = d0 - 2 * d1 + d2
        offset = np.where(curv > 0, 0.5 * (d0 - d2) / np.where(curv > 0, curv, 1.0), 0.0)
        offset = np.where(i == j, np.clip(offset, -1.0, 1.0), 0.0)
        pred[start:start + 500] = grid[i] + offset * TEMPLATE_STEP_NM
    return pred


@torch.no_grad()
def cnn_predict(model, norm, X):
    X_norm = (X - norm["mean"][None]) / norm["std"][None]
    out = []
    for start in range(0, len(X_norm), 1000):
        pred = model(torch.from_numpy(X_norm[start:start + 1000]).float()).numpy()
        out.append(pred[:, 0] if pred.ndim == 2 else pred)  # output 0 is CD
    return np.concatenate(out) * norm["y_std"] + norm["y_mean"]


def summarize(err, threshold):
    a = np.abs(err)
    return {
        "bias": float(err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(a.mean()),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
        "fraction_over_threshold": float((a > threshold).mean()),
    }


def main():
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    model, cfg, norm = load_checkpoint(os.path.join(PROJECT_ROOT, "outputs", "multichannel_cnn.pt"))

    data = generate_dataset(
        n_samples=N_EVAL, n_channels=cfg.n_channels, n_wavelength=cfg.n_wavelength,
        wavelength_min=cfg.wavelength_min, wavelength_max=cfg.wavelength_max,
        cd_min=cfg.cd_min, cd_max=cfg.cd_max, n_shared_bases=cfg.n_shared_bases,
        noise_std=cfg.noise_std, seed=EVAL_SEED, physics_seed=cfg.physics_seed,
    )
    y = data.y.astype(np.float64)

    err_cnn = cnn_predict(model, norm, data.X) - y
    err_mle = mle_predict(data.X, data.wavelength, cfg) - y
    bound_per_sample = cramer_rao_bound(y, data.wavelength, cfg)

    cd_grid = np.linspace(cfg.cd_min, cfg.cd_max, 601)
    bound_grid = cramer_rao_bound(cd_grid, data.wavelength, cfg)

    edges = np.linspace(cfg.cd_min, cfg.cd_max, N_BINS + 1)
    which = np.clip(np.digitize(y, edges) - 1, 0, N_BINS - 1)
    bins = []
    for b in range(N_BINS):
        m = which == b
        bins.append({
            "cd_lo": float(edges[b]), "cd_hi": float(edges[b + 1]), "n": int(m.sum()),
            "bound_rms": float(np.sqrt((bound_per_sample[m] ** 2).mean())),
            "cnn_rmse": float(np.sqrt((err_cnn[m] ** 2).mean())),
            "cnn_bias": float(err_cnn[m].mean()),
            "mle_rmse": float(np.sqrt((err_mle[m] ** 2).mean())),
        })

    bound_rms = float(np.sqrt((bound_per_sample ** 2).mean()))
    result = {
        "n_eval": N_EVAL,
        "noise_std": cfg.noise_std,
        "threshold_nm": cfg.error_threshold_nm,
        "bound": {"rms": bound_rms, "min": float(bound_grid.min()), "max": float(bound_grid.max())},
        "cnn": summarize(err_cnn, cfg.error_threshold_nm),
        "mle": summarize(err_mle, cfg.error_threshold_nm),
        "cnn_rmse_over_bound": float(np.sqrt((err_cnn ** 2).mean()) / bound_rms),
        "bins": bins,
    }
    with open(os.path.join(EXPERIMENT_DIR, "cramer_rao.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "bins"}, indent=2))
    for b in bins:
        print(f"CD {b['cd_lo']:5.1f}-{b['cd_hi']:5.1f}  bound {b['bound_rms']:.4f}  "
              f"MLE {b['mle_rmse']:.4f}  CNN {b['cnn_rmse']:.4f}  CNN bias {b['cnn_bias']:+.4f}")

    plt = setup_matplotlib()
    centers = [(b["cd_lo"] + b["cd_hi"]) / 2 for b in bins]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(cd_grid, bound_grid, color=AQUA, label="Cramér–Rao 下限")
    ax.plot(centers, [b["mle_rmse"] for b in bins], color=ORANGE, marker="o", markersize=6,
            label="最大概似估計 (MLE) RMSE")
    ax.plot(centers, [b["cnn_rmse"] for b in bins], color=BLUE, marker="o", markersize=6,
            label="CNN RMSE")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("CD (nm)")
    ax.set_ylabel("CD 誤差標準差 (nm)")
    ax.set_title("CNN 誤差與理論下限的差距", loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(EXPERIMENT_DIR, "cramer_rao.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
