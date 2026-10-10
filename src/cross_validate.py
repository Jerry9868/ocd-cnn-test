"""k-fold cross-validation: how good is the model on samples it has not seen?

    python cross_validate.py --folds 5 --data_source folder --spectra_dir D:/data/spectra --labels_csv D:/data/labels.csv
    python cross_validate.py --folds 5 --n_samples 300          # synthetic, to try it out

train.py holds out one 15% test split, which is fine with thousands of
spectra. With the tens to hundreds of reference measurements real data
usually comes with, one split is both wasteful and noisy. Here every sample
is held out exactly once: the data is cut into `folds` parts, and each part
is predicted by a model trained on the others (a val_ratio share of those is
used to pick the epoch and calibrate the refinement, as in train.py).

If the labels table names a wafer / lot per sample (real_data.py), whole
wafers are held out together, so the score is for wafers the model has
never seen -- the situation it will be used in.

Accepts every Config field as an option, like train.py. Writes to
<output_dir>/cross_validation/:
  predictions.csv -- every sample's held-out prediction and error
  summary.json    -- acceptance (worst case vs. error_threshold_nm first,
                     RMSE second) overall and per fold
  report.png      -- predicted vs. reference CD, and error vs. reference CD

No checkpoint is written; train.py makes the model to deploy.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from config import Config
from dataset import kfold_indices, split_indices
from train import acceptance_line, add_config_arguments, error_metrics, fit, load_dataset, predict_cd, set_seed

# Which of predict_cd's outputs get a metrics block, and what to call them.
SCORED = {"cd": "reported", "cd_network": "network", "cd_refined": "refined"}


def cross_validate(cfg: Config, folds: int, progress_cb=None, log=print) -> dict:
    """Returns {"predictions": DataFrame, "summary": dict} and writes them
    (see module docstring). progress_cb(fold, folds), if given, is called
    as each fold starts."""
    data = load_dataset(cfg)
    n = len(data.y)
    groups = getattr(data, "groups", None)
    names = getattr(data, "names", None) or [f"sample_{i:05d}" for i in range(n)]
    held_out = kfold_indices(n, folds, cfg.seed, groups)

    columns = {}
    fold_of = np.empty(n, dtype=int)
    per_fold = []
    for k, test_idx in enumerate(held_out, start=1):
        if progress_cb is not None:
            progress_cb(k, folds)
        rest = np.setdiff1d(np.arange(n), test_idx)
        # Split the remaining samples into train / val, wafers kept whole.
        train_part, val_part, _ = split_indices(len(rest), cfg.val_ratio, 0.0, cfg.seed + k,
                                                None if groups is None else groups[rest])
        set_seed(cfg.seed + k)
        fitted = fit(cfg, data, rest[train_part], rest[val_part], log=lambda _line: None)
        pred = predict_cd(fitted, data.X[test_idx])
        for key, values in pred.items():
            columns.setdefault(key, np.full(n, np.nan))[test_idx] = values
        fold_of[test_idx] = k
        m = error_metrics(data.y[test_idx].astype(np.float64), pred["cd"], cfg)
        refinement = fitted["refinement"]
        per_fold.append({"fold": k, "n_train": len(train_part), "n_val": len(val_part), "n_test": len(test_idx),
                         "refinement_used": bool(refinement is not None and refinement.recommended), **m})
        log(f"fold {k}/{folds}  n_test {len(test_idx):4d}  {acceptance_line(m, cfg)}")

    y = data.y.astype(np.float64)
    predictions = pd.DataFrame({"filename": names, "fold": fold_of, "reference_cd_nm": y})
    if groups is not None:
        predictions.insert(1, "group", groups)
    predictions["predicted_cd_nm"] = columns["cd"]
    predictions["error_nm"] = columns["cd"] - y
    predictions["network_cd_nm"] = columns["cd_network"]
    if "cd_refined" in columns:
        predictions["refined_cd_nm"] = columns["cd_refined"]
        predictions["refined_cd_2sigma_nm"] = 2 * columns["cd_sd"]
        predictions["flagged"] = columns["flagged"].astype(bool)

    summary = {
        "n_samples": n, "folds": folds, "grouped_by": "group column" if groups is not None else None,
        "n_groups": int(len(np.unique(groups))) if groups is not None else None,
        "error_threshold_nm": cfg.error_threshold_nm,
        **{label: error_metrics(y, columns[key], cfg) for key, label in SCORED.items() if key in columns},
        "per_fold": per_fold,
    }
    for label in SCORED.values():
        if label in summary:
            log(f"{label:9s} {acceptance_line(summary[label], cfg)}")

    out_dir = os.path.join(cfg.output_dir, "cross_validation")
    os.makedirs(out_dir, exist_ok=True)
    predictions.to_csv(os.path.join(out_dir, "predictions.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plot_report(predictions, summary, cfg, os.path.join(out_dir, "report.png"))
    return {"predictions": predictions, "summary": summary}


def plot_report(predictions: pd.DataFrame, summary: dict, cfg: Config, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y, pred, err = (predictions[c].to_numpy() for c in ("reference_cd_nm", "predicted_cd_nm", "error_nm"))
    over = np.abs(err) > cfg.error_threshold_nm
    m = summary["reported"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    lims = [min(y.min(), pred.min()), max(y.max(), pred.max())]
    axes[0].plot(lims, lims, color="0.6", linewidth=1)
    axes[0].scatter(y[~over], pred[~over], s=12, color="#0e7c7b", alpha=0.7)
    axes[0].scatter(y[over], pred[over], s=16, color="#c8553d")
    axes[0].set_xlabel("reference CD (nm)")
    axes[0].set_ylabel("held-out prediction (nm)")
    axes[0].set_title(f"{summary['folds']}-fold cross-validation, n = {summary['n_samples']}", loc="left", fontsize=10)

    for sign in (-1, 1):
        axes[1].axhline(sign * cfg.error_threshold_nm, color="#c8553d", linewidth=1, linestyle="--")
    axes[1].axhline(0, color="0.6", linewidth=1)
    axes[1].scatter(y[~over], err[~over], s=12, color="#0e7c7b", alpha=0.7)
    axes[1].scatter(y[over], err[over], s=16, color="#c8553d")
    axes[1].set_xlabel("reference CD (nm)")
    axes[1].set_ylabel("error (nm)")
    axes[1].set_title(f"max |error| {m['max_abs_err_nm']:.3f} nm, {m['fraction_over_threshold']:.1%} over "
                      f"{cfg.error_threshold_nm:g} nm, RMSE {m['rmse']:.3f} nm", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=5)
    add_config_arguments(p, Config())
    args = vars(p.parse_args())
    folds = args.pop("folds")
    cross_validate(Config(**args), folds)


if __name__ == "__main__":
    main()
