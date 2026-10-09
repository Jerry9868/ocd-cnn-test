"""Learning curve: CD error vs. amount of training data.

    python experiments/learning_curve.py
    python experiments/learning_curve.py --sizes 1000 4000 --epochs 150

Trains one model (--model, the CNN unless told otherwise) at several dataset sizes and scores every model on
the same fresh evaluation set, so the numbers are comparable across sizes
(train.py's own test split shrinks with n_samples). Tells whether the
remaining gap to the Cramer-Rao bound (experiments/cramer_rao.py) is a data
problem or a model/training problem.

More data also means more gradient steps at a fixed epoch count, so
--control_epochs retrains the default-size dataset for that many epochs as a
control for "is it the data or just the longer optimisation".

Each run's checkpoint goes to outputs/experiments/learning_curve/<run>/, not
outputs/multichannel_cnn.pt -- the app's model is left alone. Finished runs
are kept in learning_curve.json and skipped on re-run.
"""
import argparse
import contextlib
import dataclasses
import json
import os
import time

import numpy as np

from _common import AQUA, BLUE, EVAL_SEED, EXPERIMENT_DIR, ORANGE, setup_matplotlib

from config import Config
from cramer_rao import N_EVAL, cnn_predict, cramer_rao_bound, summarize
from data_generation import generate_dataset
from train import run_training

RESULT_PATH = os.path.join(EXPERIMENT_DIR, "learning_curve.json")
RUN_DIR = os.path.join(EXPERIMENT_DIR, "learning_curve")


def train_and_score(name, cfg, eval_set):
    cfg = dataclasses.replace(cfg, output_dir=os.path.join(RUN_DIR, name))
    os.makedirs(cfg.output_dir, exist_ok=True)
    start = time.time()
    with open(os.path.join(cfg.output_dir, "train.log"), "w") as log, contextlib.redirect_stdout(log):
        out = run_training(cfg)
    norm = out["norm"]
    norm = {"mean": norm.mean, "std": norm.std, "y_mean": norm.y_mean, "y_std": norm.y_std}
    model = out["model"].eval()
    err = cnn_predict(model, norm, eval_set.X) - eval_set.y.astype(np.float64)
    n_train = cfg.n_samples - int(cfg.n_samples * cfg.val_ratio) - int(cfg.n_samples * cfg.test_ratio)
    return {"n_samples": cfg.n_samples, "n_train": n_train, "epochs": cfg.epochs,
            "minutes": (time.time() - start) / 60, **summarize(err, cfg.error_threshold_nm)}


def plot(results, bound_rms, threshold, epochs):
    plt = setup_matplotlib()
    curve = sorted((r for k, r in results.items() if k.startswith("n") and r["epochs"] == epochs),
                   key=lambda r: r["n_train"])
    x = [r["n_train"] for r in curve]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, [r["max"] for r in curve], color=ORANGE, marker="o", markersize=6, label="最大誤差")
    ax.plot(x, [r["rmse"] for r in curve], color=BLUE, marker="o", markersize=6, label="RMSE")
    ax.axhline(bound_rms, color=AQUA, linestyle="--", label="Cramér–Rao 下限 (RMS)")
    ax.axhline(threshold, color="#52514e", linestyle=":", linewidth=1.2)
    ax.annotate(f"目標 {threshold:.1f} nm", (x[0], threshold), textcoords="offset points",
                xytext=(0, 4), color="#52514e", fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Plain numbers: the default 10^-1 labels need a minus glyph the CJK
    # font doesn't have.
    ax.set_xticks(x, [f"{v:,}" for v in x], minor=False)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    yticks = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    ax.set_yticks(yticks, [f"{v:g}" for v in yticks], minor=False)
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("訓練樣本數")
    ax.set_ylabel(f"CD 誤差 (nm)，{N_EVAL} 筆評估光譜")
    ax.set_title("訓練資料量對 CD 誤差的影響", loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(EXPERIMENT_DIR, "learning_curve.png"), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[1000, 2000, 4000, 8000, 12000, 24000, 48000])
    p.add_argument("--epochs", type=int, default=Config().epochs)
    p.add_argument("--control_epochs", type=int, default=0)
    p.add_argument("--model", choices=["cnn", "mlp"], default="cnn")
    args = p.parse_args()

    os.makedirs(RUN_DIR, exist_ok=True)
    base = Config(epochs=args.epochs, model=args.model, refine="off")
    eval_set = generate_dataset(
        n_samples=N_EVAL, n_channels=base.n_channels, n_wavelength=base.n_wavelength,
        wavelength_min=base.wavelength_min, wavelength_max=base.wavelength_max,
        cd_min=base.cd_min, cd_max=base.cd_max, n_shared_bases=base.n_shared_bases,
        noise_std=base.noise_std, seed=EVAL_SEED, physics_seed=base.physics_seed,
    )
    bound = cramer_rao_bound(eval_set.y.astype(np.float64), eval_set.wavelength, base)
    bound_rms = float(np.sqrt((bound ** 2).mean()))

    results = {}
    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH) as f:
            results = json.load(f)["runs"]

    runs = [(f"n{n}_e{args.epochs}", dataclasses.replace(base, n_samples=n)) for n in args.sizes]
    if args.control_epochs:
        runs.append((f"control_n{base.n_samples}_e{args.control_epochs}",
                     dataclasses.replace(base, epochs=args.control_epochs)))

    for name, cfg in runs:
        if name not in results:
            results[name] = train_and_score(name, cfg, eval_set)
            with open(RESULT_PATH, "w") as f:
                json.dump({"n_eval": N_EVAL, "bound_rms": bound_rms, "runs": results}, f, indent=2)
        r = results[name]
        print(f"{name:24s} n_train {r['n_train']:6d}  rmse {r['rmse']:.4f} ({r['rmse'] / bound_rms:.2f}x bound)  "
              f"p99 {r['p99']:.3f}  max {r['max']:.3f}  >thr {r['fraction_over_threshold']:.2%}  "
              f"{r['minutes']:.1f} min", flush=True)

    plot(results, bound_rms, base.error_threshold_nm, args.epochs)


if __name__ == "__main__":
    main()
