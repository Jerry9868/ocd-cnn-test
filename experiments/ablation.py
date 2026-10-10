"""Ablation suites: train several Config variants and score them alike.

    python experiments/ablation.py arch
    python experiments/ablation.py arch --workers 4 --only no_bn no_downsample

Every variant is the default Config with a few fields overridden (see
SUITES). Each is trained with train.run_training and then scored on the same
fresh evaluation spectra as experiments/cramer_rao.py, plus the bias / noise
split from experiments/error_decomposition.py, so variants are comparable
with each other and with the Cramer-Rao bound.

Results accumulate in outputs/experiments/ablation_<suite>.json (finished
variants are skipped on re-run); each variant's checkpoint and training log
go to outputs/experiments/ablation_<suite>/<variant>/. The app's
outputs/multichannel_cnn.pt is never touched.
"""
import argparse
import contextlib
import dataclasses
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from _common import EVAL_SEED, EXPERIMENT_DIR

from config import Config

N_EVAL = 20000

# 8000 samples: the learning curve (experiments/learning_curve.py) is flat
# from there on, so this is the cheapest size that isn't data-limited.
# model="cnn": variants that don't name a model mean the CNN, whatever
# Config's default is. refine="off": the suites score the network alone
# (experiments/refine.py measures the refinement).
BASE = dict(n_samples=8000, model="cnn", refine="off")

SUITES = {
    # What keeps the default network at ~1.5x the Cramer-Rao bound?
    "arch": {
        "base": {},
        "base_seed1": dict(seed=1),
        "base_seed2": dict(seed=2),
        "no_bn": dict(norm="none"),
        "no_downsample": dict(downsample="none"),
        "avgpool": dict(downsample="avgpool"),
        "no_bn_no_downsample": dict(norm="none", downsample="none"),
        "wd0": dict(weight_decay=0.0),
        "bs16": dict(batch_size=16),
        "bs256": dict(batch_size=256),
        "lr3e-4": dict(lr=3e-4),
        "lr3e-3": dict(lr=3e-3),
        "hidden48": dict(hidden_dim=48),
        "hidden192": dict(hidden_dim=192),
        "mlp": dict(model="mlp"),
        "noise_aug0.005": dict(noise_augment_std=0.005),
        "noise_aug0.01": dict(noise_augment_std=0.01),
        "per_channel_conv": dict(conv_groups="per_channel"),
    },
    # Three hidden nuisance parameters also move the spectrum. How much data
    # does CD regression need then, and does predicting the nuisances too
    # (multi-task, "mt") help over predicting CD alone ("cd")?
    "nuisance": {
        **{f"cd_n{n}": dict(n_nuisance=3, n_samples=n) for n in (8000, 24000, 48000)},
        **{f"mt_n{n}": dict(n_nuisance=3, aux_targets=3, n_samples=n) for n in (8000, 24000, 48000)},
        "mt_w0.3_n24000": dict(n_nuisance=3, aux_targets=3, aux_loss_weight=0.3, n_samples=24000),
        "mt_w3_n24000": dict(n_nuisance=3, aux_targets=3, aux_loss_weight=3.0, n_samples=24000),
        **{f"mlp_cd_n{n}": dict(model="mlp", n_nuisance=3, n_samples=n) for n in (8000, 24000, 48000)},
        **{f"mlp_mt_n{n}": dict(model="mlp", n_nuisance=3, aux_targets=3, n_samples=n)
           for n in (8000, 24000, 48000)},
    },
    # Do the conclusions hold for a different random "instrument" (another
    # CD<->spectrum relationship), or only for physics_seed 2024?
    "physics": {
        **{f"cnn_p{p}": dict(physics_seed=p) for p in (7, 99, 314)},
        **{f"mlp_p{p}": dict(model="mlp", physics_seed=p) for p in (7, 99, 314)},
    },
    # Closer to real labelling: few reference measurements, each with its
    # own error (label_noise = std of the CD label error in nm).
    "labels": {
        **{f"n{n}": dict(model="mlp", n_samples=n) for n in (150, 300, 600, 1000)},
        **{f"label_noise{s}_n{n}": dict(model="mlp", label_noise_std=s, n_samples=n)
           for s in (0.3, 1.0) for n in (1000, 8000, 24000)},
        **{f"cnn_label_noise{s}_n8000": dict(label_noise_std=s) for s in (0.3, 1.0)},
    },
    # The rest of the planned comparison, against "arch"'s base (Multi-channel
    # 1D CNN), mlp and per_channel_conv (independent 1D CNNs + fusion):
    # Channel Attention, 2D / 3D CNN, derivative features.
    # Measured 2026-10-10 (base 0.105 nm, mlp 0.077): attention 0.108, cnn2d
    # 0.111, cnn3d 0.119, deriv1 0.121, deriv2 0.149, mlp_deriv1 0.096 -- none
    # helps. The attention gates all end at 0.98-1.01: every synthetic channel
    # is an equally informative random mixture, so there is nothing to weight.
    "stage": {
        "attention": dict(attention="input"),
        "cnn2d": dict(model="cnn2d"),
        "cnn3d": dict(model="cnn3d"),
        "deriv1": dict(derivative=1),
        "deriv2": dict(derivative=2),
        "mlp_deriv1": dict(model="mlp", derivative=1),
    },
    # The plain MLP beat every CNN variant in "arch" -- how far does it go?
    "mlp": {
        "base": dict(model="mlp"),
        "seed1": dict(model="mlp", seed=1),
        "seed2": dict(model="mlp", seed=2),
        "h32": dict(model="mlp", hidden_dim=32),
        "h192": dict(model="mlp", hidden_dim=192),
        "h384": dict(model="mlp", hidden_dim=384),
        "n2000": dict(model="mlp", n_samples=2000),
        "n4000": dict(model="mlp", n_samples=4000),
        "n24000": dict(model="mlp", n_samples=24000),
        "n48000": dict(model="mlp", n_samples=48000),
        "noise_aug0.01": dict(model="mlp", noise_augment_std=0.01),
        "wd0": dict(model="mlp", weight_decay=0.0),
        "bs16": dict(model="mlp", batch_size=16),
        "lr3e-4": dict(model="mlp", lr=3e-4),
        "e300": dict(model="mlp", epochs=300),
    },
}


def run_variant(suite, name, overrides, threads):
    """Runs in a worker process."""
    import torch
    torch.set_num_threads(threads)

    from cramer_rao import cnn_predict, cramer_rao_bound_joint, summarize
    from data_generation import generate_dataset
    from error_decomposition import decompose
    from train import run_training

    out_dir = os.path.join(EXPERIMENT_DIR, f"ablation_{suite}", name)
    os.makedirs(out_dir, exist_ok=True)
    cfg = Config(**{**BASE, **overrides, "output_dir": out_dir})
    n_train = cfg.n_samples - int(cfg.n_samples * cfg.val_ratio) - int(cfg.n_samples * cfg.test_ratio)

    start = time.time()
    with open(os.path.join(out_dir, "train.log"), "w") as log, contextlib.redirect_stdout(log):
        out = run_training(cfg)
    minutes = (time.time() - start) / 60

    model = out["model"].eval()
    norm = out["norm"]
    norm = {"mean": norm.mean, "std": norm.std, "y_mean": norm.y_mean, "y_std": norm.y_std}
    def make_eval(noise_std):
        return generate_dataset(
            n_samples=N_EVAL, n_channels=cfg.n_channels, n_wavelength=cfg.n_wavelength,
            wavelength_min=cfg.wavelength_min, wavelength_max=cfg.wavelength_max,
            cd_min=cfg.cd_min, cd_max=cfg.cd_max, n_shared_bases=cfg.n_shared_bases,
            noise_std=noise_std, seed=EVAL_SEED, physics_seed=cfg.physics_seed,
            n_nuisance=cfg.n_nuisance, nuisance_strength=cfg.nuisance_strength,
        )

    eval_set = make_eval(cfg.noise_std)
    y = eval_set.y.astype(np.float64)
    err = cnn_predict(model, norm, eval_set.X) - y
    rms = lambda a: float(np.sqrt((a ** 2).mean()))
    if cfg.n_nuisance == 0:
        parts = decompose(model, cfg, norm)["summary"]
    else:
        # Same split, over the whole (CD, nuisance) space: error on the
        # noise-free version of the evaluation spectra is the systematic
        # part, the rest of the mean square is the noise response.
        clean_err = cnn_predict(model, norm, make_eval(0.0).X) - y
        known, unknown = cramer_rao_bound_joint(y, eval_set.nuisance, eval_set.wavelength, cfg)
        parts = {
            "bias_rms": rms(clean_err),
            "noise_sd_rms": float(np.sqrt(max(rms(err) ** 2 - rms(clean_err) ** 2, 0.0))),
            "bound_rms": rms(unknown),
            "bound_known_rms": rms(known),
        }
    best = min(out["history"], key=lambda h: h["val_loss"])
    extra = {}
    if getattr(model, "attention", None) is not None:
        # Mean input gate per channel over evaluation spectra.
        X_norm = (eval_set.X[:2000] - norm["mean"][None]) / norm["std"][None]
        with torch.no_grad():
            gates = model.channel_attention(torch.from_numpy(X_norm).float())
        extra["channel_attention_mean"] = gates.mean(dim=0).tolist()
        extra["channel_attention_sd"] = gates.std(dim=0).tolist()
    return {
        **extra,
        "overrides": overrides,
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_train": n_train,
        "minutes": minutes,
        "best_epoch": best["epoch"],
        "train_loss_at_best": best["train_loss"],
        "val_loss_at_best": best["val_loss"],
        **summarize(err, cfg.error_threshold_nm),
        "bias_rms": parts["bias_rms"],
        "noise_sd_rms": parts["noise_sd_rms"],
        "bound_rms": parts["bound_rms"],
        "bound_known_rms": parts.get("bound_known_rms", parts["bound_rms"]),
    }


def fmt(name, r):
    return (f"{name:22s} rmse {r['rmse']:.4f} ({r['rmse'] / r['bound_rms']:.2f}x)  "
            f"bias {r['bias_rms']:.4f}  noise_sd {r['noise_sd_rms']:.4f}  "
            f"p99 {r['p99']:.3f}  max {r['max']:.3f}  >thr {r['fraction_over_threshold']:.2%}  "
            f"params {r['n_params'] / 1e3:.0f}k  {r['minutes']:.1f} min")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("suite", choices=sorted(SUITES))
    p.add_argument("--only", nargs="+")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=3)
    args = p.parse_args()

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    result_path = os.path.join(EXPERIMENT_DIR, f"ablation_{args.suite}.json")
    results = {}
    if os.path.exists(result_path):
        with open(result_path) as f:
            results = json.load(f)

    variants = SUITES[args.suite]
    todo = [n for n in (args.only or variants) if n not in results]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_variant, args.suite, n, variants[n], args.threads): n for n in todo}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                results[name] = fut.result()
            except Exception as e:  # one bad variant shouldn't sink the suite
                print(f"{name:22s} FAILED: {e!r}", flush=True)
                continue
            with open(result_path, "w") as f:
                json.dump(results, f, indent=2)
            print(fmt(name, results[name]), flush=True)

    print("\n== summary (sorted by rmse)")
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["rmse"]):
        print(fmt(name, r))


if __name__ == "__main__":
    main()
