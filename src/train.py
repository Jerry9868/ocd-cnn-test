"""Stage 1: Mueller spectrum -> preprocessing -> Multi-channel 1D CNN -> CD regression.

Uses synthetic data (data_generation.py) as a stand-in until real measured
spectra are available. Run from the `src` directory or with `src` on
PYTHONPATH:

    python train.py
    python train.py --epochs 50 --n_samples 4000

`run_training(cfg)` is the reusable entry point -- app.py's "訓練/超參數"
tab calls it directly with a Config built from the UI, so hyperparameters
don't have to be edited in this file or passed on the CLI.

Trains with plain MSE and selects the checkpoint with the lowest validation
MSE -- this is stable and reproducible. Default setup (2026-09-30): flatten
head (keeps wavelength position), no dropout, cosine LR over a fixed number
of epochs -- see config.py. The legacy setup (avg pooling, dropout 0.2,
ReduceLROnPlateau + early stopping) is still selectable via Config. An earlier attempt to select by
worst-case (p99) validation error, with a loss penalty for residuals beyond
cfg.error_threshold_nm, made training *less* stable and produced worse
results on this architecture/data. The per-epoch and final nm-scale error
stats (mean/p95/p99/max) are still reported so the worst-case gap is
visible, just not optimized against directly.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from data_generation import generate_dataset
from dataset import MuellerSpectrumDataset, Normalizer, split_indices
from metrics import regression_metrics
from models import MultiChannelCNN1D


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args(cfg: Config) -> Config:
    p = argparse.ArgumentParser()
    for field, default in cfg.__dict__.items():
        p.add_argument(f"--{field}", type=type(default), default=default)
    args = p.parse_args()
    return Config(**vars(args))


def model_arch(cfg: Config) -> dict:
    """Constructor kwargs for MultiChannelCNN1D -- saved into the checkpoint
    so inference rebuilds exactly the architecture that was trained."""
    return dict(n_channels=cfg.n_channels, hidden=cfg.hidden_dim, dropout=cfg.dropout,
                pooling=cfg.pooling, n_wavelength=cfg.n_wavelength)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss = 0.0
    n = 0
    with torch.set_grad_enabled(train):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            n += len(y)
    return total_loss / n


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    for X, y in loader:
        X = X.to(device)
        pred = model(X).cpu().numpy()
        preds.append(pred)
        trues.append(y.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def nm_error_stats(model, loader, norm: Normalizer, device) -> dict:
    pred_norm, true_norm = predict(model, loader, device)
    pred = norm.inverse_transform_y(pred_norm)
    true = norm.inverse_transform_y(true_norm)
    err = np.abs(pred - true)
    return {
        "mean": float(err.mean()),
        "p95": float(np.percentile(err, 95)),
        "p99": float(np.percentile(err, 99)),
        "max": float(err.max()),
    }


def run_training(cfg: Config, progress_cb=None) -> dict:
    """Runs the full train/val/test pipeline for one Config and writes the
    checkpoint + history/metrics json to cfg.output_dir.

    progress_cb(epoch, max_epochs, epoch_log: dict), if given, is called
    after every epoch -- e.g. so a UI can show live progress.

    Returns {"model", "cfg", "norm", "history", "metrics"}.
    """
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.output_dir, exist_ok=True)

    data = generate_dataset(
        n_samples=cfg.n_samples,
        n_channels=cfg.n_channels,
        n_wavelength=cfg.n_wavelength,
        wavelength_min=cfg.wavelength_min,
        wavelength_max=cfg.wavelength_max,
        cd_min=cfg.cd_min,
        cd_max=cfg.cd_max,
        n_shared_bases=cfg.n_shared_bases,
        noise_std=cfg.noise_std,
        seed=cfg.seed,
        physics_seed=cfg.physics_seed,
    )

    train_idx, val_idx, test_idx = split_indices(cfg.n_samples, cfg.val_ratio, cfg.test_ratio, cfg.seed)

    norm = Normalizer.fit(data.X[train_idx], data.y[train_idx])
    X_norm = norm.transform_X(data.X)
    y_norm = norm.transform_y(data.y)

    train_ds = MuellerSpectrumDataset(X_norm[train_idx], y_norm[train_idx])
    val_ds = MuellerSpectrumDataset(X_norm[val_idx], y_norm[val_idx])
    test_ds = MuellerSpectrumDataset(X_norm[test_idx], y_norm[test_idx])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    arch = model_arch(cfg)
    model = MultiChannelCNN1D(**arch).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)
    elif cfg.lr_schedule == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)
    else:
        raise ValueError(f"lr_schedule must be 'cosine' or 'plateau', got {cfg.lr_schedule!r}")

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        val_stats = nm_error_stats(model, val_loader, norm, device)
        if cfg.lr_schedule == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()
        epoch_log = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                     **{f"val_{k}_nm": v for k, v in val_stats.items()}}
        history.append(epoch_log)
        print(
            f"epoch {epoch:3d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
            f"val_mean {val_stats['mean']:.3f}nm  val_p99 {val_stats['p99']:.3f}nm  val_max {val_stats['max']:.3f}nm  "
            f"lr {optimizer.param_groups[0]['lr']:.2e}"
        )
        if progress_cb is not None:
            progress_cb(epoch, cfg.epochs, epoch_log)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if cfg.lr_schedule == "plateau" and epochs_no_improve >= cfg.patience:
                print(f"Early stopping at epoch {epoch} (no val_loss improvement for {cfg.patience} epochs).")
                break

    model.load_state_dict(best_state)

    pred_norm, true_norm = predict(model, test_loader, device)
    pred = norm.inverse_transform_y(pred_norm)
    true = norm.inverse_transform_y(true_norm)
    metrics = regression_metrics(true, pred)
    err = np.abs(pred - true)
    metrics.update({
        "p95_abs_err_nm": float(np.percentile(err, 95)),
        "p99_abs_err_nm": float(np.percentile(err, 99)),
        "max_abs_err_nm": float(err.max()),
        "fraction_over_threshold": float((err > cfg.error_threshold_nm).mean()),
    })
    print("Test metrics:", metrics)

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.__dict__,
            "arch": arch,
            "norm_mean": norm.mean,
            "norm_std": norm.std,
            "norm_y_mean": norm.y_mean,
            "norm_y_std": norm.y_std,
        },
        os.path.join(cfg.output_dir, "multichannel_cnn.pt"),
    )
    with open(os.path.join(cfg.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(cfg.output_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="train")
        axes[0].plot([h["epoch"] for h in history], [h["val_loss"] for h in history], label="val")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("MSE loss (normalized y)")
        axes[0].legend()
        axes[0].set_title("Training curve")

        axes[1].scatter(true, pred, s=8, alpha=0.5)
        lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
        axes[1].plot(lims, lims, "r--", linewidth=1)
        axes[1].set_xlabel("true CD (nm)")
        axes[1].set_ylabel("predicted CD (nm)")
        axes[1].set_title(f"Test: MAE={metrics['mae']:.3f}  max_err={metrics['max_abs_err_nm']:.3f}")

        fig.tight_layout()
        fig.savefig(os.path.join(cfg.output_dir, "train_report.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        pass

    return {"model": model, "cfg": cfg, "norm": norm, "history": history, "metrics": metrics}


def main():
    cfg = parse_args(Config())
    run_training(cfg)


if __name__ == "__main__":
    main()
