"""Mueller spectrum -> preprocessing -> regression network -> CD, plus the
optional refinement step (refinement.py) trained on the same split.

The network is Config.model: "mlp" (default), "cnn" (Multi-channel 1D CNN),
"cnn2d" or "cnn3d".

Data comes from Config.data_source: "synthetic" (data_generation.py, the
stand-in used so far) or "folder" (real spectra plus a table of reference
CD, see real_data.py). Run from the `src` directory or with `src` on
PYTHONPATH:

    python train.py
    python train.py --epochs 50 --n_samples 4000
    python train.py --data_source folder --spectra_dir D:/data/spectra --labels_csv D:/data/labels.csv

`run_training(cfg)` is the reusable entry point -- app.py's "訓練/超參數"
tab calls it directly with a Config built from the UI, so hyperparameters
don't have to be edited in this file or passed on the CLI. cross_validate.py
reuses `fit` and `predict_cd` on its own splits.

Acceptance is judged on the worst case first -- every sample within
cfg.error_threshold_nm (0.3 nm) -- and on RMSE second; error_metrics
reports both.

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
from models import build_model
from refinement import Refinement, decoder_epochs, fit_decoder


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def add_config_arguments(parser: argparse.ArgumentParser, cfg: Config) -> None:
    """One --<field> option per Config field, defaulting to cfg's value."""
    for field, default in cfg.__dict__.items():
        parser.add_argument(f"--{field}", type=type(default), default=default)


def parse_args(cfg: Config) -> Config:
    p = argparse.ArgumentParser()
    add_config_arguments(p, cfg)
    return Config(**vars(p.parse_args()))


def model_arch(cfg: Config) -> dict:
    """models.build_model kwargs -- saved into the checkpoint so inference
    rebuilds exactly the architecture that was trained."""
    arch = dict(model=cfg.model, n_channels=cfg.n_channels, hidden=cfg.hidden_dim,
                dropout=cfg.dropout, n_wavelength=cfg.n_wavelength, n_outputs=1 + cfg.aux_targets)
    if cfg.model == "cnn":
        arch.update(pooling=cfg.pooling, downsample=cfg.downsample, norm=cfg.norm,
                    conv_groups=cfg.conv_groups, attention=cfg.attention)
    if cfg.derivative:
        arch["derivative"] = cfg.derivative
    return arch


class CDLoss(nn.Module):
    """MSE on CD, plus aux_weight x MSE on the auxiliary outputs of a
    multi-task model (output 0 is CD, the rest are nuisance parameters)."""

    def __init__(self, aux_weight: float = 0.0):
        super().__init__()
        self.aux_weight = aux_weight
        self.mse = nn.MSELoss()

    def forward(self, pred, y):
        if pred.ndim == 1:
            return self.mse(pred, y)
        loss = self.mse(pred[:, 0], y[:, 0])
        if self.aux_weight > 0:
            loss = loss + self.aux_weight * self.mse(pred[:, 1:], y[:, 1:])
        return loss


def run_epoch(model, loader, criterion, optimizer, device, train: bool, augment_std=None):
    """augment_std: per-channel noise std in normalized units, shaped
    (n_channels, 1), added fresh to every training batch; None disables."""
    model.train(mode=train)
    total_loss = 0.0
    n = 0
    with torch.set_grad_enabled(train):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            if train and augment_std is not None:
                X = X + torch.randn_like(X) * augment_std
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
    """Normalized CD predictions and targets (auxiliary outputs dropped)."""
    model.eval()
    preds, trues = [], []
    for X, y in loader:
        X = X.to(device)
        pred = model(X).cpu().numpy()
        y = y.numpy()
        if pred.ndim == 2:
            pred, y = pred[:, 0], y[:, 0]
        preds.append(pred)
        trues.append(y)
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


def add_label_noise(y: np.ndarray, train_idx, val_idx, cfg: Config) -> None:
    """Adds cfg.label_noise_std of Gaussian error to the train/val CD labels,
    in place. Deterministic in cfg.seed, so anything that rebuilds the
    training set (experiments/refine.py) gets the very same noisy labels."""
    if cfg.label_noise_std > 0:
        label_rng = np.random.default_rng([cfg.seed, 1])
        noisy = np.concatenate([train_idx, val_idx])
        y[noisy] += label_rng.normal(0, cfg.label_noise_std, size=len(noisy)).astype(np.float32)


def error_metrics(true: np.ndarray, pred: np.ndarray, cfg: Config) -> dict:
    """The acceptance figures -- worst case against cfg.error_threshold_nm
    first, RMSE second -- plus the usual regression metrics."""
    err = np.abs(pred - true)
    metrics = {
        "max_abs_err_nm": float(err.max()),
        "fraction_over_threshold": float((err > cfg.error_threshold_nm).mean()),
        "meets_threshold": bool(err.max() <= cfg.error_threshold_nm),
    }
    metrics.update(regression_metrics(true, pred))
    metrics.update({
        "p95_abs_err_nm": float(np.percentile(err, 95)),
        "p99_abs_err_nm": float(np.percentile(err, 99)),
    })
    return metrics


def acceptance_line(metrics: dict, cfg: Config) -> str:
    return (f"{'PASS' if metrics['meets_threshold'] else 'FAIL'} (every sample within {cfg.error_threshold_nm:g} nm): "
            f"max |error| {metrics['max_abs_err_nm']:.3f} nm, {metrics['fraction_over_threshold']:.2%} over; "
            f"RMSE {metrics['rmse']:.3f} nm")


def load_dataset(cfg: Config):
    """The labelled spectra cfg points at: .X, .y, .wavelength, .nuisance
    and, for real data, .groups / .names."""
    if cfg.data_source == "synthetic":
        return generate_dataset(
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
            n_nuisance=cfg.n_nuisance,
            nuisance_strength=cfg.nuisance_strength,
        )
    if cfg.data_source == "folder":
        from real_data import load_labelled_folder
        if not cfg.spectra_dir or not cfg.labels_csv:
            raise ValueError("data_source='folder' needs spectra_dir and labels_csv")
        return load_labelled_folder(cfg.spectra_dir, cfg.labels_csv, cfg)
    raise ValueError(f"data_source must be 'synthetic' or 'folder', got {cfg.data_source!r}")


def train_refinement(model, cfg: Config, norm: Normalizer, data, X_norm, train_idx, val_idx,
                     progress_cb=None) -> tuple:
    """Trains the decoder on the training split, removes the fitted CD's
    systematic error on the same split (Refinement.calibrate) and sets the
    misfit limit / recommendation on the validation split. Returns (Refinement, {"val_rmse_network",
    "val_rmse_refined"}).

    Decoder inputs are CD plus any nuisance parameters the dataset labels
    (data.nuisance); with none, CD alone.
    """
    params = norm.transform_y(data.y[train_idx])[:, None]
    if data.nuisance is not None and data.nuisance.shape[1] > 0:
        params = np.column_stack([params, data.nuisance[train_idx]])
    epochs = decoder_epochs(cfg.decoder_epochs, len(train_idx))
    decoder = fit_decoder(X_norm[train_idx].reshape(len(train_idx), -1).astype(np.float32),
                          params.astype(np.float32), epochs, seed=cfg.seed, progress_cb=progress_cb)

    norm_dict = {"mean": norm.mean, "std": norm.std, "y_mean": norm.y_mean, "y_std": norm.y_std}
    step = (cfg.wavelength_max - cfg.wavelength_min) / (cfg.n_wavelength - 1)
    refinement = Refinement(decoder, norm_dict, cfg.n_channels, cfg.n_wavelength, step)
    refinement.calibrate(model, data.X[train_idx], data.y[train_idx])

    val = refinement.refine(model, data.X[val_idx])
    rmse = lambda pred: float(np.sqrt(np.mean((pred - data.y[val_idx]) ** 2)))
    scores = {"val_rmse_network": rmse(val["cd_network"]), "val_rmse_refined": rmse(val["cd"])}
    # Too few training spectra make the decoder too rough to help; let the
    # validation split decide which prediction to prefer.
    refinement.recommended = scores["val_rmse_refined"] < scores["val_rmse_network"]
    # Spectra like the training data should essentially never be flagged.
    refinement.misfit_limit = float(np.percentile(val["misfit"], 99.9))
    return refinement, scores


def fit(cfg: Config, data, train_idx, val_idx, progress_cb=None, decoder_cb=None, log=print) -> dict:
    """Trains the network (and, with cfg.refine == "on", the refinement) on
    data[train_idx], selecting the epoch by loss on data[val_idx].

    Returns {"model" (on the CPU, in eval mode), "norm", "arch", "history",
    "refinement" (None when off), "refinement_scores"}.
    """
    if cfg.refine not in ("on", "off"):
        raise ValueError(f"refine must be 'on' or 'off', got {cfg.refine!r}")
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(f"need at least one training and one validation sample, got {len(train_idx)} / {len(val_idx)}")
    n_labelled = data.nuisance.shape[1] if data.nuisance is not None else 0
    if cfg.aux_targets > n_labelled:
        raise ValueError(f"aux_targets ({cfg.aux_targets}) cannot exceed the {n_labelled} labelled nuisance parameters")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    norm = Normalizer.fit(data.X[train_idx], data.y[train_idx])
    X_norm = norm.transform_X(data.X)
    y_norm = norm.transform_y(data.y)
    if cfg.aux_targets > 0:
        # Nuisances have standard deviation 1/sqrt(3) (uniform in [-1, 1]);
        # x sqrt(3) gives them unit variance, the scale of the normalized CD.
        y_norm = np.column_stack([y_norm, data.nuisance[:, :cfg.aux_targets] * np.sqrt(3.0)])

    train_loader = DataLoader(MuellerSpectrumDataset(X_norm[train_idx], y_norm[train_idx]),
                              batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(MuellerSpectrumDataset(X_norm[val_idx], y_norm[val_idx]), batch_size=cfg.batch_size)

    arch = model_arch(cfg)
    model = build_model(arch).to(device)
    criterion = CDLoss(cfg.aux_loss_weight)
    val_criterion = CDLoss()  # CD only, so val loss is comparable across runs
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)
    elif cfg.lr_schedule == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)
    else:
        raise ValueError(f"lr_schedule must be 'cosine' or 'plateau', got {cfg.lr_schedule!r}")

    augment_std = None
    if cfg.noise_augment_std > 0:
        augment_std = torch.from_numpy(cfg.noise_augment_std / norm.std).float().to(device)

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True,
                               augment_std=augment_std)
        val_loss = run_epoch(model, val_loader, val_criterion, optimizer, device, train=False)
        val_stats = nm_error_stats(model, val_loader, norm, device)
        if cfg.lr_schedule == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()
        epoch_log = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                     **{f"val_{k}_nm": v for k, v in val_stats.items()}}
        history.append(epoch_log)
        log(
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
                log(f"Early stopping at epoch {epoch} (no val_loss improvement for {cfg.patience} epochs).")
                break

    model.load_state_dict(best_state)
    model = model.cpu().eval()

    refinement, scores = None, {}
    if cfg.refine == "on":
        refinement, scores = train_refinement(model, cfg, norm, data, X_norm, train_idx, val_idx, decoder_cb)
    return {"model": model, "norm": norm, "arch": arch, "history": history,
            "refinement": refinement, "refinement_scores": scores}


def predict_cd(fitted: dict, X: np.ndarray) -> dict:
    """CD for raw spectra X (N, n_channels, L) from fit()'s result.

    Returns (N,) arrays: "cd_network"; with a refinement also "cd_refined",
    "cd_sd" (its 1-sigma error bar) and "flagged"; and "cd", the one to
    report -- the refined value when the refinement was recommended on the
    validation split, else the network's.
    """
    model, norm, refinement = fitted["model"], fitted["norm"], fitted["refinement"]
    if refinement is None:
        with torch.no_grad():
            out = model(torch.from_numpy(norm.transform_X(X)).float()).numpy()
        cd = norm.inverse_transform_y(out.reshape(len(X), -1)[:, 0].astype(np.float64))
        return {"cd_network": cd, "cd": cd}
    r = refinement.refine(model, X)
    return {"cd_network": r["cd_network"], "cd_refined": r["cd"], "cd_sd": r["cd_sd"], "flagged": r["flagged"],
            "cd": r["cd"] if refinement.recommended else r["cd_network"]}


def checkpoint_dict(cfg: Config, fitted: dict) -> dict:
    norm, refinement = fitted["norm"], fitted["refinement"]
    return {
        "model_state": fitted["model"].state_dict(),
        "config": cfg.__dict__,
        "arch": fitted["arch"],
        "norm_mean": norm.mean,
        "norm_std": norm.std,
        "norm_y_mean": norm.y_mean,
        "norm_y_std": norm.y_std,
        **({"refinement": refinement.state()} if refinement is not None else {}),
    }


def run_training(cfg: Config, progress_cb=None, decoder_cb=None) -> dict:
    """Runs the full train/val/test pipeline for one Config and writes the
    checkpoint + history/metrics json to cfg.output_dir.

    progress_cb(epoch, max_epochs, epoch_log: dict), if given, is called
    after every network epoch -- e.g. so a UI can show live progress;
    decoder_cb(epoch, max_epochs) likewise for the refinement decoder.

    With cfg.test_ratio == 0 every sample goes to training / validation and
    no test metrics are reported -- for the model to deploy once
    cross_validate.py has estimated the performance.

    Returns {"model", "cfg", "norm", "history", "metrics", "refinement"}
    ("refinement" is None with cfg.refine == "off").
    """
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    data = load_dataset(cfg)
    train_idx, val_idx, test_idx = split_indices(len(data.y), cfg.val_ratio, cfg.test_ratio, cfg.seed,
                                                 groups=getattr(data, "groups", None))
    add_label_noise(data.y, train_idx, val_idx, cfg)

    fitted = fit(cfg, data, train_idx, val_idx, progress_cb, decoder_cb)
    model, norm, refinement, history = fitted["model"], fitted["norm"], fitted["refinement"], fitted["history"]

    has_test = len(test_idx) > 0
    metrics = {}
    if has_test:
        test = predict_cd(fitted, data.X[test_idx])
        true = data.y[test_idx]
        pred = test["cd_network"].astype(np.float32)
        metrics = error_metrics(true, pred, cfg)
        print("Test metrics:", metrics)
        print("Network:", acceptance_line(metrics, cfg))
    else:
        print("No test split: every sample was used for training / validation. "
              "cross_validate.py estimates the performance.")
    metrics["n_train"], metrics["n_val"], metrics["n_test"] = len(train_idx), len(val_idx), len(test_idx)

    if refinement is not None:
        metrics.update(fitted["refinement_scores"])
        metrics["refinement_recommended"] = bool(refinement.recommended)
        if not refinement.recommended:
            print("Refinement NOT recommended: worse than the network on validation")
    if refinement is not None and has_test:
        metrics["refined"] = error_metrics(true, test["cd_refined"].astype(np.float32), cfg)
        metrics["refined"]["fraction_flagged"] = float(test["flagged"].mean())
        print("Refined test metrics:", metrics["refined"])
        print("Refined:", acceptance_line(metrics["refined"], cfg))

    torch.save(checkpoint_dict(cfg, fitted), os.path.join(cfg.output_dir, "multichannel_cnn.pt"))
    with open(os.path.join(cfg.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(cfg.output_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2 if has_test else 1, figsize=(10 if has_test else 5, 4), squeeze=False)
        axes = axes[0]
        axes[0].plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="train")
        axes[0].plot([h["epoch"] for h in history], [h["val_loss"] for h in history], label="val")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("MSE loss (normalized y)")
        axes[0].legend()
        axes[0].set_title("Training curve")

        if has_test:
            axes[1].scatter(true, pred, s=8, alpha=0.5)
            lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
            axes[1].plot(lims, lims, "r--", linewidth=1)
            axes[1].set_xlabel("true CD (nm)")
            axes[1].set_ylabel("predicted CD (nm)")
            title = f"Test (network): MAE={metrics['mae']:.3f}  max_err={metrics['max_abs_err_nm']:.3f}"
            if refinement is not None:
                title += (f"\nrefined: MAE={metrics['refined']['mae']:.3f}  "
                          f"max_err={metrics['refined']['max_abs_err_nm']:.3f}")
            axes[1].set_title(title, fontsize=10)

        fig.tight_layout()
        fig.savefig(os.path.join(cfg.output_dir, "train_report.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        pass

    return {"model": model, "cfg": cfg, "norm": norm, "history": history, "metrics": metrics,
            "refinement": refinement}


def main():
    cfg = parse_args(Config())
    run_training(cfg)


if __name__ == "__main__":
    main()
