import dataclasses

import numpy as np
import torch

from config import Config
from models import build_model
from refinement import Decoder, Refinement


def load_checkpoint(path: str):
    # weights_only=False: this checkpoint is always one we produced ourselves
    # (train.py) and includes numpy normalization arrays alongside the model
    # state dict, which PyTorch's default safe unpickler rejects.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    known = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(**{k: v for k, v in ckpt["config"].items() if k in known})
    arch = ckpt.get("arch")
    if arch is None:
        # Checkpoint from before the architecture was configurable: global
        # average pooling + dropout 0.2 (dropout is inactive in eval anyway).
        arch = dict(n_channels=cfg.n_channels, hidden=cfg.hidden_dim, dropout=0.2,
                    pooling="avg", n_wavelength=cfg.n_wavelength)
        cfg.pooling, cfg.dropout = "avg", 0.2
    cfg.model = arch.get("model", "cnn")  # checkpoints from before Config.model existed are CNNs
    model = build_model(arch)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    norm = {
        "mean": ckpt["norm_mean"],
        "std": ckpt["norm_std"],
        "y_mean": ckpt["norm_y_mean"],
        "y_std": ckpt["norm_y_std"],
    }
    return model, cfg, norm


def load_refinement(path: str):
    """The Refinement stored in a checkpoint, or None if it was trained
    without one (cfg.refine == "off", or before refinement existed)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("refinement")
    if state is None:
        return None
    cfg = ckpt["config"]
    decoder = Decoder(state["n_in"], state["n_out"], state["width"])
    decoder.load_state_dict(state["decoder_state"])
    decoder.eval()
    norm = {"mean": ckpt["norm_mean"], "std": ckpt["norm_std"],
            "y_mean": ckpt["norm_y_mean"], "y_std": ckpt["norm_y_std"]}
    step = (cfg["wavelength_max"] - cfg["wavelength_min"]) / (cfg["n_wavelength"] - 1)
    return Refinement(decoder, norm, cfg["n_channels"], cfg["n_wavelength"], step,
                      misfit_limit=state["misfit_limit"], recommended=state["recommended"],
                      calibration=state.get("calibration"))


def predict_cd_refined(model, refinement: Refinement, X: np.ndarray, fit_offset: bool = False) -> dict:
    """X: (n_channels, L) raw spectrum. Refinement.refine for one spectrum,
    with plain Python values instead of arrays."""
    out = refinement.refine(model, X[None].astype(np.float32), fit_offset=fit_offset)
    return {k: (bool(v[0]) if k == "flagged" else float(v[0])) for k, v in out.items()}


@torch.no_grad()
def predict_cd(model, norm: dict, X: np.ndarray) -> float:
    """X: (n_channels, L) raw (unnormalized) spectrum."""
    X_norm = (X - norm["mean"]) / norm["std"]
    x = torch.from_numpy(X_norm[None]).float()
    pred_norm = model(x).reshape(-1)[0].item()  # output 0 is CD (multi-task models have more)
    return pred_norm * norm["y_std"] + norm["y_mean"]


def sensitivity_map(model, norm: dict, X: np.ndarray) -> np.ndarray:
    """d(predicted CD in nm) / d(raw spectrum value), per (channel, wavelength).

    Used to highlight which wavelength regions the trained model actually
    relies on for a given spectrum -- a gradient-based saliency map, so it
    reflects what the model learned rather than an assumption baked into the
    synthetic generator (and will keep working once real spectra replace it).
    """
    model.eval()
    X_norm = (X - norm["mean"]) / norm["std"]
    x = torch.from_numpy(X_norm[None]).float()
    x.requires_grad_(True)
    pred_norm = model(x).reshape(-1)[0]
    pred_norm.backward()
    grad_norm = x.grad[0].numpy()  # (C, L), d(pred_norm)/d(X_norm)
    # X_norm = (X_raw - mean) / std  =>  d(X_norm)/d(X_raw) = 1/std
    sensitivity = grad_norm * norm["y_std"] / norm["std"]
    return np.abs(sensitivity)
