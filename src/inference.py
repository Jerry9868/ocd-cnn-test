import numpy as np
import torch

from config import Config
from models import MultiChannelCNN1D


def load_checkpoint(path: str):
    # weights_only=False: this checkpoint is always one we produced ourselves
    # (train.py) and includes numpy normalization arrays alongside the model
    # state dict, which PyTorch's default safe unpickler rejects.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(**ckpt["config"])
    model = MultiChannelCNN1D(n_channels=cfg.n_channels, hidden=cfg.hidden_dim)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    norm = {
        "mean": ckpt["norm_mean"],
        "std": ckpt["norm_std"],
        "y_mean": ckpt["norm_y_mean"],
        "y_std": ckpt["norm_y_std"],
    }
    return model, cfg, norm


@torch.no_grad()
def predict_cd(model, norm: dict, X: np.ndarray) -> float:
    """X: (n_channels, L) raw (unnormalized) spectrum."""
    X_norm = (X - norm["mean"]) / norm["std"]
    x = torch.from_numpy(X_norm[None]).float()
    pred_norm = model(x).item()
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
    pred_norm = model(x)
    pred_norm.backward()
    grad_norm = x.grad[0].numpy()  # (C, L), d(pred_norm)/d(X_norm)
    # X_norm = (X_raw - mean) / std  =>  d(X_norm)/d(X_raw) = 1/std
    sensitivity = grad_norm * norm["y_std"] / norm["std"]
    return np.abs(sensitivity)
