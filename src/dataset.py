"""Splitting, normalization and torch Dataset wrapper for Mueller spectra.

Swapping in real data later: replace the array source (currently
`data_generation.generate_dataset`) with a loader that returns the same
(X: [N, 16, L] float32, y: [N] float32, wavelength: [L] float32) shapes --
everything below is agnostic to where X/y came from.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Normalizer:
    """Per-channel z-score, fit on the training split only."""
    mean: np.ndarray  # (n_channels, 1)
    std: np.ndarray   # (n_channels, 1)
    y_mean: float
    y_std: float

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray) -> "Normalizer":
        mean = X.mean(axis=(0, 2), keepdims=True)[0]
        std = X.std(axis=(0, 2), keepdims=True)[0]
        std[std == 0] = 1.0
        return cls(mean=mean, std=std, y_mean=float(y.mean()), y_std=float(y.std() or 1.0))

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean[None]) / self.std[None]

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / self.y_std

    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.y_std + self.y_mean


class MuellerSpectrumDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def split_indices(n: int, val_ratio: float, test_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    val_idx = idx[:n_val]
    test_idx = idx[n_val:n_val + n_test]
    train_idx = idx[n_val + n_test:]
    return train_idx, val_idx, test_idx
