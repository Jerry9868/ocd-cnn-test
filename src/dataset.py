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


def _shuffled_by_group(n: int, seed: int, groups) -> np.ndarray:
    """A random order of range(n). With groups (one label per sample), the
    members of a group stay next to each other, so cutting the order into
    pieces between groups never spreads a group over two pieces."""
    rng = np.random.default_rng(seed)
    if groups is None:
        return rng.permutation(n)
    groups = np.asarray(groups)
    labels = np.unique(groups)
    return np.concatenate([np.flatnonzero(groups == g) for g in rng.permutation(labels)])


def _group_cuts(order: np.ndarray, groups, targets: list) -> list:
    """Positions at which to cut `order` into len(targets) + 1 pieces of
    about the cumulative sizes in `targets`, moved to the nearest boundary
    between groups."""
    if groups is None:
        return targets
    labels = np.asarray(groups)[order]
    boundaries = np.concatenate([[0], np.flatnonzero(labels[1:] != labels[:-1]) + 1, [len(order)]])
    return [int(boundaries[np.abs(boundaries - t).argmin()]) for t in targets]


def split_indices(n: int, val_ratio: float, test_ratio: float, seed: int, groups=None):
    """Random (train, val, test) index arrays. groups: optional label per
    sample (e.g. wafer); samples sharing one all land in the same split, at
    the cost of split sizes only approximately matching the ratios."""
    order = _shuffled_by_group(n, seed, groups)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    a, b = _group_cuts(order, groups, [n_val, n_val + n_test])
    return order[b:], order[:a], order[a:b]


def kfold_indices(n: int, k: int, seed: int, groups=None) -> list:
    """k disjoint held-out index arrays covering range(n), for k-fold
    cross-validation. groups: as in split_indices."""
    order = _shuffled_by_group(n, seed, groups)
    cuts = [0] + _group_cuts(order, groups, [round(n * i / k) for i in range(1, k)]) + [n]
    folds = [order[a:b] for a, b in zip(cuts[:-1], cuts[1:])]
    if any(len(f) == 0 for f in folds):
        raise ValueError(f"cannot make {k} non-empty folds from {n} samples"
                         + (f" in {len(np.unique(groups))} groups" if groups is not None else ""))
    return folds
