"""Spectrum loading / on-demand simulation for the viewer app.

Real-data file format is not finalized yet. Until then, two placeholder
formats are accepted so the app is usable end-to-end:

  * .csv  -- header row "wavelength,M00,M01,...,M33" (16 element columns),
             one row per wavelength sample.
  * .npy  -- array of shape (16, L) or (L, 16); wavelength axis is inferred
             from Config's wavelength_min/max/n_wavelength (no wavelength
             values stored in the file).

Replace `load_spectrum_file` once the real instrument/RCWA export format is
known -- everything downstream (`resample_to_grid`, the model, the app)
only depends on getting back (X: (16, L) float32, wavelength: (L,) float32).
"""
from typing import Optional, Tuple

import numpy as np

from config import Config
from data_generation import generate_dataset

MUELLER_LABELS = [f"M{i}{j}" for i in range(4) for j in range(4)]


def generate_single_spectrum(cd_value: float, cfg: Config, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate one spectrum at a chosen CD, for testing the app without real data."""
    ds = generate_dataset(
        n_samples=1,
        n_channels=cfg.n_channels,
        n_wavelength=cfg.n_wavelength,
        wavelength_min=cfg.wavelength_min,
        wavelength_max=cfg.wavelength_max,
        cd_min=cd_value,
        cd_max=cd_value,
        n_shared_bases=cfg.n_shared_bases,
        noise_std=cfg.noise_std,
        seed=seed if seed is not None else np.random.randint(0, 1_000_000),
        physics_seed=cfg.physics_seed,
    )
    return ds.X[0], ds.wavelength


def load_spectrum_file(file, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """file: a file-like object (e.g. Streamlit UploadedFile). Returns (X, wavelength)."""
    name = getattr(file, "name", "")
    if name.lower().endswith(".npy"):
        arr = np.load(file)
        if arr.shape[0] == cfg.n_channels:
            X = arr
        elif arr.shape[-1] == cfg.n_channels:
            X = arr.T
        else:
            raise ValueError(f"Expected an axis of length {cfg.n_channels} (Mueller elements), got shape {arr.shape}")
        wavelength = np.linspace(cfg.wavelength_min, cfg.wavelength_max, X.shape[1])
        return X.astype(np.float32), wavelength.astype(np.float32)

    if name.lower().endswith(".csv"):
        import csv
        import io

        text = file.read()
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        reader = csv.reader(io.StringIO(text))
        header = next(reader)
        header = [h.strip() for h in header]
        if header[0].lower() != "wavelength":
            raise ValueError('CSV must start with a "wavelength" column, followed by 16 Mueller element columns')
        rows = [row for row in reader if row]
        data = np.array(rows, dtype=np.float32)
        wavelength = data[:, 0]
        X = data[:, 1:].T  # (n_channels, L)
        if X.shape[0] != cfg.n_channels:
            raise ValueError(f"Expected {cfg.n_channels} Mueller element columns, got {X.shape[0]}")
        return X, wavelength

    raise ValueError(f"Unsupported file type: {name}. Use .csv or .npy")


def resample_to_grid(X: np.ndarray, wavelength: np.ndarray, target_wavelength: np.ndarray) -> np.ndarray:
    """Linearly interpolate each channel onto the model's training wavelength grid."""
    if X.shape[1] == len(target_wavelength) and np.allclose(wavelength, target_wavelength, atol=1e-6):
        return X
    out = np.zeros((X.shape[0], len(target_wavelength)), dtype=np.float32)
    for c in range(X.shape[0]):
        out[c] = np.interp(target_wavelength, wavelength, X[c])
    return out
