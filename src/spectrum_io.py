"""Spectrum loading / on-demand simulation for the viewer app.

Real-data file format is not finalized yet. Until then, several placeholder
formats are accepted so the app is usable end-to-end:

  * .csv  -- header row "wavelength,M00,M01,...,M33,TE,TM" (cfg.n_channels
             element columns), one row per wavelength sample.
  * .npy  -- array of shape (n_channels, L) or (L, n_channels); wavelength
             axis is inferred from Config's wavelength_min/max/n_wavelength
             (no wavelength values stored in the file).
  * .dat  -- whitespace/tab-delimited table, same column layout as .csv.
             GUESSED format -- the real instrument's .dat layout is not yet
             confirmed. Update `_parse_table_text` once a real sample is
             available.
  * .zip  -- a zip containing .dat file(s), matching how the real export
             will apparently be packaged. Two layouts are accepted:
               - exactly one .dat file: parsed as the whitespace-delimited
                 table above (wavelength + cfg.n_channels columns).
               - exactly cfg.n_channels .dat files: each a 2-column
                 (wavelength, value) file for ONE channel, assigned to
                 channels in filename-sorted order (assumed to match
                 CHANNEL_LABELS order -- M00..M33, TE, TM). This is a guess;
                 confirm the real per-channel naming/order once known.
             Any other file count raises with an explanation, rather than
             silently guessing wrong.

Replace/adjust these once the real instrument export format is confirmed --
everything downstream (`resample_to_grid`, the model, the app) only depends
on getting back (X: (n_channels, L) float32, wavelength: (L,) float32).
"""
import io
import re
import zipfile
from typing import Optional, Tuple

import numpy as np

from config import Config
from data_generation import generate_dataset

# The 16 Mueller elements (65-degree oblique incidence) plus the two
# normal-incidence reference spectra (TE, TM) -- see data_generation.py.
# Index order must match Config.n_channels / the trained model's channel axis.
CHANNEL_LABELS = [f"M{i}{j}" for i in range(4) for j in range(4)] + ["TE", "TM"]
MUELLER_LABELS = CHANNEL_LABELS  # backward-compatible alias


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


def _decode(raw) -> str:
    return raw.decode("utf-8") if isinstance(raw, bytes) else raw


def _parse_table_text(text: str) -> np.ndarray:
    """Parse a comma- or whitespace-delimited numeric table.

    Skips blank lines and '#' comment lines, and a non-numeric header row
    if present. Returns shape (n_rows, n_cols).
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",") if "," in line else re.split(r"\s+", line)
        parts = [p for p in parts if p != ""]
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue  # header row or other non-numeric line
    if not rows:
        raise ValueError("No numeric data rows found")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("Inconsistent column count across rows")
    return np.array(rows, dtype=np.float32)


def _table_to_X_wavelength(data: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    wavelength = data[:, 0]
    X = data[:, 1:].T  # (n_channels, L)
    if X.shape[0] != cfg.n_channels:
        raise ValueError(f"Expected {cfg.n_channels} element columns (16 Mueller + TE + TM), got {X.shape[0]}")
    return X, wavelength


def _load_zip(raw_bytes: bytes, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        dat_names = sorted(n for n in zf.namelist() if n.lower().endswith(".dat"))
        if len(dat_names) == 1:
            text = _decode(zf.read(dat_names[0]))
            return _table_to_X_wavelength(_parse_table_text(text), cfg)

        if len(dat_names) == cfg.n_channels:
            channels = []
            wavelength = None
            for name in dat_names:
                data = _parse_table_text(_decode(zf.read(name)))
                if data.shape[1] < 2:
                    raise ValueError(f"{name}: expected at least 2 columns (wavelength, value)")
                if wavelength is None:
                    wavelength = data[:, 0]
                channels.append(data[:, 1])
            return np.stack(channels, axis=0).astype(np.float32), wavelength.astype(np.float32)

        raise ValueError(
            f"zip 內有 {len(dat_names)} 個 .dat 檔，預期是 1 個（多欄位表格）或 {cfg.n_channels} 個"
            f"（每個 channel 一個檔案）。目前的解析規則是暫時猜測，請確認實際的 zip 內容結構。"
        )


def load_spectrum_file(file, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """file: a file-like object (e.g. Streamlit UploadedFile). Returns (X, wavelength)."""
    name = getattr(file, "name", "")
    lower = name.lower()

    if lower.endswith(".npy"):
        arr = np.load(file)
        if arr.shape[0] == cfg.n_channels:
            X = arr
        elif arr.shape[-1] == cfg.n_channels:
            X = arr.T
        else:
            raise ValueError(f"Expected an axis of length {cfg.n_channels} (16 Mueller elements + TE + TM), got shape {arr.shape}")
        wavelength = np.linspace(cfg.wavelength_min, cfg.wavelength_max, X.shape[1])
        return X.astype(np.float32), wavelength.astype(np.float32)

    if lower.endswith(".zip"):
        return _load_zip(file.read(), cfg)

    if lower.endswith(".csv") or lower.endswith(".dat"):
        text = _decode(file.read())
        return _table_to_X_wavelength(_parse_table_text(text), cfg)

    raise ValueError(f"Unsupported file type: {name}. Use .csv, .dat, .npy, or .zip")


def resample_to_grid(X: np.ndarray, wavelength: np.ndarray, target_wavelength: np.ndarray) -> np.ndarray:
    """Linearly interpolate each channel onto the model's training wavelength grid."""
    if X.shape[1] == len(target_wavelength) and np.allclose(wavelength, target_wavelength, atol=1e-6):
        return X
    out = np.zeros((X.shape[0], len(target_wavelength)), dtype=np.float32)
    for c in range(X.shape[0]):
        out[c] = np.interp(target_wavelength, wavelength, X[c])
    return out
