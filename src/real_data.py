"""Labelled real spectra: a folder of spectrum files plus a table of reference CD.

    Config(data_source="folder", spectra_dir=".../spectra", labels_csv=".../labels.csv")

The labels table is a .csv with a header row and one row per sample:

    filename,CD_nm,wafer,height_nm
    W01_site03.csv,45.21,W01,62.0
    ...

  * filename -- a file inside spectra_dir, in any format
    spectrum_io.load_spectrum_file reads (.csv / .dat / .npy / .zip).
  * the reference CD in nm (e.g. from TEM) -- the first column present of
    LABEL_COLUMNS.
  * optional: which wafer / lot the sample is from -- the first column
    present of GROUP_COLUMNS. Samples sharing a value are kept together when
    data is split (dataset.split_indices, dataset.kfold_indices), so a model
    is never scored on a wafer it trained on.
  * optional: any further numeric columns are other reference structure
    parameters (height, sidewall angle, ...). They become decoder inputs for
    the refinement and can be multi-task targets (Config.aux_targets).

Column names are matched ignoring case. Every spectrum is resampled onto
the Config wavelength grid, which it must cover.

Nothing here leaves the machine: files are read from local paths only.
"""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import Config
from spectrum_io import load_spectrum_file, resample_to_grid

FILENAME_COLUMNS = ("filename", "file")
LABEL_COLUMNS = ("cd_nm", "reference_cd_nm", "cd_tem_nm", "tem_cd_nm", "cd")
GROUP_COLUMNS = ("group", "wafer", "wafer_id", "lot")


@dataclass
class LabelledDataset:
    """Same fields training reads from data_generation.SyntheticDataset,
    plus where each sample came from."""
    X: np.ndarray           # (n_samples, n_channels, n_wavelength)
    y: np.ndarray           # (n_samples,) reference CD (nm)
    wavelength: np.ndarray  # (n_wavelength,)
    nuisance: np.ndarray    # (n_samples, n_extra) other reference parameters, standardized
    names: list             # filename of each sample
    groups: np.ndarray = None       # (n_samples,) wafer / lot label, or None
    extra_columns: tuple = ()       # labels-table column behind each nuisance column


def _first_present(columns: dict, candidates: tuple):
    return next((columns[c] for c in candidates if c in columns), None)


def read_labels(labels_csv: str) -> tuple:
    """Returns (table, filename column, CD column, group column or None,
    list of extra numeric columns)."""
    table = pd.read_csv(labels_csv, encoding="utf-8-sig")
    lower = {str(c).strip().lower(): c for c in table.columns}
    name_col = _first_present(lower, FILENAME_COLUMNS)
    cd_col = _first_present(lower, LABEL_COLUMNS)
    if name_col is None or cd_col is None:
        raise ValueError(
            f"{labels_csv}: need a filename column ({', '.join(FILENAME_COLUMNS)}) and a reference CD column "
            f"({', '.join(LABEL_COLUMNS)}); found {list(table.columns)}"
        )
    group_col = _first_present(lower, GROUP_COLUMNS)
    extras = [c for c in table.columns
              if c not in (name_col, cd_col, group_col) and pd.api.types.is_numeric_dtype(table[c])]
    return table, name_col, cd_col, group_col, extras


def load_labelled_folder(spectra_dir: str, labels_csv: str, cfg: Config) -> LabelledDataset:
    table, name_col, cd_col, group_col, extras = read_labels(labels_csv)
    if table[cd_col].isna().any() or table[extras].isna().any().any():
        raise ValueError(f"{labels_csv}: empty cells in the CD or reference-parameter columns")

    names = [str(n).strip() for n in table[name_col]]
    missing = [n for n in names if not os.path.isfile(os.path.join(spectra_dir, n))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} file(s) listed in {labels_csv} are not in {spectra_dir}: "
                                f"{missing[:5]}{' ...' if len(missing) > 5 else ''}")

    grid = np.linspace(cfg.wavelength_min, cfg.wavelength_max, cfg.n_wavelength)
    X = np.empty((len(names), cfg.n_channels, cfg.n_wavelength), dtype=np.float32)
    for i, name in enumerate(names):
        with open(os.path.join(spectra_dir, name), "rb") as f:
            spectrum, wavelength = load_spectrum_file(f, cfg)
        # np.interp would silently hold the end values flat outside the measured range.
        if wavelength.min() > grid[0] + 1e-6 or wavelength.max() < grid[-1] - 1e-6:
            raise ValueError(
                f"{name}: measured {wavelength.min():g}-{wavelength.max():g} nm does not cover the model grid "
                f"{grid[0]:g}-{grid[-1]:g} nm. Set Config.wavelength_min / wavelength_max inside the measured range."
            )
        X[i] = resample_to_grid(spectrum, wavelength, grid)

    nuisance = np.zeros((len(names), 0), dtype=np.float32)
    if extras:
        values = table[extras].to_numpy(dtype=np.float64)
        spread = values.std(axis=0)
        spread[spread == 0] = 1.0
        # Standard deviation 1/sqrt(3), the scale of the synthetic
        # generator's uniform [-1, 1] nuisances that train.py assumes.
        nuisance = ((values - values.mean(axis=0)) / spread / np.sqrt(3.0)).astype(np.float32)

    return LabelledDataset(
        X=X, y=table[cd_col].to_numpy(dtype=np.float32), wavelength=grid, nuisance=nuisance, names=names,
        groups=table[group_col].astype(str).to_numpy() if group_col is not None else None,
        extra_columns=tuple(extras),
    )
