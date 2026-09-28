import os
from dataclasses import dataclass

# src/ -> project root, so output_dir is stable regardless of the caller's cwd.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


@dataclass
class Config:
    # Spectrum shape
    n_channels: int = 16          # M00..M33
    n_wavelength: int = 101
    wavelength_min: float = 200.0  # nm
    wavelength_max: float = 1000.0  # nm

    # Ground-truth CD range (nm), synthetic stand-in until real data is available
    cd_min: float = 20.0
    cd_max: float = 80.0

    # Synthetic data generation.
    # physics_seed fixes the CD<->spectrum relationship itself and must stay
    # the same across every generate_dataset call for a given model (training
    # set, app-side single-spectrum preview, etc.) -- see data_generation.py.
    # seed only controls which CD values get sampled and the noise draw.
    n_samples: int = 12000
    n_shared_bases: int = 4
    noise_std: float = 0.01
    seed: int = 42
    physics_seed: int = 2024

    # Split ratios
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Model
    hidden_dim: int = 96

    # Training
    batch_size: int = 64
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 25  # early stopping on val loss

    # Reporting only -- what counts as an "acceptable" error, for the
    # fraction_over_threshold test metric. Not currently met on every sample;
    # see PROJECT_NOTES.md.
    error_threshold_nm: float = 0.30

    # Output
    output_dir: str = DEFAULT_OUTPUT_DIR
