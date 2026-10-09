import os
from dataclasses import dataclass

# src/ -> project root, so output_dir is stable regardless of the caller's cwd.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


@dataclass
class Config:
    # Spectrum shape: 16 Mueller elements (65-degree oblique incidence,
    # M00..M33) + 2 normal-incidence reference spectra (TE, TM).
    n_channels: int = 18
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
    # Extra hidden structure parameters that also move the spectrum (stand-ins
    # for height / sidewall angle / film thickness); 0 = CD is the only thing
    # that varies. See data_generation.py.
    n_nuisance: int = 0
    nuisance_strength: float = 1.0
    # Std (nm) of Gaussian error added to the CD labels of the train and
    # validation splits -- a stand-in for reference-metrology (e.g. TEM)
    # error. The test split keeps exact labels so its metrics stay meaningful.
    label_noise_std: float = 0.0

    # Split ratios
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Model
    hidden_dim: int = 96
    # "flatten" keeps wavelength-position info in the head (CD is encoded in
    # where spectral features sit); "avg" is the legacy global-average-pool
    # variant. See models/multichannel_cnn.py.
    pooling: str = "flatten"
    dropout: float = 0.0
    # Architecture-ablation switches (experiments/ablation.py); the defaults
    # are the original network. See models/multichannel_cnn.py.
    # "mlp" (models/mlp.py, plain fully-connected) is the default since
    # 2026-10-09: on the synthetic data it sits at ~1.15x the Cramer-Rao bound
    # against ~1.6x for the CNN, at a quarter of the training data. "cnn" is
    # the Multi-channel 1D CNN (models/multichannel_cnn.py); the three
    # switches below and `pooling` only apply to it. Re-compare on real
    # spectra before trusting the ranking (experiments/ablation.py arch).
    model: str = "mlp"            # "mlp" | "cnn"
    downsample: str = "maxpool"   # "maxpool" | "avgpool" | "none"
    norm: str = "batch"           # "batch" | "none"
    conv_groups: str = "full"     # "full" | "per_channel"

    # Training
    batch_size: int = 64
    epochs: int = 150
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # "cosine": LR decays smoothly from lr to lr*0.01 over exactly `epochs`
    # epochs, no early stopping (val loss is too noisy epoch-to-epoch for
    # plateau detection to be reliable -- the legacy run stopped at epoch 65
    # of 300). The lowest-val-loss epoch is still the one kept.
    # "plateau": legacy ReduceLROnPlateau + early stopping after `patience`.
    lr_schedule: str = "cosine"
    patience: int = 25  # early stopping on val loss (plateau mode only)
    # Std (raw spectrum units) of fresh Gaussian noise added to every
    # training batch; 0 disables. Applied to the training split only.
    noise_augment_std: float = 0.0

    # Refinement against a learned forward model (refinement.py): after the
    # network, train a decoder CD -> spectrum on the same training split and
    # store it in the checkpoint, so predictions can be refined and get an
    # error bar. "off" skips it. decoder_epochs is raised automatically for
    # small training sets.
    refine: str = "on"            # "on" | "off"
    decoder_epochs: int = 300
    # Multi-task: also regress the first `aux_targets` nuisance parameters
    # (needs n_nuisance >= aux_targets), weighted by aux_loss_weight in the
    # training loss. Output 0 is always CD; checkpoint selection and every
    # reported metric use CD only.
    aux_targets: int = 0
    aux_loss_weight: float = 1.0

    # Reporting only -- what counts as an "acceptable" error, for the
    # fraction_over_threshold test metric. Not currently met on every sample
    # (see train.py's module docstring for what was tried and why it was
    # reverted in favor of stability).
    error_threshold_nm: float = 0.30

    # Output
    output_dir: str = DEFAULT_OUTPUT_DIR
