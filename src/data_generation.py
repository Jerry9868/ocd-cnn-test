"""Synthetic Mueller-matrix (+ TE/TM) spectrum generator.

Stands in for real OCD measurement/simulation data until that is available.
Not a physical solver (no RCWA) -- it fabricates n_channels-channel spectra
(Config.n_channels defaults to 18: 16 Mueller elements + TE + TM) that share
an underlying set of CD-dependent basis functions, so that:

Real-instrument context (for whoever wires up the actual data loader):
  * the 16-channel Mueller matrix (M00..M33, channels 0-15) is measured at
    65-degree oblique incidence onto the wafer.
  * channels 16-17 stand in for two additional normal-incidence (0-degree)
    spectra -- TE and TM -- that will also be provided. They're generated
    here the same way as any other channel (just two more rows of the
    n_channels x n_bases mixing matrix), which is almost certainly not how
    they actually relate to the Mueller channels physically (different
    incidence angle, different measurement principle) -- treat this purely
    as a placeholder to exercise the 18-channel pipeline end-to-end, not as
    a claim about the real TE/TM relationship. Swap in a real loader once
    the format is confirmed.

  * each of the n_channels "elements" is a distinct linear mixture of the
    same bases (i.e. genuine cross-channel correlation, not independent
    per-channel noise)
  * the mixture weights are fixed per element, while the bases themselves
    shift with CD (resonance-like center/amplitude dependence on CD), so CD
    is recoverable from the spectrum shape
  * channel 0 (M00) is forced to sit near 1 with only mild variation, matching
    the usual M00-normalized convention for measured Mueller matrices

`physics_seed` fixes the CD-to-spectrum relationship itself (basis shapes,
mixing weights, normalization scale) -- it must be the SAME value every time
data is generated for a given model, whether that's a 6000-sample training
set or a single spectrum in the viewer app, or the two won't be describing
the same "instrument". `seed` only controls which CD values get sampled and
the per-sample measurement noise draw, and is fine to vary between calls
(e.g. the app drawing a fresh noise realization on every click).

Nuisance parameters (n_nuisance > 0, off by default): real spectra depend on
more than CD -- height, sidewall angle, film thicknesses all vary sample to
sample and move the same spectral features CD does. `n_nuisance` adds that
many extra hidden structure parameters, each uniform in [-1, 1], that shift
every basis's center / amplitude / phase with their own fixed coefficients
(scaled by `nuisance_strength`). They are generic stand-ins, not models of
any particular real parameter; the point is to have CD partially confounded
with something, which is what makes real CD inversion hard. Their
coefficients come from a separate random stream, so n_nuisance = 0
reproduces the original generator exactly.

This is deliberately simple. It exists so the preprocessing pipeline and
model architectures can be built and sanity-checked before real spectra are
plugged in -- swap `generate_dataset` for a real data loader later without
touching `dataset.py` or the models.
"""
from dataclasses import dataclass

import numpy as np

# Fixed CD range used only to compute a stable per-channel normalization
# scale from `physics_seed`, independent of whatever cd_min/cd_max a
# particular call requests. Without this, normalizing against the min/max of
# just the current batch would make single-spectrum calls (n_samples=1) and
# the training batch (n_samples=6000) rescale differently even under the same
# physics_seed.
_NORMALIZATION_CD_REFERENCE = np.linspace(0.0, 150.0, 64)


# Effect of a nuisance parameter going from 0 to 1 at nuisance_strength = 1,
# as the upper limit of its random per-basis coefficient. For scale: CD
# moves a basis center by up to 2 nm per nm of CD.
_NUISANCE_CENTER_NM = 20.0
_NUISANCE_AMP = 0.10
_NUISANCE_PHASE_RAD = 0.30


@dataclass
class SyntheticDataset:
    X: np.ndarray          # (n_samples, n_channels, n_wavelength)
    y: np.ndarray          # (n_samples,) ground-truth CD
    wavelength: np.ndarray  # (n_wavelength,)
    nuisance: np.ndarray = None  # (n_samples, n_nuisance), each in [-1, 1]


def _physics_params(n_channels: int, n_bases: int, physics_seed: int, n_nuisance: int = 0) -> dict:
    """All parameters that define the fixed CD<->spectrum relationship."""
    rng = np.random.default_rng(physics_seed)
    # Separate stream: drawing these must not shift any of the draws below.
    nrng = np.random.default_rng([physics_seed, 1])
    nuisance = dict(
        center=nrng.uniform(-1, 1, size=(n_nuisance, n_bases)) * _NUISANCE_CENTER_NM,
        amp=nrng.uniform(-1, 1, size=(n_nuisance, n_bases)) * _NUISANCE_AMP,
        phase=nrng.uniform(-1, 1, size=(n_nuisance, n_bases)) * _NUISANCE_PHASE_RAD,
        element_phase=nrng.uniform(-1, 1, size=(n_nuisance, n_channels)) * _NUISANCE_PHASE_RAD,
    )

    bases_params = [
        dict(
            center0=rng.uniform(300, 900),
            center_slope=rng.uniform(-2.0, 2.0),   # nm shift per nm of CD
            width=rng.uniform(60, 180),
            amp0=rng.uniform(0.3, 1.0),
            amp_slope=rng.uniform(-0.01, 0.01),
            freq=rng.uniform(0.01, 0.04),           # rad/nm
            phase0=rng.uniform(0, 2 * np.pi),
            phase_slope=rng.uniform(-0.02, 0.02),
        )
        for _ in range(n_bases)
    ]

    mix_weights = rng.normal(0, 1, size=(n_channels, n_bases))
    mix_weights /= np.linalg.norm(mix_weights, axis=1, keepdims=True)

    element_phase = rng.uniform(0, 2 * np.pi, size=n_channels)
    element_freq = rng.uniform(0.01, 0.05, size=n_channels)
    element_scale = rng.uniform(0.02, 0.08, size=n_channels)

    return dict(
        bases_params=bases_params,
        mix_weights=mix_weights,
        element_phase=element_phase,
        element_freq=element_freq,
        element_scale=element_scale,
        nuisance=nuisance,
    )


def _evaluate_bases(wavelength: np.ndarray, cd: np.ndarray, bases_params: list,
                    shifts: dict = None) -> np.ndarray:
    """Return (n_samples, n_bases, n_wavelength), deterministic given `bases_params`.

    shifts: per-sample nuisance offsets, {"center"/"amp"/"phase": (N, n_bases)}."""
    l = wavelength[None, :]  # (1, L)
    cd = cd[:, None]         # (N, 1)

    bases = []
    for k, p in enumerate(bases_params):
        center = p["center0"] + p["center_slope"] * cd
        amp = p["amp0"] + p["amp_slope"] * cd
        phase = p["phase0"] + p["phase_slope"] * cd
        if shifts is not None:
            center = center + shifts["center"][:, k:k + 1]
            amp = amp + shifts["amp"][:, k:k + 1]
            phase = phase + shifts["phase"][:, k:k + 1]

        envelope = np.exp(-0.5 * ((l - center) / p["width"]) ** 2)
        oscillation = np.cos(p["freq"] * l + phase)
        bases.append(amp * envelope * oscillation)

    return np.stack(bases, axis=1)  # (N, n_bases, L)


def _raw_spectrum(wavelength: np.ndarray, cd: np.ndarray, physics: dict,
                  nuisance: np.ndarray = None) -> np.ndarray:
    """Unnormalized (n_samples, n_channels, n_wavelength) spectrum.

    nuisance: (n_samples, n_nuisance) already scaled by nuisance_strength, or None."""
    shifts = None
    element_shift = 0.0
    if nuisance is not None and nuisance.shape[1] > 0:
        coef = physics["nuisance"]
        shifts = {k: nuisance @ coef[k] for k in ("center", "amp", "phase")}  # (N, K) each
        element_shift = (nuisance @ coef["element_phase"])[:, :, None]         # (N, C, 1)

    bases = _evaluate_bases(wavelength, cd, physics["bases_params"], shifts)  # (N, K, L)
    X = np.einsum("ck,nkl->ncl", physics["mix_weights"], bases)  # (N, C, L)

    l = wavelength[None, None, :]
    cd3 = cd[:, None, None]
    element_term = physics["element_scale"][None, :, None] * np.cos(
        physics["element_freq"][None, :, None] * l + physics["element_phase"][None, :, None] + 0.01 * cd3
        + element_shift
    )
    return X + element_term


def _normalization_scale(wavelength: np.ndarray, physics: dict) -> np.ndarray:
    """Fixed per-channel scale, independent of any particular call's n_samples/cd range."""
    ref = _raw_spectrum(wavelength, _NORMALIZATION_CD_REFERENCE, physics)
    max_abs = np.max(np.abs(ref), axis=(0, 2), keepdims=True)[0]  # (C, 1)
    max_abs[max_abs == 0] = 1.0
    return max_abs


def clean_spectrum(
    cd: np.ndarray,
    wavelength: np.ndarray,
    n_channels: int,
    n_shared_bases: int = 4,
    physics_seed: int = 2024,
    nuisance: np.ndarray = None,
    nuisance_strength: float = 1.0,
) -> np.ndarray:
    """Noise-free (n_samples, n_channels, n_wavelength) float64 spectrum for
    the given CD values -- the deterministic CD->spectrum map on its own, for
    anything that needs it without the noise draw (e.g. the Cramer-Rao bound
    in experiments/cramer_rao.py differentiates it w.r.t. CD).

    nuisance: optional (n_samples, n_nuisance) values in [-1, 1]."""
    n_nuisance = 0 if nuisance is None else nuisance.shape[1]
    physics = _physics_params(n_channels, n_shared_bases, physics_seed, n_nuisance)
    if nuisance is not None:
        nuisance = np.asarray(nuisance, dtype=np.float64) * nuisance_strength

    X = _raw_spectrum(wavelength, np.asarray(cd, dtype=np.float64), physics, nuisance)

    scale = _normalization_scale(wavelength, physics)  # (C, 1)
    X = X / scale[None]

    # M00 (channel 0) sits near 1 with only mild CD-dependent ripple, matching
    # the usual M00-normalized convention for measured Mueller matrices.
    X[:, 0, :] = 1.0 + 0.05 * X[:, 0, :]
    return X


def generate_dataset(
    n_samples: int,
    n_channels: int,
    n_wavelength: int,
    wavelength_min: float,
    wavelength_max: float,
    cd_min: float,
    cd_max: float,
    n_shared_bases: int = 4,
    noise_std: float = 0.01,
    seed: int = 42,
    physics_seed: int = 2024,
    n_nuisance: int = 0,
    nuisance_strength: float = 1.0,
) -> SyntheticDataset:
    sample_rng = np.random.default_rng(seed)

    wavelength = np.linspace(wavelength_min, wavelength_max, n_wavelength)
    cd = sample_rng.uniform(cd_min, cd_max, size=n_samples)
    nuisance = np.zeros((n_samples, 0))
    if n_nuisance > 0:
        nuisance = sample_rng.uniform(-1.0, 1.0, size=(n_samples, n_nuisance))

    X = clean_spectrum(cd, wavelength, n_channels, n_shared_bases, physics_seed,
                       nuisance=nuisance, nuisance_strength=nuisance_strength)

    X = X + sample_rng.normal(0, noise_std, size=X.shape)

    return SyntheticDataset(X=X.astype(np.float32), y=cd.astype(np.float32), wavelength=wavelength,
                            nuisance=nuisance.astype(np.float32))
