"""Refinement of the network's CD against a learned forward model.

The regression network (models/) maps spectrum -> CD in one shot. This adds
a second, optional step that needs no physical solver:

  1. A decoder is trained on the same training split the other way round,
     CD (+ any labelled nuisance parameters) -> spectrum. Regressing noisy
     spectra on the parameters averages the noise out, so the decoder
     approximates the noise-free spectrum.
  2. At prediction time a few Gauss-Newton steps, started from the network's
     CD, minimise || spectrum - decoder(params) ||^2 -- a local
     maximum-likelihood fit against the learned forward model.

Besides a more accurate CD this gives, per spectrum: a 1-sigma error bar
(from the fit's curvature and leftover misfit), a misfit that flags spectra
the forward model cannot explain, and optionally a fitted wavelength-axis
offset so a calibration error is not read as a CD change.

It is not always better: with only a few hundred training spectra the
decoder is too rough and the refined CD is worse than the network's. So
train.run_training scores both on the validation split and records which
one to prefer (`Refinement.recommended`).

experiments/refine.py and experiments/robustness.py hold the measurements
behind these claims.
"""
import numpy as np
import torch
import torch.nn as nn

# Small training sets get more epochs, so the decoder always sees at least
# this many gradient steps.
MIN_DECODER_STEPS = 12000
DECODER_BATCH_SIZE = 128
CALIBRATION_DEGREE = 3


class Decoder(nn.Module):
    """(CD, latents) -> flattened normalized spectrum. Smooth activations:
    Gauss-Newton differentiates this w.r.t. its inputs."""

    def __init__(self, n_in, n_out, width=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, n_out),
        )

    def forward(self, p):
        return self.net(p)


class ShiftedDecoder(nn.Module):
    """Decoder plus one extra input: a wavelength-axis offset (nm).

    A spectrum measured with its wavelength axis off by delta samples the
    true curve at lambda + delta, to first order decoder + delta * d/dlambda.
    Fitting delta alongside CD lets the refinement absorb a calibration
    offset instead of reading it as a CD change."""

    def __init__(self, decoder, n_channels, n_wavelength, wavelength_step):
        super().__init__()
        self.decoder = decoder
        self.shape = (n_channels, n_wavelength)
        self.step = wavelength_step

    def forward(self, q):
        out = self.decoder(q[:, :-1])
        y = out.reshape(len(q), *self.shape)
        slope = torch.cat([
            (y[:, :, 1:2] - y[:, :, 0:1]) / self.step,
            (y[:, :, 2:] - y[:, :, :-2]) / (2 * self.step),
            (y[:, :, -1:] - y[:, :, -2:-1]) / self.step,
        ], dim=2)
        return out + q[:, -1:] * slope.reshape(len(q), -1)


def decoder_epochs(epochs: int, n_train: int) -> int:
    """`epochs`, raised if needed so training runs MIN_DECODER_STEPS steps."""
    steps_per_epoch = -(-n_train // DECODER_BATCH_SIZE)
    return max(epochs, -(-MIN_DECODER_STEPS // steps_per_epoch))


def fit_decoder(X, params, epochs, seed=0, lr=2e-3, progress_cb=None):
    """Train a Decoder. X: (N, D) normalized spectra; params: (N, P) its
    inputs, normalized CD first. progress_cb(epoch, epochs), if given, is
    called after every epoch."""
    torch.manual_seed(seed)
    X = torch.from_numpy(X).float()
    params = torch.from_numpy(params).float()
    decoder = Decoder(params.shape[1], X.shape[1])
    opt = torch.optim.Adam(decoder.parameters(), lr=lr)
    steps = epochs * (-(-len(X) // DECODER_BATCH_SIZE))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(X))
        for s in range(0, len(X), DECODER_BATCH_SIZE):
            idx = perm[s:s + DECODER_BATCH_SIZE]
            loss = ((decoder(params[idx]) - X[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if progress_cb is not None:
            progress_cb(epoch, epochs)
    return decoder.eval()


def gauss_newton(decoder, X, p0, weight, n_iter=8, damping=1e-6, batch=500, n_free=None):
    """Minimise sum(weight * (X - decoder(p))^2) over p, from p0.

    X: (N, D) normalized spectra; p0: (N, P); weight: (D,) inverse noise
    variance of each normalized spectrum point (relative scale is enough).
    n_free: fit only the first n_free parameters, holding the rest at p0.
    Returns (p, final weighted misfit per sample, var0): var0 is the first
    parameter's variance per unit weighted-residual variance, [(J^T W J)^-1]_00
    at the solution -- multiply by the residual variance for its error bar.
    """
    out = np.empty_like(p0)
    out_cost = np.empty(len(p0))
    out_var0 = np.empty(len(p0))
    w = torch.from_numpy(weight).float()
    # Forward mode: few inputs (CD + latents), thousands of outputs.
    jac = torch.func.vmap(torch.func.jacfwd(lambda q: decoder(q[None])[0]))
    for s in range(0, len(X), batch):
        x = torch.from_numpy(X[s:s + batch]).float()
        p = torch.from_numpy(p0[s:s + batch]).float()
        with torch.no_grad():
            best_cost = (w * (x - decoder(p)) ** 2).sum(dim=1)
        for _ in range(n_iter):
            J = jac(p).detach()                                    # (B, D, P)
            if n_free is not None:
                J = J[:, :, :n_free]
            with torch.no_grad():
                r = x - decoder(p)                                  # (B, D)
                Jw = J * w[None, :, None]
                A = J.transpose(1, 2) @ Jw                          # (B, P, P)
                g = (Jw.transpose(1, 2) @ r[:, :, None])[:, :, 0]   # (B, P)
                eye = torch.eye(A.shape[1])[None] * damping * A.diagonal(dim1=1, dim2=2).mean(dim=1)[:, None, None]
                step = torch.linalg.solve(A + eye, g)
                if n_free is not None:
                    step = torch.cat([step, torch.zeros(len(p), p.shape[1] - n_free)], dim=1)
                cand = p + step
                cost = (w * (x - decoder(cand)) ** 2).sum(dim=1)
                # Keep a step only where it lowers the misfit.
                better = cost < best_cost
                p = torch.where(better[:, None], cand, p)
                best_cost = torch.where(better, cost, best_cost)
        out[s:s + batch] = p.numpy()
        out_cost[s:s + batch] = best_cost.numpy()
        J = jac(p).detach()
        if n_free is not None:
            J = J[:, :, :n_free]
        A = J.transpose(1, 2) @ (J * w[None, :, None])
        out_var0[s:s + batch] = torch.linalg.inv(A)[:, 0, 0].numpy()
    return out, out_cost, out_var0


class Refinement:
    """A trained decoder plus what is needed to refine a network prediction.

    norm: {"mean", "std": (n_channels, 1) arrays, "y_mean", "y_std": floats}
    -- the same normalization the regression network uses.
    wavelength_step: spacing (nm) of the model's wavelength grid.
    misfit_limit: mean-squared raw-unit residual above which a spectrum is
    flagged as not explained by the forward model (None = never flag).
    recommended: whether refined CD beat the network's on the validation
    split when this was trained.
    calibration: polynomial coefficients (lowest order first, in normalized
    CD) of the correction added to the fitted CD -- see calibrate(). None =
    no correction.
    """

    def __init__(self, decoder, norm, n_channels, n_wavelength, wavelength_step,
                 misfit_limit=None, recommended=True, calibration=None):
        self.decoder = decoder
        self.norm = norm
        self.n_channels = n_channels
        self.n_wavelength = n_wavelength
        self.wavelength_step = wavelength_step
        self.misfit_limit = misfit_limit
        self.recommended = recommended
        self.calibration = calibration
        # Decoder inputs beyond CD (labelled nuisance parameters).
        self.n_extra = decoder.net[0].in_features - 1
        # normalized -> raw units, per flattened spectrum point
        self.raw_std = np.repeat(norm["std"][:, 0], n_wavelength)
        # Raw noise is taken as uniform, so normalized noise variance ~ 1 / std^2.
        self.weight = (self.raw_std ** 2).astype(np.float32)

    def flat(self, X):
        return ((X - self.norm["mean"][None]) / self.norm["std"][None]).reshape(len(X), -1).astype(np.float32)

    def refine(self, model, X, fit_offset=False):
        """X: (N, n_channels, L) raw spectra on the model's wavelength grid.

        Returns a dict of (N,) arrays:
          cd_network -- the regression network's CD (nm)
          cd         -- refined CD (nm)
          cd_sd      -- 1-sigma error bar on `cd` (nm)
          misfit     -- mean squared raw-unit residual of the fit
          flagged    -- misfit above misfit_limit
          offset     -- fitted wavelength offset (nm); only with fit_offset
        """
        Xf = self.flat(X)
        with torch.no_grad():
            out = model(torch.from_numpy(Xf.reshape(len(X), self.n_channels, -1))).numpy()
        out = out.reshape(len(X), -1)
        cd_network = out[:, 0].astype(np.float64) * self.norm["y_std"] + self.norm["y_mean"]

        p0 = np.zeros((len(X), 1 + self.n_extra), dtype=np.float32)
        p0[:, 0] = out[:, 0]
        # A multi-task network also predicts nuisances (scaled by sqrt(3) in training).
        n_aux = min(out.shape[1] - 1, self.n_extra)
        p0[:, 1:1 + n_aux] = out[:, 1:1 + n_aux] / np.sqrt(3.0)

        decoder = self.decoder
        if fit_offset:
            decoder = ShiftedDecoder(self.decoder, self.n_channels, self.n_wavelength, self.wavelength_step)
            p0 = np.column_stack([p0, np.zeros(len(X), dtype=np.float32)])
        p, cost, var0 = gauss_newton(decoder, Xf, p0, self.weight)

        dof = Xf.shape[1] - p.shape[1]
        misfit = cost / Xf.shape[1]
        cd_fit = p[:, 0].astype(np.float64)
        if self.calibration is not None:
            cd_fit = cd_fit + np.polynomial.polynomial.polyval(cd_fit, self.calibration)
        result = {
            "cd_network": cd_network,
            "cd": cd_fit * self.norm["y_std"] + self.norm["y_mean"],
            "cd_sd": np.sqrt(var0 * cost / dof) * self.norm["y_std"],
            "misfit": misfit,
            "flagged": misfit > self.misfit_limit if self.misfit_limit is not None else np.zeros(len(X), dtype=bool),
        }
        if fit_offset:
            result["offset"] = p[:, -1].astype(np.float64)
        return result

    def calibrate(self, model, X, cd):
        """Fit the correction that removes the fitted CD's systematic error.

        A decoder error too small to matter as a spectrum residual can still
        lie along d(spectrum)/d(CD) and shift every fitted CD the same way
        (a few 0.01 nm, comparable to the noise-limited error). Refining the
        training spectra X, whose CD is known, measures that shift; a cubic
        in the fitted CD is fitted to it and added from then on."""
        self.calibration = None
        fitted = (self.refine(model, X)["cd"] - self.norm["y_mean"]) / self.norm["y_std"]
        target = (np.asarray(cd, dtype=np.float64) - self.norm["y_mean"]) / self.norm["y_std"]
        self.calibration = np.polynomial.polynomial.polyfit(fitted, target - fitted, CALIBRATION_DEGREE)

    def state(self) -> dict:
        """What train.py stores in the checkpoint (see inference.load_refinement)."""
        first, last = self.decoder.net[0], self.decoder.net[-1]
        return {
            "decoder_state": self.decoder.state_dict(),
            "n_in": first.in_features, "n_out": last.out_features, "width": first.out_features,
            "misfit_limit": self.misfit_limit,
            "recommended": self.recommended,
            "calibration": self.calibration,
        }
