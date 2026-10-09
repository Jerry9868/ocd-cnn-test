"""Analysis-by-synthesis refinement with a *learned* forward model.

    python experiments/refine.py outputs/experiments/ablation_arch/base

The regression network maps spectrum -> CD in one shot and ends up more
noise-sensitive than the Cramer-Rao bound allows. This tests a way to
recover the difference without a physical solver:

  1. Train a small decoder on the same training set, the other way round:
     (CD [, nuisance parameters]) -> spectrum. Regressing noisy spectra on
     the parameters averages the noise out, so the decoder approximates the
     noise-free spectrum.
  2. At prediction time, start from the network's CD and take a few
     Gauss-Newton steps that minimise || spectrum - decoder(params) ||^2,
     i.e. a local maximum-likelihood fit against the learned forward model.

The decoder and the Gauss-Newton fit themselves live in src/refinement.py,
which train.py and the app also use; this script adds the scoring against
the Cramer-Rao bound and the experimental latent-variable variants below.

With nuisance parameters the decoder needs them as inputs. `--latent`
chooses where they come from:
  * "labels"  -- the true training nuisances (an upper bound on what this
                 can do; real data rarely has those labels).
  * "learned" -- nothing but CD is labelled: an encoder infers free latent
                 variables per spectrum, trained jointly with the decoder to
                 reconstruct the spectrum. Only CD is tied to a label.
  * "cvae"    -- as "learned", but as a conditional VAE: the encoder also
                 sees the CD label and its latents are noisy and pay a KL
                 cost (--kl_beta). Information CD already supplies is then
                 expensive to duplicate in the latents, which should keep
                 them to what CD does not explain.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from _common import EVAL_SEED, EXPERIMENT_DIR

from cramer_rao import cnn_predict, cramer_rao_bound, cramer_rao_bound_joint, summarize
from data_generation import generate_dataset
from dataset import split_indices
from inference import load_checkpoint
from refinement import MIN_DECODER_STEPS, Decoder, ShiftedDecoder, gauss_newton
from train import add_label_noise

N_EVAL = 20000


def make_data(cfg, n_samples, seed, noise_std=None):
    return generate_dataset(
        n_samples=n_samples, n_channels=cfg.n_channels, n_wavelength=cfg.n_wavelength,
        wavelength_min=cfg.wavelength_min, wavelength_max=cfg.wavelength_max,
        cd_min=cfg.cd_min, cd_max=cfg.cd_max, n_shared_bases=cfg.n_shared_bases,
        noise_std=cfg.noise_std if noise_std is None else noise_std, seed=seed,
        physics_seed=cfg.physics_seed, n_nuisance=cfg.n_nuisance, nuisance_strength=cfg.nuisance_strength,
    )


class LatentEncoder(nn.Module):
    """Flattened normalized spectrum -> free latent variables."""

    def __init__(self, n_in, n_latent, width=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, n_latent),
        )

    def forward(self, x):
        return self.net(x)


class CVAEEncoder(nn.Module):
    """(Flattened normalized spectrum, normalized CD) -> latent mean, log-variance."""

    def __init__(self, n_in, n_latent, width=256):
        super().__init__()
        self.n_latent = n_latent
        self.net = nn.Sequential(
            nn.Linear(n_in + 1, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 2 * n_latent),
        )

    def forward(self, x, cd):
        out = self.net(torch.cat([x, cd], dim=1))
        return out[:, :self.n_latent], out[:, self.n_latent:]


def train_decoder(X, params, n_latent, epochs, seed=0, lr=2e-3, batch_size=128, log=print, kl_weight=None):
    """X: (N, D) normalized spectra. params: (N, P) labelled decoder inputs
    (CD first). n_latent extra inputs are inferred by a jointly trained
    encoder -- a plain one, or with kl_weight set a conditional-VAE one whose
    KL term is weighted by it. Returns (decoder, encoder or None)."""
    torch.manual_seed(seed)
    X = torch.from_numpy(X).float()
    params = torch.from_numpy(params).float()
    decoder = Decoder(params.shape[1] + n_latent, X.shape[1])
    encoder = None
    if n_latent:
        encoder = (LatentEncoder if kl_weight is None else CVAEEncoder)(X.shape[1], n_latent)
    weights = list(decoder.parameters()) + (list(encoder.parameters()) if encoder else [])
    opt = torch.optim.Adam(weights, lr=lr)
    steps = epochs * ((len(X) + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(X))
        total = 0.0
        for s in range(0, len(X), batch_size):
            idx = perm[s:s + batch_size]
            p = params[idx]
            kl = 0.0
            if encoder is not None and kl_weight is None:
                p = torch.cat([p, encoder(X[idx])], dim=1)
            elif encoder is not None:
                mu, logvar = encoder(X[idx], p[:, :1])
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
                kl = 0.5 * (mu ** 2 + logvar.exp() - logvar - 1).sum(dim=1).mean()
                p = torch.cat([p, z], dim=1)
            loss = ((decoder(p) - X[idx]) ** 2).mean() + (kl_weight or 0.0) * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        if epoch % 50 == 0 or epoch == epochs:
            log(f"  decoder epoch {epoch:4d}  recon mse {total / len(X):.5f}")
    decoder.eval()
    if encoder is not None:
        encoder.eval()
    return decoder, encoder


class Refiner:
    """A trained regression network plus a forward model learned from the
    same training split, and the Gauss-Newton refinement between them."""

    def __init__(self, run_dir, latent="labels", n_latent=None, epochs=300, log=print, fix_latents=False,
                 kl_beta=1.0):
        # fix_latents (learned latents only): refine CD alone, keeping the
        # latents at the encoder's values.
        self.fix_latents = fix_latents
        self.model, self.cfg, self.norm = load_checkpoint(os.path.join(run_dir, "multichannel_cnn.pt"))
        cfg = self.cfg
        self.latent = latent if cfg.n_nuisance else "none"

        # The exact training split the regression network saw.
        data = make_data(cfg, cfg.n_samples, cfg.seed)
        train_idx, val_idx, _ = split_indices(cfg.n_samples, cfg.val_ratio, cfg.test_ratio, cfg.seed)
        add_label_noise(data.y, train_idx, val_idx, cfg)
        params = self.cd_norm(data.y[train_idx])[:, None]
        self.n_latent = 0
        if self.latent == "labels":
            params = np.column_stack([params, data.nuisance[train_idx]])
        elif self.latent in ("learned", "cvae"):
            self.n_latent = cfg.n_nuisance if n_latent is None else n_latent
        kl_weight = None
        if self.latent == "cvae":
            # ELBO scaling: per-point MSE + (2 sigma^2 / D) * KL, sigma^2 the
            # noise variance of a normalized spectrum point.
            noise_var = float(np.mean((cfg.noise_std / self.norm["std"][:, 0]) ** 2))
            kl_weight = kl_beta * 2 * noise_var / (cfg.n_channels * cfg.n_wavelength)
        # Small training sets get more epochs, so the decoder always sees a
        # comparable number of gradient steps.
        epochs = max(epochs, -(-MIN_DECODER_STEPS // -(-len(train_idx) // 128)))
        self.decoder, self.encoder = train_decoder(self.flat(data.X[train_idx]), params, self.n_latent,
                                                   epochs, log=log, kl_weight=kl_weight)

        # normalized -> raw units, per flattened spectrum point
        self.raw_std = np.repeat(self.norm["std"][:, 0], cfg.n_wavelength)
        # Raw noise is uniform, so normalized noise variance ~ 1 / std^2.
        self.weight = (self.raw_std ** 2).astype(np.float32)

    def encode(self, Xf, cd_norm):
        """Latents for flattened normalized spectra (cd_norm: (N, 1), used by the cvae encoder)."""
        x = torch.from_numpy(Xf)
        if self.latent == "cvae":
            return self.encoder(x, torch.from_numpy(cd_norm))[0].numpy()
        return self.encoder(x).numpy()

    def flat(self, X):
        return ((X - self.norm["mean"][None]) / self.norm["std"][None]).reshape(len(X), -1).astype(np.float32)

    def cd_norm(self, cd):
        return ((cd - self.norm["y_mean"]) / self.norm["y_std"]).astype(np.float32)

    def decode(self, cd, X_for_latents=None, nuisance=None):
        """Decoder's normalized spectrum for given CD (+ true nuisances, or
        latents encoded from X_for_latents)."""
        p = self.cd_norm(cd)[:, None]
        with torch.no_grad():
            if self.latent == "labels":
                p = np.column_stack([p, nuisance])
            elif self.latent in ("learned", "cvae"):
                p = np.column_stack([p, self.encode(self.flat(X_for_latents), p)])
            return self.decoder(torch.from_numpy(p.astype(np.float32))).numpy()

    def predict(self, X, fit_offset=False):
        """X: (N, n_channels, L) raw spectra. Returns (network CD, refined
        CD, misfit, offset): misfit is the mean squared raw-unit residual of
        the refined fit -- about noise_std^2 for a spectrum the forward
        model can explain, larger for one it cannot. With fit_offset a
        wavelength-axis offset (nm) is fitted too and returned, else None.

        Also sets self.last_cd_sd: a per-spectrum 1-sigma error bar (nm) on
        the refined CD, from the fit's own curvature and misfit -- no
        knowledge of the true noise level needed."""
        cfg = self.cfg
        Xf = self.flat(X)
        net_cd = cnn_predict(self.model, self.norm, X)
        p0 = self.cd_norm(net_cd)[:, None]
        with torch.no_grad():
            if self.latent == "labels":
                init = np.zeros((len(X), cfg.n_nuisance), dtype=np.float32)
                if getattr(self.model, "n_outputs", 1) > 1:  # multi-task net also predicts nuisances
                    aux = self.model(torch.from_numpy(Xf.reshape(len(X), cfg.n_channels, -1))).numpy()[:, 1:]
                    init[:, :aux.shape[1]] = aux / np.sqrt(3.0)
                p0 = np.column_stack([p0, init])
            elif self.latent in ("learned", "cvae"):
                p0 = np.column_stack([p0, self.encode(Xf, p0)])
        decoder, offset = self.decoder, None
        if fit_offset:
            step = (cfg.wavelength_max - cfg.wavelength_min) / (cfg.n_wavelength - 1)
            decoder = ShiftedDecoder(self.decoder, cfg.n_channels, cfg.n_wavelength, step)
            p0 = np.column_stack([p0, np.zeros(len(X), dtype=np.float32)])
        n_free = 1 if (self.fix_latents and self.latent in ("learned", "cvae") and not fit_offset) else None
        p, cost, var0 = gauss_newton(decoder, Xf, p0.astype(np.float32), self.weight, n_free=n_free)
        dof = Xf.shape[1] - (p.shape[1] if n_free is None else n_free)
        self.last_cd_sd = np.sqrt(var0 * cost / dof) * self.norm["y_std"]
        if fit_offset:
            offset = p[:, -1].astype(np.float64)
        refined_cd = p[:, 0].astype(np.float64) * self.norm["y_std"] + self.norm["y_mean"]
        return net_cd, refined_cd, cost / Xf.shape[1], offset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="folder holding a multichannel_cnn.pt trained by train.run_training")
    ap.add_argument("--latent", choices=["labels", "learned", "cvae"], default="labels")
    ap.add_argument("--kl_beta", type=float, default=1.0, help="KL weight for --latent cvae (1 = plain ELBO)")
    ap.add_argument("--n_latent", type=int, default=None, help="free latents for --latent learned (default: n_nuisance)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--fix_latents", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    ref = Refiner(args.run_dir, args.latent, args.n_latent, args.epochs, fix_latents=args.fix_latents,
                  kl_beta=args.kl_beta)
    cfg = ref.cfg
    rms = lambda a: float(np.sqrt((np.asarray(a) ** 2).mean()))

    ev = make_data(cfg, N_EVAL, EVAL_SEED)
    ev_clean = make_data(cfg, N_EVAL, EVAL_SEED, noise_std=0.0)
    y = ev.y.astype(np.float64)

    # How close is the learned forward model to the truth, relative to noise?
    recon = ref.decode(ev.y, X_for_latents=ev_clean.X, nuisance=ev.nuisance)
    decoder_err_raw = rms((recon - ref.flat(ev_clean.X)) * ref.raw_std)

    net_cd, refined_cd, misfit, _ = ref.predict(ev.X)

    if cfg.n_nuisance:
        known, unknown = cramer_rao_bound_joint(y, ev.nuisance, ev.wavelength, cfg)
        bounds = {"bound_known_rms": rms(known), "bound_rms": rms(unknown)}
    else:
        bounds = {"bound_rms": rms(cramer_rao_bound(y, ev.wavelength, cfg))}

    result = {
        "run_dir": args.run_dir, "latent": ref.latent, "n_latent": ref.n_latent,
        "decoder_epochs": args.epochs, "noise_std": cfg.noise_std,
        "decoder_error_raw_rms": decoder_err_raw,
        "misfit_rms_raw": float(np.sqrt(misfit.mean())),
        # Is the per-spectrum error bar honest? z = error / predicted sd
        # should have sd 1, and 95.4% of |z| should be under 2.
        "error_bar": {
            "mean_sd_nm": float(ref.last_cd_sd.mean()),
            "z_sd": float(((refined_cd - y) / ref.last_cd_sd).std()),
            "within_2sd": float((np.abs(refined_cd - y) < 2 * ref.last_cd_sd).mean()),
        },
        **bounds,
        "network": summarize(net_cd - y, cfg.error_threshold_nm),
        "refined": summarize(refined_cd - y, cfg.error_threshold_nm),
    }
    print(json.dumps(result, indent=2))
    tag = args.tag or f"{os.path.basename(os.path.normpath(args.run_dir))}_{ref.latent}"
    os.makedirs(os.path.join(EXPERIMENT_DIR, "refine"), exist_ok=True)
    with open(os.path.join(EXPERIMENT_DIR, "refine", f"{tag}.json"), "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
