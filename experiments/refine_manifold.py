"""Refinement when only CD is labelled: fit the spectrum manifold, then read CD off it.

    python experiments/refine_manifold.py outputs/experiments/ablation_nuisance/mlp_cd_n24000

refine.py's "learned"/"cvae" variants feed the decoder the CD label plus
free latents. Nothing stops the latents from carrying CD information too,
so (CD, latents) is not a unique coordinate for a spectrum and the fitted
CD is partly arbitrary -- those variants barely beat the network.

Here CD is not a decoder input at all:

  1. A plain autoencoder (no labels) learns the manifold the noise-free
     spectra live on: encoder spectrum -> z, decoder z -> spectrum, with
     n_latent = number of things that vary (CD + nuisances).
  2. Gauss-Newton fits every latent, started from the encoder's z. That is
     the maximum-likelihood point on the manifold, whatever coordinates the
     autoencoder happened to pick for it.
  3. A small readout network, trained on the fitted latents of the training
     split with the CD labels, maps z -> CD.

Since CD is a function of the point on the manifold, step 3 loses nothing,
and the labels are only asked to pin down a smooth function of n_latent
variables instead of of the whole spectrum.

Also scored: the original network applied to the fitted (denoised) spectrum
decoder(z) instead of the measured one.

--polish N then alternates N times: with the fitted latents of the training
split held fixed, train a fresh decoder z -> spectrum on them (the same
supervised problem refinement.fit_decoder solves with labelled parameters),
and refit the latents against it. The jointly trained autoencoder's decoder
is ~2.5x less accurate than a decoder given the true parameters; this is
meant to close that gap. Every round is scored.

Measured 2026-10-10 on ablation_nuisance/mlp_cd_n24000 (3 nuisances, CD the
only label, bound 0.090 nm, network 0.134 nm):
  * 4 latents: 0.125 nm overall -- no better than refine.py --latent learned
    (0.122). 6 latents 0.123, 8 latents 0.177; --polish 2 changes nothing.
  * The overall figure hides a split by position in the training range
    ("readout_by_edge"). Where every parameter is within the central 70% of
    its range: 0.098 nm against the network's 0.122 (1.09x the bound). In
    the outermost 3%: 0.187 against 0.161, i.e. worse than the network.
  * The learned manifold is ~3x less accurate at the edge of the range than
    in the middle. A decoder given the true nuisances shows the same trend
    at half the size, so it is mostly a matter of how thinly 16,800 spectra
    cover a 4-parameter range, not of the missing labels.
So with CD-only labels this helps only if the training range is clearly
wider than the range the model is used on.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from _common import EVAL_SEED, EXPERIMENT_DIR

from cramer_rao import cnn_predict, cramer_rao_bound, cramer_rao_bound_joint, summarize
from dataset import split_indices
from inference import load_checkpoint
from refine import N_EVAL, LatentEncoder, make_data
from refinement import MIN_DECODER_STEPS, Decoder, fit_decoder, gauss_newton
from train import add_label_noise

BATCH_SIZE = 128
EDGE_BINS = [0.0, 0.7, 0.9, 0.97, 1.0001]


def train_autoencoder(X, n_latent, epochs, seed=0, lr=2e-3, log=print):
    """X: (N, D) normalized spectra. Returns (encoder, decoder)."""
    torch.manual_seed(seed)
    X = torch.from_numpy(X).float()
    encoder = LatentEncoder(X.shape[1], n_latent)
    decoder = Decoder(n_latent, X.shape[1])
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    steps = epochs * (-(-len(X) // BATCH_SIZE))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(X))
        total = 0.0
        for s in range(0, len(X), BATCH_SIZE):
            idx = perm[s:s + BATCH_SIZE]
            loss = ((decoder(encoder(X[idx])) - X[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        if epoch % 50 == 0 or epoch == epochs:
            log(f"  autoencoder epoch {epoch:4d}  recon mse {total / len(X):.5f}")
    return encoder.eval(), decoder.eval()


class Readout(nn.Module):
    """Fitted latents -> normalized CD: a linear fit plus a small MLP for
    what the linear fit leaves over."""

    def __init__(self, n_latent, width=128):
        super().__init__()
        self.linear = nn.Linear(n_latent, 1)
        self.net = nn.Sequential(
            nn.Linear(n_latent, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(self, z):
        return (self.linear(z) + self.net(z))[:, 0]


def train_readout(z, y, z_val, y_val, epochs, seed=0, lr=2e-3, log=print):
    """z: (N, n_latent) standardized fitted latents; y: (N,) normalized CD.
    Keeps the epoch with the lowest validation MSE."""
    torch.manual_seed(seed)
    z, y = torch.from_numpy(z).float(), torch.from_numpy(y).float()
    z_val, y_val = torch.from_numpy(z_val).float(), torch.from_numpy(y_val).float()
    readout = Readout(z.shape[1])
    # Start the linear part at the least-squares solution.
    A = torch.cat([z, torch.ones(len(z), 1)], dim=1)
    sol = torch.linalg.lstsq(A, y[:, None]).solution[:, 0]
    with torch.no_grad():
        readout.linear.weight.copy_(sol[None, :-1])
        readout.linear.bias.copy_(sol[-1:])
        readout.net[-1].weight.zero_()
        readout.net[-1].bias.zero_()
    opt = torch.optim.Adam(readout.parameters(), lr=lr)
    steps = epochs * (-(-len(z) // BATCH_SIZE))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.01)
    best, best_state = float("inf"), None
    for epoch in range(1, epochs + 1):
        readout.train()
        perm = torch.randperm(len(z))
        for s in range(0, len(z), BATCH_SIZE):
            idx = perm[s:s + BATCH_SIZE]
            loss = ((readout(z[idx]) - y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        readout.eval()
        with torch.no_grad():
            val = ((readout(z_val) - y_val) ** 2).mean().item()
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in readout.state_dict().items()}
        if epoch % 50 == 0 or epoch == epochs:
            log(f"  readout epoch {epoch:4d}  val mse {val:.3e}  best {best:.3e}")
    readout.load_state_dict(best_state)
    return readout.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="folder holding a multichannel_cnn.pt trained by train.run_training")
    ap.add_argument("--n_latent", type=int, default=None, help="autoencoder latents (default: 1 + n_nuisance)")
    ap.add_argument("--epochs", type=int, default=300, help="autoencoder epochs")
    ap.add_argument("--readout_epochs", type=int, default=300)
    ap.add_argument("--polish", type=int, default=0, help="decoder re-fit rounds after the autoencoder")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    model, cfg, norm = load_checkpoint(os.path.join(args.run_dir, "multichannel_cnn.pt"))
    n_latent = args.n_latent or 1 + cfg.n_nuisance
    rms = lambda a: float(np.sqrt((np.asarray(a) ** 2).mean()))
    flat = lambda X: ((X - norm["mean"][None]) / norm["std"][None]).reshape(len(X), -1).astype(np.float32)
    cd_norm = lambda cd: ((cd - norm["y_mean"]) / norm["y_std"]).astype(np.float32)
    raw_std = np.repeat(norm["std"][:, 0], cfg.n_wavelength)
    weight = (raw_std ** 2).astype(np.float32)

    # The exact training / validation split the regression network saw.
    data = make_data(cfg, cfg.n_samples, cfg.seed)
    train_idx, val_idx, _ = split_indices(cfg.n_samples, cfg.val_ratio, cfg.test_ratio, cfg.seed)
    add_label_noise(data.y, train_idx, val_idx, cfg)

    epochs = max(args.epochs, -(-MIN_DECODER_STEPS // -(-len(train_idx) // BATCH_SIZE)))
    encoder, decoder = train_autoencoder(flat(data.X[train_idx]), n_latent, epochs,
                                         log=lambda m: print(m, flush=True))

    ev = make_data(cfg, N_EVAL, EVAL_SEED)
    ev_clean = make_data(cfg, N_EVAL, EVAL_SEED, noise_std=0.0)
    y = ev.y.astype(np.float64)
    net_cd = cnn_predict(model, norm, ev.X)
    if cfg.n_nuisance:
        known, unknown = cramer_rao_bound_joint(y, ev.nuisance, ev.wavelength, cfg)
        bounds = {"bound_known_rms": rms(known), "bound_rms": rms(unknown)}
    else:
        unknown = cramer_rao_bound(y, ev.wavelength, cfg)
        bounds = {"bound_rms": rms(unknown)}

    # How close each evaluation sample sits to the edge of the training
    # range: 0 = centre, 1 = some parameter (CD or a nuisance) at its limit.
    # The learned manifold is least accurate there.
    half_range = (cfg.cd_max - cfg.cd_min) / 2
    edge = np.abs(y - (cfg.cd_min + cfg.cd_max) / 2) / half_range
    if cfg.n_nuisance:
        edge = np.maximum(edge, np.abs(ev.nuisance).max(axis=1))

    def by_edge(err):
        out = []
        for lo, hi in zip(EDGE_BINS[:-1], EDGE_BINS[1:]):
            m = (edge >= lo) & (edge < hi)
            out.append({"edge_lo": lo, "edge_hi": hi, "n": int(m.sum()), "bound_rms": rms(unknown[m]),
                        "network_rmse": rms((net_cd - y)[m]), "rmse": rms(err[m])})
        return out

    Xf_train = flat(data.X[train_idx])
    # Starting latents for any spectrum, in the current decoder's coordinates.
    start = lambda Xf: encoder(torch.from_numpy(Xf)).detach().numpy()

    def fit(X, z0=None):
        """Fitted latents and weighted misfit for raw spectra X."""
        Xf = flat(X)
        z, cost, _ = gauss_newton(decoder, Xf, start(Xf) if z0 is None else z0, weight)
        return z, cost / Xf.shape[1]

    rounds = []
    for r in range(args.polish + 1):
        z_train, _ = fit(data.X[train_idx])
        if r > 0:
            # New decoder on standardized fitted latents; the encoder's
            # output is mapped into the same coordinates.
            z_mean, z_std = z_train.mean(axis=0), z_train.std(axis=0)
            to_std = lambda z, m=z_mean, s=z_std: ((z - m) / s).astype(np.float32)
            decoder = fit_decoder(Xf_train, to_std(z_train), epochs, seed=r)
            start = lambda Xf, prev=start, f=to_std: f(prev(Xf))
            z_train, _ = fit(data.X[train_idx])
        z_val, _ = fit(data.X[val_idx])
        r_mean, r_std = z_train.mean(axis=0), z_train.std(axis=0)
        scale = lambda z: ((z - r_mean) / r_std).astype(np.float32)
        readout = train_readout(scale(z_train), cd_norm(data.y[train_idx]),
                                scale(z_val), cd_norm(data.y[val_idx]), args.readout_epochs)

        z_ev, misfit = fit(ev.X)
        with torch.no_grad():
            readout_cd = readout(torch.from_numpy(scale(z_ev))).numpy().astype(np.float64) * norm["y_std"] + norm["y_mean"]
            denoised = decoder(torch.from_numpy(z_ev)).numpy()
        denoised_raw = (denoised * raw_std).reshape(len(y), cfg.n_channels, -1) + norm["mean"][None]
        net_on_denoised_cd = cnn_predict(model, norm, denoised_raw.astype(np.float32))
        rounds.append({
            "polish_round": r,
            # How well the fitted spectrum recovers the noise-free one. A
            # perfect manifold leaves noise_std * sqrt(n_latent / D) in it.
            "denoised_error_raw_rms": rms((denoised - flat(ev_clean.X)) * raw_std),
            "misfit_rms_raw": float(np.sqrt(misfit.mean())),
            "network_on_denoised": summarize(net_on_denoised_cd - y, cfg.error_threshold_nm),
            "readout": summarize(readout_cd - y, cfg.error_threshold_nm),
            "readout_by_edge": by_edge(readout_cd - y),
        })
        print(json.dumps(rounds[-1], indent=2), flush=True)

    result = {
        "run_dir": args.run_dir, "n_latent": n_latent, "n_train": len(train_idx),
        "autoencoder_epochs": epochs, "noise_std": cfg.noise_std,
        "denoised_error_ideal": cfg.noise_std * float(np.sqrt(n_latent / len(raw_std))),
        **bounds,
        "network": summarize(net_cd - y, cfg.error_threshold_nm),
        "rounds": rounds,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "rounds"}, indent=2))
    tag = args.tag or f"{os.path.basename(os.path.normpath(args.run_dir))}_manifold{n_latent}_polish{args.polish}"
    os.makedirs(os.path.join(EXPERIMENT_DIR, "refine"), exist_ok=True)
    with open(os.path.join(EXPERIMENT_DIR, "refine", f"{tag}.json"), "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
