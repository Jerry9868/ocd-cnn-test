"""Write synthetic spectra to disk in the layout real data is expected in.

    python make_example_dataset.py                       # 300 samples -> data/example/
    python make_example_dataset.py --n_samples 60 --wafers 6 --label_noise 0.3 --out data/example_small

For trying out the real-data path (Config.data_source = "folder", see
real_data.py) -- train.py, cross_validate.py and the app -- before any real
measurement is available, and as a template for laying real data out:

    <out>/spectra/sample_0001.csv ...   wavelength,M00,...,M33,TE,TM
    <out>/labels.csv                    filename,CD_nm,wafer

Each "wafer" is just a block of consecutive samples; the generator has no
wafer-to-wafer variation. --label_noise adds Gaussian error (nm) to the
reference CD in labels.csv, a stand-in for TEM error.

data/ is git-ignored.
"""
import argparse
import os

import numpy as np
import pandas as pd

from config import PROJECT_ROOT, Config
from spectrum_io import CHANNEL_LABELS
from train import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=300)
    p.add_argument("--wafers", type=int, default=10)
    p.add_argument("--label_noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "example"))
    args = p.parse_args()

    cfg = Config(n_samples=args.n_samples, seed=args.seed)
    data = load_dataset(cfg)
    spectra_dir = os.path.join(args.out, "spectra")
    os.makedirs(spectra_dir, exist_ok=True)

    header = ",".join(["wavelength"] + CHANNEL_LABELS[:cfg.n_channels])
    names = [f"sample_{i + 1:04d}.csv" for i in range(args.n_samples)]
    for name, spectrum in zip(names, data.X):
        np.savetxt(os.path.join(spectra_dir, name), np.column_stack([data.wavelength, spectrum.T]),
                   delimiter=",", header=header, comments="", fmt="%.6f")

    cd = data.y + np.random.default_rng(args.seed).normal(0, args.label_noise, size=args.n_samples)
    wafer = [f"W{i * args.wafers // args.n_samples + 1:02d}" for i in range(args.n_samples)]
    pd.DataFrame({"filename": names, "CD_nm": np.round(cd, 3), "wafer": wafer}).to_csv(
        os.path.join(args.out, "labels.csv"), index=False)
    print(f"wrote {args.n_samples} spectra to {spectra_dir} and {os.path.join(args.out, 'labels.csv')}")


if __name__ == "__main__":
    main()
