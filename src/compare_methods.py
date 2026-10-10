"""Which method works best on a given data set? Cross-validates each and tabulates.

    python compare_methods.py --folds 5 --data_source folder --spectra_dir D:/data/spectra --labels_csv D:/data/labels.csv
    python compare_methods.py --methods mlp mlp_refine cnn --folds 5 --n_samples 300      # synthetic

Every method in METHODS is the Config given on the command line with a few
fields overridden, run through cross_validate.cross_validate on the same
folds, so the methods are compared on identical held-out samples.

This is the real-data counterpart of experiments/ablation.py, which scores
against freshly generated spectra and the Cramer-Rao bound and so only works
with the synthetic generator.

Writes <output_dir>/compare/<method>/cross_validation/ per method and
<output_dir>/compare/comparison.csv, ordered by the acceptance criterion:
fraction of samples over error_threshold_nm, then RMSE.
"""
import argparse
import dataclasses
import os

import pandas as pd

from config import Config
from cross_validate import cross_validate
from train import add_config_arguments

METHODS = {
    "mlp": dict(model="mlp", refine="off"),
    "mlp_refine": dict(model="mlp", refine="on"),
    "mlp_deriv1": dict(model="mlp", derivative=1, refine="off"),
    "cnn": dict(model="cnn", refine="off"),
    "cnn_refine": dict(model="cnn", refine="on"),
    "cnn_per_channel": dict(model="cnn", conv_groups="per_channel", refine="off"),
    "cnn_attention": dict(model="cnn", attention="input", refine="off"),
    "cnn_deriv1": dict(model="cnn", derivative=1, refine="off"),
    "cnn_deriv2": dict(model="cnn", derivative=2, refine="off"),
    "cnn2d": dict(model="cnn2d", refine="off"),
    "cnn3d": dict(model="cnn3d", refine="off"),
}
COLUMNS = ["max_abs_err_nm", "fraction_over_threshold", "meets_threshold", "rmse", "mae", "p95_abs_err_nm"]


def compare(cfg: Config, folds: int, methods: list, log=print) -> pd.DataFrame:
    base_dir = os.path.join(cfg.output_dir, "compare")
    rows = []
    for name in methods:
        log(f"== {name}")
        method_cfg = dataclasses.replace(cfg, **METHODS[name], output_dir=os.path.join(base_dir, name))
        summary = cross_validate(method_cfg, folds, log=log)["summary"]
        rows.append({"method": name, "n_samples": summary["n_samples"],
                     **{c: summary["reported"][c] for c in COLUMNS}})
    table = pd.DataFrame(rows).sort_values(["fraction_over_threshold", "rmse"]).reset_index(drop=True)
    table.to_csv(os.path.join(base_dir, "comparison.csv"), index=False)
    log("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=list(METHODS))
    add_config_arguments(p, Config())
    args = vars(p.parse_args())
    folds, methods = args.pop("folds"), args.pop("methods")
    compare(Config(**args), folds, methods)


if __name__ == "__main__":
    main()
