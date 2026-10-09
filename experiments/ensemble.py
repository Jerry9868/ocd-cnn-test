"""Average the CD predictions of several trained checkpoints.

    python experiments/ensemble.py outputs/experiments/ablation_mlp/base outputs/experiments/ablation_mlp/seed1 ...

Scores each member and their mean on the same fresh evaluation spectra as
the other experiments. Members must share the generator settings (they may
differ in seed, architecture and training set).
"""
import json
import os
import sys

import numpy as np

from _common import EVAL_SEED, EXPERIMENT_DIR

from cramer_rao import cnn_predict, cramer_rao_bound, summarize
from inference import load_checkpoint
from refine import N_EVAL, make_data


def main():
    run_dirs = sys.argv[1:]
    preds, eval_set, cfg = [], None, None
    for run_dir in run_dirs:
        model, cfg, norm = load_checkpoint(os.path.join(run_dir, "multichannel_cnn.pt"))
        if eval_set is None:
            eval_set = make_data(cfg, N_EVAL, EVAL_SEED)
        preds.append(cnn_predict(model, norm, eval_set.X))
    y = eval_set.y.astype(np.float64)
    bound = float(np.sqrt((cramer_rao_bound(y, eval_set.wavelength, cfg) ** 2).mean()))

    result = {"bound_rms": bound, "members": {}}
    for run_dir, pred in zip(run_dirs, preds):
        result["members"][run_dir] = summarize(pred - y, cfg.error_threshold_nm)
    result["ensemble"] = summarize(np.mean(preds, axis=0) - y, cfg.error_threshold_nm)
    # How alike are the members' errors? (1 = identical, averaging gains nothing)
    errs = np.array(preds) - y
    result["mean_error_correlation"] = float(np.corrcoef(errs)[np.triu_indices(len(preds), 1)].mean())

    for name, r in [*result["members"].items(), ("ensemble", result["ensemble"])]:
        print(f"{os.path.basename(os.path.normpath(name)):12s} rmse {r['rmse']:.4f} ({r['rmse'] / bound:.2f}x)  "
              f"max {r['max']:.3f}  >thr {r['fraction_over_threshold']:.2%}")
    print(f"mean error correlation between members: {result['mean_error_correlation']:.2f}")
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    tag = "_".join(os.path.basename(os.path.dirname(os.path.normpath(d))) for d in run_dirs[:1])
    with open(os.path.join(EXPERIMENT_DIR, f"ensemble_{tag}.json"), "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
