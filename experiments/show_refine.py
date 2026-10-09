"""Print one line per refine.py result.

    python experiments/show_refine.py                 # every result
    python experiments/show_refine.py base_none ...   # the named ones
"""
import glob
import json
import os
import sys

from _common import EXPERIMENT_DIR


def main():
    folder = os.path.join(EXPERIMENT_DIR, "refine")
    names = sys.argv[1:] or sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(folder, "*.json")))
    for name in names:
        path = os.path.join(folder, name + ".json")
        if not os.path.exists(path):
            print(f"{name:34s} (no result)")
            continue
        with open(path) as f:
            r = json.load(f)
        net, ref = r["network"], r["refined"]
        bar = r.get("error_bar")
        print(f"{name:34s} bound {r['bound_rms']:.3f}  decoder_err {r['decoder_error_raw_rms']:.4f}  "
              f"network {net['rmse']:.3f} (>thr {net['fraction_over_threshold']:.2%})  ->  "
              f"refined {ref['rmse']:.3f} max {ref['max']:.2f} (>thr {ref['fraction_over_threshold']:.2%})"
              + (f"  z_sd {bar['z_sd']:.2f}" if bar else ""))


if __name__ == "__main__":
    main()
