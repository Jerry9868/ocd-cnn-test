"""One chart of where each approach lands relative to the Cramer-Rao bound.

    python experiments/summary_plot.py

Reads the result files the other experiment scripts wrote under
outputs/experiments/; run those first (ablation.py arch / mlp / nuisance,
ensemble.py, refine.py).
"""
import json
import os

from _common import AQUA, BLUE, EXPERIMENT_DIR, TEXT, TEXT_SECONDARY, setup_matplotlib


def load(name):
    with open(os.path.join(EXPERIMENT_DIR, name)) as f:
        return json.load(f)


def panel(ax, rows, bound, title):
    labels = [r[0] for r in rows][::-1]
    values = [r[1] for r in rows][::-1]
    bars = ax.barh(labels, values, color=BLUE, height=0.55, zorder=3)
    ax.axvline(bound, color=AQUA, linestyle="--", zorder=4)
    ax.annotate(f"理論下限 {bound:.3f}", (bound, len(rows) - 0.45), textcoords="offset points",
                xytext=(4, 0), color=TEXT_SECONDARY, fontsize=9, va="center")
    for bar, v in zip(bars, values):
        ax.annotate(f"{v:.3f}", (v, bar.get_y() + bar.get_height() / 2), textcoords="offset points",
                    xytext=(4, 0), va="center", color=TEXT, fontsize=9)
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("CD RMSE (nm)")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)


def main():
    arch, mlp, nuis = load("ablation_arch.json"), load("ablation_mlp.json"), load("ablation_nuisance.json")
    ens_cnn, ens_mlp = load("ensemble_ablation_arch.json"), load("ensemble_ablation_mlp.json")
    ref = load(os.path.join("refine", "base_none.json"))
    ref_nuis = load(os.path.join("refine", "mlp_mt_n48000_labels.json"))

    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    panel(axes[0], [
        ("CNN（原預設）", arch["base"]["rmse"]),
        ("CNN × 3 集成", ens_cnn["ensemble"]["rmse"]),
        ("MLP", mlp["base"]["rmse"]),
        ("MLP × 3 集成", ens_mlp["ensemble"]["rmse"]),
        ("MLP，3 倍訓練資料", mlp["n24000"]["rmse"]),
        ("網路 + 正向模型擬合修正", ref["refined"]["rmse"]),
    ], ref["bound_rms"], "只有 CD 變動")
    panel(axes[1], [
        ("CNN，只預測 CD", nuis["cd_n48000"]["rmse"]),
        ("CNN，多任務", nuis["mt_n48000"]["rmse"]),
        ("MLP，只預測 CD", nuis["mlp_cd_n48000"]["rmse"]),
        ("MLP，多任務", nuis["mlp_mt_n48000"]["rmse"]),
        ("MLP 多任務 + 擬合修正", ref_nuis["refined"]["rmse"]),
    ], ref_nuis["bound_rms"], "另有 3 個干擾參數同時變動（48,000 筆）")
    fig.tight_layout()
    fig.savefig(os.path.join(EXPERIMENT_DIR, "summary.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
