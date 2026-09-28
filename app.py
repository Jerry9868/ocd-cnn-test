"""Spectrum viewer / CD predictor / trainer.

Run with:
    streamlit run app.py

Two tabs:
  * 光譜檢視與預測 -- simulate a Mueller-matrix spectrum (for testing, before
    the real measurement format is finalized) or upload a spectrum file,
    choose which of the 16 Mueller elements to display, see the model's
    CD-sensitive wavelength regions highlighted on each curve, and get a CD
    prediction from the trained Multi-channel 1D CNN.
  * 模型訓練 / 超參數 -- adjust training hyperparameters and retrain, without
    touching config.py or the command line.

outputs/multichannel_cnn.pt is the checkpoint both tabs read/write.
"""
import dataclasses
import os
import sys

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import Config  # noqa: E402
from spectrum_io import MUELLER_LABELS, generate_single_spectrum, load_spectrum_file, resample_to_grid  # noqa: E402
from inference import load_checkpoint, predict_cd, sensitivity_map  # noqa: E402
from train import run_training  # noqa: E402

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "outputs", "multichannel_cnn.pt")
SENSITIVITY_PERCENTILE = 75  # per-channel: shade the top 25% most CD-sensitive wavelength points

st.set_page_config(page_title="Mueller Matrix CD Viewer", layout="wide")
st.title("Mueller Matrix 光譜檢視與 CD 預測")

default_cfg = Config()


@st.cache_resource
def get_model():
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    return load_checkpoint(CHECKPOINT_PATH)


def highlighted_spans(mask: np.ndarray, wavelength: np.ndarray):
    """Contiguous (start_wl, end_wl) spans where mask is True."""
    spans = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            spans.append((wavelength[start], wavelength[i - 1]))
            start = None
    if start is not None:
        spans.append((wavelength[start], wavelength[-1]))
    return spans


tab_viewer, tab_train = st.tabs(["光譜檢視與預測", "模型訓練 / 超參數"])

# ---------------------------------------------------------------- viewer ---
with tab_viewer:
    model_bundle = get_model()
    model_cfg = model_bundle[1] if model_bundle else default_cfg

    if "spectrum" not in st.session_state:
        st.session_state.spectrum = None
        st.session_state.wavelength = None
        st.session_state.true_cd = None
        st.session_state.source = None

    with st.sidebar:
        st.header("資料來源")
        mode = st.radio("選擇來源", ["生成模擬光譜（測試用）", "上傳光譜檔"])

        if mode == "生成模擬光譜（測試用）":
            cd_value = st.slider("目標 CD (nm)", float(default_cfg.cd_min), float(default_cfg.cd_max), 50.0)
            if st.button("產生新的模擬光譜"):
                X, wl = generate_single_spectrum(cd_value, model_cfg)
                st.session_state.spectrum = X
                st.session_state.wavelength = wl
                st.session_state.true_cd = cd_value
                st.session_state.source = "simulated"
        else:
            st.caption("目前接受格式：.csv（header: wavelength, M00...M33）或 .npy（shape (16, L)）。實際量測格式待確認後再對接。")
            uploaded = st.file_uploader("上傳光譜檔", type=["csv", "npy"])
            if uploaded is not None:
                try:
                    X, wl = load_spectrum_file(uploaded, model_cfg)
                    st.session_state.spectrum = X
                    st.session_state.wavelength = wl
                    st.session_state.true_cd = None
                    st.session_state.source = "uploaded"
                except Exception as e:
                    st.error(f"讀取失敗：{e}")

        if model_bundle is None:
            st.warning("尚未找到訓練好的模型（outputs/multichannel_cnn.pt）。請先到「模型訓練」頁籤訓練一次。")

        st.header("顯示的 Mueller Elements")

        def _apply_select_all():
            for lbl in MUELLER_LABELS:
                st.session_state[f"ch_{lbl}"] = st.session_state["select_all"]

        st.checkbox("全選 / 全不選", value=True, key="select_all", on_change=_apply_select_all)
        selected = []
        for row in range(4):
            cols = st.columns(4)
            for col in range(4):
                idx = row * 4 + col
                label = MUELLER_LABELS[idx]
                with cols[col]:
                    if st.checkbox(label, value=True, key=f"ch_{label}"):
                        selected.append(idx)

    if st.session_state.spectrum is None:
        st.info("請從左側產生模擬光譜，或上傳光譜檔案。")
    elif not selected:
        st.info("請至少勾選一個 Mueller element 來顯示光譜。")
    else:
        X = st.session_state.spectrum
        wl = st.session_state.wavelength

        sens = None
        if model_bundle is not None:
            model, cfg, norm = model_bundle
            X_grid = resample_to_grid(X, wl, np.linspace(cfg.wavelength_min, cfg.wavelength_max, cfg.n_wavelength))
            sens = sensitivity_map(model, norm, X_grid)  # (16, L_grid), on the model's own wavelength grid

        col_plot, col_pred = st.columns([3, 1])

        with col_plot:
            n = len(selected)
            ncols = min(4, n)
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows), squeeze=False)
            for i, ch in enumerate(selected):
                ax = axes[i // ncols][i % ncols]
                ax.plot(wl, X[ch], linewidth=1, color="tab:blue")
                if sens is not None:
                    sens_wl = np.linspace(model_cfg.wavelength_min, model_cfg.wavelength_max, model_cfg.n_wavelength)
                    mask = sens[ch] >= np.percentile(sens[ch], SENSITIVITY_PERCENTILE)
                    for lo, hi in highlighted_spans(mask, sens_wl):
                        ax.axvspan(lo, hi, color="orange", alpha=0.25, linewidth=0)
                ax.set_title(MUELLER_LABELS[ch], fontsize=9)
                ax.tick_params(labelsize=7)
            for i in range(n, nrows * ncols):
                axes[i // ncols][i % ncols].axis("off")
            if sens is not None:
                fig.suptitle("選定的 Mueller Elements（橘色區域 = 模型判斷的 CD 特徵敏感波段）")
            else:
                fig.suptitle("選定的 Mueller Elements（尚無模型，無法標示特徵區域）")
            fig.tight_layout()
            st.pyplot(fig)

        with col_pred:
            st.subheader("預測結果")
            if st.session_state.true_cd is not None:
                st.metric("模擬 Ground-truth CD (nm)", f"{st.session_state.true_cd:.2f}")

            if model_bundle is not None:
                pred_cd = predict_cd(model, norm, X_grid)
                st.metric("模型預測 CD (nm)", f"{pred_cd:.2f}")
                if st.session_state.true_cd is not None:
                    st.metric("誤差 (nm)", f"{pred_cd - st.session_state.true_cd:+.2f}")
            else:
                st.write("尚無可用模型。")

# ----------------------------------------------------------------- train ---
with tab_train:
    st.caption(
        "調整超參數後按「開始訓練」，會用目前選的合成資料設定重新訓練 Multi-channel 1D CNN，"
        "完成後自動更新「光譜檢視與預測」頁籤使用的模型。訓練約需數十秒到數分鐘，畫面會卡住直到完成。"
    )

    with st.form("train_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            n_samples = st.number_input("n_samples（合成資料筆數）", 500, 100000, default_cfg.n_samples, step=500)
            noise_std = st.number_input("noise_std（量測雜訊標準差）", 0.0, 1.0, default_cfg.noise_std, step=0.005, format="%.3f")
            seed = st.number_input("seed（取樣/雜訊隨機種子）", 0, 999999, default_cfg.seed)
        with c2:
            hidden_dim = st.number_input("hidden_dim（模型寬度）", 8, 512, default_cfg.hidden_dim, step=8)
            batch_size = st.number_input("batch_size", 8, 1024, default_cfg.batch_size, step=8)
            weight_decay = st.number_input("weight_decay", 0.0, 1.0, default_cfg.weight_decay, step=1e-4, format="%.5f")
        with c3:
            lr = st.number_input("learning rate", 1e-6, 1.0, default_cfg.lr, step=1e-4, format="%.5f")
            epochs = st.number_input("epochs（上限）", 1, 2000, default_cfg.epochs, step=10)
            patience = st.number_input("patience（early stopping）", 1, 500, default_cfg.patience, step=5)

        submitted = st.form_submit_button("開始訓練")

    if submitted:
        cfg = dataclasses.replace(
            default_cfg,
            n_samples=int(n_samples),
            noise_std=float(noise_std),
            seed=int(seed),
            hidden_dim=int(hidden_dim),
            batch_size=int(batch_size),
            weight_decay=float(weight_decay),
            lr=float(lr),
            epochs=int(epochs),
            patience=int(patience),
        )
        progress_bar = st.progress(0.0, text="準備開始訓練...")

        def _on_epoch(epoch, max_epochs, log):
            progress_bar.progress(
                min(epoch / max_epochs, 1.0),
                text=f"epoch {epoch}/{max_epochs}  val_loss={log['val_loss']:.4f}  val_mean_err={log['val_mean_nm']:.3f}nm",
            )

        with st.spinner("訓練中..."):
            result = run_training(cfg, progress_cb=_on_epoch)

        progress_bar.progress(1.0, text="訓練完成")
        st.success("訓練完成，模型已更新。")

        m = result["metrics"]
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Test MAE (nm)", f"{m['mae']:.3f}")
        mc2.metric("Test RMSE (nm)", f"{m['rmse']:.3f}")
        mc3.metric("Test R²", f"{m['r2']:.4f}")
        mc4.metric("Max abs err (nm)", f"{m['max_abs_err_nm']:.3f}")

        report_path = os.path.join(cfg.output_dir, "train_report.png")
        if os.path.exists(report_path):
            st.image(report_path)

        get_model.clear()
        st.info("已清除模型快取，切換回「光譜檢視與預測」頁籤即可使用新模型。")
