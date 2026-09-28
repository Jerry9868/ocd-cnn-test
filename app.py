"""Spectrum viewer / CD predictor / trainer.

Run with:
    streamlit run app.py

Two tabs:
  * 光譜檢視與預測 -- simulate a spectrum (16 Mueller elements + TE + TM, for
    testing, before the real measurement format is finalized) or upload a
    spectrum file, choose which channels to display, see the model's
    CD-sensitive wavelength regions highlighted on each curve, and get a CD
    prediction from the trained Multi-channel 1D CNN.
  * 模型訓練 / 超參數 -- adjust training hyperparameters and retrain, without
    touching config.py or the command line.

outputs/multichannel_cnn.pt is the checkpoint both tabs read/write.
"""
import dataclasses
import os
import re
import sys
import time

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# matplotlib's default font has no CJK glyphs, so any Chinese text on a
# figure (titles, suptitle) renders as mojibake boxes -- use a Traditional
# Chinese-capable font that ships with Windows.
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import Config  # noqa: E402
from spectrum_io import CHANNEL_LABELS, generate_single_spectrum, load_spectrum_file, resample_to_grid  # noqa: E402
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
    if "tem_cd" not in st.session_state:
        st.session_state.tem_cd = None

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
                st.session_state.tem_cd = None
        else:
            st.caption(
                "目前接受格式：.csv / .dat（header: wavelength, M00...M33, TE, TM）、"
                ".npy（shape (18, L)）、.zip（內含 .dat，見 src/spectrum_io.py 說明）。"
                "實際量測格式待確認後再對接，.dat/.zip 的解析規則目前是暫定猜測。"
            )
            uploaded = st.file_uploader("上傳光譜檔", type=["csv", "dat", "npy", "zip"])
            if uploaded is not None:
                try:
                    X, wl = load_spectrum_file(uploaded, model_cfg)
                    st.session_state.spectrum = X
                    st.session_state.wavelength = wl
                    st.session_state.true_cd = None
                    st.session_state.source = "uploaded"
                    st.session_state.tem_cd = None
                except Exception as e:
                    st.error(f"讀取失敗：{e}")

        st.header("TEM CD 參考值（選填）")
        st.caption("上傳 TEM 實際量測到的 CD 數值（檔案內第一個數字會被當作 CD，單位 nm），或直接手動輸入，用來跟模型預測值比較。")
        tem_file = st.file_uploader("上傳 TEM CD 參考檔", type=["csv", "txt", "dat"], key="tem_file")
        if tem_file is not None:
            try:
                content = tem_file.read()
                content = content.decode("utf-8") if isinstance(content, bytes) else content
                nums = re.findall(r"[-+]?\d*\.?\d+", content)
                if nums:
                    st.session_state.tem_cd = float(nums[0])
                else:
                    st.error("在檔案裡找不到數字。")
            except Exception as e:
                st.error(f"讀取 TEM 參考值失敗：{e}")
        manual_tem_cd = st.number_input(
            "或手動輸入 TEM CD (nm)", min_value=0.0, max_value=1000.0,
            value=float(st.session_state.tem_cd) if st.session_state.tem_cd is not None else 0.0,
            step=0.1,
        )
        if manual_tem_cd > 0:
            st.session_state.tem_cd = manual_tem_cd

        if model_bundle is None:
            st.warning("尚未找到訓練好的模型（outputs/multichannel_cnn.pt）。請先到「模型訓練」頁籤訓練一次。")

        st.header("顯示的量測通道")
        st.caption("M00–M33：65° 斜入射 Mueller matrix；TE / TM：正向入射參考光譜")

        def _apply_select_all():
            for lbl in CHANNEL_LABELS:
                st.session_state[f"ch_{lbl}"] = st.session_state["select_all"]

        st.checkbox("全選 / 全不選", value=True, key="select_all", on_change=_apply_select_all)
        selected = []
        n_cols = 4
        for row_start in range(0, len(CHANNEL_LABELS), n_cols):
            row_labels = CHANNEL_LABELS[row_start:row_start + n_cols]
            cols = st.columns(n_cols)
            for col, label in zip(cols, row_labels):
                idx = CHANNEL_LABELS.index(label)
                with col:
                    if st.checkbox(label, value=True, key=f"ch_{label}"):
                        selected.append(idx)

    if st.session_state.spectrum is None:
        st.info("請從左側產生模擬光譜，或上傳光譜檔案。")
    elif not selected:
        st.info("請至少勾選一個通道來顯示光譜。")
    else:
        X = st.session_state.spectrum
        wl = st.session_state.wavelength

        sens = None
        if model_bundle is not None:
            model, cfg, norm = model_bundle
            X_grid = resample_to_grid(X, wl, np.linspace(cfg.wavelength_min, cfg.wavelength_max, cfg.n_wavelength))
            sens = sensitivity_map(model, norm, X_grid)  # (n_channels, L_grid), on the model's own wavelength grid

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
                ax.set_title(CHANNEL_LABELS[ch], fontsize=9)
                ax.tick_params(labelsize=7)
            for i in range(n, nrows * ncols):
                axes[i // ncols][i % ncols].axis("off")
            if sens is not None:
                fig.suptitle("選定的量測通道（橘色區域 = 模型判斷的 CD 特徵敏感波段）")
            else:
                fig.suptitle("選定的量測通道（尚無模型，無法標示特徵區域）")
            fig.tight_layout()
            st.pyplot(fig)

        with col_pred:
            st.subheader("預測結果")
            if st.session_state.true_cd is not None:
                st.metric("模擬 Ground-truth CD (nm)", f"{st.session_state.true_cd:.2f}")
            if st.session_state.tem_cd is not None:
                st.metric("TEM 參考 CD (nm)", f"{st.session_state.tem_cd:.2f}")

            if model_bundle is not None:
                pred_cd = predict_cd(model, norm, X_grid)
                st.metric("模型預測 CD (nm)", f"{pred_cd:.2f}")
                if st.session_state.true_cd is not None:
                    st.metric("誤差 vs 模擬 (nm)", f"{pred_cd - st.session_state.true_cd:+.2f}")
                if st.session_state.tem_cd is not None:
                    st.metric("誤差 vs TEM (nm)", f"{pred_cd - st.session_state.tem_cd:+.2f}")
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
        train_start = time.time()

        def _fmt_secs(s: float) -> str:
            m, s = divmod(max(0, int(s)), 60)
            return f"{m:02d}:{s:02d}"

        def _on_epoch(epoch, max_epochs, log):
            elapsed = time.time() - train_start
            eta = (elapsed / epoch) * (max_epochs - epoch)
            progress_bar.progress(
                min(epoch / max_epochs, 1.0),
                text=(
                    f"epoch {epoch}/{max_epochs}  val_loss={log['val_loss']:.4f}  val_mean_err={log['val_mean_nm']:.3f}nm  "
                    f"已耗時 {_fmt_secs(elapsed)}，預估剩餘 ≤{_fmt_secs(eta)}（可能因 early stopping 提早結束）"
                ),
            )

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
