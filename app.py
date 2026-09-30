"""Spectrum viewer / CD predictor / trainer.

Run with:
    streamlit run app.py

Three tabs:
  * 光譜檢視與預測 -- simulate a spectrum (16 Mueller elements + TE + TM, for
    testing, before the real measurement format is finalized) or upload a
    spectrum file, choose which channels to display, see the model's
    CD-sensitive wavelength regions highlighted on each curve, and get a CD
    prediction from the trained Multi-channel 1D CNN.
  * 批次驗證 -- upload many spectrum files at once plus an optional TEM CD
    reference table (matched by filename), get a numbered results table with
    per-sample predictions/errors, download it as CSV, see every sample's
    spectrum overlaid per channel, and inspect any one sample's channels
    either raw or as a deviation from the batch mean or from another chosen
    sample.
  * 模型訓練 / 超參數 -- adjust training hyperparameters and retrain, without
    touching config.py or the command line.

outputs/multichannel_cnn.pt is the checkpoint all tabs read/write.
"""
import dataclasses
import os
import re
import sys
import time

import numpy as np
import pandas as pd
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


def _shade_sensitivity(ax, ch: int, sens, model_cfg):
    """Shade the top SENSITIVITY_PERCENTILE% most CD-sensitive wavelength
    regions for channel `ch` (see inference.sensitivity_map). No-op if
    sens is None."""
    if sens is None:
        return
    sens_wl = np.linspace(model_cfg.wavelength_min, model_cfg.wavelength_max, model_cfg.n_wavelength)
    mask = sens[ch] >= np.percentile(sens[ch], SENSITIVITY_PERCENTILE)
    for lo, hi in highlighted_spans(mask, sens_wl):
        ax.axvspan(lo, hi, color="orange", alpha=0.25, linewidth=0)


def plot_channels_overlay(samples: list, wl: np.ndarray, selected: list, title: str, sens=None, model_cfg=None):
    """One line per sample (color-coded, by 編號), per selected channel --
    an at-a-glance view of an entire uploaded batch. `sens`, if given, is a
    single (n_channels, L) sensitivity map (e.g. averaged across the batch)
    shaded behind every sample's lines."""
    n = len(selected)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows), squeeze=False)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(samples) - 1, 1)) for i in range(len(samples))]
    for i, ch in enumerate(selected):
        ax = axes[i // ncols][i % ncols]
        _shade_sensitivity(ax, ch, sens, model_cfg)
        for s, color in zip(samples, colors):
            ax.plot(wl, s["X_grid"][ch], linewidth=0.8, color=color, label=f"#{s['編號']}")
        ax.set_title(CHANNEL_LABELS[ch], fontsize=9)
        ax.tick_params(labelsize=7)
    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")
    if len(samples) <= 12:
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize=7, ncol=min(len(samples), 4))
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_channels(X: np.ndarray, wl: np.ndarray, selected: list, sens, model_cfg, title: str):
    """Grid of line plots for the selected channel indices, with optional
    sensitivity shading (see sensitivity_map). Returns the Figure."""
    n = len(selected)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows), squeeze=False)
    for i, ch in enumerate(selected):
        ax = axes[i // ncols][i % ncols]
        ax.plot(wl, X[ch], linewidth=1, color="tab:blue")
        _shade_sensitivity(ax, ch, sens, model_cfg)
        ax.set_title(CHANNEL_LABELS[ch], fontsize=9)
        ax.tick_params(labelsize=7)
    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


tab_viewer, tab_batch, tab_train = st.tabs(["光譜檢視與預測", "批次驗證", "模型訓練 / 超參數"])

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
            title = "選定的量測通道（橘色區域 = 模型判斷的 CD 特徵敏感波段）" if sens is not None \
                else "選定的量測通道（尚無模型，無法標示特徵區域）"
            fig = plot_channels(X, wl, selected, sens, model_cfg, title)
            st.pyplot(fig)
            plt.close(fig)

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

# ----------------------------------------------------------------- batch ---
with tab_batch:
    batch_model_bundle = get_model()
    if batch_model_bundle is not None:
        b_model, b_cfg, b_norm = batch_model_bundle

    batch_source = st.radio(
        "批次資料來源", ["上傳光譜檔案", "生成模擬批次資料（測試用）"], horizontal=True, key="batch_source",
    )

    if batch_source == "上傳光譜檔案":
        st.caption(
            "批次上傳多筆光譜檔（每個檔案代表一筆樣本，格式同「光譜檢視與預測」頁籤：.csv/.dat/.npy/.zip），"
            "可選擇搭配一份 TEM CD 參考對照表（.csv，需有 filename 與 CD_TEM_nm 兩欄，依檔名比對），"
            "一次計算模型在多筆樣本上的預測與誤差。"
        )

        batch_files = st.file_uploader(
            "上傳多筆光譜檔", type=["csv", "dat", "npy", "zip"], accept_multiple_files=True, key="batch_spectra",
        )
        ref_file = st.file_uploader("上傳 TEM CD 參考對照表（選填）", type=["csv"], key="batch_ref")

        ref_map = {}
        if ref_file is not None:
            try:
                ref_df = pd.read_csv(ref_file)
                cols_lower = {c.lower().strip(): c for c in ref_df.columns}
                fname_col = cols_lower.get("filename") or cols_lower.get("sample_id")
                cd_col = cols_lower.get("cd_tem_nm") or cols_lower.get("cd_tem") or cols_lower.get("tem_cd_nm")
                if not fname_col or not cd_col:
                    st.error("參考表需要包含 filename（或 sample_id）與 CD_TEM_nm 欄位。")
                else:
                    ref_map = dict(zip(ref_df[fname_col].astype(str), ref_df[cd_col].astype(float)))
            except Exception as e:
                st.error(f"讀取參考表失敗：{e}")

        if batch_files and st.button("執行批次預測"):
            if batch_model_bundle is None:
                st.error("尚無可用模型，請先到「模型訓練」頁籤訓練一次。")
            else:
                target_wl = np.linspace(b_cfg.wavelength_min, b_cfg.wavelength_max, b_cfg.n_wavelength)

                samples = []  # list of dicts: idx, filename, X_grid, wl, ref_cd, pred_cd, error, err_msg
                for i, f in enumerate(batch_files, start=1):
                    entry = {"編號": i, "filename": f.name}
                    try:
                        X, wl = load_spectrum_file(f, b_cfg)
                        X_grid = resample_to_grid(X, wl, target_wl)
                        pred = predict_cd(b_model, b_norm, X_grid)
                        ref_cd = ref_map.get(f.name)
                        entry.update({
                            "X_grid": X_grid,
                            "Reference_CD_nm": ref_cd,
                            "Predicted_CD_nm": pred,
                            "Error_nm": (pred - ref_cd) if ref_cd is not None else None,
                        })
                    except Exception as e:
                        entry.update({"X_grid": None, "Reference_CD_nm": None, "Predicted_CD_nm": None,
                                      "Error_nm": None, "讀取錯誤": str(e)})
                    samples.append(entry)

                st.session_state.batch_results = samples
                st.session_state.batch_target_wl = target_wl
                st.session_state.batch_cfg = b_cfg
                st.session_state.batch_ref_label = "TEM"

    else:
        st.caption(
            "不需要真實資料，用目前的模型設定直接生成多筆模擬光譜，CD 值在您指定的範圍內平均分布，"
            "方便確認批次驗證頁籤（總覽圖、逐筆檢視、差值顯示）功能是否正常。"
        )
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            n_sim = st.number_input("樣本數量", min_value=2, max_value=50, value=6, step=1)
        with sc2:
            sim_cd_min = st.number_input("CD 下限 (nm)", value=float(default_cfg.cd_min))
        with sc3:
            sim_cd_max = st.number_input("CD 上限 (nm)", value=float(default_cfg.cd_max))

        if st.button("生成模擬批次資料"):
            if batch_model_bundle is None:
                st.error("尚無可用模型，請先到「模型訓練」頁籤訓練一次。")
            else:
                target_wl = np.linspace(b_cfg.wavelength_min, b_cfg.wavelength_max, b_cfg.n_wavelength)
                cd_values = np.linspace(sim_cd_min, sim_cd_max, int(n_sim))

                samples = []
                for i, cd in enumerate(cd_values, start=1):
                    X, wl = generate_single_spectrum(float(cd), b_cfg, seed=10_000 + i)
                    X_grid = resample_to_grid(X, wl, target_wl)
                    pred = predict_cd(b_model, b_norm, X_grid)
                    samples.append({
                        "編號": i,
                        "filename": f"simulated_CD{cd:.1f}nm",
                        "X_grid": X_grid,
                        "Reference_CD_nm": float(cd),
                        "Predicted_CD_nm": pred,
                        "Error_nm": pred - float(cd),
                    })

                st.session_state.batch_results = samples
                st.session_state.batch_target_wl = target_wl
                st.session_state.batch_cfg = b_cfg
                st.session_state.batch_ref_label = "模擬 Ground-truth"

    if st.session_state.get("batch_results"):
        ref_label = st.session_state.get("batch_ref_label", "參考")
        samples = st.session_state.batch_results
        display_cols = ["編號", "filename", "Reference_CD_nm", "Predicted_CD_nm", "Error_nm"]
        table_df = pd.DataFrame([{k: s.get(k) for k in display_cols + ["讀取錯誤"]} for s in samples])
        st.dataframe(table_df, width="stretch", hide_index=True)

        matched = table_df.dropna(subset=["Error_nm"])
        if len(matched):
            errs = matched["Error_nm"].abs().to_numpy()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"已有{ref_label}值的樣本數", f"{len(matched)} / {len(samples)}")
            c2.metric("MAE (nm)", f"{errs.mean():.3f}")
            c3.metric("RMSE (nm)", f"{np.sqrt((errs ** 2).mean()):.3f}")
            c4.metric("Max |Error| (nm)", f"{errs.max():.3f}")
        else:
            st.info(f"沒有樣本比對到{ref_label}值（檔名需與參考表一致），僅顯示模型預測結果。")

        csv_bytes = table_df.drop(columns=["讀取錯誤"]).to_csv(index=False).encode("utf-8")
        st.download_button("下載結果 CSV", data=csv_bytes, file_name="batch_predictions.csv", mime="text/csv")

        st.divider()
        ok_samples = [s for s in samples if s.get("X_grid") is not None]
        if ok_samples:
            st.subheader("所有樣本光譜總覽")
            overlay_baseline_mode = st.radio(
                "顯示方式", ["原始光譜", "與批次平均光譜的差值", "與指定樣本光譜的差值"],
                horizontal=True, key="overlay_baseline_mode",
            )

            overlay_baseline = None
            overlay_suffix = ""
            if overlay_baseline_mode == "與批次平均光譜的差值":
                overlay_baseline = np.mean(np.stack([s["X_grid"] for s in ok_samples]), axis=0)
                overlay_suffix = "；每條線 = 該樣本與批次平均光譜的差值"
            elif overlay_baseline_mode == "與指定樣本光譜的差值":
                overlay_options = {f"{s['編號']} - {s['filename']}": s for s in ok_samples}
                overlay_ref_label = st.selectbox("選擇作為基準的樣本", list(overlay_options.keys()), key="overlay_ref")
                overlay_baseline = overlay_options[overlay_ref_label]["X_grid"]
                overlay_suffix = f"；每條線 = 該樣本與 {overlay_ref_label} 的差值"

            display_samples = ok_samples if overlay_baseline is None else [
                {**s, "X_grid": s["X_grid"] - overlay_baseline} for s in ok_samples
            ]

            overlay_sens = None
            if batch_model_bundle is not None:
                sens_maps = [sensitivity_map(b_model, b_norm, s["X_grid"]) for s in ok_samples]
                overlay_sens = np.mean(np.stack(sens_maps), axis=0)  # (n_channels, L), averaged across the batch
            overlay_title = (
                f"批次內全部 {len(ok_samples)} 筆樣本（依編號上色"
                f"{'；橘色區域 = 批次平均後的 CD 特徵敏感波段' if overlay_sens is not None else ''}"
                f"{overlay_suffix}）"
            )
            overlay_fig = plot_channels_overlay(
                display_samples, st.session_state.batch_target_wl, selected, overlay_title,
                sens=overlay_sens, model_cfg=st.session_state.batch_cfg,
            )
            st.pyplot(overlay_fig)
            plt.close(overlay_fig)

            st.divider()
            st.subheader("單一樣本細看")
            options = {f"{s['編號']} - {s['filename']}": s for s in ok_samples}
            picked_label = st.selectbox("選擇要檢視的樣本", list(options.keys()))
            picked = options[picked_label]

            baseline_mode = st.radio(
                "顯示方式", ["原始光譜", "與批次平均光譜的差值", "與指定樣本光譜的差值"], horizontal=True,
            )

            baseline_spectrum = None
            title_suffix = ""
            if baseline_mode == "與批次平均光譜的差值":
                baseline_spectrum = np.mean(np.stack([s["X_grid"] for s in ok_samples]), axis=0)  # (n_channels, L)
                title_suffix = "（與批次平均光譜的差值）"
            elif baseline_mode == "與指定樣本光譜的差值":
                ref_options = {k: v for k, v in options.items() if v is not picked}
                if ref_options:
                    ref_label = st.selectbox("選擇作為基準的樣本", list(ref_options.keys()))
                    baseline_spectrum = ref_options[ref_label]["X_grid"]
                    title_suffix = f"（與 {ref_label} 的差值）"
                else:
                    st.info("批次裡只有一筆樣本，無法選擇其他樣本作為基準。")

            X_show = (picked["X_grid"] - baseline_spectrum) if baseline_spectrum is not None else picked["X_grid"]

            b_sens = sensitivity_map(b_model, b_norm, picked["X_grid"]) if batch_model_bundle is not None else None
            title = f"#{picked['編號']} {picked['filename']}{title_suffix}"
            fig = plot_channels(X_show, st.session_state.batch_target_wl, selected, b_sens, st.session_state.batch_cfg, title)
            st.pyplot(fig)
            plt.close(fig)

            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("模型預測 CD (nm)", f"{picked['Predicted_CD_nm']:.2f}")
            if picked["Reference_CD_nm"] is not None:
                pc2.metric(f"{ref_label} CD (nm)", f"{picked['Reference_CD_nm']:.2f}")
                pc3.metric("誤差 (nm)", f"{picked['Error_nm']:+.2f}")

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
            epochs = st.number_input("epochs", 1, 2000, default_cfg.epochs, step=10)
            lr_schedule = st.selectbox(
                "學習率排程", ["cosine", "plateau"],
                index=["cosine", "plateau"].index(default_cfg.lr_schedule),
                help="cosine：學習率在固定 epochs 內平滑遞減，不做 early stopping（建議）。"
                     "plateau：舊版 ReduceLROnPlateau + early stopping。",
            )
            patience = st.number_input("patience（early stopping，僅 plateau 模式）", 1, 500, default_cfg.patience, step=5)

        with st.expander("模型架構（進階）"):
            a1, a2 = st.columns(2)
            with a1:
                pooling = st.selectbox(
                    "pooling", ["flatten", "avg"], index=["flatten", "avg"].index(default_cfg.pooling),
                    help="flatten：保留波長位置資訊（建議，CD 主要反映在特徵峰的位置）。avg：舊版全域平均池化。",
                )
            with a2:
                dropout = st.number_input("dropout", 0.0, 0.9, default_cfg.dropout, step=0.05, format="%.2f")

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
            lr_schedule=str(lr_schedule),
            patience=int(patience),
            pooling=str(pooling),
            dropout=float(dropout),
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
                    f"已耗時 {_fmt_secs(elapsed)}，預估剩餘 ≤{_fmt_secs(eta)}"
                    + ("（可能因 early stopping 提早結束）" if cfg.lr_schedule == "plateau" else "")
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
