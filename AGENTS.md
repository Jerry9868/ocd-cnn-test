# 給接手 agent 的交接文件

這份文件寫給接手這個專案的 AI agent（或工程師）。假設你沒有先前對話的任何脈絡。使用者以繁體中文溝通，請用繁體中文回覆。

操作說明（安裝、資料格式、指令）在 [README.md](README.md)，這裡不重複，只寫 README 沒有的：目的、規則、現況、接下來該做什麼。

## 目的

1. 在公司電腦上，用**真實光譜與 TEM** 訓練模型並達到驗收標準。
2. 研究各種 ML 方法對這個問題是否有幫助，結果寫進報告（`reports/OCD_ML_report.pptx`，10 頁以內）。

驗收標準（使用者於 2026-10-10 決定）：**每一筆樣本的 CD 誤差都在 0.3 nm 以內**為主，RMSE 為輔。程式裡是 `Config.error_threshold_nm`，所有指標輸出都先列最大誤差與超過門檻的比例，再列 RMSE。不要把標準改寫成 3σ 或其他定義。

## 不可違反的規則

- **光譜與 TEM 資料不得離開公司內網。** 不上傳到任何外部服務，不貼進會送往外部的請求，不放進 commit。
- **不加入任何會連網的程式碼或套件**（遙測、雲端同步、外部 API）。使用者對這點零容忍；連框架內建的匿名統計都必須關掉（Streamlit 的已在 `.streamlit/config.toml` 關閉）。
- 真實資料放在 `data/`、`real_data/`、`raw_data/` 或專案之外；這些資料夾與 `outputs/` 都被 git 忽略。**commit 前確認沒有資料檔或模型檔被加入。** 用真實資料訓練出的模型與結果也視同資料。
- 報告裡的數字要標明來自合成資料或真實資料，兩者不可混用。

## 現況（2026-10-10）

**到目前為止，沒有任何一個數字來自真實資料。** 全部結果都是用 `src/data_generation.py` 的合成光譜產生的。這個產生器不是物理模擬（沒有 RCWA），只是讓 18 個通道共用幾個隨 CD 移動的基底函數，用來驗證流程與比較方法。

已完成並實際執行過的：

- 訓練、預測、擬合修正、操作介面（合成資料）。
- 真實資料的讀取、訓練、交叉驗證、方法比較流程，指令列與 app 都能用——但**只用合成光譜寫成的檔案測試過**（`src/make_example_dataset.py` 產生的 `data/example/`）。
- 專案沒有自動化測試。改動後請照 README「確認安裝正確」的三個指令手動確認，並用 `streamlit run app.py` 實際操作一次。
- 一系列方法比較實驗（`experiments/`），結論見下方。

尚未驗證、屬於猜測的：

- **真實光譜檔的格式。** `src/spectrum_io.py` 對 `.dat` 與 `.zip` 的解析是猜的，沒有看過真實檔案。
- **通道數與波長範圍。** 預設 18 個通道（16 個 Mueller 元素 + 正向入射的 TE、TM）、200–1000 nm、101 點，都是暫定值。Mueller 光譜是 65 度斜入射量測。
- **真實雜訊大小、TEM 誤差大小、樣本數。** 都未知。

## 合成資料上的結論

理論下限指 Cramér–Rao 下限：給定量測雜訊，任何無偏估計方法的最小 CD 誤差（`experiments/cramer_rao.py`）。只有 CD 變動、雜訊 0.01 時下限為 0.067 nm。

| 方法 | CD RMSE | 相對下限 |
|---|---|---|
| 多通道 1D CNN | 0.105 nm | 1.57 倍 |
| MLP | 0.077 nm | 1.15 倍 |
| MLP + 擬合修正 | 0.068 nm | 1.02 倍 |

- **擬合修正**（`src/refinement.py`）是改善最大的方法：用同一批訓練資料反過來訓練「CD → 光譜」的網路，再以網路預測為起點做 Gauss-Newton 擬合。附帶逐筆誤差棒、異常光譜警示，並可把波長軸偏移當未知數一起解。
- **訓練樣本少於約 700 筆時，擬合修正反而比網路差**（正向模型不夠準）。訓練時會用驗證資料比較兩者並記錄在模型檔（`recommended`），預測時自動選用。
- **樣本數是主要限制。** MLP 在 106 / 210 / 420 / 700 / 5,600 筆訓練樣本時的 RMSE 為 0.305 / 0.240 / 0.178 / 0.132 / 0.077 nm。
- **標註誤差**：TEM 誤差 0.3 nm 時擬合修正仍有 0.076 nm；1 nm 時明顯變差，且訓練範圍邊緣出現偏差。
- **其他結構參數同時變動**（以 3 個干擾參數模擬）：下限升到 0.090 nm。有這些參數的標註時擬合修正可到 0.096 nm；只有 CD 標註時約 0.125 nm，且在訓練範圍邊緣比網路還差。
- 加深加寬網路、改正規化或池化、超過 5,600 筆的更多資料：沒有幫助。
- CNN 內各通道獨立卷積（0.138 nm）比跨通道卷積（0.105 nm）差，表示跨通道資訊有用。
- **Channel Attention、2D / 3D CNN、微分特徵都沒有幫助**（`experiments/ablation.py stage`，對照 1D CNN 的 0.105 nm）：Channel Attention 0.108、2D CNN 0.111、3D CNN 0.119、一階微分 0.121、一階加二階微分 0.149 nm；MLP 加一階微分 0.096 nm（MLP 本身 0.077）。Channel Attention 學到的通道權重全部停在 0.98–1.01，等於沒有作用——合成資料的 18 個通道本來就是同等重要的隨機混合，所以這一項在真實資料上最值得重測。2D / 3D CNN 因為 CPU 運算成本，寬度只有 1D CNN 的 1/3 與 1/6。

MLP 優於 CNN 這一點最可能在真實資料上翻轉：合成光譜沒有真實光譜的局部結構。**不要假設任何排序在真實資料上成立，要重跑。**

## 在公司電腦上的第一步

依序進行，每一步確認後再往下：

1. **安裝並確認能跑**：照 README「安裝」與「確認安裝正確」。
2. **拿一份真實光譜檔，確認讀取正確。** 用 `src/spectrum_io.py` 的 `load_spectrum_file` 讀它，檢查通道數、通道順序、波長範圍。解析規則不對就修改 `_parse_table_text` / `_load_zip`，並把檔案開頭「GUESSED」的說明改成實際格式。通道順序必須是 M00、M01 … M33、TE、TM（`CHANNEL_LABELS`），否則要在讀取時重新排列。
3. **依實際光譜設定 `n_channels`、`wavelength_min`、`wavelength_max`、`n_wavelength`。** 若真實資料沒有 TE / TM 或通道數不同，只要改這些設定，模型架構不必動（3D CNN 除外，它需要前 16 個通道是 Mueller 元素）。確定後把 `src/config.py` 的預設值改掉，避免每次都要下參數。
4. **整理 `labels.csv`**（格式見 README）。有 wafer 資訊一定要填 `wafer` 欄，否則交叉驗證會高估表現。
5. **跑交叉驗證**取得第一組真實數字：`python src/cross_validate.py --folds 5 --data_source folder ...`。
6. **向使用者回報**：樣本數、最大誤差、超過 0.3 nm 的比例、RMSE，以及是否通過。

## 之後的工作，依優先順序

1. **對照樣本數判斷結果是否合理。** 依合成資料的經驗，幾百筆以下很難達到 0.3 nm。若未通過，先確認是不是資料量的問題（畫 RMSE 對訓練樣本數：`experiments/learning_curve.py` 目前只支援合成資料，需要改成支援 `data_source="folder"`）。
2. **估計真實雜訊。** 需要同一點的重複量測。有了雜訊大小才能判斷目前的誤差離極限多遠。真實資料沒有正向模型，無法直接算 Cramér–Rao 下限；可用擬合修正輸出的誤差棒（`cd_sd`）當作近似。
3. **在真實資料上重跑方法比較**：`python src/compare_methods.py --folds 5 --data_source folder ...`，對 MLP、1D CNN、2D / 3D CNN、Channel Attention、微分特徵、擬合修正各做一次交叉驗證並排序。`experiments/ablation.py` 的評分方式綁定合成資料（需要產生新的評估光譜），不能用在真實資料上。
4. **若 TEM 樣本太少**，和使用者討論其他標註來源（機台既有的 OCD 結果、物理模擬資料）。這是使用者要決定的事，不要自行引入。使用者先前決定不以 RCWA 為必要條件。
5. **更新報告**：把真實資料的結果加進 `reports/OCD_ML_report.pptx`，維持 10 頁以內，並標明哪些數字來自真實資料。

## 程式地圖

| 檔案 | 作用 |
|---|---|
| `src/config.py` | 所有設定，一個 `Config` dataclass。每個欄位都可用 `--欄位名` 從指令列覆寫 |
| `src/train.py` | `run_training(cfg)`：載入資料、切分、訓練、評分、存檔。`fit` 與 `predict_cd` 供交叉驗證重用 |
| `src/cross_validate.py` | k 折交叉驗證，可依 wafer 分組 |
| `src/compare_methods.py` | 用同一組交叉驗證切分比較各種方法；要加入新方法就在 `METHODS` 加一行 |
| `src/real_data.py` | 讀取「光譜資料夾 + labels.csv」 |
| `src/spectrum_io.py` | 讀取單一光譜檔；**格式解析是猜測** |
| `src/dataset.py` | 切分（可分組）、正規化 |
| `src/refinement.py` | 擬合修正 |
| `src/inference.py` | 載入模型檔、單筆預測 |
| `src/models/` | `mlp.py`、`multichannel_cnn.py`（含 Channel Attention）、`cnn_nd.py`（2D / 3D）、`derivative.py`（微分特徵） |
| `src/data_generation.py` | 合成光譜產生器 |
| `app.py` | Streamlit 介面，三個頁籤 |
| `experiments/` | 研究實驗，各檔案開頭有說明與量測結果；結果寫到 `outputs/experiments/` |
| `reports/build_report.js` | 產生報告的腳本，僅供參考：需要 Node、pptxgenjs 與原機器上的輔助腳本，公司電腦上多半跑不起來，請直接編輯 pptx |

模型檔 `outputs/multichannel_cnn.pt` 內含權重、訓練時的 `Config`、架構參數、正規化參數、擬合修正的 decoder。檔名沿用舊名，實際內容是訓練時選的任何模型。

## 慣例與陷阱

- **合成資料有兩個亂數種子，不可混用。** `physics_seed` 決定「CD 與光譜的關係」，同一個模型的訓練與測試必須用同一個值；`seed` 只決定取樣與雜訊。混用會讓預測差到數十 nm。
- **擬合修正需要校正。** decoder 的微小誤差會讓擬合出的 CD 整體偏移約 0.03 nm；`Refinement.calibrate()` 用訓練資料修掉。若看到擬合修正「不建議使用」，先看是偏差還是雜訊（預測值減參考值的平均）。
- **新增 `Config` 欄位時預設值要維持原本的行為**，舊模型檔才載得回來（`inference.load_checkpoint` 只取模型檔中仍存在的欄位）。
- **評估方法時不要只看整體 RMSE。** 最大誤差、以及依「離訓練範圍邊緣多遠」分組的誤差，常常給出不同結論。
- **不要回到已知無效的做法**：對超過門檻的殘差加懲罰的 loss、依 p99 選模型、dropout、ReduceLROnPlateau + early stopping，都試過且讓訓練更不穩定（見 `src/train.py` 開頭）。
- matplotlib 圖表有中文時要設定中文字型（`experiments/_common.py` 的 `setup_matplotlib`），否則顯示成方框。
- 長時間運算請控制 CPU 用量；`experiments/ablation.py` 有 `--workers` 與 `--threads`。
- commit 訊息用英文祈使句，說明做了什麼與為什麼。改動功能後同步更新本文件與 README。
