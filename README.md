# OCD CNN test：Mueller Matrix 光譜反推 CD

用量測到的光譜（16 個 Mueller 元素加上 TE、TM，共 18 個通道）訓練模型，預測 CD（nm），並以 TEM 等參考量測值驗證。

- 驗收標準：**每一筆樣本的誤差都在 0.3 nm 以內**為主，RMSE 為輔。
- 目前所有結果都來自合成光譜；真實光譜與 TEM 尚未驗證過。現況與待辦見 [AGENTS.md](AGENTS.md)，研究結果見 `reports/OCD_ML_report.pptx`。

## 資料安全

- 光譜與 TEM 資料只放在本機，**不得上傳到外部網路**。
- 程式執行時不連任何網路：沒有遙測、沒有外部 API。Streamlit 的使用統計已關閉，且只監聽 `localhost`（`.streamlit/config.toml`）。
- `data/`、`real_data/`、`raw_data/`、`outputs/` 都被 `.gitignore` 排除。真實資料請放在這些資料夾，或放在專案資料夾之外。訓練好的模型（`outputs/`）含有訓練資料的資訊，同樣不進版控。

## 安裝

需要 Python 3.13。

```
pip install -r requirements.txt
```

公司電腦不能連外網時，二選一：

- 內部套件鏡像：`pip install -r requirements.txt --index-url <內部鏡像網址>`
- 離線安裝：在能連外網的電腦（相同作業系統與 Python 版本）執行
  `pip download -r requirements.txt -d wheelhouse`，把 `wheelhouse` 資料夾帶到公司電腦後執行
  `pip install --no-index --find-links wheelhouse -r requirements.txt`

沒有 GPU 也能用；結果都是用 CPU 版的 PyTorch 產生的。

## 確認安裝正確

以下三個指令都只用合成資料，幾分鐘內完成：

```
python src/train.py --n_samples 2000 --epochs 30 --output_dir outputs/_check
python src/make_example_dataset.py
python src/cross_validate.py --folds 5 --epochs 30 --data_source folder --spectra_dir data/example/spectra --labels_csv data/example/labels.csv --output_dir outputs/_check
```

第一個指令的最後會印出 `Network: PASS/FAIL ...` 一行；第三個會在 `outputs/_check/cross_validation/` 產生 `predictions.csv`、`summary.json`、`report.png`。這裡的樣本很少，印出 FAIL 是正常的，只要能跑完即可。

## 使用真實資料

### 1. 擺放資料

```
data/my_lot/
  spectra/            每個樣本一個光譜檔（.csv / .dat / .npy / .zip）
  labels.csv          參考 CD 對照表
```

`labels.csv` 的欄位（欄名不分大小寫）：

| 欄位 | 必要 | 說明 |
|---|---|---|
| `filename` | 是 | `spectra/` 內的檔名 |
| `CD_nm` | 是 | 參考 CD（nm），例如 TEM 量測值 |
| `wafer` | 否 | 樣本來自哪一片 wafer；有這一欄時，同一片的樣本在切分時會放在一起 |
| 其他數值欄位 | 否 | 其他參考結構參數（高度、側壁角等），會用於擬合修正 |

`python src/make_example_dataset.py` 會在 `data/example/` 產生一份同樣格式的合成範例，可直接對照。

光譜檔的預設格式是一個表格：第一欄波長，之後依序是 M00、M01 … M33、TE、TM。**`.dat` 與 `.zip` 的讀取規則是在沒有真實檔案的情況下猜的**，拿到真實檔案後請先確認能正確讀取（見 `src/spectrum_io.py` 開頭的說明）。

### 2. 設定通道數與波長範圍

預設是 18 個通道、200–1000 nm、101 個波長點（`src/config.py`）。真實光譜不同時，用指令列參數或 app 上的欄位調整；設定的波長範圍必須落在量測範圍之內。

### 3. 估計表現：交叉驗證

樣本只有數十到數百筆時，用交叉驗證而不是單次切分：

```
python src/cross_validate.py --folds 5 --data_source folder --spectra_dir data/my_lot/spectra --labels_csv data/my_lot/labels.csv
```

每個樣本都會被留出、由沒看過它的模型預測一次。結果在 `outputs/cross_validation/`，`summary.json` 的 `reported` 區塊就是驗收數字（`max_abs_err_nm`、`fraction_over_threshold`、`meets_threshold`、`rmse`）。

### 4. 比較不同方法

```
python src/compare_methods.py --folds 5 --data_source folder --spectra_dir data/my_lot/spectra --labels_csv data/my_lot/labels.csv
```

對每一種方法（MLP、1D CNN、2D / 3D CNN、Channel Attention、微分特徵、擬合修正）各做一次交叉驗證，用的是同一組切分。結果依驗收標準排序，存到 `outputs/compare/comparison.csv`。全部跑完需要一段時間，可用 `--methods mlp mlp_refine cnn` 只跑其中幾種。

### 5. 訓練要使用的模型

```
python src/train.py --test_ratio 0 --data_source folder --spectra_dir data/my_lot/spectra --labels_csv data/my_lot/labels.csv
```

模型存到 `outputs/multichannel_cnn.pt`。樣本有數千筆時可以不加 `--test_ratio 0`，會保留 15% 當測試資料並直接印出驗收數字。

### 6. 操作介面

```
streamlit run app.py
```

- 光譜檢視與預測：上傳單一光譜，顯示各通道曲線與預測 CD（含誤差棒）。
- 批次驗證：一次上傳多個光譜與參考 CD 對照表，逐筆比較。
- 模型訓練 / 超參數：上述第 3、5 步也可以在這裡完成（選「真實資料」，填入兩個路徑，設定交叉驗證折數）。

## 把模型搬到另一台電腦

模型不進版控。兩種做法：在新電腦上用同一份資料重新訓練，或把 `outputs/multichannel_cnn.pt` 這一個檔案複製到新電腦專案的 `outputs/` 資料夾。模型檔內含訓練時的設定與正規化參數，不需要其他檔案。

在不同電腦上重新訓練，結果會有微小差異（浮點運算順序不同），不會逐位元相同。

## 專案結構

```
app.py                    Streamlit 操作介面
src/
  config.py               所有設定（Config）
  train.py                訓練一個模型
  cross_validate.py       k 折交叉驗證
  compare_methods.py      用交叉驗證比較各種方法
  real_data.py            讀取真實資料（光譜資料夾 + 參考 CD 對照表）
  spectrum_io.py          讀取單一光譜檔
  refinement.py           擬合修正：學出的正向模型 + Gauss-Newton
  inference.py            載入模型、預測
  models/                 MLP、多通道 1D CNN、2D / 3D CNN、微分特徵
  data_generation.py      合成光譜產生器
  make_example_dataset.py 產生真實資料格式的合成範例
experiments/              研究用的比較實驗（理論下限、消融、穩健性等）
reports/                  報告簡報
outputs/                  模型與結果（不進版控）
```
