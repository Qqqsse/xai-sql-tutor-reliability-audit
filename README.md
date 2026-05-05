# xAI Week 10：AI SQL 助教可靠度分析

本專案為 xAI 課程 Week 10 作業，模擬「AI SQL 語法助教」情境，重點在分析模型預測是否可靠、信心是否可校準、以及不同子群體是否有系統性錯誤。

## 研究背景（AI SQL 語法助教）

在資料庫課程中，學生提交 SQL 後若出錯，系統會提供：
- `actionable`：可立即執行的修正提示
- `conceptual`：偏概念引導的提示

研究問題為：不同提示型態、學生經驗與互動行為，如何影響學生是否採納提示（`adopted`）與模型判斷的可靠度。

## 專案結構

```text
xai-sql-tutor-reliability-audit/
├── data/
│   └── generate_data.py
├── src/
│   ├── train_model.py
│   └── reliability_analysis.py
├── dashboard/
│   └── app.py
├── outputs/
├── requirements.txt
└── README.md
```

## 資料說明

`data/generate_data.py` 會產生 200 筆 session 模擬資料，主要欄位如下：
- `student_id`：學生編號
- `session_id`：互動 session 編號
- `hint_condition`：提示型態（actionable / conceptual）
- `db_experience`：資料庫經驗（0=初學、1=有基礎）
- `hint_latency`：收到提示到回應的秒數
- `revision_count`：修訂次數
- `trust_score`：對助教信任分數（1~5）
- `adopted`：是否採納提示（目標變項）
- `task_success`：任務是否成功

## 如何執行

```bash
conda create -n xai_week10 python=3.10 -y
conda activate xai_week10
pip install -r requirements.txt
python data/generate_data.py
python src/train_model.py
streamlit run dashboard/app.py
```

### 一鍵執行（Windows）

```bash
run_all.bat
```

### Makefile（選用）

```bash
make data
make train
make dashboard
make all
```

## 輸出檔案說明

- `outputs/simulated_dataset.csv`：模擬資料集
- `outputs/confidence_distribution.png`：信心分佈（正確 vs 錯誤）
- `outputs/reliability_diagram.png`：校準曲線（Reliability Diagram）
- `outputs/error_slicing.csv`：Error slicing 指標
- `outputs/error_slicing_chart.png`：切片結果圖
- `outputs/subgroup_metrics.csv`：子群體 precision/recall/f1/accuracy
- `outputs/subgroup_heatmap.png`：子群體 recall 熱圖
- `outputs/high_confidence_errors.csv`：高信心錯誤個案分析
- `outputs/feature_importance.png`：特徵重要性圖
- `outputs/rf_model.joblib`：訓練完成模型
- `outputs/test_predictions.csv`：測試集逐筆預測結果（供 dashboard 篩選）
- `outputs/analysis_report.md`：可直接提交的分析報告（繁中）