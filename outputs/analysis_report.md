# xAI Week 10 分析報告（AI SQL 語法助教可靠度）

## 1. 研究目的

本研究模擬 AI SQL 語法助教在教學平台中的提示採納情境，檢驗模型在「是否採納提示（adopted）」預測上的：
- 整體效能（Accuracy、F1、AUC）
- 信心校準能力（Brier Score、ECE、Reliability Diagram）
- 子群體與切片穩定性（Error Slicing、Subgroup Metrics）
- 高信心錯誤風險（High-Confidence Errors）

## 2. 資料與方法

- 資料筆數：`200` 筆 session（測試集 `40` 筆）
- 特徵：`hint_condition`、`db_experience`、`hint_latency`、`revision_count`、`trust_score`
- 模型：`RandomForestClassifier(n_estimators=100, random_state=42)`
- 切分：`train/test = 80/20`，並以 `adopted` 做 stratify

## 3. 整體指標

根據 `outputs/overall_metrics.json`：

- Accuracy：`0.8250`
- F1-macro：`0.6311`
- AUC-ROC：`0.6494`
- Brier Score：`0.1517`
- ECE (10 bins)：`0.1615`

### 解讀

- 準確率達 82.5%，代表在此模擬場景下模型有一定判別能力。
- AUC 約 0.65，顯示排序能力中等，仍有提升空間。
- ECE 約 0.16，表示模型信心與實際正確率存在可觀差距，需持續校準。

## 4. Error Slicing 重點

根據 `outputs/error_slicing.csv`：

- `hint_condition=actionable`：accuracy `0.85`，recall(class=1) `1.00`
- `hint_condition=conceptual`：accuracy `0.80`，recall(class=1) `0.875`
- `db_experience=0`：accuracy `0.778`，`db_experience=1`：accuracy `0.923`
- `confidence>=0.7`：accuracy `0.839`，`confidence<0.7`：accuracy `0.778`

### 解讀

- 有基礎學生（`db_experience=1`）切片表現較佳。
- `actionable` 提示在測試切片中的召回較高，支持「可執行提示較容易促成採納」的假設。
- 低信心區段錯誤率較高，可作為「拒答/人工覆核」候選區。

## 5. Subgroup Metrics（hint_condition × db_experience）

根據 `outputs/subgroup_metrics.csv`：

- 最佳組合：`actionable × db_experience=1`（accuracy `1.00`, recall `1.00`）
- 相對弱勢：`actionable × db_experience=0`（accuracy `0.75`）
- `conceptual × db_experience=0/1` 皆為中等穩定（accuracy `0.80`）

### 解讀

- 初學者在某些提示條件下仍可能出現較高誤判風險。
- 子群體熱圖可協助教學設計上進行差異化策略（例如對初學者增加具體引導）。

## 6. 高信心錯誤分析（High-Confidence Errors）

根據 `outputs/high_confidence_errors.csv`，共觀察到 3 筆代表性案例（confidence >= 0.75 且預測錯誤）：

1. `S_073`：False Positive，possible_source=`shortcut on db_experience`
2. `S_083`：False Positive，possible_source=`hint_condition insensitivity`
3. `S_157`：False Positive，possible_source=`shortcut on db_experience`

### 解讀

- 模型在高信心情境仍可能過度依賴部分特徵（如 `db_experience`）而產生過度樂觀預測。
- 建議加入更多行為脈絡特徵（例如 edit pattern、SQL 錯誤型態）降低捷徑學習風險。

## 7. 研究結論與建議

1. 模型具備基礎可用性，但 AUC 與 ECE 顯示可靠度仍有改善空間。  
2. 切片分析顯示不同提示條件與學生背景存在績效落差，需進行子群體監控。  
3. 高信心錯誤顯示模型尚有「過度自信」現象，建議導入 reject option 與後續校準策略。  

## 8. 交付物對照

- 圖表：`confidence_distribution.png`、`reliability_diagram.png`、`error_slicing_chart.png`、`subgroup_heatmap.png`、`feature_importance.png`
- 表格：`error_slicing.csv`、`subgroup_metrics.csv`、`high_confidence_errors.csv`
- 模型：`rf_model.joblib`
- 儀表板資料：`test_predictions.csv`、`overall_metrics.json`
