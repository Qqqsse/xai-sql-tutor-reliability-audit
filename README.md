# xAI SQL Tutor Reliability Audit

> Reliability diagnosis for an AI-powered SQL hint tutor — simulating model predictions on student hint adoption, with confidence calibration, error slicing, subgroup metrics, high-confidence error analysis, and an interactive Streamlit dashboard.
> Built as part of an Explainable AI (xAI) graduate course project.

---

## Research Background

This project simulates a database course scenario where students practice SQL queries on a learning platform. When a query fails, an AI tutor provides one of two types of hints:

- **Conceptual** hints — guide the student toward understanding the underlying concept
- **Actionable** hints — provide specific debugging instructions

The core research question: *Which hint type more effectively prompts students to revise their code and succeed?*

The target variable is `adopted` (whether a student acted on the hint), predicted from behavioral features including `hint_latency`, `revision_count`, `db_experience`, and `trust_score`.


## Project Structure

```
xai-sql-tutor-reliability-audit/
├── data/
│   └── generate_data.py        # Simulated dataset generation
├── src/
│   ├── train_model.py          # Model training + full reliability analysis
│   └── reliability_analysis.py # Helper functions
├── dashboard/
│   └── app.py                  # Streamlit reliability dashboard
├── outputs/                    # Generated figures, CSVs, model artifacts
├── requirements.txt
└── README.md
```

## How to Run

### 1. Create and activate conda environment

```bash
conda create -n xai_week10 python=3.10 -y
conda activate xai_week10
pip install -r requirements.txt
```

### 2. Generate simulated dataset

```bash
python data/generate_data.py
```

### 3. Run model training and reliability analysis

```bash
python src/train_model.py
```

### 4. Launch the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

---

## Key Outputs

| File | Description |
|---|---|
| `outputs/simulated_dataset.csv` | Simulated student session data |
| `outputs/confidence_distribution.png` | Confidence score distribution (correct vs. error) |
| `outputs/reliability_diagram.png` | Calibration curve with ECE |
| `outputs/error_slicing.csv` | Error rates by subgroup and data condition |
| `outputs/subgroup_metrics.csv` | Precision / Recall / F1 by subgroup |
| `outputs/high_confidence_errors.csv` | Detailed analysis of high-confidence errors |
| `outputs/feature_importance.png` | Random Forest feature importance |

---

## Tech Stack

- **Language**: Python 3.10
- **ML**: scikit-learn (Random Forest)
- **Visualization**: Matplotlib · Seaborn · Plotly
- **Dashboard**: Streamlit
- **XAI Context**: Explainable AI graduate course, National Chung Hsing University

---

## Course Context

This project is the Week 10 assignment for the Explainable AI (xAI) graduate course, focusing on:
- Model uncertainty and confidence calibration
- Error slicing and subgroup failure analysis
- High-confidence error diagnosis
- Reliability dashboard design