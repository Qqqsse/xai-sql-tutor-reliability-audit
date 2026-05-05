import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    if total == 0:
        return 0.0
    for i in range(n_bins):
        left, right = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= left) & (y_prob <= right) if i == n_bins - 1 else (y_prob >= left) & (y_prob < right)
        if np.any(mask):
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += np.abs(acc - conf) * (mask.sum() / total)
    return float(ece)


@st.cache_data
def load_data():
    pred_df = pd.read_csv("outputs/test_predictions.csv")
    subgroup_df = pd.read_csv("outputs/subgroup_metrics.csv")
    high_err_df = pd.read_csv("outputs/high_confidence_errors.csv")
    metrics_path = "outputs/overall_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            overall_metrics = json.load(f)
    else:
        y_true = pred_df["actual"].values
        y_pred = pred_df["predicted"].values
        y_prob = pred_df["prob_adopted"].values
        overall_metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "auc": roc_auc_score(y_true, y_prob),
            "ece": compute_ece(y_true, y_prob),
            "brier": brier_score_loss(y_true, y_prob),
        }
    return pred_df, subgroup_df, high_err_df, overall_metrics


def apply_filters(df: pd.DataFrame, subgroup_choice: str, exp_choice: str) -> pd.DataFrame:
    filtered = df.copy()
    if subgroup_choice != "All":
        filtered = filtered[filtered["hint_condition"] == subgroup_choice]
    if exp_choice != "All":
        filtered = filtered[filtered["db_experience"] == int(exp_choice)]
    return filtered


st.set_page_config(page_title="SQL Tutor Reliability Dashboard", layout="wide")
st.title("AI SQL 助教 Reliability Dashboard")

pred_df, subgroup_df, high_err_df, overall_metrics = load_data()

st.sidebar.header("篩選條件")
subgroup_choice = st.sidebar.selectbox(
    "選擇 subgroup (hint_condition)", ["All", "actionable", "conceptual"]
)
exp_choice = st.sidebar.selectbox("選擇 db_experience", ["All", "0", "1"])

filtered_df = apply_filters(pred_df, subgroup_choice, exp_choice)
filtered_high_err = apply_filters(high_err_df, subgroup_choice, exp_choice)

st.subheader("區塊 1：Overall Performance")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy", f"{overall_metrics['accuracy']:.3f}")
col2.metric("F1 (macro)", f"{overall_metrics['f1_macro']:.3f}")
col3.metric("AUC", f"{overall_metrics['auc']:.3f}")
col4.metric("ECE", f"{overall_metrics['ece']:.3f}")
col5.metric("Brier Score", f"{overall_metrics['brier']:.3f}")

st.subheader("區塊 2：Confidence Distribution")
if filtered_df.empty:
    st.warning("目前篩選條件無資料。")
else:
    dist_df = filtered_df.copy()
    dist_df["預測結果"] = np.where(dist_df["is_correct"] == 1, "正確", "錯誤")
    fig_hist = px.histogram(
        dist_df,
        x="prob_adopted",
        color="預測結果",
        barmode="overlay",
        nbins=12,
        labels={"prob_adopted": "P(adopted)"},
        color_discrete_map={"正確": "green", "錯誤": "red"},
        title="信心分佈圖（依篩選條件）",
    )
    st.plotly_chart(fig_hist, width="stretch")

st.subheader("區塊 3：Reliability Diagram")
if len(filtered_df) > 1 and filtered_df["actual"].nunique() > 1:
    frac_pos, mean_pred = calibration_curve(
        filtered_df["actual"], filtered_df["prob_adopted"], n_bins=10, strategy="uniform"
    )
    ece_filtered = compute_ece(
        filtered_df["actual"].values, filtered_df["prob_adopted"].values, n_bins=10
    )
    fig_cal = go.Figure()
    fig_cal.add_trace(
        go.Scatter(
            x=mean_pred,
            y=frac_pos,
            mode="lines+markers",
            name="校準曲線",
        )
    )
    fig_cal.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="完美校準",
            line=dict(dash="dash"),
        )
    )
    fig_cal.update_layout(
        title=f"Reliability Diagram（ECE={ece_filtered:.3f}）",
        xaxis_title="平均預測機率",
        yaxis_title="實際正類比例",
    )
    st.plotly_chart(fig_cal, width="stretch")
else:
    st.info("目前篩選資料不足以計算校準曲線。")

st.subheader("區塊 4：Subgroup Metrics")
if subgroup_df.empty:
    st.info("尚無 subgroup metrics。")
else:
    heatmap_df = subgroup_df.pivot(
        index="hint_condition", columns="db_experience", values="recall"
    )
    fig_heatmap = px.imshow(
        heatmap_df,
        text_auto=True,
        color_continuous_scale="YlGnBu",
        zmin=0,
        zmax=1,
        labels=dict(color="Recall"),
        title="Subgroup Recall Heatmap",
    )
    st.plotly_chart(fig_heatmap, width="stretch")

st.subheader("區塊 5：High-Confidence Errors")
show_cols = [
    "session_id",
    "hint_condition",
    "db_experience",
    "hint_latency",
    "revision_count",
    "trust_score",
    "model_output",
    "confidence",
    "correct_answer",
    "error_type",
    "possible_source",
    "observable_log",
]
if filtered_high_err.empty:
    st.info("目前篩選條件下沒有高信心錯誤樣本。")
else:
    st.dataframe(filtered_high_err[show_cols], width="stretch")

st.subheader("區塊 6：Reject Option 模擬")
if filtered_df.empty:
    st.info("目前篩選條件下無法進行 Reject Option 模擬。")
else:
    threshold = st.slider("設定 confidence threshold", 0.5, 0.95, 0.75, 0.01)
    accepted = filtered_df[filtered_df["confidence"] >= threshold]
    coverage = len(accepted) / len(filtered_df) if len(filtered_df) > 0 else 0
    risk = (
        1 - accuracy_score(accepted["actual"], accepted["predicted"])
        if len(accepted) > 0
        else 0
    )

    left, right = st.columns(2)
    left.metric("Coverage（覆蓋率）", f"{coverage:.3f}")
    right.metric("Risk（拒答後錯誤率）", f"{risk:.3f}")
