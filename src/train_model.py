import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from reliability_analysis import compute_ece

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


def ensure_output_dir() -> str:
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def add_error_source_reason(row: pd.Series) -> str:
    """依規則推估高信心錯誤可能來源。"""
    if row["db_experience"] == 0 and row["confidence"] >= 0.75:
        return "shortcut on db_experience"
    if row["hint_latency"] <= 10:
        return "latency timing artifact"
    if row["hint_condition"] == "conceptual" and row["model_output"] == 1:
        return "hint_condition insensitivity"
    return "mixed factors"


def build_observable_log(row: pd.Series) -> str:
    """建立可觀察 log 敘述，模擬真實教學平台事件紀錄。"""
    return (
        f"學生在收到提示後 {int(round(row['hint_latency']))} 秒送出 edit_code，"
        f"revision_count={int(row['revision_count'])}，最終 task_success={int(row['task_success'])}，"
        "可能存在快速嘗試或策略性修改行為。"
    )


def main() -> None:
    output_dir = ensure_output_dir()
    data_path = os.path.join(output_dir, "simulated_dataset.csv")
    df = pd.read_csv(data_path)

    # 研究意涵：將提示型態轉成可學習訊號，檢驗「可執行提示」是否提高採納
    df["hint_condition_encoded"] = df["hint_condition"].map(
        {"actionable": 1, "conceptual": 0}
    )

    feature_cols = [
        "hint_condition_encoded",
        "db_experience",
        "hint_latency",
        "revision_count",
        "trust_score",
    ]
    X = df[feature_cols]
    y = df["adopted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_conf = np.where(y_pred == 1, y_prob, 1 - y_prob)
    correct_mask = y_pred == y_test.values

    # 4-3 整體 performance
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    ece = compute_ece(y_true=y_test.values, y_prob=y_prob, n_bins=10)
    cls_report = classification_report(y_test, y_pred, digits=4)

    print("=== Overall Performance ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score (macro): {f1_macro:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"ECE (10 bins): {ece:.4f}")
    print("\n=== Classification Report ===")
    print(cls_report)

    # dashboard 重用
    with open(os.path.join(output_dir, "overall_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": acc,
                "f1_macro": f1_macro,
                "auc": auc,
                "brier": brier,
                "ece": ece,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 建立 test analysis dataframe
    test_df = X_test.copy()
    test_df["session_id"] = df.loc[X_test.index, "session_id"]
    test_df["hint_condition"] = df.loc[X_test.index, "hint_condition"]
    test_df["task_success"] = df.loc[X_test.index, "task_success"]
    test_df["actual"] = y_test.values
    test_df["predicted"] = y_pred
    test_df["prob_adopted"] = y_prob
    test_df["confidence"] = y_conf
    test_df["is_correct"] = correct_mask.astype(int)
    test_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

    # 4-4 信心分佈圖
    plt.figure(figsize=(9, 5))
    plt.hist(y_prob[correct_mask], bins=12, alpha=0.7, color="green", label="Correct")
    plt.hist(y_prob[~correct_mask], bins=12, alpha=0.7, color="red", label="Incorrect")
    plt.title("Confidence Distribution (Correct vs Incorrect)")
    plt.xlabel("P(adopted)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_distribution.png"), dpi=200)
    plt.close()

    # 4-5 reliability diagram
    frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")
    plt.figure(figsize=(6, 6))
    plt.plot(mean_pred, frac_pos, marker="o", label="Model Calibration")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.title(f"Reliability Diagram (ECE={ece:.3f})")
    plt.xlabel("平均預測機率")
    plt.ylabel("Fraction of Positives")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reliability_diagram.png"), dpi=200)
    plt.close()

    # 4-6 Error slicing
    slicing_df = test_df.copy()
    slicing_df["confidence_group"] = np.where(
        slicing_df["confidence"] >= 0.7, "confidence>=0.7", "confidence<0.7"
    )
    slices = [
        ("hint_condition", "actionable"),
        ("hint_condition", "conceptual"),
        ("db_experience", 0),
        ("db_experience", 1),
        ("confidence_group", "confidence>=0.7"),
        ("confidence_group", "confidence<0.7"),
    ]
    slicing_rows = []
    for col, value in slices:
        part = slicing_df[slicing_df[col] == value]
        if part.empty:
            continue
        slicing_rows.append(
            {
                "slice_type": col,
                "slice_value": value,
                "n_samples": len(part),
                "accuracy": accuracy_score(part["actual"], part["predicted"]),
                "recall_class1": recall_score(
                    part["actual"], part["predicted"], pos_label=1, zero_division=0
                ),
            }
        )
    error_slicing = pd.DataFrame(slicing_rows).sort_values(["slice_type", "slice_value"])
    error_slicing.to_csv(os.path.join(output_dir, "error_slicing.csv"), index=False)

    plt.figure(figsize=(9, 5))
    chart_data = error_slicing.copy()
    chart_data["label"] = chart_data["slice_type"] + "=" + chart_data["slice_value"].astype(str)
    plt.barh(chart_data["label"], chart_data["accuracy"], alpha=0.8, label="Accuracy")
    plt.xlabel("Accuracy")
    plt.title("Error Slicing Accuracy Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_slicing_chart.png"), dpi=200)
    plt.close()

    # 4-7 subgroup metrics
    subgroup_rows = []
    for hint in ["actionable", "conceptual"]:
        for exp in [0, 1]:
            subgroup = test_df[
                (test_df["hint_condition"] == hint) & (test_df["db_experience"] == exp)
            ]
            if subgroup.empty:
                continue
            p, r, f1, _ = precision_recall_fscore_support(
                subgroup["actual"], subgroup["predicted"], average="binary", zero_division=0
            )
            subgroup_rows.append(
                {
                    "hint_condition": hint,
                    "db_experience": exp,
                    "n_samples": len(subgroup),
                    "precision": p,
                    "recall": r,
                    "f1": f1,
                    "accuracy": accuracy_score(subgroup["actual"], subgroup["predicted"]),
                }
            )
    subgroup_metrics = pd.DataFrame(subgroup_rows)
    subgroup_metrics.to_csv(os.path.join(output_dir, "subgroup_metrics.csv"), index=False)

    heatmap_data = subgroup_metrics.pivot(
        index="hint_condition", columns="db_experience", values="recall"
    )
    plt.figure(figsize=(6, 4))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
    plt.title("Subgroup Recall Heatmap")
    plt.xlabel("db_experience")
    plt.ylabel("hint_condition")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "subgroup_heatmap.png"), dpi=200)
    plt.close()

    # 4-8 high-confidence errors
    high_conf_errors = test_df[
        (test_df["confidence"] >= 0.75) & (test_df["predicted"] != test_df["actual"])
    ].copy()
    high_conf_errors["model_output"] = high_conf_errors["predicted"]
    high_conf_errors["correct_answer"] = high_conf_errors["actual"]
    high_conf_errors["error_type"] = np.where(
        (high_conf_errors["model_output"] == 1) & (high_conf_errors["correct_answer"] == 0),
        "False Positive",
        "False Negative",
    )
    high_conf_errors["possible_source"] = high_conf_errors.apply(
        add_error_source_reason, axis=1
    )
    high_conf_errors["observable_log"] = high_conf_errors.apply(build_observable_log, axis=1)

    required_cols = [
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
    high_conf_errors_report = high_conf_errors[required_cols].copy()
    if len(high_conf_errors_report) < 3:
        # 若樣本不足，降門檻補足至少 3 筆供教學分析使用
        fallback = test_df[
            (test_df["confidence"] >= 0.70) & (test_df["predicted"] != test_df["actual"])
        ].copy()
        fallback["model_output"] = fallback["predicted"]
        fallback["correct_answer"] = fallback["actual"]
        fallback["error_type"] = np.where(
            (fallback["model_output"] == 1) & (fallback["correct_answer"] == 0),
            "False Positive",
            "False Negative",
        )
        fallback["possible_source"] = fallback.apply(add_error_source_reason, axis=1)
        fallback["observable_log"] = fallback.apply(build_observable_log, axis=1)
        high_conf_errors_report = fallback[required_cols].head(3)

    high_conf_errors_report.to_csv(
        os.path.join(output_dir, "high_confidence_errors.csv"), index=False
    )

    print("\n=== High-Confidence Errors 分析 ===")
    if high_conf_errors_report.empty:
        print("無高信心錯誤樣本。")
    else:
        print(high_conf_errors_report.to_string(index=False))

    # 4-9 feature importance
    feature_importance = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=feature_importance, x="importance", y="feature", orient="h")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=200)
    plt.close()

    # 4-10 儲存模型
    joblib.dump(model, os.path.join(output_dir, "rf_model.joblib"))
    print("\n模型已儲存至 outputs/rf_model.joblib")


if __name__ == "__main__":
    main()
