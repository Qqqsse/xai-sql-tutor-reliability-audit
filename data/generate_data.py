import os
import numpy as np
import pandas as pd


def main() -> None:
    """建立 AI SQL 助教模擬資料集。"""
    np.random.seed(42)
    n_sessions = 200

    student_ids = [f"ST_{i:03d}" for i in range(1, 101)]
    session_ids = [f"S_{i:03d}" for i in range(1, n_sessions + 1)]

    df = pd.DataFrame(
        {
            "student_id": np.random.choice(student_ids, size=n_sessions, replace=True),
            "session_id": session_ids,
            "hint_condition": np.random.choice(
                ["actionable", "conceptual"], size=n_sessions, p=[0.5, 0.5]
            ),
            "db_experience": np.random.choice([0, 1], size=n_sessions, p=[0.6, 0.4]),
            "hint_latency": np.clip(
                np.random.exponential(scale=40, size=n_sessions), 5, 300
            ),
            # revision_count 偏低：較高機率集中在 1-3
            "revision_count": np.random.choice(
                [1, 2, 3, 4, 5, 6, 7, 8],
                size=n_sessions,
                p=[0.23, 0.22, 0.18, 0.13, 0.10, 0.07, 0.04, 0.03],
            ),
            # trust_score 偏高：4-5 出現較多
            "trust_score": np.random.choice(
                [1, 2, 3, 4, 5], size=n_sessions, p=[0.05, 0.10, 0.20, 0.33, 0.32]
            ),
        }
    )

    # 研究意涵：以規則化方式模擬「提示類型、經驗與行為時序」對採納機率的影響
    base_prob = np.full(shape=n_sessions, fill_value=0.5, dtype=float)
    base_prob += (df["hint_condition"] == "actionable").astype(float) * 0.25
    base_prob += (df["db_experience"] == 1).astype(float) * 0.15
    base_prob += (df["hint_latency"] < 30).astype(float) * 0.10
    base_prob += (df["revision_count"] <= 2).astype(float) * 0.05

    # 適量噪音：模擬真實互動中不可觀測因素（動機、疲勞、題目難度）
    noise = np.random.normal(loc=0.0, scale=0.15, size=n_sessions)
    adopted_prob = np.clip(base_prob + noise, 0.05, 0.95)
    df["adopted"] = np.random.binomial(1, adopted_prob, size=n_sessions)

    # 任務成功率設定：採納且低修訂數時成功率較高
    success_prob = np.where(
        (df["adopted"] == 1) & (df["revision_count"] <= 3),
        0.8,
        0.3,
    )
    df["task_success"] = np.random.binomial(1, success_prob, size=n_sessions)

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "simulated_dataset.csv")
    df.to_csv(output_path, index=False)

    print("=== 模擬資料集已建立 ===")
    print(f"檔案位置: {output_path}")
    print("\n=== 基本統計摘要 ===")
    print(df.describe(include="all").transpose())
    print("\n=== 採納率與成功率 ===")
    print(
        pd.Series(
            {
                "adopted_rate": round(df["adopted"].mean(), 4),
                "task_success_rate": round(df["task_success"].mean(), 4),
            }
        )
    )


if __name__ == "__main__":
    main()
