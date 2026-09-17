"""用验证集最优 MLP 配置进行一次全量重训，并预测无标签测试集。"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from project1_train import ROOT, load_prepared_data, refit_and_predict


# ==================== 可修改的文件名 ====================
RESULT_FILE = "mlp_training_results.csv"
PREDICTION_FILE = "mlp_predictions_final.csv"
MODEL_FILE = "mlp_best_model_full_final.joblib"


def parse_layers(value: object) -> list[int]:
    """把 CSV 中的 '[100]' 或 '[100, 50]' 转成整数列表。"""

    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    parsed = ast.literal_eval(str(value))
    if isinstance(parsed, int):
        return [parsed]
    return [int(item) for item in parsed]


def main() -> None:
    result_path = ROOT / RESULT_FILE
    if not result_path.exists():
        raise FileNotFoundError(
            f"找不到 {RESULT_FILE}。请先运行 train_mlp.py 完成验证集调参。"
        )

    result_df = pd.read_csv(result_path)
    if result_df.empty:
        raise ValueError(f"{RESULT_FILE} 没有实验结果。")
    if "kind" in result_df.columns:
        result_df = result_df[result_df["kind"].eq("mlp")]
    if result_df.empty:
        raise ValueError(f"{RESULT_FILE} 中没有 MLP 结果。")

    best = result_df.sort_values(
        ["macro_f1", "accuracy"], ascending=False
    ).iloc[0]
    best_config = {
        "kind": "mlp",
        "hidden_layers": parse_layers(best["hidden_layers"]),
        "alpha": float(best["alpha"]),
        "max_iter": int(best["max_iter"]),
        "batch_size": int(best["batch_size"]),
        "learning_rate_init": float(best["learning_rate_init"]),
        "early_stopping": str(best["early_stopping"]).lower() == "true",
    }

    print("将使用验证集最佳配置进行一次全量重训：")
    print(f"  hidden_layers = {best_config['hidden_layers']}")
    print(f"  alpha = {best_config['alpha']}")
    print(f"  learning_rate_init = {best_config['learning_rate_init']}")
    print(f"  验证集 macro-F1 = {float(best['macro_f1']):.4f}")
    print("训练数据：全部 7,368 条有标签文本；预测数据：2,457 条无标签测试文本。")

    bundle = load_prepared_data()
    refit_and_predict(
        bundle,
        best_config,
        ROOT / PREDICTION_FILE,
        ROOT / MODEL_FILE,
    )
    print(f"全量模型已保存：{MODEL_FILE}")
    print(f"最终预测已保存：{PREDICTION_FILE}")


if __name__ == "__main__":
    main()
