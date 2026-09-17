"""训练和调试 SVM。直接修改文件顶部的参数即可。"""

import joblib
import pandas as pd

from project1_train import (
    ROOT,
    RANDOM_STATE,
    build_svm,
    evaluate_model,
    load_prepared_data,
    refit_and_predict,
)


# ==================== SVM 参数区 ====================
# 按数量级从小到大搜索 C，想减少或增加组合时直接修改这里。
C_VALUES = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
KERNEL = "linear"  # 可选: "linear" 或 "rbf"
GAMMA = "scale"  # rbf 时可改为 "auto" 或具体数值

# 是否用完整有标签数据重新训练最佳模型并生成测试集预测。
MAKE_TEST_PREDICTIONS = False
PREDICTION_FILE = "svm_predictions.csv"
RESULT_FILE = "svm_training_results.csv"
SPLIT_MODEL_FILE = "svm_best_model_split.joblib"
FULL_MODEL_FILE = "svm_best_model_full.joblib"


def main() -> None:
    bundle = load_prepared_data()
    results = []
    trained_models = []

    for c in C_VALUES:
        config = {
            "kind": "svm",
            "c": c,
            "kernel": KERNEL,
            "gamma": GAMMA,
        }
        result, model = evaluate_model(
            f"svm_C{c:g}_{KERNEL}",
            build_svm(c, KERNEL, GAMMA),
            bundle["X_train"],
            bundle["y_train"],
            bundle["X_valid"],
            bundle["y_valid"],
        )
        result.update(config)
        results.append(result)
        trained_models.append((result, model))

    result_df = pd.DataFrame(results).sort_values(
        ["macro_f1", "accuracy"], ascending=False
    )
    result_df.to_csv(ROOT / RESULT_FILE, index=False)

    best_result, best_model = max(
        trained_models,
        key=lambda item: (item[0]["macro_f1"], item[0]["accuracy"]),
    )
    joblib.dump(best_model, ROOT / SPLIT_MODEL_FILE, compress=3)
    print("\nSVM 验证集结果:")
    print(result_df.to_string(index=False))
    print(f"最佳模型已保存: {SPLIT_MODEL_FILE}")

    if MAKE_TEST_PREDICTIONS:
        refit_and_predict(
            bundle,
            best_result,
            ROOT / PREDICTION_FILE,
            ROOT / FULL_MODEL_FILE,
        )


if __name__ == "__main__":
    main()
