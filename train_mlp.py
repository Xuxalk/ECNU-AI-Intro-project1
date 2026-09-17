"""训练和调试 MLP。直接修改文件顶部的参数即可。"""

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from project1_train import (
    ROOT,
    build_mlp,
    evaluate_model,
    load_prepared_data,
    refit_and_predict,
)


# ==================== MLP 参数区 ====================
HIDDEN_LAYER_CONFIGS = [(50,), (100,), (100, 50)]
ALPHA_VALUES = [0.0001, 0.001, 0.01]
LEARNING_RATE_INIT_VALUES = [0.001]
MAX_ITER = 300
BATCH_SIZE = 64
EARLY_STOPPING = True
# 激活函数固定为 relu，优化器固定为 adam。

# 是否用完整有标签数据重新训练最佳模型并生成测试集预测。
MAKE_TEST_PREDICTIONS = True
PREDICTION_FILE = "mlp_predictions.csv"
RESULT_FILE = "mlp_training_results.csv"
SPLIT_MODEL_FILE = "mlp_best_model_split.joblib"
FULL_MODEL_FILE = "mlp_best_model_full.joblib"
LOSS_PLOT_FILE = "mlp_loss_curves.png"
LOSS_DATA_FILE = "mlp_loss_curves.csv"


def main() -> None:
    bundle = load_prepared_data()
    results = []
    trained_models = []
    loss_rows = []
    loss_curves = []

    for hidden_layers in HIDDEN_LAYER_CONFIGS:
        for alpha in ALPHA_VALUES:
            for learning_rate_init in LEARNING_RATE_INIT_VALUES:
                config = {
                    "kind": "mlp",
                    "hidden_layers": list(hidden_layers),
                    "alpha": alpha,
                    "max_iter": MAX_ITER,
                    "batch_size": BATCH_SIZE,
                    "learning_rate_init": learning_rate_init,
                    "early_stopping": EARLY_STOPPING,
                }
                layer_name = "x".join(str(layer) for layer in hidden_layers)
                result, model = evaluate_model(
                    f"mlp_{layer_name}_alpha{alpha:g}_lr{learning_rate_init:g}",
                    build_mlp(
                        hidden_layers,
                        alpha,
                        MAX_ITER,
                        BATCH_SIZE,
                        learning_rate_init,
                        EARLY_STOPPING,
                    ),
                    bundle["X_train"],
                    bundle["y_train"],
                    bundle["X_valid"],
                    bundle["y_valid"],
                )
                result.update(config)
                results.append(result)
                trained_models.append((result, model))

                # 保存每次迭代的训练 loss，便于绘图和写实验报告。
                loss_curve = getattr(model, "loss_curve_", None)
                if loss_curve:
                    loss_curves.append((result["model"], loss_curve))
                    loss_rows.extend(
                        {
                            "model": result["model"],
                            "iteration": iteration,
                            "loss": loss,
                        }
                        for iteration, loss in enumerate(loss_curve, start=1)
                    )

    result_df = pd.DataFrame(results).sort_values(
        ["macro_f1", "accuracy"], ascending=False
    )
    result_df.to_csv(ROOT / RESULT_FILE, index=False)
    pd.DataFrame(loss_rows).to_csv(ROOT / LOSS_DATA_FILE, index=False)

    if loss_curves:
        fig, ax = plt.subplots(figsize=(9, 5))
        for name, loss_curve in loss_curves:
            ax.plot(range(1, len(loss_curve) + 1), loss_curve, label=name)
        ax.set_title("MLP training loss")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(ROOT / LOSS_PLOT_FILE, dpi=180)
        plt.close(fig)
        print(f"MLP loss 图已保存: {LOSS_PLOT_FILE}")

    best_result, best_model = max(
        trained_models,
        key=lambda item: (item[0]["macro_f1"], item[0]["accuracy"]),
    )
    joblib.dump(best_model, ROOT / SPLIT_MODEL_FILE, compress=3)
    print("\nMLP 验证集结果:")
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
