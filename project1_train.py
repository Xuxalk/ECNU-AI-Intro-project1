"""读取预处理文件，训练并比较 SVM 或 MLP。"""

from __future__ import annotations

import argparse
import time
from itertools import product
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42


def parse_hidden_layers(value: str) -> tuple[int, ...]:
    """把 100 或 100,50 转成网络层结构。"""

    try:
        layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"隐藏层参数无效: {value}") from exc
    if not layers or any(layer <= 0 for layer in layers):
        raise argparse.ArgumentTypeError(f"隐藏层参数无效: {value}")
    return layers


def load_prepared_data() -> dict:
    """读取 prepare.py 生成的特征文件。"""

    path = ROOT / "prepared_data.joblib"
    if not path.exists():
        raise FileNotFoundError("找不到 prepared_data.joblib，请先运行 project1_prepare.py")
    return joblib.load(path)


def evaluate_model(
    name: str,
    model: object,
    X_train: object,
    y_train: object,
    X_valid: object,
    y_valid: object,
) -> tuple[dict, object]:
    """训练一个模型，并返回验证集结果。"""

    print(f"\n开始训练: {name}")
    start = time.perf_counter()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    elapsed = time.perf_counter() - start

    result = {
        "model": name,
        "accuracy": accuracy_score(y_valid, y_pred),
        "macro_f1": f1_score(y_valid, y_pred, average="macro"),
        "weighted_f1": f1_score(y_valid, y_pred, average="weighted"),
        "seconds": elapsed,
    }
    print(f"accuracy: {result['accuracy']:.4f}")
    print(f"macro-F1: {result['macro_f1']:.4f}")
    print(f"weighted-F1: {result['weighted_f1']:.4f}")
    print(classification_report(y_valid, y_pred, zero_division=0))
    return result, model


def build_svm(c: float, kernel: str, gamma: str) -> SVC:
    """创建 SVM。"""

    return SVC(
        kernel=kernel,
        C=c,
        gamma=gamma,
        random_state=RANDOM_STATE,
        cache_size=4096,
    )


def build_mlp(
    hidden_layers: tuple[int, ...],
    alpha: float,
    max_iter: int,
    batch_size: int,
    learning_rate_init: float,
    early_stopping: bool,
    activation: str = "relu",
    solver: str = "adam",
    learning_rate: str = "constant",
    momentum: float = 0.9,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-8,
    validation_fraction: float = 0.1,
    n_iter_no_change: int = 10,
    tol: float = 1e-4,
    shuffle: bool = True,
) -> MLPClassifier:
    """创建 MLP。"""

    return MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver=solver,
        alpha=alpha,
        batch_size=batch_size,
        learning_rate_init=learning_rate_init,
        learning_rate=learning_rate,
        momentum=momentum,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        max_iter=max_iter,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        tol=tol,
        shuffle=shuffle,
        random_state=RANDOM_STATE,
    )


def refit_and_predict(
    bundle: dict,
    best_config: dict,
    output_path: Path,
    model_output_path: Path | None = None,
) -> None:
    """用完整有标签数据重新拟合最佳模型，再预测测试集。"""

    all_text = bundle["train_cleaned"] + bundle["valid_cleaned"]
    all_y = list(bundle["y_train"]) + list(bundle["y_valid"])
    vectorizer = TfidfVectorizer(**bundle["vectorizer_params"])
    X_all = vectorizer.fit_transform(all_text)
    X_test = vectorizer.transform(bundle["test_cleaned"])

    if best_config["kind"] == "svm":
        model = build_svm(
            c=best_config["c"],
            kernel=best_config["kernel"],
            gamma=best_config["gamma"],
        )
    else:
        model = build_mlp(
            hidden_layers=tuple(best_config["hidden_layers"]),
            alpha=best_config["alpha"],
            max_iter=best_config["max_iter"],
            batch_size=best_config["batch_size"],
            learning_rate_init=best_config["learning_rate_init"],
            early_stopping=best_config["early_stopping"],
            activation=best_config.get("activation", "relu"),
            solver=best_config.get("solver", "adam"),
            learning_rate=best_config.get("learning_rate", "constant"),
            momentum=best_config.get("momentum", 0.9),
            beta_1=best_config.get("beta_1", 0.9),
            beta_2=best_config.get("beta_2", 0.999),
            epsilon=best_config.get("epsilon", 1e-8),
            validation_fraction=best_config.get("validation_fraction", 0.1),
            n_iter_no_change=best_config.get("n_iter_no_change", 10),
            tol=best_config.get("tol", 1e-4),
            shuffle=best_config.get("shuffle", True),
        )

    print("\n使用完整有标签数据重新训练最佳模型...")
    model.fit(X_all, all_y)
    predictions = model.predict(X_test)
    pd.DataFrame(predictions).to_csv(output_path, index=False, header=False)
    joblib.dump(
        {"model": model, "vectorizer": vectorizer, "config": best_config},
        model_output_path or ROOT / "best_model_full.joblib",
        compress=3,
    )
    print(f"预测文件已保存: {output_path.name}")


def build_parser() -> argparse.ArgumentParser:
    """定义命令行参数。"""

    parser = argparse.ArgumentParser(description="训练并比较 SVM/MLP 文本分类模型")
    parser.add_argument("--model", choices=("svm", "mlp", "both"), default="svm")
    parser.add_argument("--c", type=float, nargs="+", default=[1.0])
    parser.add_argument("--kernel", choices=("linear", "rbf"), default="linear")
    parser.add_argument("--gamma", default="scale")
    parser.add_argument("--hidden-layers", nargs="+", default=["100"])
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.001])
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate-init", type=float, default=0.001)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--results", default="training_results.csv")
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--prediction-output", default="predictions_tuned.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bundle = load_prepared_data()
    X_train = bundle["X_train"]
    X_valid = bundle["X_valid"]
    y_train = bundle["y_train"]
    y_valid = bundle["y_valid"]

    hidden_layers = [parse_hidden_layers(value) for value in args.hidden_layers]
    results: list[dict] = []
    trained_models: list[tuple[dict, object]] = []

    if args.model in ("svm", "both"):
        for c in args.c:
            config = {
                "kind": "svm",
                "c": c,
                "kernel": args.kernel,
                "gamma": args.gamma,
            }
            result, model = evaluate_model(
                f"svm_C{c:g}_{args.kernel}",
                build_svm(c, args.kernel, args.gamma),
                X_train,
                y_train,
                X_valid,
                y_valid,
            )
            result.update(config)
            results.append(result)
            trained_models.append((result, model))

    if args.model in ("mlp", "both"):
        for layers, alpha in product(hidden_layers, args.alpha):
            config = {
                "kind": "mlp",
                "hidden_layers": list(layers),
                "alpha": alpha,
                "max_iter": args.max_iter,
                "batch_size": args.batch_size,
                "learning_rate_init": args.learning_rate_init,
                "early_stopping": not args.no_early_stopping,
            }
            layer_name = "x".join(str(layer) for layer in layers)
            result, model = evaluate_model(
                f"mlp_{layer_name}_alpha{alpha:g}",
                build_mlp(
                    layers,
                    alpha,
                    args.max_iter,
                    args.batch_size,
                    args.learning_rate_init,
                    not args.no_early_stopping,
                ),
                X_train,
                y_train,
                X_valid,
                y_valid,
            )
            result.update(config)
            results.append(result)
            trained_models.append((result, model))

    if not results:
        raise ValueError("至少选择一个模型")

    result_df = pd.DataFrame(results).sort_values(
        ["macro_f1", "accuracy"], ascending=False
    )
    result_df.to_csv(ROOT / args.results, index=False)
    best_result, best_model = max(
        trained_models, key=lambda item: (item[0]["macro_f1"], item[0]["accuracy"])
    )
    joblib.dump(best_model, ROOT / "best_model_split.joblib", compress=3)
    print("\n验证集最佳结果:")
    print(result_df.iloc[0].to_string())
    print("模型已保存: best_model_split.joblib")

    if args.predict_test:
        refit_and_predict(
            bundle,
            best_result,
            ROOT / args.prediction_output,
        )


if __name__ == "__main__":
    main()
