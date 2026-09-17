"""准备实验数据和模型参数，暂不开始分类器训练。"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

# 保留 Subject，删除技术性邮件头、引用、分隔线、URL 和邮箱地址。
HEADER_LINE_RE = re.compile(
    r"^(?:from|organization|lines|reply-to|nntp-posting-host|distribution|"
    r"xref|article|references|keywords|summary|followup-to|path|message-id|"
    r"sender|date|approved|control|content-type|mime-version|user-agent|"
    r"in-reply-to):.*$",
    flags=re.IGNORECASE,
)
SUBJECT_RE = re.compile(r"^subject:\s*(.*)$", flags=re.IGNORECASE)
QUOTE_RE = re.compile(r"^\s*>+")
ATTRIBUTION_RE = re.compile(r"^\s*in article .* wrote:\s*$", flags=re.IGNORECASE)
SEPARATOR_RE = re.compile(r"^\s*(?:[=-]{3,}|\.{3,})\s*$")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", flags=re.IGNORECASE)
def clean_text(value: object) -> str:
    """清洗文本，保留 Subject，避免产生空样本。"""

    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    header_part, separator, body_part = text.partition("\n\n")

    subjects: list[str] = []
    for line in header_part.split("\n"):
        match = SUBJECT_RE.match(line.strip())
        if match:
            subjects.append(match.group(1))

    body_lines: list[str] = []
    for line in (body_part if separator else text).split("\n"):
        line = line.strip()
        if not line:
            continue
        if SUBJECT_RE.match(line) or HEADER_LINE_RE.match(line):
            continue
        if QUOTE_RE.match(line) or ATTRIBUTION_RE.match(line):
            continue
        if SEPARATOR_RE.match(line):
            continue
        body_lines.append(line)

    cleaned = " ".join(subjects + body_lines)
    cleaned = URL_RE.sub(" ", cleaned)
    cleaned = EMAIL_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned or "empty_message"


def load_and_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取数据并进行可复现的分层划分。"""

    train_path = ROOT / "train_data.csv"
    test_path = ROOT / "test_data_unlabeled.csv"
    train_df = pd.read_csv(train_path, usecols=["text", "target"])
    test_df = pd.read_csv(test_path, usecols=["text"])

    if train_df[["text", "target"]].isna().any().any():
        raise ValueError("train_data.csv contains missing text or target values")
    if test_df["text"].isna().any():
        raise ValueError("test_data_unlabeled.csv contains missing text values")
    if train_df["target"].nunique() != 10:
        raise ValueError("Expected exactly 10 target classes")
    if train_df["text"].duplicated().any():
        raise ValueError(
            "Exact duplicate texts were found. Resolve duplicate groups before splitting "
            "to avoid train/validation leakage."
        )

    # 先划分原始数据，避免验证集和测试集参与规则学习。
    row_ids = pd.Series(range(len(train_df)), name="source_index")
    train_ids, valid_ids = train_test_split(
        row_ids,
        test_size=VALIDATION_SIZE,
        stratify=train_df["target"],
        random_state=RANDOM_STATE,
    )

    train_split = train_df.iloc[train_ids.to_numpy()].copy()
    valid_split = train_df.iloc[valid_ids.to_numpy()].copy()
    train_split.insert(0, "source_index", train_ids.to_numpy())
    valid_split.insert(0, "source_index", valid_ids.to_numpy())

    train_split.to_csv(ROOT / "train_split.csv", index=False)
    valid_split.to_csv(ROOT / "validation_split.csv", index=False)

    split_manifest = {
        "random_state": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
        "train_rows": int(len(train_split)),
        "validation_rows": int(len(valid_split)),
        "train_class_counts": {
            str(k): int(v) for k, v in train_split["target"].value_counts().sort_index().items()
        },
        "validation_class_counts": {
            str(k): int(v)
            for k, v in valid_split["target"].value_counts().sort_index().items()
        },
        "duplicate_texts_in_supplied_train": int(train_df["text"].duplicated().sum()),
    }
    (ROOT / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return train_split, valid_split, test_df


def prepare_features(
    train_split: pd.DataFrame,
    valid_split: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[TfidfVectorizer, object, object, object]:
    """清洗数据，并且只用训练集拟合 TF-IDF。"""

    train_text = train_split["text"].map(clean_text)
    valid_text = valid_split["text"].map(clean_text)
    test_text = test_df["text"].map(clean_text)

    vectorizer = TfidfVectorizer(
        lowercase=False,  # clean_text 已经统一大小写
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_]+\b",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=20_000,
        sublinear_tf=True,
        norm="l2",
    )

    # 这里只做特征准备，不训练分类器；词表和 IDF 只从训练集学习。
    X_train_tfidf = vectorizer.fit_transform(train_text)
    X_valid_tfidf = vectorizer.transform(valid_text)
    X_test_tfidf = vectorizer.transform(test_text)

    # 保存清洗后的文本，方便复查和后续重新拟合。
    train_cleaned = train_split.copy()
    valid_cleaned = valid_split.copy()
    test_cleaned = test_df.copy()
    train_cleaned["cleaned_text"] = train_text
    valid_cleaned["cleaned_text"] = valid_text
    test_cleaned.insert(0, "source_index", range(len(test_cleaned)))
    test_cleaned["cleaned_text"] = test_text
    train_cleaned.to_csv(ROOT / "prepared_train.csv", index=False)
    valid_cleaned.to_csv(ROOT / "prepared_validation.csv", index=False)
    test_cleaned.to_csv(ROOT / "prepared_test.csv", index=False)

    # 保存特征矩阵和向量器，训练脚本可以直接读取，不必重复预处理。
    prepared_bundle = {
        "X_train": X_train_tfidf,
        "X_valid": X_valid_tfidf,
        "X_test": X_test_tfidf,
        "y_train": train_split["target"].to_numpy(),
        "y_valid": valid_split["target"].to_numpy(),
        "train_cleaned": train_text.tolist(),
        "valid_cleaned": valid_text.tolist(),
        "test_cleaned": test_text.tolist(),
        "vectorizer": vectorizer,
        "vectorizer_params": vectorizer.get_params(deep=False),
        "random_state": RANDOM_STATE,
    }
    joblib.dump(prepared_bundle, ROOT / "prepared_data.joblib", compress=3)

    cleaning_report = {
        "train_empty_after_cleaning": int((train_text == "empty_message").sum()),
        "validation_empty_after_cleaning": int((valid_text == "empty_message").sum()),
        "test_empty_after_cleaning": int((test_text == "empty_message").sum()),
        "train_median_characters_after_cleaning": float(train_text.str.len().median()),
        "validation_median_characters_after_cleaning": float(valid_text.str.len().median()),
        "test_median_characters_after_cleaning": float(test_text.str.len().median()),
        "tfidf_vocabulary_size": int(len(vectorizer.vocabulary_)),
        "train_matrix_shape": list(X_train_tfidf.shape),
        "validation_matrix_shape": list(X_valid_tfidf.shape),
        "test_matrix_shape": list(X_test_tfidf.shape),
        "outputs": [
            "prepared_train.csv",
            "prepared_validation.csv",
            "prepared_test.csv",
            "prepared_data.joblib",
        ],
    }
    (ROOT / "preprocessing_manifest.json").write_text(
        json.dumps(cleaning_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return vectorizer, X_train_tfidf, X_valid_tfidf, X_test_tfidf


def build_models() -> tuple[SVC, MLPClassifier]:
    """创建 SVM 和 MLP，暂不调用 fit。"""

    svm_classifier = SVC(
        kernel="linear",
        C=1.0,
        random_state=RANDOM_STATE,
        cache_size=4096,
    )
    mlp_classifier = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=RANDOM_STATE,
    )
    return svm_classifier, mlp_classifier


SVM_CANDIDATE_C = (0.1, 1.0, 10.0)
MLP_CANDIDATE_CONFIGS = (
    {"hidden_layer_sizes": (100,), "alpha": 0.0001},
    {"hidden_layer_sizes": (100,), "alpha": 0.001},
    {"hidden_layer_sizes": (100, 50), "alpha": 0.001},
)

# 预先固定后续评价指标，避免根据测试结果临时改标准。
EVALUATION_METRICS = ("accuracy", "macro_f1", "weighted_f1", "per_class_report")


def main() -> None:
    train_split, valid_split, test_df = load_and_split()
    vectorizer, X_train, X_valid, X_test = prepare_features(
        train_split, valid_split, test_df
    )
    svm_classifier, mlp_classifier = build_models()

    print("--- 数据准备完成，尚未开始分类器训练 ---")
    print(f"训练集: {len(train_split)} 条, 验证集: {len(valid_split)} 条")
    print(f"无标签测试集: {len(test_df)} 条")
    print(f"TF-IDF: 词表 {len(vectorizer.vocabulary_)} 个, train={X_train.shape}")
    print(f"TF-IDF validation={X_valid.shape}, test={X_test.shape}")
    print(f"SVM 参数: {svm_classifier.get_params()}")
    print(f"MLP 参数: {mlp_classifier.get_params()}")
    print(f"SVM 候选 C: {SVM_CANDIDATE_C}")
    print(f"MLP 候选结构/alpha: {MLP_CANDIDATE_CONFIGS}")
    print("已生成: prepared_train.csv, prepared_validation.csv, prepared_test.csv")
    print("已生成: prepared_data.joblib")
    print("没有调用 SVC.fit、MLPClassifier.fit，也没有生成新的 predictions.csv。")


if __name__ == "__main__":
    main()
