# ECNU AI Intro Project 1

这是《当代人工智能》大作业 1 的文本分类代码。项目使用 TF-IDF 表示新闻文本，并比较 SVM 与 MLP 两种分类器。

## 1. 环境配置

推荐使用 Conda：

```bash
conda env create -f environment.yml
conda activate project1
```

如果环境已经存在，可以执行：

```bash
conda env update -f environment.yml --prune
```

## 2. 数据文件

请将原始数据放在项目根目录：

- `train_data.csv`：有标签训练集，至少包含 `text` 和 `target` 两列；
- `test_data_unlabeled.csv`：无标签测试集，至少包含 `text` 列。

数据集和训练生成的大文件没有提交到仓库。这样可以避免仓库过大，也避免把本地实验结果误当成代码的一部分。

## 3. 运行流程

### 预处理、清洗和划分数据

```bash
python project1_prepare.py
```

该脚本会清理邮件/新闻文本，按类别分层划分训练集和验证集，并且只用训练子集拟合 TF-IDF，再转换验证集和测试集。输出的中间文件包括 `prepared_*.csv`、`prepared_data.joblib` 和数据划分说明文件。

### 训练 SVM

```bash
python train_svm.py
```

直接在 `train_svm.py` 顶部修改待尝试的 `C_VALUES`、核函数和其他参数即可。脚本会在验证集上比较多组配置，并保存 `svm_training_results.csv`、最佳模型和统计图。

### 训练 MLP

```bash
python train_mlp.py
```

直接在 `train_mlp.py` 顶部修改 `HIDDEN_LAYER_CONFIGS`、`ALPHA_VALUES`、`LEARNING_RATE_VALUES`、`MAX_ITER` 等参数即可。脚本会保存验证结果、loss 曲线和最佳模型。

### 用最佳 MLP 配置进行最终预测

在完成 `train_mlp.py` 后运行：

```bash
python predict_mlp_best.py
```

该脚本读取验证集上 macro-F1 最优的 MLP 配置，用全部有标签数据重新拟合 TF-IDF 和模型，最后预测无标签测试集，生成 `mlp_predictions_final.csv`。验证集只用于选择参数，不参与此前的模型选择拟合；参数确定后，最终重训才使用全部有标签数据。

## 4. 代码说明

| 文件 | 作用 |
| --- | --- |
| `project1_prepare.py` | 文本清洗、分层划分、TF-IDF 特征生成 |
| `project1_train.py` | 公共数据读取、SVM/MLP 构建、评估和最终重训函数 |
| `train_svm.py` | SVM 参数实验和验证集评估 |
| `train_mlp.py` | MLP 参数实验、loss 曲线和验证集评估 |
| `predict_mlp_best.py` | 读取最佳 MLP 配置并进行最终预测 |
| `environment.yml` | Conda 环境依赖 |
| `TUNING.md` | 参数调整建议和实验记录说明 |

## 5. 数据泄漏控制

参数比较阶段先划分训练集和验证集，再只使用训练子集拟合 TF-IDF；验证集仅用于评估和选择参数。只有在模型配置确定后，最终预测脚本才会将全部有标签数据用于重训。这保证了验证集没有参与参数选择阶段的模型拟合。

## 6. 结果说明

本项目的最终测试集没有提供标签，因此不能在本地计算测试准确率或 macro-F1。`mlp_predictions_final.csv` 只包含最终预测标签；验证集指标和参数对比保存在训练脚本生成的结果 CSV 中。
