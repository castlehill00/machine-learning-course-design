# 任务四：眼底图像血管分割（报告第四章）

20 张训练图、20 张测试图（565 × 584），预测血管 mask，以 Dice 评分，提交为 RLE 编码。

## 目录

- `src/`：分割模型与各组实验的实现。除 `task4_pipeline.py` 外，均由对应的 Kaggle Notebook 调用其中的函数运行。
- `notebooks/`：实际在 Kaggle GPU 上运行的版本，已内嵌所需模块，可单独上传。
- `analysis/`：读取 Kaggle 下载的结果，统计报告中的表格与配对比较。

## 代码依赖关系

```
task4_pipeline.py                 数据审计、标签处理、官方 RLE 转换、初版随机初始化 U-Net 基线（基础模块）
└── task4_kaggle_v2.py            ResNet34 U-Net：原尺寸训练、EMA、五折、D4 TTA
    └── task4_controlled_ablation.py      受控消融框架：阈值/膨胀（E1）、EMA（E2）、标签监督 L0/L1/L2
        ├── task4_cldice_ablation.py          L2 + soft-clDice 结构损失
        ├── task4_core_band.py                核心 / 边带 / 背景三分类
        ├── task4_hyperparameter_ablation.py  学习率与 BCE 权重（含 AMP 数值保护）
        └── task4_recovery.py                 原尺寸与 L1 补跑、固定 logit 融合、最终测试推理与提交
```

## 报告章节与代码对照

| 报告小节 | 实验 | 代码 | Kaggle Notebook | 结果分析 |
|---|---|---|---|---|
| 第一节 任务定义、标签口径、评估协议 | 数据审计、JPEG 标签与官方转换器、初版 U-Net | `task4_pipeline.py` | `task4.ipynb` | `summarize_results.py`、`diagnose_label_boundaries.py` |
| 第二节 ResNet34 U-Net（图 4-1、表 4-2） | 原尺寸五折、EMA、D4 TTA | `task4_kaggle_v2.py` | `task4_resnet34_5fold_kaggle.ipynb` | `analyze_kaggle_v2_results.py` |
| 第三节 学习率与损失权重（表 4-4、图 4-2、图 4-3） | 学习率 ×0.5 / ×2，BCE 权重 0.25 / 0.75 | `task4_hyperparameter_ablation.py` | `task4_hyperparameter_ablation_kaggle.ipynb` | `analyze_hyperparameter_results.py` |
| 第四节 1. TTA 与预测宽度匹配（表 4-5） | 阈值 0.4 / 0.1 / 固定膨胀；EMA 对照 | `task4_controlled_ablation.py` | `task4_controlled_ablation_kaggle.ipynb` | `analyze_controlled_ablation_results.py` |
| 第四节 2. 标签监督（表 4-6） | L0 干净 / L1 代理 / L2 软边带 | `task4_controlled_ablation.py` | `task4_label_supervision_5fold_kaggle.ipynb` | `analyze_label_supervision_results.py` |
| 第四节 2. 结构损失（表 4-7） | L2 + 0.1 × soft-clDice | `task4_cldice_ablation.py` | `task4_cldice_5fold_kaggle.ipynb` | `analyze_structure_ablation_results.py` |
| 第四节 2. 三分类代理输出（表 4-8） | 核心 / 边带 / 背景 softmax | `task4_core_band.py` | `task4_core_band_kaggle.ipynb` | `analyze_core_band_results.py` |
| 第四节 3. 固定等权 logit 融合（表 4-9） | 原尺寸模型 + L1 融合 | `task4_recovery.py` | `task4_recovery_and_fusion_kaggle.ipynb` | `analyze_recovery_oof.py` |
| 第五节 最终测试结果（表 4-10） | 十个折模型融合推理并导出 RLE | `task4_recovery.py` 中的 `finish_inference()` | — | — |

报告中的“干净目标（C）”指原 JPEG 标签灰度 > 127 后最近邻缩放；“代理目标（P）”指原 JPEG 经官方转换器（PIL 双线性缩放到 512 后取 > 0）得到的标签，两者分别评价。

## 运行方式

数据解压到 `task4/2026task-4/`。初版基线可在本地运行：

```bash
python src/task4_pipeline.py --phase audit --output-dir audit_results
python src/task4_pipeline.py --phase full --output-dir baseline_results
```

其余实验上传 `notebooks/` 中对应的 Notebook 到 Kaggle，挂载比赛数据、开启 GPU 和 Internet（下载 ImageNet 预训练权重）后从头运行。
