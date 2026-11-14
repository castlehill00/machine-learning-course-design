# 任务三：眼底图像中央凹定位（报告第三章）

预测眼底图像中中央凹中心的像素坐标，训练集 80 张、测试集 20 张，按坐标 RMSE 评分（越小越好）。

## 目录

- `src/`：训练与实验代码。主要实验在 Kaggle GPU 上运行，本地也可直接运行。
- `notebooks/`：实际在 Kaggle 上运行的版本，已内嵌 `src/` 中所需的模块，可单独上传。
- `analysis/`：读取 Kaggle 下载的结果，统计报告中的表格与诊断。

## 代码依赖关系

后面的实验在前面的基础上扩展，直接导入前者：

```
task3_pipeline.py            数据读取、热图定位网络、训练循环、评估与提交导出（基础模块）
└── task3_kaggle_cv.py       ResNet34 五折：320 / 320+翻转TTA / 512，及双尺度集成
    ├── task3_kaggle_refine.py          粗定位 + 局部精修，固定回退保护
    │   └── task3_refine_diagnostic_v3.py      精修诊断（探索）
    │       └── task3_context_refine_v4.py     加入整图上下文的精修（探索）
    ├── task3_flair_transfer.py         FLAIR 眼底预训练编码器迁移对照
    └── task3_kaggle_multiseed.py       全量多种子对照（探索）
pretrained_ensemble.py / run_resnet34_fulltrain.py / evaluate_fit_pretrained.py   直接基于 task3_pipeline.py
```

## 报告章节与代码对照

| 报告小节 | 实验 | 代码 | Kaggle Notebook | 结果分析 |
|---|---|---|---|---|
| 第一节 任务与协议 | 数据读取、热图、坐标 RMSE、五折划分；ResNet18 单折基线 | `task3_pipeline.py` | `task3.ipynb` | — |
| 第二节 2. 骨干网络对比（表 3-2） | ResNet18 / ResNet34 / ConvNeXt / RETFound-Green 及集成 | `pretrained_ensemble.py` | `task3_pretrained_ensemble.ipynb` | — |
| 第三节 训练轮数与收敛（图 3-2） | 第 0 折逐轮曲线；统一第 80 轮对照 | `task3_kaggle_cv.py` | `task3_resnet34_5fold_kaggle.ipynb` | `analyze_common_epoch.py` |
| | ResNet34 全量 43 轮单模型 | `run_resnet34_fulltrain.py` | `task3_resnet34_fulltrain.ipynb` | — |
| 第四节 1. 分辨率、双尺度集成、TTA（表 3-4） | 320 / 512 / 翻转 TTA / 320+512 等权 | `task3_kaggle_cv.py` | `task3_resnet34_5fold_kaggle.ipynb` | `analyze_kaggle_cv_results.py` |
| 第四节 2. 局部精修与回退（表 3-5） | 粗定位 + 768 像素 ROI 精修，256 像素回退 | `task3_kaggle_refine.py` | `task3_resnet34_refine_kaggle.ipynb` | `analyze_refine_results.py`、`diagnose_refinement_reversal.py` |
| 第五节 榜单成绩（表 3-6） | 提交文件之间的预测差异、留出模型与全量模型的预测稳定性 | — | — | `audit_public_gap.py` |
| 第六节 2. FLAIR 迁移 | FLAIR 编码器 vs ImageNet ResNet50 | `task3_flair_transfer.py` | `task3_flair_transfer_kaggle.ipynb` | `analyze_flair_transfer.py` |
| 第六节 2. ZIAN 探索 | 直接推理与微调 | `run_zian_transfer.py`、`finetune_zian.py` | — | — |
| 第七节 小结（上下文融合精修） | 局部 ROI + 整图上下文 | `task3_context_refine_v4.py` | `task3_context_refine_v4_kaggle.ipynb` | `analyze_context_v4_results.py` |

## 报告未收录的探索实验

| 文件 | 内容 |
|---|---|
| `src/task3_refine_diagnostic_v3.py`（`notebooks/task3_refine_diagnostic_v3_kaggle.ipynb`、`analysis/analyze_refine_v3_results.py`） | 精修诊断：内层选轮次、裁剪扰动、位置先验 |
| `src/task3_kaggle_multiseed.py`（`notebooks/task3_resnet34_multiseed_kaggle.ipynb`） | 固定 43 轮的全量多随机种子对照 |
| `src/evaluate_fit_pretrained.py` | 直接评估作者发布的 EfficientNet-B3 中央凹权重（不训练） |

## 运行方式

数据解压到 `task3/2026task3/`，在任务目录下运行：

```bash
python src/task3_pipeline.py --phase audit                     # 检查数据与坐标
python src/task3_pipeline.py --phase full --epochs 80 --image-size 320
python src/task3_kaggle_cv.py --allow-local                    # 五折对照（默认在 Kaggle 上运行）
```

在 Kaggle 上运行时，上传 `notebooks/` 中对应的 Notebook，挂载比赛数据并开启 GPU 即可。ZIAN 相关脚本需要先准备作者的源码与权重（放在 `zian_transfer_20261027/` 下）。
