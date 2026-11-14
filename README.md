# 机器学习课程设计代码

课程设计报告共四章，每章对应一个任务。本仓库存放各任务的训练、实验与结果分析代码，报告中的每项实验都能在对应任务的 README 里找到入口文件。

| 目录 | 报告章节 | 任务 | 一句话概括 |
|---|---|---|---|
| [`task1/`](task1/) | 第一章 | 植物幼苗分类（5 类，500 张） | ConvNeXt-T / EfficientNetV2-S / Swin-T 三种骨干 × 5 折集成，8 视角 TTA |
| [`task2/`](task2/) | 第二章 | 人脸表情分类（6 类） | 表情预训练 EfficientNet-B2 + ResNet18 等权融合，12 视图 TTA |
| [`task3/`](task3/) | 第三章 | 眼底图像中央凹定位 | ResNet34 热图定位，320/512 双尺度五折集成，局部精修 |
| [`task4/`](task4/) | 第四章 | 眼底图像血管分割 | ResNet34 U-Net，五折 + D4 TTA，标签监督与损失消融，固定 logit 融合 |

## 目录约定

任务二～四按用途分成三个子目录（任务一代码量小，`train.py` 与 `ablation.py` 直接放在任务目录下）：

```
taskN/
├── README.md     报告章节与代码对照、运行方式（先看这里）
├── src/          训练与实验代码（可直接运行，或被 Notebook 调用）
├── notebooks/    在 Kaggle GPU 上实际运行的 Notebook，已内嵌所需源码，可单独上传
└── analysis/     对实验结果做统计、生成报告表格的分析脚本
```

## 使用说明

- **数据集未包含**：从各任务的 Kaggle 比赛页面下载数据，解压到对应任务目录下（例如 `task3/2026task3/`），代码会自动查找。
- **只使用比赛提供的数据**，没有用外部数据集扩充训练集；用到的预训练权重（ImageNet、EmotiEffLib、RETFound、FLAIR 等）见报告各章参考文献。
- **模型权重与中间结果未包含**（体积过大）。任务一的消融实验原始结果较小，保存在 `task1/logs/`。
- `analysis/` 中的脚本读取的是 Kaggle 运行后下载的结果目录（如 `kaggle_cv_results_20261025/`），这些结果同样未包含在仓库中。

## 环境

Python 3.11，PyTorch 2.x + torchvision，numpy、pandas、pillow。任务二另需 `timm`；任务四另需 `opencv-python`、`scikit-image`。
