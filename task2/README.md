# 任务二：人脸表情分类（报告第二章）

48 × 48 灰度人脸图像的六分类（愤怒、恐惧、快乐、悲伤、惊讶、中性），按准确率评分。

## 目录

- `src/`：本地运行的实验脚本。`run_pretraining_local.py` 是公共基础模块（数据读取、模型构建、训练与评估），其余脚本都导入它。
- `notebooks/`：ResNet18/34 的训练与正则化消融、ResNet34 全量训练。
- 本任务没有单独的分析脚本，结果分析写在各实验脚本和 `notebooks/task2.ipynb` 中。

## 报告章节与代码对照

| 报告小节 | 实验 | 入口 |
|---|---|---|
| 第二节 骨干网络与预训练对比（表 2-1） | ResNet18 / ResNet34 | `notebooks/task2.ipynb` |
| | EfficientNet-B2：表情预训练（EmotiEffLib）对比通用预训练（Noisy Student） | `src/run_pretraining_local.py` |
| 第三节 训练收敛性（图 2-2） | ResNet18 逐轮曲线、标签平滑 | `notebooks/task2.ipynb` |
| 第四节 1. 正则化与 Dropout 消融（表 2-3） | Weight Decay × Dropout、标签平滑 | `notebooks/task2.ipynb` |
| 第四节 2. 融合权重（表 2-4） | B2 + ResNet18 概率加权融合 | `src/run_ensemble_local.py` |
| 第四节 3. TTA 视图数量（表 2-5） | 翻转 / 五裁剪 / 12 视图 | `src/run_multicrop_local.py` |
| 第五节 榜单与最终方案（表 2-6） | ResNet34 全量训练 | `notebooks/task2_fulltrain.ipynb`（本地用 `src/run_fulltrain_local.py` 运行） |
| | B2 全量 9 轮、ResNet18 全量 20 轮，等权融合得到最终提交 | `src/run_b2_fulltrain_local.py`、`src/run_resnet18_fulltrain_local.py` |
| | B2 留出集权重导出提交 | `src/export_b2_submission.py` |
| 第六节 2. 探索性尝试 | 多尺度裁剪 TTA | `src/run_crop_scales_local.py` |
| | 温度校准 | `src/run_calibration_local.py` |

## 报告未收录的探索实验

| 文件 | 内容 |
|---|---|
| `src/run_semantic_head_local.py` | 迁移表情预训练的语义分类头 |
| `src/run_resnet34_third_local.py` | 加入 ResNet34 作为第三个融合模型 |

## 运行方式

数据解压到 `task2/2026task2/`，在任务目录下运行，例如：

```bash
python src/run_pretraining_local.py      # B2 预训练对比
python src/run_ensemble_local.py         # 融合权重
python src/run_multicrop_local.py        # TTA 视图数量
```

每个实验的输出写在任务目录下以实验命名的文件夹中，支持中断后续跑。
