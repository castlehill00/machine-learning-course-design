# 任务二：人脸表情分类（报告第二章）

48 × 48 灰度人脸图像的六分类（愤怒、恐惧、快乐、悲伤、惊讶、中性），按准确率评分。

## 目录

- `src/`：本地运行的实验脚本。`run_pretraining_local.py` 是公共基础模块（数据读取、模型构建、训练与评估），其余脚本都导入它。
- `notebooks/`：ResNet18/34 的训练与正则化消融代码、ResNet34 全量训练。`task2.ipynb` 保存的是恢复前两轮实验并追加 ResNet34 的版本，需要第二轮实验备份。
- 本任务没有单独的分析脚本，结果分析写在各实验脚本和 `notebooks/task2.ipynb` 中。

## 报告章节与代码对照

| 报告小节 | 实验 | 入口 |
|---|---|---|
| 第二节 骨干网络与预训练对比（表 2-2） | ResNet18 / ResNet34 | `notebooks/task2.ipynb` |
| | EfficientNet-B2：表情预训练（EmotiEffLib）对比通用预训练（Noisy Student） | `src/run_pretraining_local.py` |
| 第三节 训练收敛性（图 2-2） | ResNet18 逐轮曲线、标签平滑 | `notebooks/task2.ipynb` |
| 第四节 1. 正则化与 Dropout 消融（表 2-4） | Weight Decay × Dropout、标签平滑 | `notebooks/task2.ipynb` |
| 第四节 2. 融合权重（表 2-5） | B2 + ResNet18 概率加权融合 | `src/run_ensemble_local.py` |
| 第四节 3. TTA 视图数量（表 2-6） | 翻转 / 五裁剪 / 12 视图 | `src/run_multicrop_local.py` |
| 第五节 榜单与最终方案（表 2-7） | ResNet34 全量训练 | `notebooks/task2_fulltrain.ipynb`（本地用 `src/run_fulltrain_local.py` 运行） |
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

先准备以下文件（比赛数据和预训练权重另行下载，第二轮实验备份通过 GitHub Release 提供）：

1. 比赛数据：`task2/2026task2/fer_data/fer_data/train/`、`test/`，以及 `task2/2026task2/submission.csv`。
2. 原实验备份：[下载 task2_ablation_backup_round2.zip](https://github.com/castlehill00/machine-learning-course-design/releases/download/task2-ablation-round2/task2_ablation_backup_round2.zip)（210,704,034 字节，约 211 MB；[Release 说明](https://github.com/castlehill00/machine-learning-course-design/releases/tag/task2-ablation-round2)）。下载后将 ZIP 原样放到 `task2/` 目录，无需手动解压。其中保留 `task2_ablation_seed42/split.csv`，以及 baseline、weight_decay、dropout、combined、label_smoothing 五组实验的配置、指标、训练历史、混淆矩阵、验证概率和 `best.pt`。后续脚本直接读取该压缩包，复用划分和 ResNet18 权重。
3. B2 表情预训练权重：[enet_b2_7.pt](https://raw.githubusercontent.com/sb-ai-lab/EmotiEffLib/main/models/affectnet_emotions/enet_b2_7.pt)，保存至 `task2/.torch/hub/checkpoints/enet_b2_7.pt`。
4. B2 通用预训练权重：[tf_efficientnet_b2_ns-00306e48.pth](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2_ns-00306e48.pth)，保存至同一检查点目录。`prepare_weights()` 校验文件哈希并转换表情权重，不负责自动下载。

备份 ZIP 的 SHA-256（上传的是未修改的原始备份）：

```text
cfbc717fb14983a3c2435b23a23ddf6d49d0437f4ade4e4ac93158e8776dc649
```

在 Kaggle 运行 `task2.ipynb` 时，还需挂载第二轮备份数据集 `task2-ablation-backup-round2`。本地 B2、融合与 TTA 脚本依次运行；后续步骤读取前一步产生的结果与权重，例如：

```bash
python src/run_pretraining_local.py      # B2 预训练对比
python src/run_ensemble_local.py         # 融合权重
python src/run_multicrop_local.py        # TTA 视图数量
```

每个实验的输出写在任务目录下以实验命名的文件夹中，支持中断后续跑。
