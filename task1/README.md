# 任务一：植物幼苗分类（报告第一章）

5 类植物幼苗，训练集 500 张（每类 100 张），测试集 378 张，按分类准确率评分。

## 文件

| 文件 | 作用 |
|---|---|
| `train.py` | 完整流程（单文件）：数据增强 → 三种骨干 × 5 折训练 → OOF 评估 → 15 个折模型 8 视角平均 → 写出 `submission.csv` |
| `ablation.py` | 消融实验调度与汇总：每次只改 `train.py` 的一个设置，在 ConvNeXt-Tiny 上做 5 折交叉验证 |
| `logs/` | 11 组消融实验的原始结果：`*.json` 汇总、`*.pt` 逐视角 OOF 概率、`*.log` 训练日志 |
| `analysis/facts.py` | 从 `logs/` 统计报告第一章用到的全部数字（含困难样本与背景线索分析），输出 `analysis/facts.json` |
| `notebooks/task1.ipynb` | 早期从零训练的自定义 CNN 基线（未写入报告） |

## 报告章节与代码对照

| 报告小节 | 对应代码 |
|---|---|
| 第二节 骨干网络选择（表 1-3、图 1-2） | `train.py`（`MODELS` 列表中的三种骨干） |
| 第三节 分辨率、训练轮数、学习率（表 1-5、图 1-4） | `ablation.py` 中的 `img_224`、`epochs_5/15`、`lr_3e-5/3e-4` |
| 第四节 预训练、旋转增强、标签平滑、随机深度（表 1-6、图 1-5） | `ablation.py` 中的 `no_pretrain`、`no_rotate`、`no_label_smoothing`、`drop_path_1x` |
| 第四节 TTA 视角组合（表 1-7） | `ablation.py summary`（直接用已保存的逐视角概率计算，无需重训） |
| 第四节 HSV 去背景（表 1-8、图 1-6） | `train.py` 中设 `REMOVE_BG = True` |
| 第五节 困难样本与背景线索（表 1-9、图 1-7、图 1-8） | `analysis/facts.py` |

## 运行方式

把比赛数据解压到本目录（含 `train/`、`test/` 的文件夹会被自动找到），然后：

```bash
python train.py                    # 5 折交叉验证 + 三模型集成，写出 submission.csv
python ablation.py                 # 依次运行全部消融实验（已完成的会跳过）
python ablation.py summary         # 汇总 logs/ 中的结果，打印对比表
python analysis/facts.py           # 统计报告用到的数字
```

`train.py` 开头的常量都可以在命令行覆盖，例如 `python train.py 'MODELS=["swin_t"]' EPOCHS=15`。
