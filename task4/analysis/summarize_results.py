"""Write the first-run experiment record from completed artifacts."""

import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROOT = ROOT.parent if ROOT.name == "analysis" else ROOT  # 位于 analysis/ 时回到任务根目录
RESULT = ROOT / "baseline_results"


def load(name):
    return json.loads((RESULT / name).read_text(encoding="utf-8"))


def main():
    audit = load("audit_summary.json")
    metrics = load("validation_metrics.json")
    provenance = load("submission_provenance.json")
    config = load("run_config.json")
    if not provenance["official_converter_exact_match"] or not provenance["rle_roundtrip_all_20"]:
        raise AssertionError("Export checks did not pass")
    histories = {}
    for stage in ["holdout", "fulltrain"]:
        with (RESULT / stage / "history.csv").open(newline="", encoding="utf-8") as f:
            histories[stage] = list(csv.DictReader(f))
    if (
        len(histories["holdout"]) != config["config"]["epochs"]
        or len(histories["fulltrain"]) != metrics["best_epoch"]
    ):
        raise AssertionError("Incomplete training history")
    with (RESULT / "validation_per_image.csv").open(newline="", encoding="utf-8") as f:
        per_image = list(csv.DictReader(f))
    table = "\n".join(
        f"| {r['id']} | {float(r['dice_05']):.5f} | {float(r['dice']):.5f} | {float(r['native_dice']):.5f} | {float(r['converter_target_proxy_dice']):.5f} |"
        for r in per_image
    )
    hold_seconds = sum(float(r["seconds"]) for r in histories["holdout"])
    full_seconds = sum(float(r["seconds"]) for r in histories["fulltrain"])
    copy = ROOT / "submission_baseline.csv"
    shutil.copyfile(RESULT / "submission.csv", copy)
    assert hashlib.sha256(copy.read_bytes()).hexdigest() == provenance["csv_sha256"]
    text = f"""# 任务4初版实验记录

日期：2026-10-31。以下结果均来自本地GPU训练。

## 任务与数据

任务为二维眼底血管分割，以Dice评价，提交文件列名为`Id,Predicted`，预测以RLE编码。

数据目录为`2026task-4/segmentation/`，训练图及标签各20张、测试图20张，尺寸均为565×584。20张训练图互不重复，与测试图也没有完全相同的图像；数据未提供患者ID。数据哈希指纹为`{audit['data_fingerprint']}`。

固定随机种子42，按图像分组划分5折，每折4张。本轮只运行第0折：验证图为{', '.join(metrics['validation_ids'])}，其余16张用于训练。

## 标签极性与转码

比赛说明中血管像素为0、背景为255，而实际标签是血管白、背景黑；官方提供的`segmentation_to_csv.py`对非零像素编码。因此CSV按该脚本的实际约定生成，另外按说明导出原尺寸的黑血管PNG。

官方脚本先用PIL双线性插值缩放到512×512，以`>0`判定前景，按行展开，RLE位置从1开始。原始标签为有损JPEG，压缩噪声会被当作前景：20张图的平均血管比例为{audit['mean_clean_foreground']:.4%}，直接按官方脚本处理原JPEG则为{audit['mean_converter_foreground']:.4%}。因此训练标签采用原尺寸`>127`二值化后最近邻缩放；验证时另外统计“官方转换器处理原JPEG”的代理目标指标。

提交的PNG直接在512×512上生成，取值只有0和255，避免转码时引入JPEG噪声或额外缩放。

## 方法与训练

网络为随机初始化的U-Net，基础通道数24，使用GroupNorm，输入为RGB加CLAHE增强的绿色通道共4通道。训练时在512尺寸图上随机抽取256×256的patch，每轮128个、batch=8，其中30%以血管点为中心；图像与标签同步做裁剪、随机缩放、翻转和旋转，图像另做亮度与Gamma增强。

损失为BCE与前景Soft Dice各占50%；优化器AdamW，学习率0.001，权重衰减0.0001；80轮余弦退火，混合精度训练，梯度裁剪阈值5。推理在512尺寸上用步长128的滑窗加权融合，不使用TTA。未使用预训练权重和外部数据。

验证训练共80轮，以平均血管Dice@0.5选出第**{metrics['best_epoch']}轮**。在该模型上以0.025为步长扫描0.2至0.8的阈值，最终阈值为**{metrics['threshold']:.3f}**。轮数和阈值都在这4张验证图上选出，属于开发集估计。

验证训练的逐轮时间合计约{hold_seconds/60:.2f}分钟（含每轮验证）。

## 验证结果

| 指标 | 实际结果 |
|---|---:|
| 512干净血管目标，阈值0.5，逐图平均Dice | {metrics['mean_vessel_dice_05']:.6f} |
| 512干净血管目标，所选阈值，逐图平均Dice | {metrics['mean_vessel_dice']:.6f} |
| 原尺寸干净血管目标，所选阈值，逐图平均Dice | {metrics['mean_native_vessel_dice']:.6f} |
| 原JPEG按官方转换器处理的目标代理Dice | {metrics['converter_target_proxy_dice']:.6f} |

| 验证图 | Dice@0.5 | Dice@所选阈值 | 原尺寸Dice | 转换器目标代理Dice |
|---|---:|---:|---:|---:|
{table}

表中为逐图平均的前景Dice。验证集预测可视化见`baseline_results/validation_preview.png`，预测概率、逐图结果和阈值扫描均已保存。

## 全量重训与提交

用相同随机种子重新初始化模型，在全部20张训练图上训练{provenance['epochs']}轮（沿用80轮的余弦调度，只运行前{provenance['epochs']}轮），逐轮训练时间合计约{full_seconds/60:.2f}分钟。阈值沿用验证集上选出的值。

提交文件为`baseline_results/submission.csv`（副本`submission_baseline.csv`），共20行，SHA256为`{provenance['csv_sha256']}`。

20张白血管PNG经官方函数转换的结果与本实现的RLE编码完全一致，全部RLE解码后均与预测mask吻合。全量模型权重的SHA256为`{provenance['model_sha256']}`，详见`submission_provenance.json`。

`masks_native_black_vessels/`保存原尺寸、血管为0、背景为255的PNG；`masks_512_white_vessels/`保存用于官方转码的白血管PNG；测试集预测可视化见`test_preview.png`。

## 代码检查

已检查图像级划分、8种D4同步变换、CUDA前向/反向与参数更新、滑窗覆盖范围以及RLE编码的各种边界情况。

`task4.ipynb`与脚本使用相同的实现，用其重新推理导出的权重、日志和CSV与脚本运行结果完全一致。

本轮是单折、单模型的基线，后续再进行五折交叉验证、模型集成和TTA对照。
"""
    (ROOT / "实验记录.md").write_text(text, encoding="utf-8")
    print("Wrote 实验记录.md and submission_baseline.csv")


if __name__ == "__main__":
    main()
