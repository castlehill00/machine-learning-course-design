"""Audit returned E1/E2 tables without executing returned code or unpickling weights."""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ROOT = ROOT.parent if ROOT.name == "analysis" else ROOT  # 位于 analysis/ 时回到任务根目录
RESULT = ROOT / "controlled_ablation_results_20261101"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name):
    return json.loads((RESULT / name).read_text(encoding="utf-8"))


def table(name):
    with (RESULT / name).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def close(a, b):
    if not math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-10):
        raise AssertionError((a, b))


def write(name, rows):
    with (RESULT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    status, audit, config = (
        load("experiment_status.json"),
        load("replay_audit.json"),
        load("ablation_config.json"),
    )
    checks = {}
    for name, key in [
        ("task4_controlled_ablation.py", "source_sha256"),
        ("task4_kaggle_v2.py", "v2_source_sha256"),
        ("task4_pipeline.py", "base_source_sha256"),
    ]:
        assert sha(RESULT / "runtime_source" / name) == config[key] == sha(ROOT / name)
    checks["all_three_returned_sources_match_delivered_sources"] = True
    assert (
        status["inference_done"]
        and status["raw_replayed"]
        and not status["label_training_requested"]
    )
    assert (
        not status["competition_csv_generated"]
        and not status["public_scores_used"]
        and not status["automatic_model_selection"]
    )
    assert status["label_summary"] == [] and not any(RESULT.glob("L*/fold*/history.csv"))
    assert audit["raw_and_ema_replayed_in_same_environment"] and not audit["training_started"]
    checks["completed_e1_e2_only_no_label_training_or_public_score_selection"] = True

    manifest = table("manifest.csv")
    assignment = {r["id"]: int(r["fold"]) for r in manifest if r["split"] == "train"}
    assert len(manifest) == 40 and len(assignment) == 20
    stable = []
    data = ROOT / "2026task-4/segmentation"
    for row in manifest:
        assert sha(data / row["split"] / "image" / f'{row["id"]}.jpg') == row["image_sha256"]
        if row["split"] == "train":
            assert sha(data / "train/label" / f'{row["id"]}.jpg') == row["label_sha256"]
        item = {
            k: row[k]
            for k in ["split", "id", "width", "height", "image_sha256", "label_sha256", "fold"]
        }
        if row["split"] == "train":
            item["fold"] = int(item["fold"])
        stable.append(item)
    fp = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
    assert fp == config["data_fingerprint"] == load("audit_summary.json")["data_fingerprint"]
    for fold in range(5):
        record = audit["fold_records"][str(fold)]
        original = json.loads(
            (ROOT / f"kaggle_v2_results_20261101/fold{fold}/completed.json").read_text()
        )
        assert record == original and record["epoch"] == 80
        assert set(record["val_ids"]) == {i for i, f in assignment.items() if f == fold}
        cache = load(f"replay_cache/fold{fold}_identity.json")
        assert cache["identity"]["checkpoint_sha256"] == record["sha256"]["last.pt"]
        assert cache["identity"]["config"]["epochs"] == 80
    checks["data_fingerprint_frozen_splits_and_prior_checkpoint_lineage_match"] = True

    rows, fold_rows, summary = (
        table("replay_per_image.csv"),
        table("replay_per_fold.csv"),
        table("replay_summary.csv"),
    )
    metrics = [
        "dice",
        "precision",
        "recall",
        "centerline_precision",
        "centerline_recall",
        "cldice",
        "foreground_fraction",
        "thin_clean_centerline_recall",
    ]
    expected = {
        "E1_ema_t01",
        "E1_ema_t04",
        "E1_ema_t04_dilate1",
        "E2_raw_t01",
        "E2_raw_t04",
        "E2_replay_ema_t01",
        "E2_replay_ema_t04",
    }
    assert len(rows) == 280 and len(fold_rows) == 70 and len(summary) == 14
    lookup = {(r["experiment"], r["target"], r["id"]): r for r in rows}
    grouped = {(r["experiment"], r["target"], int(r["fold"])): r for r in fold_rows}
    summaries = {(r["experiment"], r["target"]): r for r in summary}
    assert len(lookup) == 280 and {r["experiment"] for r in rows} == expected
    for r in rows:
        assert int(r["fold"]) == assignment[r["id"]] and r["target"] in {"clean", "proxy"}
        assert all(math.isfinite(float(r[k])) and 0 <= float(r[k]) <= 1.0000000001 for k in metrics)
        a, b = float(r["centerline_precision"]), float(r["centerline_recall"])
        close(r["cldice"], 2 * a * b / (a + b) if a + b else 0)
    for experiment in expected:
        for target in ["clean", "proxy"]:
            group = [lookup[experiment, target, str(i)] for i in range(1, 21)]
            s = summaries[experiment, target]
            assert int(s["n_images"]) == 20 and s["complete_20_image_oof"] == "True"
            for key in metrics:
                close(s[key], np.mean([float(r[key]) for r in group]))
            fs = []
            for fold in range(5):
                subset = [r for r in group if int(r["fold"]) == fold]
                f = grouped[experiment, target, fold]
                assert len(subset) == 4 and int(f["n_images"]) == 4
                for key in metrics:
                    close(f[key], np.mean([float(r[key]) for r in subset]))
                fs.append(float(f["dice"]))
            close(s["worst_fold_dice"], min(fs))
            close(s["fold_dice_std"], np.std(fs))
            original = [
                r
                for r in status["inference_summary"]
                if r["experiment"] == experiment and r["target"] == target
            ][0]
            for key in metrics:
                close(original[key], s[key])
    for experiment in expected:
        for i in range(1, 21):
            for key in ["foreground_fraction", "thin_clean_centerline_recall"]:
                close(
                    lookup[experiment, "clean", str(i)][key],
                    lookup[experiment, "proxy", str(i)][key],
                )
    checks["all_280_oof_rows_70_fold_rows_and_14_summaries_verified"] = True

    drift = table("ema_replay_drift.csv")
    assert len(drift) == 20 and {r["id"] for r in drift} == set(assignment)
    assert all(int(r["fold"]) == assignment[r["id"]] for r in drift)
    assert all(
        float(r["mean_absolute_probability_drift"]) == 0
        and float(r["max_absolute_probability_drift"]) == 0
        for r in drift
    )
    for tag in ["t01", "t04"]:
        for target in ["clean", "proxy"]:
            for i in range(1, 21):
                for key in metrics:
                    close(
                        lookup[f"E1_ema_{tag}", target, str(i)][key],
                        lookup[f"E2_replay_ema_{tag}", target, str(i)][key],
                    )
    previous = json.loads((ROOT / "kaggle_v2_results_20261101/cv_summary.json").read_text())[
        "modes"
    ]["d4"]
    close(summaries["E1_ema_t04", "clean"]["dice"], previous["mean_dice_global_threshold"])
    close(summaries["E1_ema_t01", "proxy"]["dice"], previous["converter_proxy_best_dice"])
    checks["reported_zero_ema_replay_drift_and_prior_scores_reproduced"] = True
    checks["local_weight_or_probability_replay_possible_from_this_light_zip"] = False

    comparisons = [
        ("low_vs_high", "E1_ema_t01", "E1_ema_t04"),
        ("dilate_vs_high", "E1_ema_t04_dilate1", "E1_ema_t04"),
        ("low_vs_dilate", "E1_ema_t01", "E1_ema_t04_dilate1"),
        ("ema_vs_raw_t01", "E2_replay_ema_t01", "E2_raw_t01"),
        ("ema_vs_raw_t04", "E2_replay_ema_t04", "E2_raw_t04"),
    ]
    comparisons_summary = []
    paired = []
    paired_folds = []
    for name, left, right in comparisons:
        for target in ["clean", "proxy"]:
            for key in metrics:
                values = []
                fv = []
                for i in range(1, 21):
                    a, b = lookup[left, target, str(i)], lookup[right, target, str(i)]
                    delta = float(a[key]) - float(b[key])
                    values.append(delta)
                    paired.append(
                        dict(
                            comparison=name,
                            target=target,
                            metric=key,
                            id=i,
                            fold=assignment[str(i)],
                            delta=delta,
                        )
                    )
                for fold in range(5):
                    d = float(grouped[left, target, fold][key]) - float(
                        grouped[right, target, fold][key]
                    )
                    fv.append(d)
                    paired_folds.append(
                        dict(comparison=name, target=target, metric=key, fold=fold, delta=d)
                    )
                comparisons_summary.append(
                    dict(
                        comparison=name,
                        target=target,
                        metric=key,
                        mean_delta=float(np.mean(values)),
                        positive_images=sum(v > 0 for v in values),
                        negative_images=sum(v < 0 for v in values),
                        positive_folds=sum(v > 0 for v in fv),
                        negative_folds=sum(v < 0 for v in fv),
                        min_fold_delta=min(fv),
                        max_fold_delta=max(fv),
                    )
                )
    write("analysis_paired_summary.csv", comparisons_summary)
    write("analysis_paired_per_image.csv", paired)
    write("analysis_paired_per_fold.csv", paired_folds)
    result = dict(
        checks=checks,
        archive_sha256=load("received_archive.json")["sha256"],
        comparisons=comparisons_summary,
        scope="E1/E2 complete; L0/L1/L2 not run",
        private_score=None,
        new_training_started=False,
        new_submission_generated=False,
    )
    (RESULT / "analysis_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    s = lambda experiment, target, key: float(summaries[experiment, target][key])
    names = [
        ("阈值0.4", "E1_ema_t04"),
        ("阈值0.1", "E1_ema_t01"),
        ("0.4＋膨胀1像素", "E1_ema_t04_dilate1"),
    ]
    values = "\n".join(
        f"| {name} | {s(e,'clean','dice'):.6f} | {s(e,'proxy','dice'):.6f} | {s(e,'clean','thin_clean_centerline_recall'):.2%} | {s(e,'clean','cldice'):.6f} | {s(e,'clean','foreground_fraction'):.2%} |"
        for name, e in names
    )
    ema = "\n".join(
        f"| {t} | {target} | {s('E2_raw_'+tag,target,'dice'):.6f} | {s('E2_replay_ema_'+tag,target,'dice'):.6f} | {s('E2_replay_ema_'+tag,target,'dice')-s('E2_raw_'+tag,target,'dice'):+.6f} |"
        for t, tag in [("0.1", "t01"), ("0.4", "t04")]
        for target in ["clean", "proxy"]
    )
    body = f"""# 任务4 E1/E2受控消融结果分析

日期：2026-11-01。结果文件`ablation_analysis_results.zip`（SHA256 `{result['archive_sha256']}`）。本次在20张训练图上完成E1、E2两组实验，L0/L1/L2标签监督实验尚未运行，本分析不涉及公开榜分数。

## 主要结论

1. 固定膨胀已能带来大部分转码代理Dice的提升，说明标签边界宽度的差异是重要因素。阈值0.1的效果仍优于固定膨胀，并额外覆盖了一部分已标注的细血管中心线，因此低阈值的作用不等同于统一加粗。
2. 在干净目标上，低阈值的骨架精确率和clDice都低于固定膨胀。细血管召回提高的同时也引入了误检，属于召回与精度之间的取舍。
3. EMA与原始权重的Dice几乎相同，在当前80轮、衰减0.99的配方下没有明显额外收益，因此不作为主要提升来源。
4. 下一步执行L0/L1/L2标签监督消融，重点研究目标定义和边带不确定性，暂不调整EMA参数或阈值。

## E1：边界匹配与低阈值

| 同一EMA＋D4预测 | 干净Dice | 代理Dice | 干净细血管中心线召回 | 干净clDice | 前景比例 |
|---|---:|---:|---:|---:|---:|
{values}

从阈值0.4到固定膨胀，代理Dice由0.765125升至0.818303；从阈值0.4到0.1，则升至0.827069，说明边界匹配很重要。

阈值0.1与固定膨胀相比：代理Dice平均高0.008767，17/20张图、5/5折提高；干净细血管中心线召回高5.892个百分点，20/20张图提高；前景比例反而低0.946个百分点。可见低阈值的覆盖更有选择性，而不是把所有血管同等加粗。

另一方面，干净clDice从固定膨胀的0.823680降到0.810475（16/20张图下降），代理clDice也略有下降；干净骨架精确率从0.811217降到0.758349，骨架召回则提高。新增的细线中有一部分是伪分支或弱响应误检，中心线召回提高并不等于连通性整体改善。

细血管中心线定义为干净标注骨架上距背景不超过1.5像素的点（在512网格上计算）。所有数值为逐图平均，干净目标与代理目标分别统计。

## E2：EMA权重与同轮原始权重

| 阈值 | 评价目标 | 原始模型Dice | EMA Dice | EMA−原始 |
|---|---|---:|---:|---:|
{ema}

四项Dice的平均差异绝对值均小于0.0002，且正负不一；阈值0.1代理目标下EMA的增量为−0.000046。在当前配方下EMA没有带来可测的收益，因此不再调整EMA衰减系数。

20张图的重放结果中，EMA与原缓存概率的平均差异和最大差异均为0，E1缓存与E2重放的逐图指标一致，说明运行环境没有影响比较结果。

## 结果核对

源码、数据文件、五折划分和原80轮模型与上一轮一致。逐图表（7种配置×2个目标×20张图，共280行）、分折表（70行）和汇总表（14行）均已核对，clDice按逐图公式计算。

完整权重和概率保存在Kaggle上，本地只核对源码、表格和重放记录。

## 下一步：三组标签监督实验

三组分别为L0_clean、L1_proxy、L2_soft，统一使用512尺寸，其余设置、数据增强和采样方式保持一致，均取最终第80轮的EMA权重。先报告干净目标和代理目标在0.5阈值下的结果，再查看事先固定的OOF阈值校准和逐折表现。

重点关注L2：它用软边带处理边界的不确定性，同时约束干净的血管核心，预期能减少低阈值带来的精度损失。L1用于检验直接拟合转码目标的效果。clDice训练损失放在标签策略实验之后，避免同时改变多个因素。

运行方式：在受控消融Notebook中设置`RUN_LABEL_TRAINING=True`，其余设置不变，E1/E2的缓存会被复用；也可以直接运行`task4_label_supervision_5fold_kaggle.ipynb`，它只执行三组标签训练，需要挂载比赛数据并提供ImageNet预训练权重（或开启Internet）。

配对统计结果见`controlled_ablation_results_20261101/`目录下的`analysis_paired_summary.csv`、`analysis_paired_per_fold.csv`和`analysis_paired_per_image.csv`。
"""
    (ROOT / "受控消融E1E2结果分析_20261101.md").write_text(body, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
