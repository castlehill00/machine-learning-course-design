"""Audit the returned label ablation tables; never execute returned code or weights."""

import csv
import hashlib
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
ROOT = ROOT.parent if ROOT.name == "analysis" else ROOT  # 位于 analysis/ 时回到任务根目录
RESULT = ROOT / "label_supervision_results_20261102_bf57a639"
PREVIOUS = ROOT / "controlled_ablation_results_20261101"
VARIANTS = ("L0_clean", "L1_proxy", "L2_soft")
METRICS = (
    "dice",
    "precision",
    "recall",
    "centerline_precision",
    "centerline_recall",
    "cldice",
    "foreground_fraction",
    "thin_clean_centerline_recall",
)
GRID = (0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name):
    return json.loads((RESULT / name).read_text(encoding="utf-8"))


def table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(name, rows):
    with (RESULT / name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def close(a, b, tolerance=1e-10):
    assert math.isclose(float(a), float(b), rel_tol=0, abs_tol=tolerance), (a, b)


def reproduce_plans(train_ids, fold, cfg, clean):
    """Rebuild deterministic sampling records from original masks, without training."""
    positive = [np.argwhere(clean[i]) for i in train_ids]
    hashes = []
    for epoch in range(1, cfg["epochs"] + 1):
        plans = []
        for index in range(cfg["patches_per_epoch"]):
            rng = random.Random((cfg["seed"] + 1009 * fold) * 10**10 + epoch * 10**6 + index)
            i = rng.randrange(len(train_ids))
            patch = cfg["patch"]
            crop = rng.randint(int(patch * 0.9), min(int(patch * 1.1), 512))
            if rng.random() < 0.3 and len(positive[i]):
                y, x = positive[i][rng.randrange(len(positive[i]))]
                top = int(np.clip(y - crop // 2, 0, 512 - crop))
                left = int(np.clip(x - crop // 2, 0, 512 - crop))
            else:
                top, left = rng.randint(0, 512 - crop), rng.randint(0, 512 - crop)
            plans.append(
                (
                    i,
                    top,
                    left,
                    crop,
                    rng.randrange(4),
                    rng.random() < 0.5,
                    rng.uniform(0.85, 1.15),
                    rng.uniform(-0.04, 0.04),
                    rng.uniform(0.85, 1.15),
                    [rng.uniform(0.95, 1.05) for _ in range(3)],
                )
            )
        hashes.append(hashlib.sha256(json.dumps(plans, sort_keys=True).encode()).hexdigest())
    return hashes


def main():
    checks = {}
    config, status, audit = (
        load("ablation_config.json"),
        load("experiment_status.json"),
        load("label_audit.json"),
    )
    cfg = config["config"]
    for name, key in [
        ("task4_controlled_ablation.py", "source_sha256"),
        ("task4_kaggle_v2.py", "v2_source_sha256"),
        ("task4_pipeline.py", "base_source_sha256"),
    ]:
        assert sha(RESULT / "runtime_source" / name) == sha(ROOT / name) == config[key]
    notebook = json.loads(
        (ROOT / "notebooks" / "task4_label_supervision_5fold_kaggle.ipynb").read_text(
            encoding="utf-8"
        )
    )
    # The previously checked delivery records bind this notebook to the same core sources.
    delivery = json.loads(
        (ROOT / "label_supervision_notebook_checks.json").read_text(encoding="utf-8")
    )
    assert notebook["nbformat"] == 4
    checks["returned_sources_match_local_delivered_sources"] = True
    checks["delivery_check_record_present"] = bool(delivery)
    assert cfg == dict(
        seed=42,
        folds=5,
        fold=0,
        epochs=80,
        patch=256,
        patches_per_epoch=256,
        batch_size=8,
        encoder_lr=0.0001,
        decoder_lr=0.0007,
        weight_decay=0.0001,
        warmup_epochs=3,
        ema_decay=0.99,
        monitor_every=10,
        threads=4,
        amp=True,
        pretrained=True,
        tta_gain_required=0.001,
        soft_boundary=0.5,
    )
    assert (
        status["label_training_requested"]
        and not status["inference_done"]
        and not status["raw_replayed"]
    )
    assert (
        status["requested_folds"] == list(range(5))
        and tuple(status["requested_variants"]) == VARIANTS
    )
    assert not any(
        status[k]
        for k in ["competition_csv_generated", "public_scores_used", "automatic_model_selection"]
    )
    assert (
        tuple(audit["complete_variants"]) == VARIANTS
        and tuple(audit["requested_variants"]) == VARIANTS
    )
    assert audit["complete_all_requested"] and audit["no_automatic_winner"]
    assert not audit["competition_csv_generated"] and not audit["public_scores_used"]
    assert tuple(audit["calibration_grid"]) == GRID and len(audit["pairing"]) == 5
    assert all(
        r["fold"] == f
        and tuple(r["variants"]) == VARIANTS
        and r["same_initial_weights"]
        and r["same_rgb_plan"]
        for f, r in enumerate(audit["pairing"])
    )
    checks["three_variants_five_folds_80_epochs_preset_no_public_score_selection"] = True

    manifest = table(RESULT / "manifest.csv")
    assignment = {r["id"]: int(r["fold"]) for r in manifest if r["split"] == "train"}
    assert len(manifest) == 40 and set(assignment) == {str(i) for i in range(1, 21)}
    previous_manifest = table(PREVIOUS / "manifest.csv")
    assert manifest == previous_manifest
    clean, truth, stable = {}, {}, []
    for row in manifest:
        data = ROOT / "2026task-4/segmentation"
        assert sha(data / row["split"] / "image" / f'{row["id"]}.jpg') == row["image_sha256"]
        if row["split"] == "train":
            path = data / "train/label" / f'{row["id"]}.jpg'
            assert sha(path) == row["label_sha256"]
            with Image.open(path) as image:
                image = image.convert("L")
                c = cv2.resize(
                    (np.asarray(image) > 127).astype(np.uint8),
                    (512, 512),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                p = np.asarray(image.resize((512, 512), Image.Resampling.BILINEAR)) > 0
            assert not np.any(c & ~p)
            clean[row["id"]] = c
            truth[row["id"]] = {"clean": int(c.sum()), "proxy": int(p.sum())}
        item = {
            k: row[k]
            for k in ["split", "id", "width", "height", "image_sha256", "label_sha256", "fold"]
        }
        if row["split"] == "train":
            item["fold"] = int(item["fold"])
        stable.append(item)
    fingerprint = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
    assert (
        fingerprint == config["data_fingerprint"] == load("audit_summary.json")["data_fingerprint"]
    )
    checks["original_data_hashes_frozen_folds_and_both_target_counts_verified"] = True

    histories, records = {}, {}
    seconds = 0
    for fold in range(5):
        for variant in VARIANTS:
            folder = RESULT / variant / f"fold{fold}"
            done = json.loads((folder / "completed.json").read_text())
            history = table(folder / "history.csv")
            assert done["epoch"] == 80 and done["variant"] == variant
            assert set(done["val_ids"]) == {i for i, f in assignment.items() if f == fold}
            assert set(done["train_ids"]) == set(assignment) - set(done["val_ids"])
            assert len(done["train_ids"]) == 16 and len(done["val_ids"]) == 4
            assert len(history) == 80 and [int(r["epoch"]) for r in history] == list(range(1, 81))
            assert sha(folder / "history.csv") == done["sha256"]["history.csv"]
            assert done["plan_hashes"] == [r["plan_sha256"] for r in history]
            for r in history:
                assert math.isfinite(float(r["loss"])) and float(r["loss"]) > 0
                seconds += float(r["seconds"])
                monitored = int(r["epoch"]) % 10 == 0
                assert bool(r["monitor_clean_05"]) == bool(r["monitor_proxy_05"]) == monitored
            records[variant, fold], histories[variant, fold] = done, history
        first = records[VARIANTS[0], fold]
        for variant in VARIANTS[1:]:
            done = records[variant, fold]
            assert done["initial_model_hash"] == first["initial_model_hash"]
            assert done["train_ids"] == first["train_ids"] and done["val_ids"] == first["val_ids"]
            assert done["plan_hashes"] == first["plan_hashes"]
        assert len({records[v, fold]["sha256"]["final_ema.pt"] for v in VARIANTS}) == 3
        assert reproduce_plans(first["train_ids"], fold, cfg, clean) == first["plan_hashes"]
    checks["all_1200_logged_epochs_history_hashes_and_15_fold_partitions_verified"] = True
    checks["matched_initial_weights_and_all_400_sampling_plan_hashes_rebuilt_from_data"] = True
    assert (
        load("pretrained_source.json")["sha256"]
        == "b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f"
    )

    rows = table(RESULT / "label_per_image.csv")
    folds = table(RESULT / "label_per_fold.csv")
    summary = table(RESULT / "label_summary.csv")
    experiments = {
        f"{v}_{mode}_t{t:g}" for v in VARIANTS for mode in ["none", "d4"] for t in [0.1, 0.4, 0.5]
    }
    lookup = {(r["experiment"], r["target"], r["id"]): r for r in rows}
    fs = {(r["experiment"], r["target"], int(r["fold"])): r for r in folds}
    ss = {(r["experiment"], r["target"]): r for r in summary}
    assert (
        len(rows) == len(lookup) == 720
        and len(folds) == len(fs) == 180
        and len(summary) == len(ss) == 36
    )
    assert {r["experiment"] for r in rows} == experiments
    for r in rows:
        assert int(r["fold"]) == assignment[r["id"]]
        assert all(math.isfinite(float(r[k])) and 0 <= float(r[k]) <= 1 for k in METRICS)
        # Reconcile exact foreground/true-positive counts with the local original labels.
        n_pred = round(float(r["foreground_fraction"]) * 512 * 512)
        close(r["foreground_fraction"], n_pred / (512 * 512))
        n_true = truth[r["id"]][r["target"]]
        tp = round(float(r["recall"]) * n_true)
        close(r["recall"], tp / n_true)
        close(r["precision"], tp / n_pred)
        close(r["dice"], (2 * tp + 1e-7) / (n_pred + n_true + 1e-7))
        a, b = float(r["centerline_precision"]), float(r["centerline_recall"])
        close(r["cldice"], 2 * a * b / (a + b) if a + b else 0)
    for experiment in experiments:
        for target in ["clean", "proxy"]:
            group = [lookup[experiment, target, str(i)] for i in range(1, 21)]
            s = ss[experiment, target]
            assert s["complete_20_image_oof"] == "True" and int(s["n_images"]) == 20
            for k in METRICS:
                close(s[k], np.mean([float(r[k]) for r in group]))
            scores = []
            for fold in range(5):
                subset = [r for r in group if int(r["fold"]) == fold]
                f = fs[experiment, target, fold]
                assert len(subset) == int(f["n_images"]) == 4
                for k in METRICS:
                    close(f[k], np.mean([float(r[k]) for r in subset]))
                scores.append(float(f["dice"]))
            close(s["worst_fold_dice"], min(scores))
            close(s["fold_dice_std"], np.std(scores))
            reported = next(
                r
                for r in status["label_summary"]
                if r["experiment"] == experiment and r["target"] == target
            )
            for k in METRICS:
                close(s[k], reported[k])
        for i in assignment:
            for k in ["foreground_fraction", "thin_clean_centerline_recall"]:
                close(lookup[experiment, "clean", i][k], lookup[experiment, "proxy", i][k])
    for variant in VARIANTS:
        for fold in range(5):
            for target in ["clean", "proxy"]:
                close(
                    histories[variant, fold][-1][f"monitor_{target}_05"],
                    fs[f"{variant}_none_t0.5", target, fold]["dice"],
                )
    checks["all_720_image_metrics_180_fold_metrics_36_summaries_and_final_monitors_verified"] = True
    checks["reported_pixel_counts_precision_recall_dice_reconcile_with_original_targets"] = True

    grid, calibration = table(RESULT / "label_threshold_grid.csv"), table(
        RESULT / "label_calibration_development.csv"
    )
    assert len(grid) == 156 and len(calibration) == 12
    grid_lookup = {(r["variant"], r["mode"], r["target"], float(r["threshold"])): r for r in grid}
    assert len(grid_lookup) == 156
    calibrated = {(r["variant"], r["mode"], r["target"]): r for r in calibration}
    for variant in VARIANTS:
        for mode in ["none", "d4"]:
            for target in ["clean", "proxy"]:
                group = [grid_lookup[variant, mode, target, t] for t in GRID]
                for r in group:
                    values = [float(r[f"fold{f}"]) for f in range(5)]
                    close(r["mean_dice"], np.mean(values))
                    close(r["worst_fold_dice"], min(values))
                    t = float(r["threshold"])
                    if t in [0.1, 0.4, 0.5]:
                        close(r["mean_dice"], ss[f"{variant}_{mode}_t{t:g}", target]["dice"])
                        for fold in range(5):
                            close(
                                r[f"fold{fold}"],
                                fs[f"{variant}_{mode}_t{t:g}", target, fold]["dice"],
                            )
                best = max(
                    group, key=lambda r: (float(r["mean_dice"]), -abs(float(r["threshold"]) - 0.5))
                )
                report = calibrated[variant, mode, target]
                for k in best:
                    assert report[k] == best[k]
                assert "not nested" in report["selection_caveat"]
    checks["all_156_grid_rows_and_12_development_threshold_choices_verified"] = True

    previous = table(PREVIOUS / "replay_per_image.csv")
    lookup.update({(r["experiment"], r["target"], r["id"]): r for r in previous})
    comparisons = [
        ("L2_vs_L0_fixed05", "L2_soft_d4_t0.5", "L0_clean_d4_t0.5"),
        ("L1_vs_L0_fixed05", "L1_proxy_d4_t0.5", "L0_clean_d4_t0.5"),
        ("L2_vs_L0_clean_calibrated", "L2_soft_d4_t0.5", "L0_clean_d4_t0.4"),
        ("L1_vs_original_proxy_candidates", "L1_proxy_d4_t0.4", "E1_ema_t01"),
        ("L2_vs_original_clean_candidates", "L2_soft_d4_t0.5", "E1_ema_t04"),
    ]
    comparisons += [(f"{v}_D4_vs_none_fixed05", f"{v}_d4_t0.5", f"{v}_none_t0.5") for v in VARIANTS]
    paired, paired_fold, paired_summary = [], [], []
    for name, left, right in comparisons:
        for target in ["clean", "proxy"]:
            for key in METRICS:
                deltas = []
                for i in sorted(assignment, key=int):
                    a, b = lookup[left, target, i], lookup[right, target, i]
                    assert a["fold"] == b["fold"]
                    delta = float(a[key]) - float(b[key])
                    deltas.append(delta)
                    paired.append(
                        dict(
                            comparison=name,
                            target=target,
                            metric=key,
                            id=i,
                            fold=assignment[i],
                            delta=delta,
                        )
                    )
                fold_deltas = []
                for fold in range(5):
                    delta = float(
                        np.mean(
                            [
                                d
                                for i, d in zip(sorted(assignment, key=int), deltas)
                                if assignment[i] == fold
                            ]
                        )
                    )
                    fold_deltas.append(delta)
                    paired_fold.append(
                        dict(comparison=name, target=target, metric=key, fold=fold, delta=delta)
                    )
                paired_summary.append(
                    dict(
                        comparison=name,
                        left=left,
                        right=right,
                        target=target,
                        metric=key,
                        mean_delta=float(np.mean(deltas)),
                        positive_images=sum(d > 0 for d in deltas),
                        negative_images=sum(d < 0 for d in deltas),
                        positive_folds=sum(d > 0 for d in fold_deltas),
                        negative_folds=sum(d < 0 for d in fold_deltas),
                        min_fold_delta=min(fold_deltas),
                        max_fold_delta=max(fold_deltas),
                    )
                )
    write("analysis_paired_per_image.csv", paired)
    write("analysis_paired_per_fold.csv", paired_fold)
    write("analysis_paired_summary.csv", paired_summary)

    calibration_pairs = []
    for left, right in [("L2_soft", "L0_clean"), ("L1_proxy", "L0_clean"), ("L1_proxy", "L2_soft")]:
        for target in ["clean", "proxy"]:
            a, b = calibrated[left, "d4", target], calibrated[right, "d4", target]
            values = [float(a[f"fold{f}"]) - float(b[f"fold{f}"]) for f in range(5)]
            calibration_pairs.append(
                dict(
                    left=left,
                    right=right,
                    target=target,
                    left_threshold=a["threshold"],
                    right_threshold=b["threshold"],
                    mean_delta=float(np.mean(values)),
                    positive_folds=sum(v > 0 for v in values),
                    worst_fold_delta=float(a["worst_fold_dice"]) - float(b["worst_fold_dice"]),
                    **{f"fold{f}_delta": v for f, v in enumerate(values)},
                    caveat="OOF-selected thresholds; descriptive development comparison, not independent validation",
                )
            )
    write("analysis_calibrated_paired_folds.csv", calibration_pairs)
    difficult = []
    for experiment in experiments:
        for target in ["clean", "proxy"]:
            r = lookup[experiment, target, "12"]
            difficult.append({k: r[k] for k in ["experiment", "target", "id", "fold", *METRICS]})
    write("analysis_image12.csv", sorted(difficult, key=lambda r: (r["experiment"], r["target"])))

    checks["light_package_contains_no_weights_probabilities_or_competition_submission"] = (
        not list(RESULT.rglob("*.pt"))
        and not list(RESULT.rglob("*.npz"))
        and not list(RESULT.glob("*submission*.csv"))
    )
    assert checks["light_package_contains_no_weights_probabilities_or_competition_submission"]
    result = dict(
        archive_sha256=load("received_archive.json")["sha256"],
        checks=checks,
        actual_training_log_seconds=seconds,
        training_monitor_minutes=seconds / 60,
        checkpoint_saving_and_final_inference_included_in_time=False,
        local_actual_weight_or_probability_replay=False,
        comparisons=paired_summary,
        calibrated_fold_comparisons=calibration_pairs,
        recommendation="Keep original candidates; no new competition submission justified by this experiment.",
        public_scores_used=False,
        private_score=None,
        new_training_started_locally=False,
        new_competition_submission_generated=False,
    )
    (RESULT / "analysis_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(ss, calibrated, lookup, paired_summary, calibration_pairs, seconds, result)
    print(
        json.dumps(
            {
                "checks": checks,
                "training_monitor_minutes": seconds / 60,
                "report": str(ROOT / "标签监督消融结果分析_20261102.md"),
                "calibrated_fold_comparisons": calibration_pairs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def write_report(summary, calibrated, lookup, comparisons, calibrated_pairs, seconds, result):
    def score(experiment, target, metric="dice"):
        return float(summary[experiment, target][metric])

    fixed = "\n".join(
        f"| {v} | {score(v+'_d4_t0.5','clean'):.6f} | {score(v+'_d4_t0.5','proxy'):.6f} | "
        f"{score(v+'_d4_t0.5','clean','cldice'):.6f} | {score(v+'_d4_t0.5','clean','thin_clean_centerline_recall'):.2%} | "
        f"{score(v+'_d4_t0.5','clean','foreground_fraction'):.2%} |"
        for v in VARIANTS
    )
    calibrated_table = "\n".join(
        f"| {v} | {target} | {calibrated[v,'d4',target]['threshold']} | "
        f"{float(calibrated[v,'d4',target]['mean_dice']):.6f} | {float(calibrated[v,'d4',target]['worst_fold_dice']):.6f} |"
        for v in VARIANTS
        for target in ["clean", "proxy"]
    )
    fold_table = "\n".join(
        f"| {fold} | "
        + " | ".join(
            f"{float(next(r for r in calibrated_pairs if r['left']==a and r['right']==b and r['target']==t)[f'fold{fold}_delta']):+.6f}"
            for a, b, t in [("L2_soft", "L0_clean", "clean"), ("L1_proxy", "L0_clean", "proxy")]
        )
        + " |"
        for fold in range(5)
    )
    original_proxy = [lookup["E1_ema_t01", "proxy", str(i)] for i in range(1, 21)]
    old_folds = [
        np.mean([float(r["dice"]) for r in original_proxy if int(r["fold"]) == f]) for f in range(5)
    ]
    threshold_rows = table(RESULT / "label_threshold_grid.csv")
    l1_plateau = "、".join(
        f"{float(next(r for r in threshold_rows if r['variant']=='L1_proxy' and r['mode']=='d4' and r['target']=='proxy' and float(r['threshold'])==t)['mean_dice']):.6f}"
        for t in [0.3, 0.4, 0.5, 0.6]
    )
    original_clean = float(
        np.mean([float(lookup["E1_ema_t01", "clean", str(i)]["dice"]) for i in range(1, 21)])
    )
    text = f"""# 任务4：标签监督消融结果分析

分析日期：2026-11-02。结果文件`ablation_analysis_results (1).zip`（SHA256 `{result['archive_sha256']}`），为L0/L1/L2三组正式训练的结果。

## 主要结论

**本轮实验得到了明确结论，但不需要生成新的提交。** 在相同512设置下，L2在固定阈值下优于L0；L1明显提升了同系列的转码代理目标，而且不再依赖0.1这样低的阈值。但新系列没有在两个目标上同时超过原有的原尺寸训练候选，因此最终方案不变。这一判断依据OOF结果和逐折稳定性。

L2是有效的改进：阈值校准后干净目标Dice的提升依然存在，只是最差折略有下降。L1说明监督目标的定义对结果影响很大。

## 实验设置

- L0_clean、L1_proxy、L2_soft均完成5折×80轮训练，每折16张训练、4张验证，每张图只由未见过它的那一折模型评价。三组统一使用512尺寸RGB输入，模型、初始化、数据增强和采样方式相同，均取最终第80轮的EMA权重。
- L0：BCE(C)+Dice(C)；L1：BCE(P)+Dice(P)；L2：BCE(C+0.5×(P−C))+Dice(C)。BCE与Dice等权，辅助输出头的设置不变。L2只在BCE项中使用软标签。
- C为干净目标：原JPEG灰度>127后最近邻缩放；P为转码代理目标：原JPEG经官方PIL双线性缩放到512后取>0。两个目标分别评价。
- 运行环境：Tesla T4、Python 3.13.15、torch 2.11.0+cu128、torchvision 0.26.0+cu128、Pillow 13.0.0、skimage 0.25.2，与E1/E2一致。
- 逐轮训练与监控时间合计{seconds/60:.2f}分钟（不含检查点保存和最终D4推理）。

## 固定阈值0.5下的结果

下表均为D4推理、20张图的逐图平均值；最后两列与评价目标无关。

| 监督策略 | 干净Dice | 代理Dice | 干净clDice | 干净细中心线召回 | 前景面积 |
|---|---:|---:|---:|---:|---:|
{fixed}

L2相对L0：干净Dice提高0.009804，16/20张图、5/5折提升；干净clDice提高0.022140，20/20张图、5/5折提升；细中心线召回提高8.301个百分点，20/20张图提升。同时干净像素精确率下降约2.997个百分点，召回提高约5.044个百分点，即提升伴随着精确率与召回之间的取舍。

L1相对L0：固定0.5阈值下代理Dice提高0.130012，20/20张图、5/5折提升，但干净Dice下降0.031304。这说明学习目标改变了预测的边界和覆盖范围；由于固定阈值还受概率校准影响，下面再看阈值校准后的结果。

细中心线定义为512网格上C的骨架中距背景不超过1.5像素的点。clDice基于骨架覆盖计算。

## 阈值校准后的结果

阈值网格在运行前固定为0.05、0.075、0.1、0.125、0.15、0.2、0.3、0.4、0.5、0.6、0.7、0.8、0.9，选取平均Dice最大的阈值，完全相同时取更接近0.5者。

| 策略 | 评价目标 | 开发阈值 | 开发OOF Dice | 最差折Dice |
|---|---|---:|---:|---:|
{calibrated_table}

L2在干净目标上为0.784055，L0校准后为0.776996，提高0.007059，4/5折提高；但最差折从0.744165降到0.742793（−0.001373）。因此固定0.5下五折全部提升的结论，在阈值校准后的最差折上不完全成立。

L1在代理目标上为0.824917，L0校准后为0.767161，提高0.057756，5/5折提高。L1在阈值0.3、0.4、0.5、0.6下的代理Dice分别为{l1_plateau}，最优点附近比较平缓；L2在代理目标上最佳为0.797158，低于L1。L0的代理最佳阈值落在网格下界0.05。

| 折 | L2−L0，干净目标各自开发阈值 | L1−L0，代理目标各自开发阈值 |
|---|---:|---:|
{fold_table}

以上为在同一批OOF上选择阈值后的开发集结果；五折模型的训练集互有重叠，样本只有20张。

## 与原有候选的比较

原有候选（原尺寸训练、D4、阈值0.1）：代理Dice {np.mean([float(r['dice']) for r in original_proxy]):.6f}，干净Dice {original_clean:.6f}，最差代理折 {min(old_folds):.6f}。

L1（D4、阈值0.4）：代理Dice 0.824917，干净Dice 0.728847，最差代理折0.796798。与原候选相比，代理Dice平均下降0.002152，只有8/20张图提高；干净Dice下降0.016215，20/20张图下降。代理Dice的逐折差值为+0.000408、+0.002265、−0.005606、+0.001616、−0.009445，困难的第4折退步最多。代理clDice虽略升0.002162，但整体并未优于原候选。

L2在干净目标上的最佳为0.784055，原有干净候选（阈值0.4）为0.787453。L2的干净clDice更高，但干净细中心线召回比原候选低约1.632个百分点，综合来看不替换原候选。

需要注意，这是两套完整流程之间的比较，而不是单因素消融：原系列使用原尺寸训练和推理，新系列使用512尺寸RGB输入，采样和重采样方式也不同。标签监督方式的单因素比较应以新系列内部的三组为准。

12号困难图保留在所有统计中，单独结果见`analysis_image12.csv`。

## 后续计划

1. 保留原有的阈值0.1和0.4两个候选，本轮不生成新的测试预测和提交。
2. 下一项实验在**L2_soft相同的512配方上**只增加0.1×soft-clDice损失，骨架监督使用干净核心C，以本次L2为对照，其余设置不变，用来检验在软边带的基础上结构损失能否带来额外提升。
3. 继续同时报告C/P两个目标、固定0.5与阈值网格结果、逐折和逐图结果、最差折以及精确率和召回；重点关注L2在第4折上的表现和细中心线缺失。
4. 如需进一步提升竞赛成绩，可利用完整的概率和误差图分析不同模型的互补性；新的融合规则需事先固定，并只用对应折的OOF评估。

## 结果核对

已核对原始数据文件、五折划分、15份完成记录、1200条逐轮记录以及三组每折的初始化；在本地根据C重新生成的400个逐轮RGB采样计划与运行记录完全一致。

逐图表（720行）、逐折表（180行）、36个固定阈值汇总及156条网格结果均已核对，均值、最差折和标准差计算无误；由前景像素数还原的TP与预测前景数与Dice、clDice的公式一致。

模型权重和逐图预测mask保存在Kaggle的完整输出中。

结果目录：`{RESULT.name}`。配对统计见`analysis_paired_summary.csv`、`analysis_paired_per_fold.csv`、`analysis_paired_per_image.csv`，阈值校准后的分折对照见`analysis_calibrated_paired_folds.csv`。
"""
    (ROOT / "标签监督消融结果分析_20261102.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
