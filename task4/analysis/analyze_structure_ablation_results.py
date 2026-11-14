"""Audit formal structure ablation, including lossless D4@0.5 mask panels.

Returned Python files are hashed, never executed; no checkpoint is loaded.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROOT = ROOT.parent if ROOT.name == "analysis" else ROOT  # 位于 analysis/ 时回到任务根目录
sys.path.insert(0, str(ROOT / ".ablation_test_deps"))

import csv
import hashlib
import json
import math

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

RESULT = ROOT / "structure_ablation_results_20261102_d1ab08bb"
LABEL = ROOT / "label_supervision_results_20261102_bf57a639"
METHODS = ("L2_reference", "T1_L2_cldice")
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


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def table(p):
    with Path(p).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def close(a, b, tolerance=1e-10):
    assert math.isclose(float(a), float(b), rel_tol=0, abs_tol=tolerance), (a, b)


def write(name, rows):
    with (RESULT / name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_from_mask(mask, truth, sc, sp, thin):
    sk = skeletonize(mask)

    def div(a, b):
        return float(a / b) if b else 0.0

    out = {}
    for target, gt, sg in [("clean", truth[0], sc), ("proxy", truth[1], sp)]:
        tp = int((mask & gt).sum())
        precision = div(tp, int(mask.sum()))
        recall = div(tp, int(gt.sum()))
        cp = div(int((sk & gt).sum()), int(sk.sum()))
        cr = div(int((sg & mask).sum()), int(sg.sum()))
        out[target] = dict(
            dice=float((2 * tp + 1e-7) / (mask.sum() + gt.sum() + 1e-7)),
            precision=precision,
            recall=recall,
            centerline_precision=cp,
            centerline_recall=cr,
            cldice=2 * cp * cr / (cp + cr) if cp + cr else 0.0,
            foreground_fraction=float(mask.mean()),
            thin_clean_centerline_recall=div(int((mask & thin).sum()), int(thin.sum())),
        )
    return out


def binary_panel(rgb):
    assert rgb.shape == (512, 512, 3)
    assert np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(rgb[:, :, 0], rgb[:, :, 2])
    assert set(np.unique(rgb[:, :, 0])) <= {0, 255}
    return rgb[:, :, 0] == 255


def main():
    checks = {}
    config = load(RESULT / "structure_config.json")
    cfg = config["config"]
    status = load(RESULT / "structure_status.json")
    reference = load(RESULT / "reference_audit.json")
    expected_config = load(LABEL / "ablation_config.json")
    old_records = {str(f): load(LABEL / f"L2_soft/fold{f}/completed.json") for f in range(5)}
    assert reference["reference_records"] == old_records
    assert config["reference_expected_config"] == expected_config
    assert (
        config["reference_expected_records_hash"]
        == hashlib.sha256(json.dumps(old_records, sort_keys=True).encode()).hexdigest()
    )
    for name, key in [
        ("task4_cldice_ablation.py", "source_sha256"),
        ("task4_controlled_ablation.py", "reference_source_sha256"),
        ("task4_kaggle_v2.py", "v2_source_sha256"),
        ("task4_pipeline.py", "base_source_sha256"),
    ]:
        assert sha(ROOT / name) == sha(RESULT / "runtime_source" / name) == config[key]
    assert {k: v for k, v in cfg.items() if k in expected_config["config"]} == expected_config[
        "config"
    ]
    assert {k: v for k, v in cfg.items() if k not in expected_config["config"]} == dict(
        cldice_weight=0.1, skeleton_iterations=10, skeleton_smooth=1.0
    )
    assert not config["engineering_only"] and not status["engineering_only"]
    assert (
        status["complete_folds"] == status["requested_folds"] == list(range(5))
        and status["complete_20_image_oof"]
    )
    assert (
        status["paired_initialization_and_plans"]
        and status["new_training_arms"] == 1
        and not status["reference_retrained"]
    )
    for key in [
        "automatic_model_selection",
        "public_scores_used",
        "competition_csv_generated",
        "test_predictions_generated",
    ]:
        assert status[key] is False
    assert reference["all_five_probability_hashes_verified"]
    assert all(v == 0 for v in reference["baseline_metric_max_absolute_drift"].values())
    env = load(RESULT / "execution_environment.json")
    assert env == load(LABEL / "execution_environment.json") == reference["reference_environment"]
    assert reference["environment_differences"] == status["environment_differences"] == {}
    checks["all_four_sources_frozen_recipe_original_L2_lineage_and_recorded_environment_match"] = (
        True
    )

    manifest = table(RESULT / "manifest.csv")
    assert manifest == table(LABEL / "manifest.csv")
    assignment = {r["id"]: int(r["fold"]) for r in manifest if r["split"] == "train"}
    assert len(manifest) == 40 and len(assignment) == 20
    stable = []
    truth = {}
    data = ROOT / "2026task-4/segmentation"
    for r in manifest:
        assert sha(data / r["split"] / "image" / f'{r["id"]}.jpg') == r["image_sha256"]
        if r["split"] == "train":
            path = data / "train/label" / f'{r["id"]}.jpg'
            assert sha(path) == r["label_sha256"]
            with Image.open(path) as im:
                im = im.convert("L")
                c = cv2.resize(
                    (np.asarray(im) > 127).astype(np.uint8),
                    (512, 512),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                p = np.asarray(im.resize((512, 512), Image.Resampling.BILINEAR)) > 0
            assert not (c & ~p).any()
            truth[r["id"]] = (c, p)
        row = {
            k: r[k]
            for k in ["split", "id", "width", "height", "image_sha256", "label_sha256", "fold"]
        }
        if row["split"] == "train":
            row["fold"] = int(row["fold"])
        stable.append(row)
    fp = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
    assert fp == config["data_fingerprint"] == expected_config["data_fingerprint"]
    checks["all_original_data_hashes_frozen_folds_and_target_definitions_verified"] = True

    histories = {}
    new_seconds = 0.0
    reference_seconds = 0.0
    for fold in range(5):
        folder = RESULT / "T1_L2_cldice" / f"fold{fold}"
        done = load(folder / "completed.json")
        old = old_records[str(fold)]
        assert done["epoch"] == 80 and done["method"] == "T1_L2_cldice" and done["config"] == cfg
        for key in ["initial_model_hash", "train_ids", "val_ids", "plan_hashes"]:
            assert done[key] == old[key]
        assert set(done["val_ids"]) == {i for i, f in assignment.items() if f == fold}
        assert set(done["train_ids"]) == set(assignment) - set(done["val_ids"])
        assert (
            len(done["plan_hashes"]) == 80
            and sha(folder / "history.csv") == done["sha256"]["history.csv"]
        )
        rows = table(folder / "history.csv")
        histories[fold] = rows
        assert [int(r["epoch"]) for r in rows] == list(range(1, 81))
        assert [r["plan_sha256"] for r in rows] == done["plan_hashes"]
        for r in rows:
            assert all(
                math.isfinite(float(r[k])) and float(r[k]) >= 0
                for k in ["loss", "reference_loss", "soft_cldice_loss", "seconds"]
            )
            close(r["loss"], float(r["reference_loss"]) + 0.1 * float(r["soft_cldice_loss"]), 1e-6)
            monitored = int(r["epoch"]) % 10 == 0
            assert bool(r["monitor_clean_05"]) == bool(r["monitor_proxy_05"]) == monitored
            new_seconds += float(r["seconds"])
        reference_seconds += sum(
            float(r["seconds"]) for r in table(LABEL / f"L2_soft/fold{fold}/history.csv")
        )
    checks["five_final80_folds_all_400_history_rows_loss_arithmetic_and_pairing_verified"] = True

    rows = table(RESULT / "structure_per_image.csv")
    fold_rows = table(RESULT / "structure_per_fold.csv")
    summary = table(RESULT / "structure_summary.csv")
    lookup = {(r["experiment"], r["target"], r["id"]): r for r in rows}
    fs = {(r["experiment"], r["target"], int(r["fold"])): r for r in fold_rows}
    ss = {(r["experiment"], r["target"]): r for r in summary}
    experiments = {
        f"{m}_{mode}_t{t:g}" for m in METHODS for mode in ["none", "d4"] for t in [0.1, 0.4, 0.5]
    }
    assert (
        len(rows) == len(lookup) == 480
        and len(fold_rows) == len(fs) == 120
        and len(summary) == len(ss) == 24
    )
    assert {r["experiment"] for r in rows} == experiments
    for r in rows:
        assert int(r["fold"]) == assignment[r["id"]]
        assert all(math.isfinite(float(r[k])) and 0 <= float(r[k]) <= 1 for k in METRICS)
        gt = truth[r["id"]][0 if r["target"] == "clean" else 1]
        n_pred = round(float(r["foreground_fraction"]) * 512 * 512)
        tp = round(float(r["recall"]) * int(gt.sum()))
        close(r["foreground_fraction"], n_pred / (512 * 512))
        close(r["precision"], tp / n_pred)
        close(r["recall"], tp / int(gt.sum()))
        close(r["dice"], (2 * tp + 1e-7) / (n_pred + int(gt.sum()) + 1e-7))
        a, b = float(r["centerline_precision"]), float(r["centerline_recall"])
        close(r["cldice"], 2 * a * b / (a + b) if a + b else 0.0)
    for e in experiments:
        for target in ["clean", "proxy"]:
            group = [lookup[e, target, str(i)] for i in range(1, 21)]
            s = ss[e, target]
            assert s["complete_20_image_oof"] == "True" and int(s["n_images"]) == 20
            for k in METRICS:
                close(s[k], np.mean([float(r[k]) for r in group]))
            scores = []
            for fold in range(5):
                subset = [r for r in group if int(r["fold"]) == fold]
                f = fs[e, target, fold]
                assert len(subset) == int(f["n_images"]) == 4
                for k in METRICS:
                    close(f[k], np.mean([float(r[k]) for r in subset]))
                scores.append(float(f["dice"]))
            close(s["worst_fold_dice"], min(scores))
            close(s["fold_dice_std"], np.std(scores))
            saved = next(
                r for r in status["summary"] if r["experiment"] == e and r["target"] == target
            )
            for k in METRICS:
                close(s[k], saved[k])
        for i in assignment:
            for k in ["foreground_fraction", "thin_clean_centerline_recall"]:
                close(lookup[e, "clean", i][k], lookup[e, "proxy", i][k])
    for fold in range(5):
        for target in ["clean", "proxy"]:
            close(
                histories[fold][-1][f"monitor_{target}_05"],
                fs["T1_L2_cldice_none_t0.5", target, fold]["dice"],
            )
    for r in table(LABEL / "label_per_image.csv"):
        if r["experiment"].startswith("L2_soft_"):
            a = lookup[r["experiment"].replace("L2_soft", "L2_reference", 1), r["target"], r["id"]]
            for k in METRICS:
                close(a[k], r[k])
    checks["480_image_rows_120_fold_rows_24_summaries_and_original_L2_metrics_verified"] = True

    # These PNG panels retain all mask pixels at native 512x512, so they permit
    # independent binary-metric recomputation at this one mode/threshold.
    recomputed = []
    all_rgb_equal = True
    for i in sorted(assignment, key=int):
        with Image.open(RESULT / f"oof_error_previews/oof_{int(i):02d}.png") as im:
            canvas = np.asarray(im.convert("RGB"))
        assert canvas.shape == (1080, 2048, 3)
        c, p = truth[i]
        assert np.array_equal(binary_panel(canvas[28:540, 512:1024]), c)
        assert np.array_equal(binary_panel(canvas[28:540, 1024:1536]), p)
        masks = [binary_panel(canvas[568:1080, 0:512]), binary_panel(canvas[568:1080, 512:1024])]
        with Image.open(data / "train/image" / f"{i}.jpg") as im:
            rgb = cv2.resize(
                np.asarray(im.convert("RGB")), (512, 512), interpolation=cv2.INTER_AREA
            )
        all_rgb_equal &= np.array_equal(rgb, canvas[28:540, 0:512])
        sc, sp = skeletonize(c), skeletonize(p)
        thin = sc & (cv2.distanceTransform(c.astype(np.uint8), cv2.DIST_L2, 5) <= 1.5)
        for method, mask, x in zip(METHODS, masks, [1024, 1536]):
            color = np.repeat(((mask & c).astype(np.uint8) * 255)[:, :, None], 3, axis=2)
            color[mask & ~c] = [235, 65, 65]
            color[c & ~mask] = [70, 125, 245]
            assert np.array_equal(color, canvas[568:1080, x : x + 512])
            values = metric_from_mask(mask, (c, p), sc, sp, thin)
            for target, v in values.items():
                original = lookup[method + "_d4_t0.5", target, i]
                for k in METRICS:
                    close(v[k], original[k])
                recomputed.append(dict(method=method, id=i, fold=assignment[i], target=target, **v))
        a, b = masks
        changes = np.repeat(((a & b).astype(np.uint8) * 255)[:, :, None], 3, axis=2)
        changes[b & ~a] = [70, 200, 100]
        changes[a & ~b] = [230, 110, 70]
        assert np.array_equal(changes, canvas[28:540, 1536:2048])
    assert all_rgb_equal
    write("analysis_recomputed_d4_t05_from_png.csv", recomputed)
    checks["all20_lossless_png_masks_targets_RGB_and_error_panels_verified"] = True
    checks["independently_recomputed_all80_d4_t05_image_target_metric_rows_from_png"] = True

    grid = table(RESULT / "structure_threshold_grid.csv")
    cal = table(RESULT / "structure_calibration_development.csv")
    gg = {(r["method"], r["mode"], r["target"], float(r["threshold"])): r for r in grid}
    cc = {(r["method"], r["mode"], r["target"]): r for r in cal}
    assert len(grid) == len(gg) == 104 and len(cal) == len(cc) == 8
    for m in METHODS:
        for mode in ["none", "d4"]:
            for target in ["clean", "proxy"]:
                group = [gg[m, mode, target, t] for t in GRID]
                for r in group:
                    f = [float(r[f"fold{k}"]) for k in range(5)]
                    close(r["mean_dice"], np.mean(f))
                    close(r["worst_fold_dice"], min(f))
                    t = float(r["threshold"])
                    if t in [0.1, 0.4, 0.5]:
                        close(r["mean_dice"], ss[f"{m}_{mode}_t{t:g}", target]["dice"])
                        for k in range(5):
                            close(r[f"fold{k}"], fs[f"{m}_{mode}_t{t:g}", target, k]["dice"])
                best = max(
                    group, key=lambda r: (float(r["mean_dice"]), -abs(float(r["threshold"]) - 0.5))
                )
                for k in best:
                    assert best[k] == cc[m, mode, target][k]
                assert "not nested" in cc[m, mode, target]["selection_caveat"]
    for r in table(LABEL / "label_threshold_grid.csv"):
        if r["variant"] == "L2_soft":
            actual = gg["L2_reference", r["mode"], r["target"], float(r["threshold"])]
            for k in ["mean_dice", "worst_fold_dice", *[f"fold{f}" for f in range(5)]]:
                close(actual[k], r[k])
    checks["104_threshold_grid_rows_and8_development_choices_verified"] = True

    paired = table(RESULT / "structure_paired_per_image.csv")
    paired_folds = table(RESULT / "structure_paired_per_fold.csv")
    paired_summary = table(RESULT / "structure_paired_summary.csv")
    assert len(paired) == 1920 and len(paired_folds) == 480 and len(paired_summary) == 96
    pp = {(r["mode"], float(r["threshold"]), r["target"], r["metric"], r["id"]): r for r in paired}
    pf = {
        (r["mode"], float(r["threshold"]), r["target"], r["metric"], int(r["fold"])): r
        for r in paired_folds
    }
    assert len(pp) == 1920 and len(pf) == 480
    for r in paired_summary:
        mode, t, target, k = r["mode"], float(r["threshold"]), r["target"], r["metric"]
        ds = []
        fd = []
        for i in sorted(assignment, key=int):
            d = float(lookup[f"T1_L2_cldice_{mode}_t{t:g}", target, i][k]) - float(
                lookup[f"L2_reference_{mode}_t{t:g}", target, i][k]
            )
            close(pp[mode, t, target, k, i]["delta"], d)
            ds.append((i, d))
        for fold in range(5):
            d = float(np.mean([d for i, d in ds if assignment[i] == fold]))
            fd.append(d)
            close(pf[mode, t, target, k, fold]["delta"], d)
        close(r["mean_delta"], np.mean([d for _, d in ds]))
        assert int(r["positive_images"]) == sum(d > 0 for _, d in ds) and int(
            r["negative_images"]
        ) == sum(d < 0 for _, d in ds)
        assert int(r["positive_folds"]) == sum(d > 0 for d in fd) and int(
            r["negative_folds"]
        ) == sum(d < 0 for d in fd)
        close(r["min_fold_delta"], min(fd))
        close(r["max_fold_delta"], max(fd))
    checks["all1920_paired_image_deltas_480_fold_deltas_and96_summaries_verified"] = True

    calibrated_pairs = []
    for mode in ["none", "d4"]:
        for target in ["clean", "proxy"]:
            a, b = cc["T1_L2_cldice", mode, target], cc["L2_reference", mode, target]
            ds = [float(a[f"fold{f}"]) - float(b[f"fold{f}"]) for f in range(5)]
            calibrated_pairs.append(
                dict(
                    mode=mode,
                    target=target,
                    new_threshold=a["threshold"],
                    reference_threshold=b["threshold"],
                    mean_delta=float(a["mean_dice"]) - float(b["mean_dice"]),
                    positive_folds=sum(d > 0 for d in ds),
                    worst_fold_delta=float(a["worst_fold_dice"]) - float(b["worst_fold_dice"]),
                    **{f"fold{f}_delta": d for f, d in enumerate(ds)},
                    caveat="OOF-tuned descriptive development comparison",
                )
            )
    write("analysis_calibrated_paired_folds.csv", calibrated_pairs)
    hard = [r for r in rows if r["id"] == "12" and "_d4_" in r["experiment"]]
    write("analysis_image12.csv", hard)
    assert not list(RESULT.rglob("*.pt")) and not list(RESULT.rglob("*.npz"))
    result = dict(
        checks=checks,
        archive_sha256=load(RESULT / "received_archive.json")["sha256"],
        training_monitor_minutes=new_seconds / 60,
        reference_training_monitor_minutes=reference_seconds / 60,
        calibration_comparisons=calibrated_pairs,
        independent_binary_metric_scope="D4 threshold0.5 only, masks from lossless PNG panels",
        local_checkpoint_or_probability_replay=False,
        new_training_started=False,
        new_competition_submission_generated=False,
        private_score=None,
        decision="Do not promote this fixed clDice setting to competition candidate; retain as structural recall/precision tradeoff.",
    )
    (RESULT / "analysis_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(ss, cc, fs, lookup, paired_summary, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def write_report(ss, cc, fs, lookup, paired, result):
    def metric(method, target, key="dice", threshold=0.5):
        return float(ss[f"{method}_d4_t{threshold:g}", target][key])

    fixed = "\n".join(
        f"| {label} | {metric(m,'clean'):.6f} | {metric(m,'proxy'):.6f} | "
        f"{metric(m,'clean','cldice'):.6f} | {metric(m,'clean','thin_clean_centerline_recall'):.2%} | "
        f"{metric(m,'clean','worst_fold_dice'):.6f} |"
        for m, label in [("L2_reference", "原L2"), ("T1_L2_cldice", "L2＋0.1 soft-clDice")]
    )
    calibrated = "\n".join(
        f"| {target} | {cc['L2_reference','d4',target]['threshold']} | "
        f"{float(cc['L2_reference','d4',target]['mean_dice']):.6f} | {cc['T1_L2_cldice','d4',target]['threshold']} | "
        f"{float(cc['T1_L2_cldice','d4',target]['mean_dice']):.6f} | "
        f"{float(cc['T1_L2_cldice','d4',target]['mean_dice'])-float(cc['L2_reference','d4',target]['mean_dice']):+.6f} |"
        for target in ["clean", "proxy"]
    )
    fold_table = "\n".join(
        f"| {f} | {float(fs['L2_reference_d4_t0.5','clean',f]['dice']):.6f} | "
        f"{float(fs['T1_L2_cldice_d4_t0.5','clean',f]['dice']):.6f} | "
        f"{float(fs['T1_L2_cldice_d4_t0.5','clean',f]['dice'])-float(fs['L2_reference_d4_t0.5','clean',f]['dice']):+.6f} | "
        f"{float(fs['T1_L2_cldice_d4_t0.5','clean',f]['cldice'])-float(fs['L2_reference_d4_t0.5','clean',f]['cldice']):+.6f} |"
        for f in range(5)
    )
    image12 = "\n".join(
        f"| {m} | {float(lookup[m+'_d4_t0.5','clean','12']['dice']):.6f} | "
        f"{float(lookup[m+'_d4_t0.5','clean','12']['cldice']):.6f} | "
        f"{float(lookup[m+'_d4_t0.5','clean','12']['thin_clean_centerline_recall']):.2%} |"
        for m in METHODS
    )
    proxy_pair = next(
        r
        for r in paired
        if r["mode"] == "d4"
        and float(r["threshold"]) == 0.1
        and r["target"] == "proxy"
        and r["metric"] == "dice"
    )
    text = f"""# 任务4：结构损失单项消融结果分析

日期：2026-11-02。结果文件`structure_ablation_analysis_results.zip`（SHA256 `{result['archive_sha256']}`）。本轮T1_L2_cldice完成5折×80轮正式训练，对照组为原L2（未重新训练）。

## 结论

**在当前配方下，soft-clDice不加入最终候选，不生成新提交。** 它提高了干净目标的中心线覆盖和细线召回，也改善了最差折，但同时降低了像素精确率和平均干净Dice；阈值分别校准后，两个目标的平均Dice都低于原L2。

这一结论限于当前设置：基于L2、权重0.1、仅作用于主输出头、软骨架迭代10次、smooth=1、训练80轮。本轮不再调整这些参数。

## 固定阈值0.5：结构覆盖提高，像素精度下降

以下均为D4推理、20张图的逐图平均。

| 方法 | 干净Dice | 代理Dice | 干净clDice | 细中心线召回 | 最差干净折Dice |
|---|---:|---:|---:|---:|---:|
{fixed}

与原L2相比：

- 干净Dice下降0.002769，只有6/20张图、2/5折提高。
- 干净clDice提高0.004835，15/20张图、5/5折提高；细中心线召回提高2.006个百分点，20/20张图提高。
- 干净像素精确率下降1.518个百分点（20/20张图下降），召回提高1.158个百分点；骨架精确率下降0.498个百分点，骨架召回提高1.302个百分点。
- 前景面积增加0.326个百分点（20/20张图增加），新增区域中既有正确恢复的细血管，也有误检。
- 代理Dice提高0.001052，但只有10/20张图、2/5折提高，提升并不稳定。
- 最差干净折从0.742793提高到0.745169（+0.002376），但折2的干净Dice下降0.008346。

| 折 | 原L2干净Dice@0.5 | 新组干净Dice@0.5 | Dice增量 | 干净clDice增量 |
|---|---:|---:|---:|---:|
{fold_table}

细中心线定义为C骨架上距背景不超过1.5像素的点。

## 阈值校准后的结果

沿用运行前固定的13点阈值网格。

| 目标 | 原L2开发阈值 | 原L2 Dice | 新组开发阈值 | 新组 Dice | 新−原 |
|---|---:|---:|---:|---:|---:|
{calibrated}

干净目标：新组在阈值0.6下为0.781732，低于原L2在阈值0.5下的0.784055（−0.002323）；5折中只有折4提高。

代理目标：两者的最佳阈值都为0.1，新组0.789996，低于原L2的0.797158（−0.007162），{proxy_pair['negative_images']}/20张图、5/5折降低。因此固定0.5下代理Dice的小幅上升在阈值校准后没有保留；不使用TTA时，新组在两个目标上也都低于原L2。

以上为在OOF上选择阈值后的开发集结果。像素级的独立复算只针对D4、阈值0.5。

## 困难样本与可视化

12号困难图在固定0.5阈值下的结果：

| 方法 | 干净Dice | 干净clDice | 细中心线召回 |
|---|---:|---:|---:|
{image12}

`oof_error_previews/oof_01.png`至`oof_20.png`展示了原图、C/P目标、两种方法在512网格上的二值预测及错误图：红色为假阳性，蓝色为相对C的漏检；变化图中绿色为新增、橙色为移除。

## 与已有候选的比较

原尺寸训练候选的20图OOF：干净目标（阈值0.4）为0.787453，代理目标（阈值0.1）为0.827069；新结构组在两个目标上的最佳值分别为0.781732和0.789996，均低于原候选。两者属于不同的完整流程，结构损失本身的作用以新组与L2的对比为准。

原候选保持不变，本轮不生成新的测试预测和提交。

## 小结与后续

1. E1/E2、三组标签监督和结构损失消融均已完成：L2软边带在干净目标上有收益，而当前的clDice损失以平均像素Dice为代价换取召回。
2. 不再继续调整clDice的权重和迭代次数。
3. 如需进一步提升成绩，可对原尺寸候选与L1代理模型的完整OOF概率做误差互补性分析。两者的概率校准尺度不同，融合方法、权重和阈值需事先固定，并只用对应折的OOF评估。
4. 如需进一步验证软边带方法，可用另一个事先固定的随机种子复验L0/L2，并报告两个种子的全部结果。

## 结果核对

源码、原始数据、五折划分和原L2的完成记录均与此前一致。配置上唯一的改动是在主输出头加入0.1×soft-clDice（10次迭代、smooth=1）；五折初始化与80轮RGB采样计划和原L2逐项对齐，400行训练日志完整，各轮loss满足loss≈reference_loss+0.1×soft_cldice_loss。

两次运行环境相同；原L2概率文件的哈希一致，基线指标重算无偏差。

已核对480行逐图结果、120行分折结果、24个汇总、104条阈值网格、8个校准选择以及全部配对增量。

诊断PNG以原始512尺寸无损保存预测mask，因此可从20张PNG恢复两种方法在D4、阈值0.5下的二值预测，据此独立重算的80行像素、骨架和细线指标与表格完全一致。

完整权重和OOF概率保存在Kaggle上。

新组逐轮训练与监控时间合计{result['training_monitor_minutes']:.2f}分钟，原L2为{result['reference_training_monitor_minutes']:.2f}分钟（均不含检查点保存和最终D4推理）。

结果目录：`{RESULT.name}`。由PNG独立重算的结果见`analysis_recomputed_d4_t05_from_png.csv`，阈值校准后的分折对照见`analysis_calibrated_paired_folds.csv`。
"""
    (ROOT / "结构损失消融结果分析_20261102.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
