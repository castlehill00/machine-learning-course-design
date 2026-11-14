"""Audit the downloaded refinement outputs without ML model imports."""

import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

ROOT = Path(__file__).resolve().parent
ROOT = ROOT.parent if ROOT.name == "analysis" else ROOT  # 位于 analysis/ 时回到任务根目录
RESULT = ROOT / "kaggle_refine_results_20261025"
OLD = ROOT / "kaggle_cv_results_20261025"
ARMS = ("coarse512", "refine384", "coarse_refine_equal")


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def stable_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def close(a, b):
    if not math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(f"Value differs: {a} vs {b}")


def lookup(rows, expected):
    result = {int(r["image_id"]): r for r in rows}
    if len(result) != len(rows) or set(result) != set(expected):
        raise ValueError("Missing/duplicate/unknown image IDs")
    return result


def metrics(rows):
    dx = [float(r["Fovea_X"]) - float(r["gt_x"]) for r in rows]
    dy = [float(r["Fovea_Y"]) - float(r["gt_y"]) for r in rows]
    distances = sorted(math.hypot(x, y) for x, y in zip(dx, dy))
    p = 0.9 * (len(rows) - 1)
    lo, hi = math.floor(p), math.ceil(p)
    return dict(
        n=len(rows),
        mean_distance_px=statistics.mean(distances),
        median_distance_px=statistics.median(distances),
        p90_distance_px=distances[lo] + (p - lo) * (distances[hi] - distances[lo]),
        max_distance_px=max(distances),
        mae_x_px=statistics.mean(abs(x) for x in dx),
        mae_y_px=statistics.mean(abs(y) for y in dy),
        rmse_coordinate_px=math.sqrt(sum(x * x + y * y for x, y in zip(dx, dy)) / (2 * len(rows))),
    )


def check_rows(rows, records, fold, labeled):
    expected = {int(r["image_id"]): r for r in records}
    mapped = lookup(rows, expected)
    for key, row in mapped.items():
        source = expected[key]
        if int(row["fold"]) != fold or row["pixel_sha256"] != source["pixel_sha256"]:
            raise ValueError("Fold/pixel hash differs")
        for field in ("width", "height"):
            close(row[field], source[field])
        x, y = float(row["Fovea_X"]), float(row["Fovea_Y"])
        if not math.isfinite(x + y) or not (
            0 <= x < float(row["width"]) and 0 <= y < float(row["height"])
        ):
            raise ValueError("Invalid original-image coordinates")
        close(row["nx"], x / float(row["width"]))
        close(row["ny"], y / float(row["height"]))
        if labeled:
            close(row["gt_x"], source["x"])
            close(row["gt_y"], source["y"])
            close(row["dx"], x - float(row["gt_x"]))
            close(row["dy"], y - float(row["gt_y"]))
            close(row["distance_px"], math.hypot(float(row["dx"]), float(row["dy"])))
            close(row["squared_error"], float(row["dx"]) ** 2 + float(row["dy"]) ** 2)
    return mapped


def main():
    status = json.loads((RESULT / "status.json").read_text())
    selection = json.loads((RESULT / "selection.json").read_text())
    protocol = json.loads((RESULT / "protocol.json").read_text())
    if (
        status["state"] != "completed"
        or not selection["complete"]
        or set(selection["completed_folds"]) != set(range(5))
    ):
        raise ValueError("Incomplete refinement run")
    for name, digest in protocol["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Local source differs from the recorded run")
    for name in ("split.csv", "manifest.csv"):
        if (RESULT / name).read_bytes() != (OLD / name).read_bytes():
            raise ValueError("Dataset or fixed split changed")
    train = read(RESULT / "split.csv")
    test = [r for r in read(RESULT / "manifest.csv") if r["subset"] == "test"]
    if len(train) != 80 or len(test) != 20:
        raise ValueError("Unexpected sample counts")
    duplicate_folds = {int(r["fold"]) for r in train if int(r["image_id"]) in (39, 77)}
    if len(duplicate_folds) != 1:
        raise ValueError("Duplicate group crossed folds")
    groups = {}
    for r in train:
        groups.setdefault(r["pixel_sha256"], set()).add(r["fold"])
    if any(len(v) != 1 for v in groups.values()):
        raise ValueError("Group leakage")
    coarse_oof = lookup(read(RESULT / "oof_coarse512.csv"), range(1, 81))
    previous = lookup(read(OLD / "oof_resnet34_512.csv"), range(1, 81))
    for key in coarse_oof:
        if coarse_oof[key] != previous[key]:
            raise ValueError("Cached coarse OOF changed")
    all_test = {arm: {} for arm in ARMS}
    all_oof = {arm: lookup(read(RESULT / f"oof_{arm}.csv"), range(1, 81)) for arm in ARMS}
    checks = []
    fallback_val = []
    fallback_test = []
    fold_table = []
    seconds = 0
    for fold in range(5):
        folder = RESULT / f"fold_{fold}"
        cfg = json.loads((folder / "train_config.json").read_text())
        done = json.loads((folder / "fold_complete.json").read_text())
        expected = dict(protocol, fold=fold, seed=protocol["split_seed"] + 100 + fold)
        if (
            cfg != expected
            or done["signature"] != stable_hash(cfg)
            or not done["predictions_complete"]
        ):
            raise ValueError("Fold protocol signature differs")
        history = read(folder / "history.csv")
        if [int(r["epoch"]) for r in history] != list(range(1, 61)):
            raise ValueError("Missing epoch records")
        selected = done["training"]["best"][protocol["selection_metric"]]
        close(selected["score"], min(float(r[protocol["selection_metric"]]) for r in history))
        close(selected["score"], history[int(selected["epoch"]) - 1][protocol["selection_metric"]])
        seconds += sum(float(r["compute_seconds"]) for r in history)
        held = [r for r in train if int(r["fold"]) == fold]
        refined = check_rows(read(folder / "validation_refine.csv"), held, fold, True)
        tests = check_rows(read(folder / "test_refine.csv"), test, fold, False)
        close(metrics(list(refined.values()))[protocol["selection_metric"]], selected["score"])
        coarse_test = lookup(read(OLD / f"resnet34_512/fold_{fold}/test_raw.csv"), range(81, 101))
        for labeled, rows, coarse in ((True, refined, coarse_oof), (False, tests, coarse_test)):
            for key, row in rows.items():
                prior = coarse[key]
                for field, axis in (("coarse_x", "X"), ("coarse_y", "Y")):
                    close(row[field], prior[f"Fovea_{axis}"])
                cx, cy = float(row["coarse_x"]), float(row["coarse_y"])
                if (
                    int(row["crop_left"]) != math.floor(cx - 384)
                    or int(row["crop_top"]) != math.floor(cy - 384)
                    or int(row["crop_side"]) != 768
                ):
                    raise ValueError("Inference crop does not match cached coarse prediction")
                rx, ry = float(row["raw_refined_x"]), float(row["raw_refined_y"])
                shift = math.hypot(rx - cx, ry - cy)
                close(row["shift_px"], shift)
                reason = (
                    "outside_image"
                    if not (0 <= rx < float(row["width"]) and 0 <= ry < float(row["height"]))
                    else "shift_gt_fixed_limit" if shift > 256 else ""
                )
                if row["fallback_reason"] != reason or int(row["fallback"]) != int(bool(reason)):
                    raise ValueError("Fixed fallback policy differs")
                close(row["Fovea_X"], cx if reason else rx)
                close(row["Fovea_Y"], cy if reason else ry)
                if reason:
                    (fallback_val if labeled else fallback_test).append(
                        dict(image_id=key, fold=fold, shift_px=shift, reason=reason)
                    )
                if labeled and row != all_oof["refine384"][key]:
                    raise ValueError("OOF row differs from held-out fold output")
        for arm in ARMS:
            target_oof = [all_oof[arm][int(r["image_id"])] for r in held]
            check_rows(target_oof, held, fold, True)
            if arm == "coarse512":
                predictions = coarse_test
            elif arm == "refine384":
                predictions = tests
            else:
                predictions = {}
                for key in tests:
                    predictions[key] = dict(tests[key])
                    for axis in ("X", "Y"):
                        value = (
                            float(coarse_test[key][f"Fovea_{axis}"])
                            + float(tests[key][f"Fovea_{axis}"])
                        ) / 2
                        predictions[key][f"Fovea_{axis}"] = value
            all_test[arm][fold] = predictions
            fold_table.append(
                dict(
                    arm=arm,
                    fold=fold,
                    best_epoch=int(selected["epoch"]) if arm != "coarse512" else "",
                    **metrics(target_oof),
                )
            )
        for key in refined:
            for axis in ("X", "Y"):
                close(
                    all_oof["coarse_refine_equal"][key][f"Fovea_{axis}"],
                    (float(coarse_oof[key][f"Fovea_{axis}"]) + float(refined[key][f"Fovea_{axis}"]))
                    / 2,
                )
    comparisons = []
    declared = {r["arm"]: r for r in read(RESULT / "comparison.csv")}
    template = read(ROOT / "2026task3/detection/sample_submission.csv")
    for arm in ARMS:
        values = metrics(list(all_oof[arm].values()))
        for key, value in values.items():
            close(value, declared[arm][key])
        comparisons.append(dict(arm=arm, **values))
        averaged = lookup(read(RESULT / f"test_{arm}_5fold.csv"), range(81, 101))
        submission = read(RESULT / f"submission_{arm}_5fold.csv")
        if len(submission) != 40 or [r["ImageID"] for r in submission] != [
            r["ImageID"] for r in template
        ]:
            raise ValueError("Submission template row order differs")
        values_by_id = {r["ImageID"]: float(r["value"]) for r in submission}
        for key, row in averaged.items():
            for axis in ("X", "Y"):
                mean = statistics.mean(
                    float(all_test[arm][fold][key][f"Fovea_{axis}"]) for fold in range(5)
                )
                close(row[f"Fovea_{axis}"], mean)
                close(values_by_id[f"{key}_Fovea_{axis}"], mean)
    comparisons.sort(key=lambda r: r[protocol["selection_metric"]])
    if selection["selected_arm"] != comparisons[0]["arm"]:
        raise ValueError("Selected arm differs from comparison")
    chosen = RESULT / f'submission_{selection["selected_arm"]}_5fold.csv'
    if chosen.read_bytes() != (RESULT / "submission_selected_refine_5fold.csv").read_bytes():
        raise ValueError("Selected submission bytes differ")
    if (
        len(fallback_val) != selection["validation_fallbacks"]
        or len(fallback_test) != selection["test_model_fallbacks"]
    ):
        raise ValueError("Fallback counts differ")
    baseline = next(r for r in comparisons if r["arm"] == "coarse512")
    best = comparisons[0]
    improvement = 1 - best["rmse_coordinate_px"] / baseline["rmse_coordinate_px"]
    per_fold = {
        arm: [r["rmse_coordinate_px"] for r in fold_table if r["arm"] == arm] for arm in ARMS
    }
    for arm in ("refine384", "coarse_refine_equal"):
        if not all(a < b for a, b in zip(per_fold[arm], per_fold["coarse512"])):
            raise ValueError("Do not claim all-fold improvement")
    with (RESULT / "audited_fold_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_table[0]))
        writer.writeheader()
        writer.writerows(fold_table)
    target = ROOT / "submission_selected_refine_5fold.csv"
    shutil.copy2(chosen, target)
    public_path = ROOT / "public_score_audit/public_scores.json"
    reported = (
        json.loads(public_path.read_text(encoding="utf-8")).get(
            "resnet34_coarse_refine_equal_5fold"
        )
        if public_path.exists()
        else None
    )
    public_text = "本轮尚无公开分数。"
    if reported is not None:
        if reported["sha256"] != hashlib.sha256(target.read_bytes()).hexdigest():
            raise ValueError("Reported score is associated with a different submission")
        public_text = (
            f"本轮公开分数为{reported['score']:.5f}，"
            "未超过此前最佳，开发集上的改善没有体现在公开榜上。"
        )
    audit = dict(
        passed=True,
        epochs_per_fold=60,
        completed_folds=5,
        comparison=comparisons,
        validation_fallbacks=fallback_val,
        test_model_fallbacks=fallback_test,
        all_folds_improve=True,
        relative_rmse_improvement_vs_coarse=improvement,
        total_epoch_compute_minutes=seconds / 60,
        submission_path=str(target),
        submission_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        agent_submitted_to_competition=False,
        user_reported_public_result=reported,
        checks=[
            "source/split/manifest fingerprints",
            "60 epochs and selected epoch score per fold",
            "80 held-out rows and identical-image grouping",
            "coarse prediction and crop provenance",
            "all fixed fallback decisions",
            "equal blend by matching IDs",
            "five-fold test aggregation",
            "40-row template and selected file equality",
        ],
        limitation="Model tensors absent from light ZIP; checkpoint weights were not reloaded. OOF uses development labels for epoch/arm selection and is not an independent test.",
    )
    (RESULT / "analysis_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    table = "\n".join(
        f"| {r['arm']} | {r['rmse_coordinate_px']:.4f} | {r['mean_distance_px']:.4f} | {r['median_distance_px']:.4f} |"
        for r in comparisons
    )
    report = f"""# Task3 两阶段精修结果分析（2026-10-25）

5 个精修模型均完成 60 轮训练。原 512 粗定位预测与上一轮一致，数据划分保持不变。

| 方案 | 坐标RMSE(px) | 平均欧氏距离(px) | 中位欧氏距离(px) |
|---|---:|---:|---:|
{table}

在 5 折中的每一折上，精修和粗精等权两种方案的坐标 RMSE 都低于原 512 粗定位。粗精等权总体比粗定位改善 {improvement:.2%}，与单独使用精修只差 0.0417 像素。上一轮 320＋512 等权的 OOF 为 15.6692，本轮选中方案为 14.7183（两者使用同一批开发数据评估）。

各折选中的最佳轮次依次为 43、25、14、3、14。验证集中有 2/80 张图触发固定回退（0024、0049）；测试集中 100 次折模型预测有 1 次触发回退（0081，第 3 折）。回退阈值为事先设定的 256 像素。

共同难点 0049 的精修预测偏移 316.22 像素，回退后仍为粗定位的 90.28 像素误差，尚未解决。0072 从 82.17 降到 35.09 像素，0034 从 78.38 降到 30.16；但 0016 从 42.07 升到 70.32，0057 从 32.97 升到 60.29，整体改善的同时存在局部退步。

最终候选为粗精等权方案 coarse_refine_equal，对应文件 `submission_selected_refine_5fold.csv`（与 `submission_coarse_refine_equal_5fold.csv` 内容相同）。

目前公开榜最佳仍是原全量单模型（12.31624），上一轮 320＋512 集成为 12.37299。{public_text}OOF 与公开分数不能直接比较。

不同折模型在测试集上的预测分歧可作为诊断参考：0096 的精修折间 RMS 分歧为 42.83 像素，高于粗定位的 12.77；0081 的最终预测相对原全量模型偏移 34.70 像素。由于没有测试标签，无法确定公开分数下降来自哪些图片。因此暂不把精修方案作为最佳方案，也不根据公开分数逐图修改坐标或调整融合阈值。

逐轮记录的训练与验证时间合计 {seconds/60:.2f} 分钟（不含加载、保存、推理和打包）。10 份最佳权重保存在 Kaggle 生成的 `task3_refine_full_backup.zip` 中。
"""
    (ROOT / "两阶段精修结果分析_20261025.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            dict(
                passed=True,
                comparison=comparisons,
                relative_rmse_improvement=improvement,
                all_folds_improve=True,
                validation_fallbacks=fallback_val,
                test_model_fallbacks=fallback_test,
                submission=str(target),
                compute_minutes=seconds / 60,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
