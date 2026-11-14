"""Audit the downloaded Task3 CV output without importing ML frameworks."""

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent
BASE = BASE.parent if BASE.name == "analysis" else BASE  # 位于 analysis/ 时回到任务根目录
RESULTS = BASE / "kaggle_cv_results_20261025"
ARMS = ("resnet34_320", "resnet34_320_flip", "resnet34_512")
BLEND = "resnet34_320_512_equal"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def close(a, b, tolerance=1e-7):
    if not math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=tolerance):
        raise ValueError(f"Mismatch: {a} versus {b}")


def metrics(rows):
    errors = [
        (float(r["Fovea_X"]) - float(r["gt_x"]), float(r["Fovea_Y"]) - float(r["gt_y"]))
        for r in rows
    ]
    distances = sorted(math.hypot(x, y) for x, y in errors)
    index = 0.9 * (len(distances) - 1)
    lo, hi = math.floor(index), math.ceil(index)
    return dict(
        n=len(errors),
        mean_distance_px=statistics.mean(distances),
        median_distance_px=statistics.median(distances),
        p90_distance_px=distances[lo] + (index - lo) * (distances[hi] - distances[lo]),
        max_distance_px=max(distances),
        mae_x_px=statistics.mean(abs(x) for x, _ in errors),
        mae_y_px=statistics.mean(abs(y) for _, y in errors),
        rmse_coordinate_px=math.sqrt(sum(x * x + y * y for x, y in errors) / (2 * len(errors))),
    )


def id_map(rows, expected):
    mapped = {int(r["image_id"]): r for r in rows}
    if len(mapped) != len(rows) or set(mapped) != set(expected):
        raise ValueError("Missing or duplicated image IDs")
    return mapped


def main():
    protocol = json.loads((RESULTS / "protocol.json").read_text())
    status = json.loads((RESULTS / "status.json").read_text())
    selection = json.loads((RESULTS / "selection.json").read_text())
    if status["state"] != "completed" or not selection["complete"]:
        raise ValueError("Incomplete run")
    split = id_map(read_csv(RESULTS / "split.csv"), range(1, 81))
    labels = {
        int(r["data"]): r
        for r in read_csv(BASE / "2026task3/detection/fovea_localization_train_GT.csv")
    }
    groups = {}
    for key, r in split.items():
        close(r["x"], labels[key]["Fovea_X"])
        close(r["y"], labels[key]["Fovea_Y"])
        groups.setdefault(r["pixel_sha256"], set()).add(int(r["fold"]))
    if any(len(folds) != 1 for folds in groups.values()):
        raise ValueError("Duplicate pixels cross validation folds")
    oof, comparison = {}, []
    source_metrics = {r["arm"]: r for r in read_csv(RESULTS / "comparison.csv")}
    for arm in ARMS:
        oof[arm] = id_map(read_csv(RESULTS / f"oof_{arm}.csv"), range(1, 81))
        for key, r in oof[arm].items():
            if r["fold"] != split[key]["fold"] or r["pixel_sha256"] != split[key]["pixel_sha256"]:
                raise ValueError("OOF split mismatch")
            close(r["gt_x"], split[key]["x"])
            close(r["gt_y"], split[key]["y"])
            close(r["dx"], float(r["Fovea_X"]) - float(r["gt_x"]))
            close(r["dy"], float(r["Fovea_Y"]) - float(r["gt_y"]))
            close(r["distance_px"], math.hypot(float(r["dx"]), float(r["dy"])))
        values = metrics(list(oof[arm].values()))
        for name, value in values.items():
            close(value, source_metrics[arm][name])
        comparison.append(dict(arm=arm, **values))
    training, total_seconds = [], 0
    for size in (320, 512):
        for fold in range(5):
            folder = RESULTS / f"resnet34_{size}/fold_{fold}"
            history = read_csv(folder / "history.csv")
            complete = json.loads((folder / "fold_complete.json").read_text())
            if [int(r["epoch"]) for r in history] != list(range(1, 81)) or not complete[
                "predictions_complete"
            ]:
                raise ValueError("Missing epochs or predictions")
            best = complete["training"]["best"][protocol["selection_metric"]]
            close(best["score"], min(float(r[protocol["selection_metric"]]) for r in history))
            close(best["score"], history[best["epoch"] - 1][protocol["selection_metric"]])
            total_seconds += sum(float(r["compute_seconds"]) for r in history)
            training.append(
                dict(
                    size=size,
                    fold=fold,
                    epochs=len(history),
                    best_epoch=best["epoch"],
                    best_rmse=best["score"],
                    last_rmse=float(history[-1]["rmse_coordinate_px"]),
                )
            )
    tests = {}
    for arm in ARMS:
        tests[arm] = id_map(read_csv(RESULTS / f"test_{arm}_5fold.csv"), range(81, 101))
        size, mode = ("512" if arm.endswith("512") else "320"), (
            "flip" if arm.endswith("flip") else "raw"
        )
        fold_tests = [
            id_map(
                read_csv(RESULTS / f"resnet34_{size}/fold_{fold}/test_{mode}.csv"), range(81, 101)
            )
            for fold in range(5)
        ]
        for key, r in tests[arm].items():
            for axis in ("X", "Y"):
                close(
                    r[f"Fovea_{axis}"],
                    statistics.mean(float(f[key][f"Fovea_{axis}"]) for f in fold_tests),
                )
    blend_oof = []
    for key in range(1, 81):
        a, b = oof["resnet34_320"][key], oof["resnet34_512"][key]
        r = {name: a[name] for name in ("image_id", "fold", "width", "height", "gt_x", "gt_y")}
        for axis in ("X", "Y"):
            r[f"Fovea_{axis}"] = (float(a[f"Fovea_{axis}"]) + float(b[f"Fovea_{axis}"])) / 2
        r["dx"], r["dy"] = r["Fovea_X"] - float(r["gt_x"]), r["Fovea_Y"] - float(r["gt_y"])
        r["distance_px"] = math.hypot(r["dx"], r["dy"])
        blend_oof.append(r)
    blend_metrics = metrics(blend_oof)
    comparison.append(dict(arm=BLEND, **blend_metrics))
    template = read_csv(BASE / "2026task3/detection/sample_submission.csv")
    expected_order = [r["ImageID"] for r in template]
    submissions = []
    for row in template:
        key, field = row["ImageID"].split("_", 1)
        key = int(key)
        a, b = tests["resnet34_320"][key], tests["resnet34_512"][key]
        value = (float(a[field]) + float(b[field])) / 2
        limit = float(a["width" if field.endswith("X") else "height"])
        if not math.isfinite(value) or not 0 <= value <= limit:
            raise ValueError("Invalid submission coordinate")
        submissions.append(dict(ImageID=row["ImageID"], value=f"{value:.6f}"))
    target = BASE / "submission_resnet34_320_512_equal_5fold.csv"
    write_csv(target, submissions)
    saved = read_csv(target)
    if len(saved) != 40 or [r["ImageID"] for r in saved] != expected_order:
        raise ValueError("Submission row order mismatch")
    if (RESULTS / "submission_selected_5fold.csv").read_bytes() != (
        RESULTS / "submission_resnet34_512_5fold.csv"
    ).read_bytes():
        raise ValueError("Selected file does not match the selected 512 arm")
    comparison.sort(key=lambda r: r["rmse_coordinate_px"])
    write_csv(RESULTS / "comparison_with_equal_blend.csv", comparison)
    write_csv(RESULTS / f"oof_{BLEND}.csv", blend_oof)
    improvement = 1 - blend_metrics["rmse_coordinate_px"] / source_metrics_number(
        source_metrics, "resnet34_512", "rmse_coordinate_px"
    )
    audit = dict(
        checks_passed=True,
        train_images=80,
        test_images=20,
        completed_training_jobs=10,
        epochs_per_job=80,
        total_epoch_compute_minutes=total_seconds / 60,
        duplicate_0039_0077_fold=split[39]["fold"],
        original_selection=selection["selected_arm"],
        official_metric_confirmed=selection["official_metric_confirmed"],
        blend_policy="Fixed equal coordinates: 0.5 * mean(5 raw 320 fold models) + 0.5 * mean(5 raw 512 fold models). No weight fitting.",
        comparison=comparison,
        training=training,
        rmse_improvement_vs_512=improvement,
        submission_path=str(target),
        submission_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        limitations=[
            "OOF epoch and arm selection use validation labels; not an independent test.",
            "OOF metrics cannot be directly compared with the previous 12.31624 public score.",
            "ZIP contains no checkpoints; model tensors and GPU paths were not re-executed locally.",
        ],
    )
    (RESULTS / "analysis_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    table = "\n".join(
        f"| {r['arm']} | {r['rmse_coordinate_px']:.4f} | {r['mean_distance_px']:.4f} | {r['max_distance_px']:.2f} |"
        for r in comparison
    )
    report = f"""# Task3 五折结果分析

10 个模型均完成 80 轮训练，3 组方案均得到完整的 80 张 OOF 预测和 20 张测试预测。按验证结果选出的方案为 512 输入，导出的 selected 文件与 512 方案的提交文件一致。

| 方案 | 坐标RMSE（像素） | 平均欧氏距离（像素） | 最大欧氏距离（像素） |
|---|---:|---:|---:|
{table}

320 与 512 两个尺度的预测直接等权平均后，坐标 RMSE 比单独使用 512 降低 {improvement:.2%}，无需额外训练。五折中有两折优于 512，其余三折持平或略差。融合采用固定等权，不在这 80 张图上拟合权重。

后续优先提交 submission_resnet34_320_512_equal_5fold.csv，并以 512 方案作对照；原全量模型（公开榜 12.31624）及其权重继续保留。OOF 与公开测试集是不同的数据，15.67 与 12.31624 不能直接比较。

翻转 TTA 在 0072 和 0024 上的误差分别达到 326.37 和 319.19 像素，因此不纳入候选。代码对翻转预测的 X 坐标做了反变换，Y 方向同样出现大偏差，说明是少数翻转图像本身定位失败。

0049 是两种原图方案共同的难点：320 误差 107.82 像素，512 误差 90.28 像素。0072、0034、0044 误差也较大，可作为后续局部高分辨率精修的重点。

320 方案第 3 折最后一轮 RMSE 升至 158.57，输出使用的是第 62 轮的最佳权重。各折均按最佳轮次保存权重，单纯增加训练轮数缺少依据。

逐轮记录的训练与验证时间合计 {total_seconds/60:.2f} 分钟（不含数据准备、保存和推理）。模型权重保存在 Kaggle 的完整输出中。
"""
    (BASE / "五折结果分析_20261025.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            dict(
                checks_passed=True,
                comparison=comparison,
                submission=str(target),
                rmse_improvement_vs_512=improvement,
                compute_minutes=total_seconds / 60,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


def source_metrics_number(data, arm, metric):
    return float(data[arm][metric])


if __name__ == "__main__":
    main()
