"""Aggregate saved per-epoch validation RMSE; no training or model inference."""

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROOT = ROOT.parent if ROOT.name == "analysis" else ROOT  # 位于 analysis/ 时回到任务根目录
SOURCE = ROOT / "kaggle_cv_results_20261025"
OUTPUT = ROOT / "common_epoch_analysis"


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pooled(rows):
    return math.sqrt(
        sum(int(r["n"]) * float(r["rmse_coordinate_px"]) ** 2 for r in rows)
        / sum(int(r["n"]) for r in rows)
    )


def main():
    OUTPUT.mkdir(exist_ok=True)
    summary = {}
    curves = []
    for size in (320, 512):
        histories = []
        best_rows = []
        best_epochs = []
        for fold in range(5):
            rows = read(SOURCE / f"resnet34_{size}/fold_{fold}/history.csv")
            by_epoch = {int(r["epoch"]): r for r in rows}
            assert len(by_epoch) == len(rows) == 80
            assert set(by_epoch) == set(range(1, 81))
            assert all(
                int(r["n"]) == 16 and math.isfinite(float(r["rmse_coordinate_px"])) for r in rows
            )
            histories.append(by_epoch)
            epoch = min(by_epoch, key=lambda k: float(by_epoch[k]["rmse_coordinate_px"]))
            best_epochs.append(epoch)
            best_rows.append(by_epoch[epoch])
        results = []
        for epoch in range(1, 81):
            rows = [history[epoch] for history in histories]
            result = dict(
                image_size=size,
                epoch=epoch,
                n=sum(int(r["n"]) for r in rows),
                pooled_coordinate_rmse_px=pooled(rows),
            )
            result.update(
                {f"fold_{f}_rmse_px": float(r["rmse_coordinate_px"]) for f, r in enumerate(rows)}
            )
            results.append(result)
        best = min(results, key=lambda r: r["pooled_coordinate_rmse_px"])
        oof = read(SOURCE / f"oof_resnet34_{size}.csv")
        oof_rmse = math.sqrt(
            sum(
                (float(r["Fovea_X"]) - float(r["gt_x"])) ** 2
                + (float(r["Fovea_Y"]) - float(r["gt_y"])) ** 2
                for r in oof
            )
            / (2 * len(oof))
        )
        selected_rmse = pooled(best_rows)
        assert len(oof) == 80 and abs(oof_rmse - selected_rmse) < 1e-8
        summary[str(size)] = dict(
            best_common_epoch=best["epoch"],
            best_common_rmse=best["pooled_coordinate_rmse_px"],
            per_fold_best_epochs=best_epochs,
            per_fold_best_rmse=selected_rmse,
            relative_gap_percent=(best["pooled_coordinate_rmse_px"] / selected_rmse - 1) * 100,
            selected_common_epochs={
                str(k): results[k - 1]["pooled_coordinate_rmse_px"]
                for k in (10, 11, 20, 40, 43, 80)
            },
        )
        curves.extend(results)
    with (OUTPUT / "common_epoch_rmse.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    report = dict(
        source=str(SOURCE),
        formula="sqrt(sum(n_f * RMSE_f(k)^2) / sum(n_f))",
        folds=5,
        images_per_fold=16,
        epochs=80,
        results=summary,
        limitation="The best common epoch is selected using the same development folds; it is not an independent test estimate. Scalar fold metrics cannot reconstruct coordinate-averaged dual-scale ensemble RMSE.",
    )
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
