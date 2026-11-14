"""Audit received Kaggle v3 metadata and predictions without importing Torch."""

import ast
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ROOT = ROOT.parent if ROOT.name == "analysis" else ROOT  # 位于 analysis/ 时回到任务根目录
RESULTS = ROOT / "refine_v3_results_20261027"


def definitions(path, names):
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    nodes = [
        n
        for n in ast.parse(source).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    assert len(nodes) == len(names)
    return "\n\n".join("".join(lines[n.lineno - 1 : n.end_lineno]) for n in nodes)


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def near(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-6), (a, b)


def compare_rows(actual, expected):
    a = {int(r["image_id"]): r for r in actual}
    b = {int(r["image_id"]): r for r in expected}
    assert len(a) == len(actual) == len(b) == len(expected) and a.keys() == b.keys()
    for key in a:
        for field, value in b[key].items():
            if field in ("pixel_sha256", "fallback_reason"):
                assert str(a[key][field]) == str(value)
            elif isinstance(value, (int, float)) or field in (
                "Fovea_X",
                "Fovea_Y",
                "nx",
                "ny",
                "gt_x",
                "gt_y",
                "dx",
                "dy",
                "distance_px",
                "squared_error",
                "width",
                "height",
                "fold",
            ):
                near(a[key][field], value)


def main():
    assert "torch" not in sys.modules
    scope = dict(
        Path=Path,
        PurePosixPath=PurePosixPath,
        pd=pd,
        np=np,
        csv=csv,
        io=io,
        hashlib=hashlib,
        json=json,
        math=math,
        statistics=statistics,
        zipfile=zipfile,
        defaultdict=defaultdict,
        DIAGNOSTIC_VERSION="refine-diagnostic-v3",
    )
    sources = [
        ("task3_pipeline.py", {"assign_folds"}),
        (
            "task3_kaggle_cv.py",
            {
                "stable_hash",
                "atomic_csv",
                "read_rows",
                "quantile_linear",
                "metrics_from_rows",
                "validate_oof",
                "validate_test_rows",
                "average_test_folds",
                "write_submission",
            },
        ),
        (
            "task3_kaggle_refine.py",
            {
                "CoarseBundle",
                "validate_coarse_bundle",
                "crop_geometry",
                "point_to_patch",
                "patch_to_original",
                "make_refinement_row",
                "blend_prediction_rows",
            },
        ),
        (
            "task3_refine_diagnostic_v3.py",
            {
                "inner_partition",
                "selected_inner_epoch",
                "validate_phase_config",
                "fine_config",
                "raw_prediction_rows",
                "identity_rows",
            },
        ),
    ]
    for name, names in sources:
        exec(definitions(ROOT / name, names), scope)
    protocol = load(RESULTS / "protocol.json")
    assert not protocol["smoke_test"]
    for name, expected in protocol["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    train = pd.read_csv(RESULTS / "split.csv")
    test = pd.read_csv(RESULTS / "manifest.csv").query("subset=='test'")
    records, testing = train.to_dict("records"), test.to_dict("records")
    coarse = scope["validate_coarse_bundle"](ROOT / "kaggle_cv_results_20261025", records, testing)
    assert coarse["fingerprint"] == protocol["coarse_fingerprint"]
    selection = load(RESULTS / "selection.json")
    assert selection["complete"] and selection["completed_folds"] == list(range(5))
    assert not selection["selected_by_outer_scores"] and not selection["smoke_test"]
    assert load(RESULTS / "status.json")["state"] == "completed"
    source_oof = scope["validate_oof"](
        [r for f in range(5) for r in coarse["validation"][f].values()], records
    )
    compare_rows(rows(RESULTS / "oof_coarse512.csv"), source_oof)
    assert (RESULTS / "submission_coarse512.csv").read_bytes() == (
        ROOT / "submission_resnet34_512_control.csv"
    ).read_bytes()
    assert (RESULTS / "submission_refine_v3_candidate.csv").read_bytes() == (
        RESULTS / "submission_small_prior_equal.csv"
    ).read_bytes()
    report = {
        "completed_folds": 5,
        "official_metric": "RMSE over submitted X/Y coordinate rows",
        "formula_full_image_pairs": "sqrt(sum_i((x_pred-x_gt)^2+(y_pred-y_gt)^2)/(2*N))",
        "source_sha256_verified": protocol["source_sha256"],
        "inner": [],
        "variants": {},
        "limitations": [
            selection["limitation"],
            "Light ZIP contains no model tensors; tensor reload and exact inference reproduction were not performed.",
            "Outer selected snapshots differ between plain and prior in fold3, so their comparison is not entirely a same-weight decoder ablation.",
        ],
    }
    phase_count = 0
    for fold in range(5):
        job = RESULTS / f"outer_{fold}"
        dev, fit, held, outer = scope["inner_partition"](train, fold)
        fit_ids, held_ids, outer_ids = [
            [int(x) for x in frame.image_id] for frame in (fit, held, outer)
        ]
        design = dict(
            protocol=protocol,
            fold=fold,
            fit_ids=fit_ids,
            inner_val_ids=held_ids,
            outer_ids=outer_ids,
        )
        assert load(job / "design.json") == design
        done = load(job / "fold_complete.json")
        assert done["signature"] == scope["stable_hash"](design) and done["predictions_complete"]
        assert not done["outer_labels_used_for_epoch_selection"]
        roles = rows(job / "split_roles.csv")
        for role, frame in [
            ("inner_train", fit),
            ("inner_validation", held),
            ("outer_validation", outer),
        ]:
            # The saved role names are part of provenance, independent of the numeric fold column.
            expected = set(int(x) for x in frame.image_id)
            actual = set(int(r["image_id"]) for r in roles if r["role"] == role)
            assert actual == expected, (fold, role, actual, expected)
        folder = job / "inner_coarse"
        cfg = load(folder / "checkpoints/config.json")
        assert cfg["train_ids"] == fit_ids and cfg["inner_validation_ids"] == held_ids
        assert (
            cfg["forbidden_outer_ids"] == outer_ids
            and cfg["epochs"] == 43
            and cfg["schedule_epochs"] == 80
        )
        cd = load(folder / "coarse_complete.json")
        assert (
            cd["signature"] == scope["stable_hash"](cfg) and not cd["validation_used_for_training"]
        )
        hist = rows(folder / "checkpoints/history.csv")
        assert [int(r["epoch"]) for r in hist] == list(range(1, 44))
        centers = rows(folder / "inner_validation.csv")
        scope["validate_test_rows"](centers, held.to_dict("records"), -1)
        for c in centers:
            gt = next(
                r for r in held.to_dict("records") if int(r["image_id"]) == int(c["image_id"])
            )
            near(c["gt_x"], gt["x"])
            near(c["gt_y"], gt["y"])
        bm = scope["metrics_from_rows"](centers)
        for key, value in bm.items():
            near(value, cd["baseline_metrics"][key])
        misses = []
        for c in centers:
            box = scope["crop_geometry"]((float(c["Fovea_X"]), float(c["Fovea_Y"])), 768)
            if not (box[0] <= float(c["gt_x"]) < box[2] and box[1] <= float(c["gt_y"]) < box[3]):
                misses.append(
                    dict(image_id=int(c["image_id"]), distance_px=float(c["distance_px"]))
                )
        decisions = load(job / "inner_decisions.json")
        assert decisions == done["decisions"]
        for arm in ("wide", "small"):
            folder = job / f"inner_{arm}"
            cfg = load(folder / "config.json")
            expected = scope["fine_config"](
                protocol, fold, arm, "inner", fit_ids, held_ids, outer_ids, 60
            )
            assert cfg == expected
            phase = load(folder / "phase_complete.json")
            hist = rows(folder / "history.csv")
            assert phase["signature"] == scope["stable_hash"](cfg) and phase["epochs"] == 60
            assert [int(r["epoch"]) for r in hist] == list(range(1, 61))
            phase_count += 1
            for mode, best in phase["best"].items():
                minimum = min(hist, key=lambda r: float(r[f"{mode}_score"]))
                near(best["score"], minimum[f"{mode}_score"])
                assert best["epoch"] == int(minimum["epoch"])
                name = arm if mode == "plain" else "small_prior"
                # Match the run's recorded aggregation order; the CSV recomputation
                # above agrees within floating-point rounding.
                expected_decision = scope["selected_inner_epoch"](
                    best, cd["baseline_metrics"]["rmse_coordinate_px"], 0.05
                )
                assert decisions[name] == expected_decision
            targets = {
                m: decisions[arm if m == "plain" else "small_prior"]["epoch"]
                for m in cfg["modes"]
                if decisions[arm if m == "plain" else "small_prior"]["enabled"]
            }
            folder = job / f"outer_{arm}"
            if targets:
                expected = scope["fine_config"](
                    protocol,
                    fold,
                    arm,
                    "outer",
                    list(map(int, dev.image_id)),
                    [],
                    outer_ids,
                    max(targets.values()),
                    targets,
                )
                assert load(folder / "config.json") == expected
                phase = load(folder / "phase_complete.json")
                hist = rows(folder / "history.csv")
                assert phase["signature"] == scope["stable_hash"](expected) and not phase["best"]
                assert [int(r["epoch"]) for r in hist] == list(range(1, max(targets.values()) + 1))
                assert not any("score" in k or "rmse" in k for r in hist for k in r)
                phase_count += 1
            else:
                assert not folder.exists()
        report["inner"].append(
            dict(
                fold=fold,
                fit_n=len(fit),
                validation_n=len(held),
                outer_n=len(outer),
                coarse_rmse=bm["rmse_coordinate_px"],
                crop_misses=misses,
                decisions=decisions,
            )
        )
    comparison = {r["arm"]: r for r in rows(RESULTS / "comparison.csv")}
    for variant in ("wide", "small", "small_prior"):
        all_rows, test_folds = [], {}
        for fold in range(5):
            decision = load(RESULTS / f"outer_{fold}/inner_decisions.json")[variant]
            for subset, frame, cm in [
                ("validation", train[train.fold == fold], coarse["validation"][fold]),
                ("test", test, coarse["test"][fold]),
            ]:
                actual = rows(RESULTS / f"outer_{fold}/{subset}_{variant}.csv")
                frame_records = frame.to_dict("records")
                scope["validate_test_rows"](actual, frame_records, fold)
                if subset == "validation":
                    scope["validate_oof"](actual, frame_records)
                expected_rows = []
                for row in actual:
                    record = next(
                        r for r in frame_records if int(r["image_id"]) == int(row["image_id"])
                    )
                    center = cm[int(row["image_id"])]
                    box = scope["crop_geometry"](
                        (float(center["Fovea_X"]), float(center["Fovea_Y"])), 768
                    )
                    expected_rows.append(
                        scope["make_refinement_row"](
                            record,
                            scope["point_to_patch"](
                                (row["raw_refined_x"], row["raw_refined_y"]), box
                            ),
                            center,
                            box,
                            fold,
                            256.0,
                        )
                    )
                    assert int(row["refine_applied"]) == int(decision["enabled"])
                compare_rows(actual, expected_rows)
                if not decision["enabled"]:
                    compare_rows(actual, scope["identity_rows"](frame_records, cm, fold, decision))
                if subset == "validation":
                    all_rows.extend(actual)
                else:
                    test_folds[fold] = actual
        all_rows = scope["validate_oof"](all_rows, records)
        raw = scope["raw_prediction_rows"](all_rows)
        equal = scope["blend_prediction_rows"](source_oof, all_rows, True)
        for suffix, predicted, label in [
            ("refined", all_rows, "refined"),
            ("equal", equal, "equal"),
            ("raw", raw, "raw_diagnostic"),
        ]:
            compare_rows(rows(RESULTS / f"oof_{variant}_{suffix}.csv"), predicted)
            metrics = scope["metrics_from_rows"](predicted)
            for key, value in metrics.items():
                near(value, comparison[f"{variant}_{label}"][key])
        for suffix, folds in [
            ("refined", test_folds),
            (
                "equal",
                {
                    f: scope["blend_prediction_rows"](
                        list(coarse["test"][f].values()), test_folds[f], False
                    )
                    for f in range(5)
                },
            ),
        ]:
            mean = scope["average_test_folds"](folds, testing)
            compare_rows(rows(RESULTS / f"test_{variant}_{suffix}.csv"), mean)
            with tempfile.TemporaryDirectory(dir=ROOT) as temp:
                target = Path(temp) / "submission.csv"
                scope["write_submission"](
                    ROOT / "2026task3/detection/sample_submission.csv", mean, target
                )
                assert (
                    target.read_bytes()
                    == (RESULTS / f"submission_{variant}_{suffix}.csv").read_bytes()
                )
        report["variants"][variant] = dict(
            raw=scope["metrics_from_rows"](raw),
            refined=scope["metrics_from_rows"](all_rows),
            equal=scope["metrics_from_rows"](equal),
            fallback_ids=[int(r["image_id"]) for r in all_rows if int(r["fallback"])],
        )
    candidates = rows(RESULTS / "oof_small_prior_equal.csv")
    base = {int(r["image_id"]): r for r in source_oof}
    deltas = [
        dict(
            image_id=int(r["image_id"]),
            fold=int(r["fold"]),
            coarse_distance=float(base[int(r["image_id"])]["distance_px"]),
            candidate_distance=float(r["distance_px"]),
            delta_sse=float(r["squared_error"]) - float(base[int(r["image_id"])]["squared_error"]),
        )
        for r in candidates
    ]
    report["paired"] = dict(
        improved=sum(r["delta_sse"] < -1e-6 for r in deltas),
        degraded=sum(r["delta_sse"] > 1e-6 for r in deltas),
        unchanged=sum(abs(r["delta_sse"]) <= 1e-6 for r in deltas),
        total_delta_sse=sum(r["delta_sse"] for r in deltas),
        largest_degradations=sorted(deltas, key=lambda r: r["delta_sse"], reverse=True)[:6],
        largest_gains=sorted(deltas, key=lambda r: r["delta_sse"])[:6],
    )
    report["per_fold"] = [
        dict(
            fold=f,
            coarse=scope["metrics_from_rows"]([r for r in source_oof if int(r["fold"]) == f])[
                "rmse_coordinate_px"
            ],
            **{
                v: scope["metrics_from_rows"](
                    [r for r in rows(RESULTS / f"oof_{v}_equal.csv") if int(r["fold"]) == f]
                )["rmse_coordinate_px"]
                for v in ("wide", "small", "small_prior")
            },
        )
        for f in range(5)
    ]
    report["fine_phases_verified"] = phase_count
    report["coarse_phases_verified"] = 5
    report["coarse"] = scope["metrics_from_rows"](source_oof)
    report["all_checks_passed"] = True
    report["torch_imported"] = "torch" in sys.modules
    assert not report["torch_imported"]
    zip_path = Path("C:/Users/hp/Downloads/task3_refine_v3_results_light.zip")
    if zip_path.is_file():
        report["input_zip"] = dict(
            path=str(zip_path),
            sha256=hashlib.sha256(zip_path.read_bytes()).hexdigest(),
            size_bytes=zip_path.stat().st_size,
        )
    if (ROOT / "official_evaluation.json").is_file():
        report["metric_evidence"] = "../official_evaluation.json"
    (RESULTS / "analysis_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            dict(
                checks="passed",
                fine_phases=phase_count,
                coarse_phases=5,
                coarse_rmse=report["coarse"]["rmse_coordinate_px"],
                candidate_rmse=report["variants"]["small_prior"]["equal"]["rmse_coordinate_px"],
                torch_imported=False,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
