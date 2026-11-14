"""Audit the received v4 light results without loading Torch or model tensors."""

from collections import defaultdict
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys
import tempfile
import zipfile
import csv

import numpy as np
import pandas as pd

from analyze_refine_v3_results import definitions, rows, load, near, compare_rows

ROOT = Path(__file__).resolve().parent
ROOT = ROOT.parent if ROOT.name == "analysis" else ROOT  # 位于 analysis/ 时回到任务根目录
RESULTS = ROOT / "context_v4_results_20261027"


def main():
    assert "torch" not in sys.modules
    scope = dict(
        Path=Path,
        PurePosixPath=PurePosixPath,
        hashlib=hashlib,
        io=io,
        json=json,
        math=math,
        statistics=statistics,
        sys=sys,
        zipfile=zipfile,
        csv=csv,
        np=np,
        pd=pd,
        defaultdict=defaultdict,
        CONTEXT_VERSION="context-refine-v4",
        CONTEXT_ARMS=("local", "context"),
    )
    groups = {
        "task3_pipeline.py": {"assign_folds"},
        "task3_kaggle_cv.py": {
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
        "task3_kaggle_refine.py": {
            "CoarseBundle",
            "validate_coarse_bundle",
            "crop_geometry",
            "point_to_patch",
            "patch_to_original",
            "make_refinement_row",
            "blend_prediction_rows",
        },
        "task3_refine_diagnostic_v3.py": {
            "inner_partition",
            "selected_inner_epoch",
            "identity_rows",
            "raw_prediction_rows",
        },
        "task3_context_refine_v4.py": {
            "context_config",
            "validate_context_config",
            "context_design",
        },
    }
    for path, names in groups.items():
        exec(definitions(ROOT / path, names), scope)
    protocol = load(RESULTS / "protocol.json")
    selection = load(RESULTS / "selection.json")
    assert protocol["version"] == "context-refine-v4" and not protocol["smoke_test"]
    assert selection["complete"] and selection["completed_folds"] == list(range(5))
    assert selection["metric"] == "rmse_coordinate_px" and not selection["selected_by_outer_scores"]
    assert (
        selection["candidate_public_score"] is None
        and load(RESULTS / "status.json")["state"] == "completed"
    )
    for name, digest in protocol["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    preflight = load(RESULTS / "gpu_preflight.json")
    assert preflight["passed"] and preflight["zero_init_max_abs_difference"] <= 1e-3
    assert preflight["gradients"] == {
        "context3.weight": True,
        "context4.weight": True,
    } and math.isfinite(preflight["loss"])
    hotfix = load(RESULTS / "hotfix_application.json")
    assert hotfix["driver_sha256"] == protocol["source_sha256"]["task3_context_refine_v4.py"]
    train = pd.read_csv(RESULTS / "split.csv")
    test = pd.read_csv(RESULTS / "manifest.csv").query("subset=='test'")
    records, testing = train.to_dict("records"), test.to_dict("records")
    coarse = scope["validate_coarse_bundle"](ROOT / "kaggle_cv_results_20261025", records, testing)
    assert coarse["fingerprint"] == protocol["coarse_fingerprint"]
    baseline = scope["validate_oof"](
        [r for f in range(5) for r in coarse["validation"][f].values()], records
    )
    compare_rows(rows(RESULTS / "oof_coarse512.csv"), baseline)
    assert (RESULTS / "submission_coarse512.csv").read_bytes() == (
        ROOT / "submission_resnet34_512_control.csv"
    ).read_bytes()
    assert (RESULTS / "submission_context_v4_candidate.csv").read_bytes() == (
        RESULTS / "submission_context_equal.csv"
    ).read_bytes()
    prepared = dict(train=train, protocol=protocol)
    provenance = []
    inner_records = []
    fine_phases = 0
    training_seconds = 0.0
    expected_models = 0
    for fold in range(5):
        dev, fit, held, outer, design = scope["context_design"](prepared, fold)
        job = RESULTS / f"outer_{fold}"
        done = load(job / "fold_complete.json")
        assert load(job / "design.json") == design and done["signature"] == scope["stable_hash"](
            design
        )
        assert done["predictions_complete"] and not done["outer_labels_used_for_epoch_selection"]
        roles = rows(job / "split_roles.csv")
        for role, frame in [
            ("inner_train", fit),
            ("inner_validation", held),
            ("outer_validation", outer),
        ]:
            expected = {int(r["image_id"]): r["pixel_sha256"] for r in frame.to_dict("records")}
            actual = {int(r["image_id"]): r["pixel_sha256"] for r in roles if r["role"] == role}
            assert actual == expected
        old = ROOT / f"refine_v3_results_20261027/outer_{fold}/inner_coarse"
        inner_cfg = load(old / "checkpoints/config.json")
        inner_complete = load(old / "coarse_complete.json")
        cached = rows(old / "inner_validation.csv")
        assert inner_complete["signature"] == scope["stable_hash"](inner_cfg)
        assert (
            inner_cfg["train_ids"] == design["fit_ids"]
            and inner_cfg["inner_validation_ids"] == design["inner_val_ids"]
            and inner_cfg["forbidden_outer_ids"] == design["outer_ids"]
        )
        compare_rows(rows(RESULTS / f"inner_cache/fold_{fold}/inner_validation.csv"), cached)
        scope["validate_test_rows"](cached, held.to_dict("records"), -1)
        provenance.append(
            dict(fold=fold, config=inner_cfg, complete=inner_complete, predictions=cached)
        )
        inner_baseline = scope["metrics_from_rows"](cached)["rmse_coordinate_px"]
        decisions = load(job / "inner_decisions.json")
        assert decisions == done["decisions"]
        required = []
        for arm in ("local", "context"):
            folder = job / f"inner_{arm}"
            cfg = load(folder / "config.json")
            expected = scope["context_config"](
                protocol,
                fold,
                arm,
                "inner",
                design["fit_ids"],
                design["inner_val_ids"],
                design["outer_ids"],
                60,
            )
            assert cfg == expected
            phase = load(folder / "phase_complete.json")
            hist = rows(folder / "history.csv")
            assert phase["signature"] == scope["stable_hash"](cfg) and phase["epochs"] == 60
            assert [int(r["epoch"]) for r in hist] == list(range(1, 61))
            best_row = min(hist, key=lambda r: float(r["score"]))
            best = phase["best"]
            assert best["epoch"] == int(best_row["epoch"])
            for field in ("score", "refined_rmse", "raw_rmse", "mean_distance", "fallbacks"):
                near(best[field], best_row[field])
            decision = decisions[arm]
            near(decision["inner_baseline"], inner_baseline)
            assert decision == scope["selected_inner_epoch"](best, decision["inner_baseline"], 0.05)
            fine_phases += 1
            expected_models += 1
            required.append(f"inner_{arm}/best.pt")
            training_seconds += sum(float(r["seconds"]) for r in hist)
            folder = job / f"outer_{arm}"
            if decision["enabled"]:
                expected = scope["context_config"](
                    protocol,
                    fold,
                    arm,
                    "outer",
                    list(map(int, dev.image_id)),
                    [],
                    design["outer_ids"],
                    decision["epoch"],
                )
                assert load(folder / "config.json") == expected
                phase = load(folder / "phase_complete.json")
                hist = rows(folder / "history.csv")
                assert (
                    phase["signature"] == scope["stable_hash"](expected)
                    and phase["epochs"] == decision["epoch"]
                    and not phase["best"]
                )
                assert [int(r["epoch"]) for r in hist] == list(range(1, decision["epoch"] + 1))
                assert not any(
                    k in r for r in hist for k in ("score", "rmse_coordinate_px", "refined_rmse")
                )
                fine_phases += 1
                expected_models += 1
                required.append(f"outer_{arm}/final.pt")
                training_seconds += sum(float(r["seconds"]) for r in hist)
            else:
                assert not folder.exists()
        assert sorted(done["required_models"]) == sorted(required)
        inner_records.append(
            dict(
                fold=fold,
                baseline=inner_baseline,
                fit_n=len(fit),
                held_n=len(held),
                decisions=decisions,
            )
        )
    assert scope["stable_hash"](provenance) == protocol["inner_fingerprint"]
    comparison = {r["arm"]: r for r in rows(RESULTS / "comparison.csv")}
    summary = dict(
        completed_folds=5,
        fine_phases=fine_phases,
        required_weight_files=expected_models,
        source_sha256_verified=protocol["source_sha256"],
        gpu_preflight=preflight,
        coarse=scope["metrics_from_rows"](baseline),
        inner=inner_records,
        variants={},
        training_and_evaluation_seconds_sum=training_seconds,
        metric="Official coordinate RMSE over flattened X/Y rows",
        metric_evidence="../official_evaluation.json",
        limitations=[
            selection["limitation"],
            "Light ZIP contains no model tensors; no inference replay or learned fusion-weight inspection was performed.",
            "Local and context choose different inner epochs, so their scores compare complete protocols, not a fixed-weight context intervention.",
            "One seed/split does not establish that global context is inherently ineffective.",
        ],
    )
    for key, value in summary["coarse"].items():
        near(value, comparison["coarse512"][key])
    for arm in ("local", "context"):
        collected = []
        test_folds = {}
        active = 0
        for fold in range(5):
            job = RESULTS / f"outer_{fold}"
            decision = inner_records[fold]["decisions"][arm]
            active += int(decision["enabled"])
            for subset, frame, centers in [
                ("validation", train[train.fold == fold], coarse["validation"][fold]),
                ("test", test, coarse["test"][fold]),
            ]:
                actual = rows(job / f"{subset}_{arm}.csv")
                frame_records = frame.to_dict("records")
                scope["validate_test_rows"](actual, frame_records, fold)
                if subset == "validation":
                    scope["validate_oof"](actual, frame_records)
                record_map = {int(r["image_id"]): r for r in frame_records}
                expected = []
                for r in actual:
                    key = int(r["image_id"])
                    center = centers[key]
                    box = scope["crop_geometry"](
                        (float(center["Fovea_X"]), float(center["Fovea_Y"])), 768
                    )
                    expected.append(
                        scope["make_refinement_row"](
                            record_map[key],
                            scope["point_to_patch"]((r["raw_refined_x"], r["raw_refined_y"]), box),
                            center,
                            box,
                            fold,
                            256.0,
                        )
                    )
                    assert int(r["refine_applied"]) == int(decision["enabled"]) and r["arm"] == arm
                    assert r["decoder"] == ("prior" if decision["enabled"] else "identity")
                compare_rows(actual, expected)
                if not decision["enabled"]:
                    compare_rows(
                        actual, scope["identity_rows"](frame_records, centers, fold, decision)
                    )
                if subset == "validation":
                    collected.extend(actual)
                else:
                    test_folds[fold] = actual
        collected = scope["validate_oof"](collected, records)
        raw = scope["raw_prediction_rows"](collected)
        equal = scope["blend_prediction_rows"](baseline, collected, True)
        for suffix, preds, key in [
            ("refined", collected, "refined"),
            ("raw", raw, "raw_diagnostic"),
            ("equal", equal, "equal"),
        ]:
            compare_rows(rows(RESULTS / f"oof_{arm}_{suffix}.csv"), preds)
            for field, value in scope["metrics_from_rows"](preds).items():
                near(value, comparison[f"{arm}_{key}"][field])
            assert int(comparison[f"{arm}_{key}"]["active_folds"]) == active
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
            compare_rows(rows(RESULTS / f"test_{arm}_{suffix}.csv"), mean)
            with tempfile.TemporaryDirectory(dir=ROOT) as temp:
                target = Path(temp) / "submission.csv"
                scope["write_submission"](
                    ROOT / "2026task3/detection/sample_submission.csv", mean, target
                )
                assert (
                    target.read_bytes() == (RESULTS / f"submission_{arm}_{suffix}.csv").read_bytes()
                )
        base_map = {int(r["image_id"]): r for r in baseline}
        changes = [
            dict(
                image_id=int(r["image_id"]),
                fold=int(r["fold"]),
                coarse_distance=float(base_map[int(r["image_id"])]["distance_px"]),
                equal_distance=float(r["distance_px"]),
                delta_sse=float(r["squared_error"])
                - float(base_map[int(r["image_id"])]["squared_error"]),
            )
            for r in equal
        ]
        summary["variants"][arm] = dict(
            refined=scope["metrics_from_rows"](collected),
            raw=scope["metrics_from_rows"](raw),
            equal=scope["metrics_from_rows"](equal),
            active_folds=active,
            fallback_ids=[int(r["image_id"]) for r in collected if int(r["fallback"])],
            improved=sum(r["delta_sse"] < -1e-6 for r in changes),
            degraded=sum(r["delta_sse"] > 1e-6 for r in changes),
            unchanged=sum(abs(r["delta_sse"]) <= 1e-6 for r in changes),
            total_delta_sse=sum(r["delta_sse"] for r in changes),
            largest_degradations=sorted(changes, key=lambda r: r["delta_sse"], reverse=True)[:6],
            largest_gains=sorted(changes, key=lambda r: r["delta_sse"])[:6],
        )
    summary["per_fold"] = [
        dict(
            fold=f,
            **{
                name: scope["metrics_from_rows"](
                    [r for r in rows(RESULTS / f"oof_{file}.csv") if int(r["fold"]) == f]
                )["rmse_coordinate_px"]
                for name, file in [
                    ("coarse", "coarse512"),
                    ("local", "local_equal"),
                    ("context", "context_equal"),
                ]
            },
        )
        for f in range(5)
    ]
    local = summary["variants"]["local"]["equal"]["rmse_coordinate_px"]
    context = summary["variants"]["context"]["equal"]["rmse_coordinate_px"]
    base = summary["coarse"]["rmse_coordinate_px"]
    summary["context_rmse_increase_vs_coarse_percent"] = 100 * (context / base - 1)
    summary["context_rmse_increase_vs_local_percent"] = 100 * (context / local - 1)
    best = load(ROOT / "current_best_public.json")
    assert hashlib.sha256(Path(best["file"]).read_bytes()).hexdigest() == best["submission_sha256"]
    summary["best_public_csv_unchanged"] = True
    summary["best_public_score"] = best["score"]
    archive = Path("C:/Users/hp/Downloads/task3_context_v4_results_light.zip")
    summary["input_zip"] = dict(
        path=str(archive),
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        size_bytes=archive.stat().st_size,
    )
    summary["all_checks_passed"] = True
    summary["torch_imported"] = "torch" in sys.modules
    assert not summary["torch_imported"]
    (RESULTS / "analysis_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            dict(
                checks="passed",
                fine_phases=fine_phases,
                coarse_rmse=base,
                local_equal_rmse=local,
                context_equal_rmse=context,
                context_change_vs_coarse_pct=summary["context_rmse_increase_vs_coarse_percent"],
                torch_imported=False,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
