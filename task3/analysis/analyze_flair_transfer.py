"""Audit full local transfer outputs and optionally replay every saved model on CUDA."""

import argparse
import gc
import json
import math
from pathlib import Path
import tempfile
import zipfile

import pandas as pd
import torch
from torch.utils.data import DataLoader

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # 引用 src/ 下的实验代码
import task3_flair_transfer as t
from analyze_refine_v3_results import compare_rows, near

ROOT = Path(__file__).resolve().parent
ROOT = ROOT.parent if ROOT.name == "analysis" else ROOT  # 位于 analysis/ 时回到任务根目录


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    protocol = read_json(output / "protocol.json")
    selection = read_json(output / "selection.json")
    assert protocol["version"] == "flair-transfer-v1" and not protocol["smoke_test"]
    assert read_json(output / "status.json")["state"] == "completed"
    assert (
        selection["complete"]
        and selection["completed_folds"] == list(range(5))
        and not selection["selected_by_outer_scores"]
    )
    assert selection["candidate_public_score"] is None
    for name, digest in protocol["source_sha256"].items():
        assert t.file_sha(ROOT / name) == digest
    sources = read_json(output / "pretraining_sources.json")
    for recipe, spec in t.WEIGHT_SPECS.items():
        assert sources[recipe]["url"] == spec["url"] and sources[recipe]["sha256"].startswith(
            spec["sha256_prefix"]
        )
        assert t.file_sha(sources[recipe]["path"]) == sources[recipe]["sha256"]
    preflight = read_json(output / "gpu_preflight.json")
    assert preflight["passed"]
    data = ROOT / "2026task3/detection"
    train = pd.read_csv(output / "split.csv")
    test = pd.read_csv(output / "manifest.csv").query("subset=='test'")
    prepared = dict(data=data, train=train, test=test, output=output, protocol=protocol)
    cache = t.CanvasCache(data)
    comparison = {r["model"]: r for r in t.read_rows(output / "comparison.csv")}
    report = dict(
        all_checks_passed=False,
        pretraining_sources=sources,
        gpu_preflight=preflight,
        folds=[],
        metrics={},
        replay_requested=args.replay,
        replay_models=0,
        replay_max_coordinate_difference_px=0.0,
        limitations=[
            selection["limitation"],
            "No competition public score is available for the new candidate.",
        ],
    )
    all_predictions = {r: [] for r in t.RECIPES}
    test_predictions = {r: {} for r in t.RECIPES}
    seconds = 0.0
    torch.set_num_threads(2)
    t.seed_everything(55000)
    torch.set_num_threads(2)

    def replay(path, cfg, frame, tag, reference):
        if not args.replay:
            return
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert state["signature"] == t.stable_hash(cfg) and state["epoch"] == (
            cfg["epochs"] if cfg["role"] == "outer" else cfg["selected_epoch"]
        )
        model = t.TransferHeatmap().to("cuda:0")
        model.load_state_dict(state["model"], strict=True)
        del state
        loader = DataLoader(
            t.CanvasDataset(frame.to_dict("records"), cache, cfg, False),
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=0,
        )
        actual = t.predict_transfer(model, loader, tag)
        lookup = {int(r["image_id"]): r for r in reference}
        for row in actual:
            old = lookup[int(row["image_id"])]
            for axis in ("X", "Y"):
                delta = abs(float(row[f"Fovea_{axis}"]) - float(old[f"Fovea_{axis}"]))
                report["replay_max_coordinate_difference_px"] = max(
                    report["replay_max_coordinate_difference_px"], delta
                )
                assert delta <= 0.001, (path, tag, row["image_id"], axis, delta)
        del model, loader
        gc.collect()
        torch.cuda.empty_cache()
        return actual

    for fold in range(5):
        dev, fit, held, outer, design = t.fold_design(prepared, fold)
        job = output / f"fold_{fold}"
        done = read_json(job / "fold_complete.json")
        assert read_json(job / "design.json") == design and done["signature"] == t.stable_hash(
            design
        )
        assert done["predictions_complete"] and not done["outer_labels_used_for_epoch_selection"]
        role_rows = t.read_rows(job / "split_roles.csv")
        for role, frame in [
            ("inner_train", fit),
            ("inner_validation", held),
            ("outer_validation", outer),
        ]:
            assert {
                int(r["image_id"]): r["pixel_sha256"] for r in role_rows if r["role"] == role
            } == {int(r["image_id"]): r["pixel_sha256"] for r in frame.to_dict("records")}
        required = []
        entry = dict(fold=fold, recipes={})
        heads = {}
        for recipe in t.RECIPES:
            folder = job / f"inner_{recipe}"
            cfg = read_json(folder / "config.json")
            phase = read_json(folder / "phase_complete.json")
            hist = t.read_rows(folder / "history.csv")
            expected = t.phase_config(
                protocol, fold, recipe, "inner", fit, held, design["outer_ids"], 40
            )
            assert cfg == expected
            assert (
                phase["signature"] == t.stable_hash(cfg)
                and phase["epochs"] == 40
                and [int(r["epoch"]) for r in hist] == list(range(1, 41))
            )
            best = min(hist, key=lambda r: float(r["rmse_coordinate_px"]))
            selected = phase["best"]
            assert selected["epoch"] == int(best["epoch"])
            near(selected["score"], best["rmse_coordinate_px"])
            assert done["decisions"][recipe] == selected
            heads[recipe] = read_json(folder / "initial_head.json")["sha256"]
            seconds += sum(float(r["seconds"]) for r in hist)
            required.append(f"inner_{recipe}/best.pt")
            if args.replay:
                # Keep the checkpoint signature tied to the exact original config.
                state = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
                assert (
                    state["signature"] == t.stable_hash(cfg)
                    and state["epoch"] == selected["epoch"]
                    and state["selection"] == selected
                )
                model = t.TransferHeatmap().to("cuda:0")
                model.load_state_dict(state["model"], strict=True)
                del state
                loader = DataLoader(
                    t.CanvasDataset(held.to_dict("records"), cache, cfg, False),
                    batch_size=cfg["batch_size"],
                    shuffle=False,
                    num_workers=0,
                )
                replayed = t.predict_transfer(model, loader, -1)
                t.validate_test_rows(replayed, held.to_dict("records"), -1)
                near(t.metrics_from_rows(replayed)["rmse_coordinate_px"], selected["score"])
                report["replay_models"] += 1
                del model, loader
                gc.collect()
                torch.cuda.empty_cache()
            folder = job / f"outer_{recipe}"
            cfg = read_json(folder / "config.json")
            phase = read_json(folder / "phase_complete.json")
            hist = t.read_rows(folder / "history.csv")
            assert cfg == t.phase_config(
                protocol, fold, recipe, "outer", dev, None, design["outer_ids"], selected["epoch"]
            )
            assert (
                phase["signature"] == t.stable_hash(cfg)
                and phase["epochs"] == selected["epoch"]
                and not phase["best"]
            )
            assert [int(r["epoch"]) for r in hist] == list(range(1, selected["epoch"] + 1))
            assert not any("rmse_coordinate_px" in r for r in hist)
            assert all(
                r["backbone_stage"] == ("head_only" if int(r["epoch"]) <= 5 else "layer3_layer4")
                for r in hist
            )
            seconds += sum(float(r["seconds"]) for r in hist)
            required.append(f"outer_{recipe}/final.pt")
            val = t.validate_oof(
                t.read_rows(job / f"validation_{recipe}.csv"), outer.to_dict("records")
            )
            t.validate_test_rows(val, outer.to_dict("records"), fold)
            pred = t.read_rows(job / f"test_{recipe}.csv")
            t.validate_test_rows(pred, test.to_dict("records"), fold)
            all_predictions[recipe].extend(val)
            test_predictions[recipe][fold] = pred
            replay(folder / "final.pt", cfg, outer, fold, val)
            replay(folder / "final.pt", cfg, test, fold, pred)
            if args.replay:
                report["replay_models"] += 1
            entry["recipes"][recipe] = dict(
                selected_epoch=selected["epoch"],
                inner_rmse=selected["score"],
                outer=t.metrics_from_rows(val),
            )
        assert len(set(heads.values())) == 1
        assert sorted(done["required_models"]) == sorted(required) and all(
            (job / p).is_file() for p in required
        )
        report["folds"].append(entry)
    for recipe in t.RECIPES:
        predictions = t.validate_oof(all_predictions[recipe], train.to_dict("records"))
        compare_rows(t.read_rows(output / f"oof_{recipe}.csv"), predictions)
        metrics = t.metrics_from_rows(predictions)
        report["metrics"][recipe] = metrics
        for field, value in metrics.items():
            near(value, comparison[recipe][field])
        for r in predictions:
            near(r["dx"], float(r["Fovea_X"]) - float(r["gt_x"]))
            near(r["dy"], float(r["Fovea_Y"]) - float(r["gt_y"]))
            near(r["distance_px"], math.hypot(float(r["dx"]), float(r["dy"])))
            near(r["squared_error"], float(r["dx"]) ** 2 + float(r["dy"]) ** 2)
        mean = t.average_test_folds(test_predictions[recipe], test.to_dict("records"))
        compare_rows(t.read_rows(output / f"test_{recipe}.csv"), mean)
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            target = Path(temp) / "submission.csv"
            t.write_submission(data / "sample_submission.csv", mean, target)
            assert target.read_bytes() == (output / f"submission_{recipe}_5fold.csv").read_bytes()
    equal = t.read_rows(output / "oof_equal.csv")
    equal_metrics = t.metrics_from_rows(equal)
    report["metrics"]["fixed_equal"] = equal_metrics
    maps = {r: {int(p["image_id"]): p for p in all_predictions[r]} for r in t.RECIPES}
    for row in equal:
        key = int(row["image_id"])
        for a in ("X", "Y"):
            near(row[f"Fovea_{a}"], sum(float(maps[r][key][f"Fovea_{a}"]) for r in t.RECIPES) / 2)
    for field, value in equal_metrics.items():
        near(value, comparison["fixed_equal"][field])
    assert (output / "submission_flair_candidate.csv").read_bytes() == (
        output / "submission_flair_5fold.csv"
    ).read_bytes()
    backup = output.parent / "task3_flair_transfer_full_backup.zip"
    with zipfile.ZipFile(backup) as archive:
        names = set(archive.namelist())
        for f in range(5):
            done = read_json(output / f"fold_{f}/fold_complete.json")
            assert all(f"fold_{f}/{p}" in names for p in done["required_models"])
        assert not any("_pretrained/" in name for name in names)
    best = read_json(ROOT / "current_best_public.json")
    assert t.file_sha(best["file"]) == best["submission_sha256"]
    report.update(
        all_checks_passed=True,
        training_and_inner_evaluation_seconds=seconds,
        best_public_unchanged=best["score"],
        candidate_sha256=t.file_sha(output / "submission_flair_candidate.csv"),
    )
    (output / "analysis_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            dict(
                passed=True,
                metrics=report["metrics"],
                replayed_models=report["replay_models"],
                max_replay_coordinate_difference_px=report["replay_max_coordinate_difference_px"],
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
