"""Audit returned hyperparameter results; never execute returned source/weights."""

import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
ROOT = ROOT.parent if ROOT.name == "analysis" else ROOT  # 位于 analysis/ 时回到任务根目录
sys.path.insert(0, str(ROOT / ".ablation_test_deps"))

import csv
import json
import math
import zipfile
from collections import Counter
import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

import analyze_core_band_results as common

OUT = ROOT / "hyperparameter_results_20261107_ddbe2303"
REC = ROOT / "recovery_results_20261103_5596e091"
DATA = ROOT / "2026task-4/segmentation"
METRICS, GRID = common.METRICS, common.GRID
sha, load, table, digest, close = common.sha, common.load, common.table, common.digest, common.close
SETTINGS = {
    "H0_reference": dict(lr_multiplier=1.0, bce_weight=0.5, factor="reference"),
    "H1_lr_half": dict(lr_multiplier=0.5, bce_weight=0.5, factor="learning_rate"),
    "H2_lr_double": dict(lr_multiplier=2.0, bce_weight=0.5, factor="learning_rate"),
    "H3_bce_quarter": dict(lr_multiplier=1.0, bce_weight=0.25, factor="bce_weight"),
    "H4_bce_three_quarters": dict(lr_multiplier=1.0, bce_weight=0.75, factor="bce_weight"),
}
NEW = tuple(list(SETTINGS)[1:])


def write(name, rows):
    with (OUT / name).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    checks = {}
    archive = OUT / "received_hyperparameter_analysis.zip"
    assert sha(archive) == "ddbe230373c31332c2def53029eeada851d44623c23fd270ecde56629df42a07"
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == 198
        assert sum(n.startswith("oof_binary_masks/") for n in z.namelist()) == 100
        assert sum(n.startswith("masks_512_white_vessels/") for n in z.namelist()) == 20
        for i in z.infolist():
            p = PurePosixPath(i.filename)
            assert (
                not p.is_absolute()
                and ".." not in p.parts
                and "\\" not in i.filename
                and ":" not in i.filename
            )
            assert ((i.external_attr >> 16) & 0o170000) != 0o120000 and p.suffix not in (
                ".pt",
                ".npz",
            )
            assert (OUT / i.filename).read_bytes() == z.read(i)
    checks["zip_crc_paths_and_all198_original_files_verified_unchanged"] = True
    cfg_all = load(OUT / "hyperparameter_config.json")
    cfg = cfg_all["config"]
    status = load(OUT / "hyperparameter_status.json")
    assert cfg_all["settings"] == SETTINGS
    assert (
        cfg == load(REC / "recovery_config.json")["config"] == cfg_all["reference_info"]["config"]
    )
    assert not cfg_all["engineering_only"] and not status["engineering_only"]
    source_names = {
        "task4_hyperparameter_ablation.py": "task4_hyperparameter_ablation_before_source_copy_fix.py",
        "task4_hyperparameter_ablation_before_precision_repair.py": "task4_hyperparameter_ablation_before_precision_repair.py",
        "task4_hyperparameter_numerics.py": "task4_hyperparameter_numerics.py",
        "task4_pipeline.py": "task4_pipeline.py",
        "task4_kaggle_v2.py": "task4_kaggle_v2.py",
        "task4_controlled_ablation.py": "task4_controlled_ablation.py",
    }
    for returned, local in source_names.items():
        assert sha(OUT / "runtime_source" / returned) == sha(ROOT / local)
    assert cfg_all["source_sha256"] == sha(OUT / "runtime_source/task4_hyperparameter_ablation.py")
    assert cfg_all["numerical_source_sha256"] == sha(
        OUT / "runtime_source/task4_hyperparameter_numerics.py"
    )
    for n, h in cfg_all["helper_source_sha256"].items():
        assert sha(OUT / "runtime_source" / n) == h
    records = load(OUT / "reference_records.json")
    assert records == {str(f): load(REC / f"L1_proxy/fold{f}/completed.json") for f in range(5)}
    frozen = load(ROOT / "core_band_reference_frozen.json")
    assert (
        digest(records) == cfg_all["reference_records_hash"] == frozen["identity"]["records_hash"]
    )
    for key in ("rows", "grid"):
        assert digest(frozen[key]) == cfg_all[f"reference_{key}_hash"]
    reference_rows = table(OUT / "reference_L1_per_image.csv")
    reference_grid = table(OUT / "reference_L1_threshold_grid.csv")
    common.check_frozen_rows(reference_rows, frozen["rows"])
    common.check_frozen_rows(reference_grid, frozen["grid"])
    env = load(OUT / "execution_environment.json")
    assert (
        env == cfg_all["reference_info"]["environment"] == load(REC / "execution_environment.json")
    )
    assert (
        not status["environment_differences"] and status["environment_matches_historical_reference"]
    )
    assert (
        load(OUT / "pretrained_source.json")["sha256"]
        == "b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f"
    )
    assert status["complete_all_settings"] and status["completed_settings"] == list(NEW)
    assert status["reference_retrained"] is False and status["new_training_settings"] == 4
    for k in (
        "public_scores_used",
        "automatic_final_winner_selection",
        "submitted_to_competition",
        "model_weight_download_required",
    ):
        assert status[k] is False
    checks["six_sources_frozen_settings_reference_and_recorded_environment_match"] = True

    manifest = table(OUT / "manifest.csv")
    assert manifest == table(REC / "manifest.csv") and len(manifest) == 40
    assignment = {r["id"]: int(r["fold"]) for r in manifest if r["split"] == "train"}
    truth, clean, stable = {}, {}, []
    for r in manifest:
        image = DATA / r["split"] / "image" / f'{r["id"]}.jpg'
        assert sha(image) == r["image_sha256"]
        if r["split"] == "train":
            lp = DATA / "train/label" / f'{r["id"]}.jpg'
            assert sha(lp) == r["label_sha256"]
            with Image.open(lp) as im:
                im = im.convert("L")
                c = cv2.resize(
                    (np.asarray(im) > 127).astype(np.uint8),
                    (512, 512),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                p = np.asarray(im.resize((512, 512), Image.Resampling.BILINEAR)) > 0
            assert not (c & ~p).any()
            sc, sp = skeletonize(c), skeletonize(p)
            thin = sc & (cv2.distanceTransform(c.astype(np.uint8), cv2.DIST_L2, 5) <= 1.5)
            truth[r["id"]] = c, p, sc, sp, thin
            clean[r["id"]] = c
        row = {
            k: r[k]
            for k in ("split", "id", "width", "height", "image_sha256", "label_sha256", "fold")
        }
        if row["split"] == "train":
            row["fold"] = int(row["fold"])
        stable.append(row)
    assert digest(stable) == cfg_all["data_fingerprint"] == frozen["identity"]["data_fingerprint"]
    assert (
        sha(DATA / "segmentation_to_csv.py")
        == load(OUT / "audit_summary.json")["converter"]["source_sha256"]
    )
    plans = {
        str(f): common.rebuild_plans(records[str(f)]["train_ids"], clean, f, cfg) for f in range(5)
    }
    assert all(plans[str(f)] == records[str(f)]["plan_hashes"] for f in range(5))
    histories = {}
    seconds = Counter()
    for name in NEW:
        assert status["completed_folds"][name] == list(range(5))
        expected_config = dict(cfg)
        for k in ("encoder_lr", "decoder_lr"):
            expected_config[k] *= SETTINGS[name]["lr_multiplier"]
        for f in range(5):
            folder = OUT / name / f"fold{f}"
            done = load(folder / "completed.json")
            ref = records[str(f)]
            assert (
                done["epoch"] == 80
                and done["setting"] == name
                and done["parameters"] == SETTINGS[name]
            )
            assert done["config"] == expected_config
            for k in ("initial_model_hash", "train_ids", "val_ids", "plan_hashes"):
                assert done[k] == ref[k]
            assert done["plan_hashes"] == plans[str(f)]
            assert set(done["val_ids"]) == {i for i in assignment if assignment[i] == f}
            assert set(done["train_ids"]) == set(assignment) - set(done["val_ids"])
            assert sha(folder / "history.csv") == done["sha256"]["history.csv"]
            if "numerical_policy" in done:
                assert done["numerical_policy"] == cfg_all["numerical_policy"]
                assert done["numerical_source_sha256"] == cfg_all["numerical_source_sha256"]
            h = table(folder / "history.csv")
            histories[name, f] = h
            assert [int(r["epoch"]) for r in h] == list(range(1, 81))
            assert [r["plan_sha256"] for r in h] == plans[str(f)]
            for r in h:
                for k in ("loss", "bce_loss", "dice_loss", "seconds"):
                    assert math.isfinite(float(r[k])) and float(r[k]) >= 0
                beta = SETTINGS[name]["bce_weight"]
                close(
                    r["loss"],
                    beta * float(r["bce_loss"]) + (1 - beta) * float(r["dice_loss"]),
                    1e-6,
                )
                monitored = int(r["epoch"]) % 10 == 0
                for k in ("monitor_clean_05", "monitor_proxy_05"):
                    assert bool(r[k]) == monitored
                    if monitored:
                        assert 0 <= float(r[k]) <= 1
                step = int(r["epoch"]) * 32 - 1
                warm = 96
                total = 2560
                factor = (
                    (step + 1) / warm
                    if step < warm
                    else 0.02
                    + 0.98
                    * 0.5
                    * (1 + math.cos(math.pi * min(1, (step - warm) / (total - warm - 1))))
                )
                for k in ("encoder_lr", "decoder_lr"):
                    close(r[k], expected_config[k] * factor, 1e-14)
                seconds[name] += float(r["seconds"])
    checks["original_data_and_all20_final80_folds_1600_logs_loss_and_LR_arithmetic_verified"] = True
    checks["all1600_plan_entries_match_400_independently_rebuilt_unique_epoch_plans"] = True
    print(
        "Verified frozen sources/data and all 20 final80 folds, learning rates, losses and sampling plans.",
        flush=True,
    )

    old_config = load(OUT / "hyperparameter_config_before_precision_repair.json")
    assert old_config["source_sha256"] == sha(
        ROOT / "task4_hyperparameter_ablation_before_precision_repair.py"
    )
    assert {
        k: v
        for k, v in cfg_all.items()
        if k not in ("source_sha256", "numerical_policy", "numerical_source_sha256")
    } == {k: v for k, v in old_config.items() if k != "source_sha256"}
    migration = load(OUT / "precision_repair_migration.json")
    assert migration["original_source_sha256"] == old_config["source_sha256"]
    assert migration["repaired_source_sha256"] == cfg_all["source_sha256"]
    assert len(migration["completed_folds_preserved"]) == 7
    for folder, h in migration["completed_folds_preserved"].items():
        assert sha(OUT / folder / "completed.json") == h
    guard = load(OUT / "source_copy_guard.json")
    assert (
        guard["underlying_source_sha256"] == cfg_all["source_sha256"]
        and guard["scope"] == "HP.shutil only"
    )
    assert (
        not guard["training_hyperparameters_changed"] and not guard["model_or_optimizer_modified"]
    )
    events_path = OUT / "H2_lr_double/fold2/precision_events.json"
    events = load(events_path)["events"]
    assert len(events) == 150 and len({(e["epoch"], e["batch"]) for e in events}) == 150
    assert (
        sha(events_path)
        == load(OUT / "H2_lr_double/fold2/completed.json")["sha256"]["precision_events.json"]
    )
    assert len(list(OUT.glob("*/fold*/*precision_events.json"))) == 1
    for e in events:
        assert e["setting"] == "H2_lr_double" and e["fold"] == 2 and e["stage"] == "training"
        assert 17 <= e["epoch"] <= 28 and 1 <= e["batch"] <= 32
        assert (
            e["event"] == "amp_loss_nonfinite"
            and e["fp32_retry_succeeded"]
            and e["nonfinite_model_state"] == []
        )
        assert e["loss"]["nonfinite"] == 1 and e["fp32_loss"]["nonfinite"] == 0
        assert math.isfinite(e["fp32_loss"]["finite_abs_max"])
        assert all(x["dtype"] == "torch.float16" and x["nonfinite"] > 0 for x in e["outputs"])
        assert all(x["dtype"] == "torch.float32" and x["nonfinite"] == 0 for x in e["fp32_outputs"])
    assert (events[0]["epoch"], events[0]["batch"]) == (17, 25)
    assert (events[-1]["epoch"], events[-1]["batch"]) == (28, 11)
    ns = status["numerical_repair_summary"]
    assert ns["total_events"] == ns["fp32_retries"] == 150
    assert ns["per_file"] == [
        dict(
            file="H2_lr_double/fold2/precision_events.json",
            event_count=150,
            fp32_retries=150,
            failed_retries=0,
        )
    ]
    numerical = dict(
        events=150,
        failed_retries=0,
        first_epoch=17,
        first_batch=25,
        last_epoch=28,
        last_batch=11,
        per_epoch=dict(sorted(Counter(e["epoch"] for e in events).items())),
        fold_planned_batch_fraction=150 / (80 * 32),
        all_new_planned_batch_fraction=150 / (20 * 80 * 32),
        precise_failing_operator_unknown=True,
        independently_replayed_T4_batch=False,
    )
    checks[
        "seven_completed_records_preserved_and_all150_successful_FP32_retries_verified_from_logs"
    ] = True

    rows = table(OUT / "hyperparameter_per_image.csv")
    folds = table(OUT / "hyperparameter_per_fold.csv")
    summary = table(OUT / "hyperparameter_summary.csv")
    assert (len(rows), len(folds), len(summary)) == (1200, 300, 60)
    lookup, fl = common.check_aggregates(rows, folds, summary)
    experiments = {
        f"{s}_{m}_t{t:g}" for s in SETTINGS for m in ("none", "d4") for t in (0.1, 0.4, 0.5)
    }
    assert {r["experiment"] for r in rows} == experiments
    for r in rows:
        assert int(r["fold"]) == assignment[r["id"]]
        assert all(math.isfinite(float(r[k])) and 0 <= float(r[k]) <= 1 for k in METRICS)
        gt = truth[r["id"]][0 if r["target"] == "clean" else 1]
        n = round(float(r["foreground_fraction"]) * 512 * 512)
        tp = round(float(r["recall"]) * int(gt.sum()))
        close(r["foreground_fraction"], n / (512 * 512))
        close(r["precision"], tp / n)
        close(r["recall"], tp / int(gt.sum()))
        close(r["dice"], (2 * tp + 1e-7) / (n + int(gt.sum()) + 1e-7))
        a, b = float(r["centerline_precision"]), float(r["centerline_recall"])
        close(r["cldice"], 2 * a * b / (a + b) if a + b else 0)
        for k in ("foreground_fraction", "thin_clean_centerline_recall"):
            close(
                lookup[r["experiment"], "clean", r["id"]][k],
                lookup[r["experiment"], "proxy", r["id"]][k],
            )
    for r in reference_rows:
        for k in METRICS:
            close(
                r[k],
                lookup[
                    r["experiment"].replace("L1_proxy", "H0_reference", 1), r["target"], r["id"]
                ][k],
            )
    for name in NEW:
        for f in range(5):
            for target in ("clean", "proxy"):
                close(
                    histories[name, f][-1][f"monitor_{target}_05"],
                    fl[f"{name}_none_t0.5", target, f]["dice"],
                )
    grid = table(OUT / "hyperparameter_threshold_grid.csv")
    calibrated = table(OUT / "hyperparameter_calibration_development.csv")
    assert len(grid) == 260 and len(calibrated) == 20
    for name in SETTINGS:
        for mode in ("none", "d4"):
            for target in ("clean", "proxy"):
                group = [
                    r
                    for r in grid
                    if (r["setting"], r["mode"], r["target"]) == (name, mode, target)
                ]
                assert tuple(float(r["threshold"]) for r in group) == GRID
                for r in group:
                    scores = [float(r[f"fold{f}"]) for f in range(5)]
                    close(r["mean_dice"], np.mean(scores))
                    close(r["worst_fold_dice"], min(scores))
                    t = float(r["threshold"])
                    if t in (0.1, 0.4, 0.5):
                        for f in range(5):
                            close(scores[f], fl[f"{name}_{mode}_t{t:g}", target, f]["dice"])
                best = max(
                    group, key=lambda r: (float(r["mean_dice"]), -abs(float(r["threshold"]) - 0.5))
                )
                selected = next(
                    r
                    for r in calibrated
                    if (r["setting"], r["mode"], r["target"]) == (name, mode, target)
                )
                assert all(selected[k] == best[k] for k in best)
    paired = table(OUT / "hyperparameter_paired_per_image.csv")
    ps = table(OUT / "hyperparameter_paired_summary.csv")
    assert len(paired) == 7680 and len(ps) == 384
    assert len({(r["experiment"], r["target"], r["id"], r["metric"]) for r in paired}) == 7680
    for r in paired:
        assert int(r["fold"]) == assignment[r["id"]]
        suffix = r["experiment"][len(r["setting"]) :]
        close(
            r["delta"],
            float(lookup[r["experiment"], r["target"], r["id"]][r["metric"]])
            - float(lookup["H0_reference" + suffix, r["target"], r["id"]][r["metric"]]),
        )
    for s in ps:
        g = [
            r
            for r in paired
            if (r["experiment"], r["target"], r["metric"])
            == (s["experiment"], s["target"], s["metric"])
        ]
        assert len(g) == 20
        ds = [float(r["delta"]) for r in g]
        fs = [
            float(np.mean([float(r["delta"]) for r in g if int(r["fold"]) == f])) for f in range(5)
        ]
        close(s["mean_delta"], np.mean(ds))
        assert int(s["positive_images"]) == sum(d > 0 for d in ds)
        assert int(s["positive_folds"]) == sum(d > 0 for d in fs)
        for f in range(5):
            close(s[f"fold{f}_delta"], fs[f])
    checks[
        "1200_image_300_fold_60_summary_260_grid_20_selections_7680_deltas_384_summaries_verified"
    ] = True
    sensitivity = table(OUT / "hyperparameter_sensitivity.csv")
    choices = table(OUT / "hyperparameter_choices_development.csv")
    assert len(sensitivity) == 12 and len(choices) == 5
    for r in sensitivity:
        assert float(r["threshold"]) == 0.5
        s = next(
            x
            for x in summary
            if (x["experiment"], x["target"]) == (r["setting"] + "_d4_t0.5", r["target"])
        )
        for k in ("dice", "worst_fold_dice", "fold_dice_std"):
            close(r[k], s[k])
    baseline = next(
        r
        for r in calibrated
        if (r["setting"], r["mode"], r["target"]) == ("H0_reference", "d4", "proxy")
    )
    for r in choices:
        s = next(
            x
            for x in calibrated
            if (x["setting"], x["mode"], x["target"]) == (r["setting"], "d4", "proxy")
        )
        close(r["calibrated_proxy_dice"], s["mean_dice"])
        close(r["selected_threshold"], s["threshold"])
        close(
            r["fixed05_proxy_dice"],
            next(
                x["dice"]
                for x in summary
                if (x["experiment"], x["target"]) == (r["setting"] + "_d4_t0.5", "proxy")
            ),
        )
        ds = [float(s[f"fold{f}"]) - float(baseline[f"fold{f}"]) for f in range(5)]
        close(r["calibrated_delta_vs_reference"], np.mean(ds))
        assert int(r["positive_folds"]) == sum(d > 0 for d in ds)
        for f in range(5):
            close(r[f"fold{f}_delta"], ds[f])
    assert choices == sorted(choices, key=lambda r: -float(r["calibrated_proxy_dice"]))
    assert (
        choices[0]["setting"]
        == status["selected_setting"]
        == status["primary_fixed05_selection"]["setting"]
        == "H2_lr_double"
    )
    assert float(choices[0]["selected_threshold"]) == status["selected_threshold"] == 0.4
    checks["both_parameter_rankings_and_all_five_development_choices_match_predeclared_rules"] = (
        True
    )

    cr = table(OUT / "candidate_oof_per_image.csv")
    cf = table(OUT / "candidate_oof_per_fold.csv")
    cs = table(OUT / "candidate_oof_summary.csv")
    assert (len(cr), len(cf), len(cs)) == (40, 10, 2)
    cand, _ = common.check_aggregates(cr, cf, cs)
    masks = {}
    recomputed = []
    for name in NEW:
        for i in assignment:
            m = common.binary(OUT / f"oof_binary_masks/{name}/{i}.png")
            for target, v in common.metrics(m, truth[i]).items():
                for k in METRICS:
                    close(v[k], lookup[f"{name}_d4_t0.5", target, i][k])
                recomputed.append(
                    dict(setting=name, threshold=0.5, id=i, fold=assignment[i], target=target, **v)
                )
    for i in assignment:
        m = common.binary(OUT / f"oof_binary_masks/selected_candidate/{i}.png")
        masks[i] = m
        for target, v in common.metrics(m, truth[i]).items():
            for k in METRICS:
                close(v[k], cand["selected_development_candidate", target, i][k])
                close(v[k], lookup["H2_lr_double_d4_t0.4", target, i][k])
            recomputed.append(
                dict(
                    setting="H2_candidate",
                    threshold=0.4,
                    id=i,
                    fold=assignment[i],
                    target=target,
                    **v,
                )
            )
    assert len(recomputed) == 200
    write("local_recomputed_binary_metrics.csv", recomputed)
    checks["all100_lossless_OOF_masks_200_pixel_and_skeleton_rows_independently_recomputed"] = True

    previous_tables = {
        "L1_reference": {
            (r["target"], r["id"]): r
            for r in reference_rows
            if r["experiment"] == "L1_proxy_d4_t0.4"
        },
        "fixed_fusion": {
            (r["target"], r["id"]): r
            for r in table(REC / "fusion_per_image.csv")
            if r["experiment"] == "fixed_equal_centered_logit"
        },
        "old_native_public084056": {
            (r["target"], r["id"]): r
            for r in table(ROOT / "controlled_ablation_results_20261101/replay_per_image.csv")
            if r["experiment"] == "E1_ema_t01"
        },
    }
    comparisons = []
    comparison_summary = []
    for name, before in previous_tables.items():
        assert len(before) == 40
        for i in assignment:
            for target in ("clean", "proxy"):
                now = cand["selected_development_candidate", target, i]
                for k in METRICS:
                    a, b = float(before[target, i][k]), float(now[k])
                    comparisons.append(
                        dict(
                            reference=name,
                            id=i,
                            fold=assignment[i],
                            target=target,
                            metric=k,
                            before=a,
                            after=b,
                            delta=b - a,
                        )
                    )
        for target in ("clean", "proxy"):
            for k in METRICS:
                g = [
                    r
                    for r in comparisons
                    if (r["reference"], r["target"], r["metric"]) == (name, target, k)
                ]
                ds = [r["delta"] for r in g]
                fs = [float(np.mean([r["delta"] for r in g if r["fold"] == f])) for f in range(5)]
                comparison_summary.append(
                    dict(
                        reference=name,
                        target=target,
                        metric=k,
                        before=float(np.mean([r["before"] for r in g])),
                        after=float(np.mean([r["after"] for r in g])),
                        mean_delta=float(np.mean(ds)),
                        positive_images=sum(d > 0 for d in ds),
                        positive_folds=sum(d > 0 for d in fs),
                        **{f"fold{f}_delta": v for f, v in enumerate(fs)},
                    )
                )
    write("local_candidate_comparisons_per_image.csv", comparisons)
    write("local_candidate_comparisons_summary.csv", comparison_summary)
    fold_sensitivity = []
    for t in (0.4, 0.5):
        for target in ("clean", "proxy"):
            scores = [
                float(fl[f"H2_lr_double_d4_t{t:g}", target, f]["dice"])
                - float(fl[f"H0_reference_d4_t{t:g}", target, f]["dice"])
                for f in range(5)
            ]
            fold_sensitivity.append(
                dict(
                    threshold=t,
                    target=target,
                    full_mean_delta=float(np.mean(scores)),
                    precision_retry_fold2_delta=scores[2],
                    other_four_mean_delta=float(
                        np.mean([v for f, v in enumerate(scores) if f != 2])
                    ),
                    other_four_positive_folds=sum(v > 0 for f, v in enumerate(scores) if f != 2),
                    caveat="Posthoc sensitivity only; fold2 remains in the official 20-image result",
                )
            )
    write("local_precision_fold_sensitivity.csv", fold_sensitivity)
    checks["paired_candidate_comparisons_and_posthoc_precision_fold_sensitivity_verified"] = True

    exported = status["exported"]
    csv_path = OUT / "submission_hyperparameter_candidate.csv"
    assert (
        sha(csv_path)
        == sha(OUT / "received_submission_hyperparameter_candidate.csv")
        == exported["sha256"]
        == "4c2b067c8e8ff5cddd0ac4fc432fdfcb7e39d1e09a9822e8f446b6462a497ea3"
    )
    ids = [r["Id"] for r in table(DATA / "sample_submission.csv")]
    csv_rows = table(csv_path)
    assert [r["Id"] for r in csv_rows] == ids == [str(i) for i in range(1, 21)]
    assert list(csv_rows[0]) == ["Id", "Predicted"]
    for f in range(5):
        meta = load(OUT / f"H2_lr_double/fold{f}/test_completed.json")
        identity = meta["identity"]
        assert status["test_models"][f] == dict(fold=f, **meta)
        done = load(OUT / f"H2_lr_double/fold{f}/completed.json")
        assert (
            identity["setting"] == "H2_lr_double"
            and identity["parameters"] == SETTINGS["H2_lr_double"]
        )
        assert (
            identity["final_ema_sha256"] == done["sha256"]["final_ema.pt"]
            and identity["config"] == done["config"]
        )
        for i in ids:
            assert identity["test_image_sha256"][i] == sha(DATA / "test/image" / f"{i}.jpg")
    previous_test = {
        n: {r["Id"]: common.decode(r["Predicted"]) for r in table(ROOT / n)}
        for n in [
            "submission_fixed_equal_logit_fusion.csv",
            "submission_resnet34_5fold_converter_proxy.csv",
        ]
    }
    assert (
        sha(ROOT / "submission_fixed_equal_logit_fusion.csv")
        == "51abf18c801dab4b49e26ce51b51b1063ef67a6075a1b2790f5bb12b0fcdd075"
    )
    assert (
        sha(ROOT / "submission_resnet34_5fold_converter_proxy.csv")
        == "4274babb2e8e8e7ccfe2e11caf300ac1a4582a3f9230278a33836a2b23b9e352"
    )
    areas = []
    differences = []
    for r in csv_rows:
        i = r["Id"]
        m = common.decode(r["Predicted"])
        assert common.encode(m) == r["Predicted"]
        path = OUT / f"masks_512_white_vessels/submission_hyperparameter_candidate/{i}.png"
        assert np.array_equal(m, common.binary(path))
        with Image.open(path) as im:
            official = np.asarray(im.convert("L").resize((512, 512), Image.Resampling.BILINEAR)) > 0
        assert common.encode(official) == r["Predicted"]
        areas.append(float(m.mean()))
        for name, old in previous_test.items():
            old = old[i]
            differences.append(
                dict(
                    reference=name,
                    id=i,
                    changed_pixels=int((m != old).sum()),
                    binary_dice_agreement=float(2 * (m & old).sum() / (m.sum() + old.sum())),
                )
            )
    write("local_test_prediction_differences.csv", differences)
    checks["20_test_IDs_RLE_PNG_converter_semantics_and_five_selected_model_records_verified"] = (
        True
    )
    checks["both_previously_recommended_submission_bytes_unchanged"] = True
    audit = dict(
        date="2026-11-07",
        checks=checks,
        archive_sha256=sha(archive),
        candidate_csv_sha256=sha(csv_path),
        new_settings=4,
        new_models=20,
        epochs_per_model=80,
        formal_results_complete=True,
        logged_training_and_monitor_minutes={n: v / 60 for n, v in seconds.items()},
        development_choices=choices,
        numerical_precision_audit=numerical,
        precision_fold_sensitivity=fold_sensitivity,
        candidate_comparisons=comparison_summary,
        foreground_mean=float(np.mean(areas)),
        foreground_min=min(areas),
        foreground_max=max(areas),
        test_prediction_agreement={
            n: float(
                np.mean([r["binary_dice_agreement"] for r in differences if r["reference"] == n])
            )
            for n in previous_test
        },
        candidate_not_submitted_by_agent=True,
        weight_download_required=False,
        additional_training_required=False,
        limits=[
            "New weights and floating OOF/test probabilities absent: local checks do not replay neural inference.",
            "At new settings, only D4@0.5 and selected H2@0.4 masks independently recomputed; other thresholds checked for arithmetic consistency.",
            "Parameter and threshold selection reuse 20-image OOF with one seed, not independent or nested evaluation.",
            "H2 fold2 had 150 FP32 retries: numerical precision is an additional intervention; posthoc other-fold analysis does not replace the complete result.",
            "Public/private scores for this new CSV are unknown; historical public scores are taken from Kaggle submissions.",
        ],
    )
    (OUT / "local_analysis_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in audit.items()
                if k not in ["candidate_comparisons", "development_choices"]
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
