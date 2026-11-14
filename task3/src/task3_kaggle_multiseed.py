"""Fixed 43/80-epoch-schedule ResNet34 full-data runs, differing only by seed."""

from __future__ import annotations
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import time
import zipfile

if __name__ == "__main__" and not Path("/kaggle/working").is_dir():
    raise SystemExit("Use the generated Kaggle Notebook. No local training was started.")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from task3_pipeline import (
    FoveaDataset,
    assign_folds,
    error_metrics,
    find_data,
    load_manifest,
    predict,
    save_json,
    train_model,
    training_lock,
)
from task3_kaggle_cv import (
    ResNet34CV,
    atomic_csv,
    prediction_rows,
    read_rows,
    resolve_weights,
    restore_backup,
    stable_hash,
    write_submission,
)

MULTISEED_VERSION = "resnet34-full-multiseed-v1"
PLAN_SEEDS = (42, 2024, 3407)


def multiseed_sources():
    if "MULTISEED_SOURCE_SHA256" in globals():
        return dict(MULTISEED_SOURCE_SHA256)
    base = Path(__file__).resolve().parent
    return {
        name: hashlib.sha256((base / name).read_bytes()).hexdigest()
        for name in ("task3_pipeline.py", "task3_kaggle_cv.py", "task3_kaggle_multiseed.py")
    }


def manifest_records(records, labeled):
    result = []
    for r in records:
        row = dict(
            image_id=int(r["image_id"]),
            width=int(r["width"]),
            height=int(r["height"]),
            subset=r["subset"],
            filename=r["filename"],
            pixel_sha256=r["pixel_sha256"],
        )
        if labeled:
            row.update(x=float(r["x"]), y=float(r["y"]))
        result.append(row)
    return result


def historical_training_fingerprint(records):
    values = manifest_records(records, True)
    # Source CSV labels have six decimals; ignore parser noise below that scale.
    for row in values:
        row["x"], row["y"] = round(row["x"], 6), round(row["y"], 6)
    return stable_hash(values)


def validate_seed_predictions(rows, records, seed):
    expected = {int(r["image_id"]): r for r in records}
    result = {int(r["image_id"]): r for r in rows}
    if len(result) != len(rows) or set(result) != set(expected):
        raise ValueError("Seed prediction IDs differ or repeat.")
    for key, row in result.items():
        source = expected[key]
        if int(row["seed"]) != seed or row["pixel_sha256"] != source["pixel_sha256"]:
            raise ValueError("Seed or image hash differs.")
        for field in ("width", "height"):
            if int(row[field]) != int(source[field]):
                raise ValueError("Image dimensions differ.")
        for axis, dim in (("X", "width"), ("Y", "height")):
            value = float(row[f"Fovea_{axis}"])
            if not math.isfinite(value) or not 0 <= value < int(source[dim]):
                raise ValueError("Prediction outside original image.")
    return result


def average_seed_predictions(predictions, records, seeds=PLAN_SEEDS):
    if set(predictions) != set(seeds):
        raise ValueError("All three predeclared seeds are required for the ensemble.")
    maps = [validate_seed_predictions(predictions[seed], records, seed) for seed in seeds]
    return [
        dict(
            image_id=int(r["image_id"]),
            width=int(r["width"]),
            height=int(r["height"]),
            Fovea_X=statistics.mean(float(m[int(r["image_id"])]["Fovea_X"]) for m in maps),
            Fovea_Y=statistics.mean(float(m[int(r["image_id"])]["Fovea_Y"]) for m in maps),
        )
        for r in records
    ]


def validate_historical_reference(rows, records, expected_pixels):
    lookup = {r["ImageID"]: float(r["value"]) for r in rows}
    expected = {f'{int(r["image_id"])}_Fovea_{axis}' for r in records for axis in ("X", "Y")}
    if len(rows) != len(lookup) or set(lookup) != expected:
        raise ValueError("Historical reference submission IDs differ.")
    result = []
    for r in records:
        key = int(r["image_id"])
        prior = expected_pixels.get(str(key))
        if prior is None or (int(prior["width"]), int(prior["height"]), prior["pixel_sha256"]) != (
            int(r["width"]),
            int(r["height"]),
            r["pixel_sha256"],
        ):
            raise ValueError("Historical reference test-image fingerprint differs.")
        x, y = lookup[f"{key}_Fovea_X"], lookup[f"{key}_Fovea_Y"]
        if not (math.isfinite(x + y) and 0 <= x < int(r["width"]) and 0 <= y < int(r["height"])):
            raise ValueError("Historical reference coordinate outside image.")
        result.append(
            dict(image_id=key, width=int(r["width"]), height=int(r["height"]), Fovea_X=x, Fovea_Y=y)
        )
    return result


def full_seed_config(protocol, seed):
    if seed not in PLAN_SEEDS:
        raise ValueError("Undeclared random seed.")
    # weights is a portable identifier. The model builder receives the resolved
    # runtime path separately, so moving a complete backup does not change config.
    return dict(
        protocol,
        seed=seed,
        folds=5,
        fold="full",
        model_name="resnet34",
        weights="resnet34-b627a593.pth",
        pretrained=True,
        architecture="resnet34_same_decoder_as_legacy_fulltrain",
        normalization="imagenet",
        selection_metric="fixed_final_epoch_no_validation",
    )


def checkpoint_final(path, config):
    item = torch.load(path, map_location="cpu", weights_only=True)
    if int(item.get("epoch", 0)) != 43 or item.get("config") != config:
        raise ValueError("Expected this seed's fixed final epoch43 checkpoint.")
    del item


def seed_checkpoint_summary(folder, config, verify_model=True):
    metadata = json.loads((folder / "seed_complete.json").read_text())
    signature = stable_hash(config)
    if metadata["signature"] != signature or metadata["epoch"] != 43:
        raise ValueError("Completed seed configuration differs.")
    model_path = folder / "checkpoints/best.pt"
    if (
        not model_path.is_file()
        or hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata["checkpoint_sha256"]
    ):
        raise ValueError("Completed seed model missing/corrupt; mount its full output.")
    if verify_model:
        checkpoint_final(model_path, config)
    return metadata


def write_seed_reports(data, train, test, reference, output, protocol, verify_models=True):
    test_records = test.to_dict("records")
    predictions, summary = {}, []
    for seed in PLAN_SEEDS:
        folder = output / f"seed_{seed}"
        if not (folder / "seed_complete.json").exists():
            continue
        config = full_seed_config(protocol, seed)
        done = seed_checkpoint_summary(folder, config, verify_models)
        rows = read_rows(folder / "test_predictions.csv")
        validate_seed_predictions(rows, test_records, seed)
        predictions[seed] = rows
        history = read_rows(folder / "checkpoints/history.csv")
        if [int(r["epoch"]) for r in history] != list(range(1, 44)):
            raise ValueError("Seed history does not contain all43 epochs.")
        # These are training-fit scores, never candidate selection scores.
        fit = json.loads((folder / "train_fit_metrics.json").read_text())
        summary.append(
            dict(
                seed=seed,
                epoch=43,
                training_images=80,
                train_fit_rmse_px=float(fit["rmse_coordinate_px"]),
                train_fit_mean_distance_px=float(fit["mean_distance_px"]),
                final_heatmap_ce=float(history[-1]["train_heatmap_ce"]),
                evaluation="in_sample_training_fit_only",
                public_score="",
            )
        )
        write_submission(
            data / "sample_submission.csv", rows, output / f"submission_resnet34_seed{seed}.csv"
        )
    if summary:
        atomic_csv(output / "train_fit_summary.csv", summary)
    complete = set(predictions) == set(PLAN_SEEDS)
    plan = dict(
        complete=complete,
        completed_seeds=sorted(predictions),
        planned_seeds=list(PLAN_SEEDS),
        training_images_per_model=80,
        final_epoch=43,
        cosine_schedule_epochs=80,
        validation_used=False,
        ensemble_rule="Equal coordinate mean of exactly the three newly trained seeds; no subsets or weight tuning.",
        selection="No best seed is chosen from training fit or prediction agreement.",
        historical_reference_score=12.31624,
        historical_reference_only=True,
        retrained_seed42_public_score=None,
        agent_submitted_to_competition=False,
        limitation="No independent validation in this full-data experiment. Same seed on another platform need not reproduce the old predictions or score.",
    )
    if not complete:
        save_json(output / "selection.json", plan)
        return False
    averaged = average_seed_predictions(predictions, test_records)
    atomic_csv(output / "test_multiseed_equal.csv", averaged)
    write_submission(
        data / "sample_submission.csv", averaged, output / "submission_resnet34_multiseed_equal.csv"
    )
    reference_by_id = {int(r["image_id"]): r for r in reference}
    mappings = {s: validate_seed_predictions(predictions[s], test_records, s) for s in PLAN_SEEDS}
    diagnostics = []
    for row in averaged:
        key = int(row["image_id"])
        x, y = float(row["Fovea_X"]), float(row["Fovea_Y"])
        old = reference_by_id[key]
        control = mappings[42][key]
        rms = math.sqrt(
            statistics.mean(
                (float(mappings[s][key]["Fovea_X"]) - x) ** 2
                + (float(mappings[s][key]["Fovea_Y"]) - y) ** 2
                for s in PLAN_SEEDS
            )
        )
        diagnostics.append(
            dict(
                image_id=key,
                Fovea_X=x,
                Fovea_Y=y,
                seed_disagreement_rms_px=rms,
                ensemble_shift_vs_historical_px=math.hypot(
                    x - float(old["Fovea_X"]), y - float(old["Fovea_Y"])
                ),
                new_seed42_shift_vs_historical_px=math.hypot(
                    float(control["Fovea_X"]) - float(old["Fovea_X"]),
                    float(control["Fovea_Y"]) - float(old["Fovea_Y"]),
                ),
            )
        )
    atomic_csv(output / "test_prediction_diagnostics.csv", diagnostics)
    plan.update(
        candidate="submission_resnet34_multiseed_equal.csv",
        historical_seed42_vs_new_seed42_mean_shift_px=statistics.mean(
            r["new_seed42_shift_vs_historical_px"] for r in diagnostics
        ),
        historical_seed42_vs_new_seed42_max_shift_px=max(
            r["new_seed42_shift_vs_historical_px"] for r in diagnostics
        ),
        candidate_public_score=None,
    )
    save_json(output / "selection.json", plan)
    print("All three seeds complete. Candidate:", output / plan["candidate"], flush=True)
    return True


def multiseed_archive(output, full=False):
    target = (
        output.parent / "task3_multiseed_full_backup.zip"
        if full
        else output / "task3_multiseed_results_light.zip"
    )
    temporary = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True
    ) as archive:
        for path in sorted(output.rglob("*")):
            if not path.is_file() or path.is_symlink() or path.suffix in (".tmp", ".zip"):
                continue
            if full or path.suffix in (".csv", ".json", ".png", ".md"):
                archive.write(path, path.relative_to(output))
    temporary.replace(target)
    return target


def prepare_multiseed(
    reference_rows,
    reference_sha256,
    reference_pixels,
    reference_train_fingerprint,
    data_dir=None,
    output_dir="/kaggle/working/task3_resnet34_multiseed_full",
    resume_dir=None,
):
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("Run this Notebook on Kaggle; no local training was started.")
    if not torch.cuda.is_available():
        raise RuntimeError("Enable Kaggle GPU; CPU training is disabled.")
    data, output = find_data(data_dir), Path(output_dir).resolve()
    manifest = load_manifest(data)
    train = assign_folds(manifest[manifest.subset == "train"], 42, 5)
    test = manifest[manifest.subset == "test"].copy()
    if len(train) != 80 or len(test) != 20 or train.pixel_sha256.nunique() != 79:
        raise ValueError("Unexpected training/test counts or duplicate groups.")
    if historical_training_fingerprint(train.to_dict("records")) != reference_train_fingerprint:
        raise ValueError(
            "Training pixels/labels/order differ from the original full-data reference."
        )
    reference = validate_historical_reference(
        reference_rows, test.to_dict("records"), reference_pixels
    )
    protocol = dict(
        version=MULTISEED_VERSION,
        seeds=list(PLAN_SEEDS),
        epochs=43,
        schedule_epochs=80,
        image_size=320,
        batch_size=8,
        lr_backbone=1e-4,
        lr_decoder=1e-3,
        weight_decay=1e-4,
        sigma=1.5,
        split_seed=42,
        training_mode="all80_no_validation_fixed_epoch",
        flip_tta=False,
        dataset_fingerprint=stable_hash(manifest_records(train.to_dict("records"), True)),
        test_fingerprint=stable_hash(manifest_records(test.to_dict("records"), False)),
        historical_submission_sha256=reference_sha256,
        source_sha256=multiseed_sources(),
        training_implementation="Original task3_pipeline.train_model, including80-epoch cosine schedule and every-epoch resume saves",
    )
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 3 * 1024**3:
        raise RuntimeError("Need at least3 GiB free for models, optimizer and atomic saves.")
    restore_backup(resume_dir, output, protocol)
    if (output / "protocol.json").exists() and json.loads(
        (output / "protocol.json").read_text()
    ) != protocol:
        raise ValueError("Existing multi-seed protocol differs. Use a new output directory.")
    save_json(output / "protocol.json", protocol)
    train.to_csv(output / "train_manifest.csv", index=False)
    test.to_csv(output / "test_manifest.csv", index=False)
    write_submission(
        data / "sample_submission.csv", reference, output / "historical_reference_12p31624.csv"
    )
    atomic_csv(output / "historical_reference_predictions.csv", reference)
    save_json(
        output / "environment.json",
        dict(
            torch=torch.__version__,
            torchvision=__import__("torchvision").__version__,
            numpy=np.__version__,
            gpu=torch.cuda.get_device_name(0),
            visible_gpu_count=torch.cuda.device_count(),
            device_policy="cuda:0 sequential seeds; original CPU thread setting4; workers0",
            historical_environment="RTX4060 Laptop, torch2.11.0+cu128, numpy2.4.6; same seed is not a bitwise guarantee.",
        ),
    )
    print(
        "80 images /79 pixel groups; seeds",
        PLAN_SEEDS,
        ";43 epochs /80 cosine period; no validation/TTA.",
        flush=True,
    )
    print(
        "Historical score12.31624 belongs only to the old submission, not a new seed42 run.",
        flush=True,
    )
    return dict(
        data=data, output=output, train=train, test=test, reference=reference, protocol=protocol
    )


def multiseed_eta(output, current_seed):
    timing = []
    pending = 0
    for seed in PLAN_SEEDS:
        folder = output / f"seed_{seed}"
        history = (
            read_rows(folder / "checkpoints/history.csv")
            if (folder / "checkpoints/history.csv").exists()
            else []
        )
        timing.extend(float(r["seconds"]) for r in history[-8:])
        if not (folder / "seed_complete.json").exists():
            pending += max(0, 43 - len(history))
    if len(timing) >= 3:
        minutes = statistics.median(timing) * pending / 60
        save_json(
            output / "eta.json",
            dict(
                current_seed=current_seed,
                remaining_training_minutes_estimate=minutes,
                excludes="Preparation, future cold starts, prediction, compression and some optimizer writes.",
            ),
        )
        return minutes
    return None


def run_multiseed(prepared, run_seeds=PLAN_SEEDS, weights=None, keep_last=False, full_backup=True):
    if not Path("/kaggle/working").is_dir() or not torch.cuda.is_available():
        raise RuntimeError("Use Kaggle GPU; local/CPU training is disabled.")
    if (
        not run_seeds
        or len(set(run_seeds)) != len(run_seeds)
        or not set(run_seeds) <= set(PLAN_SEEDS)
    ):
        raise ValueError("RUN_SEEDS must be a unique subset of42,2024,3407.")
    data, output, train, test, reference, protocol = [
        prepared[k] for k in ("data", "output", "train", "test", "reference", "protocol")
    ]
    pending = [s for s in run_seeds if not (output / f"seed_{s}/seed_complete.json").exists()]
    weight_path = resolve_weights(output, weights) if pending else None
    with training_lock(output):
        try:
            for seed in run_seeds:
                folder = output / f"seed_{seed}"
                config = full_seed_config(protocol, seed)
                if (folder / "seed_complete.json").exists():
                    seed_checkpoint_summary(folder, config)
                    print(f"Reuse completed fixed43-epoch seed{seed}.", flush=True)
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                checkpoints = folder / "checkpoints"
                if (checkpoints / "best.pt").exists() and not (checkpoints / "last.pt").exists():
                    raise ValueError(
                        "Unfinished seed has a best model but no optimizer resume checkpoint. Restore full output or start a new directory."
                    )
                if (checkpoints / "last.pt").exists():
                    last = torch.load(
                        checkpoints / "last.pt", map_location="cpu", weights_only=False
                    )
                    if len(last["cuda_rng"]) != torch.cuda.device_count():
                        raise ValueError(
                            "Resume with the same visible GPU count as the saved session."
                        )
                    del last
                save_json(
                    output / "status.json",
                    dict(state="training", seed=seed, updated_unix=time.time()),
                )
                start = time.perf_counter()
                # The legacy trainer seeds before dataset/model creation, just as
                # the original43-epoch run. The ResNet34 model is structurally identical.
                model, epoch = train_model(
                    data,
                    train,
                    None,
                    checkpoints,
                    config,
                    torch.device("cuda:0"),
                    model_builder=lambda cfg: ResNet34CV(weight_path),
                    dataset_builder=FoveaDataset,
                )
                if epoch != 43:
                    raise RuntimeError("Fixed final43-epoch run incomplete.")
                checkpoint_final(checkpoints / "best.pt", config)
                test_loader = DataLoader(
                    FoveaDataset(data, test, 320), batch_size=8, shuffle=False, num_workers=0
                )
                points = predict(model, test_loader, torch.device("cuda:0"), flip_tta=False)
                rows = prediction_rows(test.to_dict("records"), points, 0, False)
                for row in rows:
                    row.pop("fold")
                    row["seed"] = seed
                atomic_csv(folder / "test_predictions.csv", rows)
                write_submission(data / "sample_submission.csv", rows, folder / "submission.csv")
                fit_loader = DataLoader(
                    FoveaDataset(data, train, 320), batch_size=8, shuffle=False, num_workers=0
                )
                fit_prediction = predict(model, fit_loader, torch.device("cuda:0"), flip_tta=False)
                fit = error_metrics(
                    fit_prediction,
                    train[["nx", "ny"]].to_numpy(),
                    train[["width", "height"]].to_numpy(),
                )
                save_json(
                    folder / "train_fit_metrics.json",
                    dict(
                        **fit,
                        evaluation="in_sample_training_fit_not_validation",
                        used_for_selection=False,
                    ),
                )
                save_json(
                    folder / "seed_complete.json",
                    dict(
                        signature=stable_hash(config),
                        seed=seed,
                        epoch=43,
                        checkpoint_sha256=hashlib.sha256(
                            (checkpoints / "best.pt").read_bytes()
                        ).hexdigest(),
                        seconds=time.perf_counter() - start,
                        public_score=None,
                        submitted_to_competition=False,
                        validation_used=False,
                    ),
                )
                if not keep_last and (checkpoints / "last.pt").exists():
                    (checkpoints / "last.pt").unlink()
                del model, test_loader, fit_loader, points, fit_prediction
                gc.collect()
                torch.cuda.empty_cache()
                eta = multiseed_eta(output, seed)
                if eta is not None:
                    print(
                        f"Approximate remaining training: {eta:.1f}min (excludes preparation/prediction/backup).",
                        flush=True,
                    )
                write_seed_reports(data, train, test, reference, output, protocol)
                multiseed_archive(output)
            complete = write_seed_reports(data, train, test, reference, output, protocol)
            save_json(
                output / "status.json",
                dict(
                    state="completed" if complete else "partial_completed", updated_unix=time.time()
                ),
            )
            multiseed_archive(output)
            if complete and full_backup:
                backup = multiseed_archive(output, full=True)
                with zipfile.ZipFile(backup) as archive:
                    if not all(
                        f"seed_{s}/checkpoints/best.pt" in archive.namelist() for s in PLAN_SEEDS
                    ):
                        raise RuntimeError("Full backup lacks a final model.")
                print("Full backup:", backup, flush=True)
        except BaseException as error:
            save_json(
                output / "status.json",
                dict(
                    state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    error=str(error),
                    updated_unix=time.time(),
                ),
            )
            multiseed_archive(output)
            raise
    return output
