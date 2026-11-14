"""Kaggle-only ResNet34 five-fold comparison: 320, 320+flip TTA, and 512.

Local preparation and tests do not run this module or import PyTorch.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import time
import zipfile

if (
    __name__ == "__main__"
    and not Path("/kaggle/working").is_dir()
    and "--allow-local" not in sys.argv
):
    raise SystemExit(
        "This experiment is intended for Kaggle. No local training was started. Use the generated Notebook."
    )

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

from task3_pipeline import (
    DecoderBlock,
    FoveaDataset,
    assign_folds,
    contact_sheet,
    decode_heatmap,
    find_data,
    heatmap_loss,
    load_manifest,
    save_json,
    seed_everything,
    training_lock,
)

CV_VERSION = "resnet34-cv-v1"
WEIGHT_NAME = "resnet34-b627a593.pth"
WEIGHT_URL = "https://download.pytorch.org/models/" + WEIGHT_NAME
ARMS = ("resnet34_320", "resnet34_320_flip", "resnet34_512")


def stable_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def atomic_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = list(fields or (list(rows[0]) if rows else []))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def prediction_rows(records, points, fold, labeled):
    if len(records) != len(points):
        raise ValueError("Prediction count differs from image count.")
    rows, seen = [], set()
    for item, point in zip(records, points):
        key, width, height = int(item["image_id"]), int(item["width"]), int(item["height"])
        if key in seen:
            raise ValueError("Duplicate image ID.")
        seen.add(key)
        nx, ny = float(point[0]), float(point[1])
        if not all(math.isfinite(v) for v in [nx, ny]) or not (0 <= nx < 1 and 0 <= ny < 1):
            raise ValueError(f"Invalid normalized prediction for image {key}.")
        row = dict(
            image_id=key,
            fold=int(fold),
            width=width,
            height=height,
            pixel_sha256=str(item["pixel_sha256"]),
            nx=nx,
            ny=ny,
            Fovea_X=nx * width,
            Fovea_Y=ny * height,
        )
        if labeled:
            x, y = float(item["x"]), float(item["y"])
            dx, dy = row["Fovea_X"] - x, row["Fovea_Y"] - y
            row.update(
                gt_x=x,
                gt_y=y,
                dx=dx,
                dy=dy,
                distance_px=math.hypot(dx, dy),
                squared_error=dx * dx + dy * dy,
            )
        rows.append(row)
    return rows


def quantile_linear(values, probability):
    values = sorted(float(v) for v in values)
    if not values:
        raise ValueError("No values.")
    position = (len(values) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def metrics_from_rows(rows):
    rows = list(rows)
    if not rows:
        raise ValueError("No labeled predictions.")
    distance, dx, dy = [], [], []
    for row in rows:
        ex = float(row["Fovea_X"]) - float(row["gt_x"])
        ey = float(row["Fovea_Y"]) - float(row["gt_y"])
        if not math.isfinite(ex) or not math.isfinite(ey):
            raise ValueError("Non-finite error.")
        dx.append(ex)
        dy.append(ey)
        distance.append(math.hypot(ex, ey))
    return dict(
        n=len(rows),
        mean_distance_px=statistics.mean(distance),
        median_distance_px=statistics.median(distance),
        p90_distance_px=quantile_linear(distance, 0.9),
        max_distance_px=max(distance),
        mae_x_px=statistics.mean(abs(v) for v in dx),
        mae_y_px=statistics.mean(abs(v) for v in dy),
        rmse_coordinate_px=math.sqrt(sum(x * x + y * y for x, y in zip(dx, dy)) / (2 * len(rows))),
    )


def validate_oof(rows, records):
    expected = {int(r["image_id"]): r for r in records}
    if len(expected) != len(records) or len(rows) != len(records):
        raise ValueError("OOF must contain exactly one row per training image.")
    seen, groups = set(), {}
    for row in rows:
        key = int(row["image_id"])
        if key not in expected or key in seen:
            raise ValueError("Unknown or repeated OOF image ID.")
        seen.add(key)
        source = expected[key]
        if int(row["fold"]) != int(source["fold"]):
            raise ValueError("OOF fold does not match the fixed split.")
        if (int(row["width"]), int(row["height"])) != (int(source["width"]), int(source["height"])):
            raise ValueError("OOF dimensions differ.")
        if str(row["pixel_sha256"]) != str(source["pixel_sha256"]):
            raise ValueError("OOF image hash differs.")
        if (
            abs(float(row["gt_x"]) - float(source["x"])) > 1e-6
            or abs(float(row["gt_y"]) - float(source["y"])) > 1e-6
        ):
            raise ValueError("OOF ground truth differs.")
        groups.setdefault(str(source["pixel_sha256"]), set()).add(int(row["fold"]))
    if seen != set(expected) or any(len(value) != 1 for value in groups.values()):
        raise ValueError("Incomplete OOF coverage or duplicate group leakage.")
    return sorted(rows, key=lambda r: int(r["image_id"]))


def validate_test_rows(rows, records, fold):
    expected = {int(r["image_id"]): r for r in records}
    if len(rows) != len(expected):
        raise ValueError("Incomplete test prediction coverage.")
    seen = set()
    for row in rows:
        key = int(row["image_id"])
        if key not in expected or key in seen or int(row["fold"]) != fold:
            raise ValueError("Invalid test ID/fold.")
        seen.add(key)
        source = expected[key]
        if (int(row["width"]), int(row["height"])) != (int(source["width"]), int(source["height"])):
            raise ValueError("Test image dimensions differ.")
        if str(row["pixel_sha256"]) != str(source["pixel_sha256"]):
            raise ValueError("Test image hash differs.")
        x, y = float(row["Fovea_X"]), float(row["Fovea_Y"])
        if not all(math.isfinite(v) for v in [x, y]) or not (
            0 <= x < int(source["width"]) and 0 <= y < int(source["height"])
        ):
            raise ValueError("Test prediction outside image.")
    return {int(r["image_id"]): r for r in rows}


def average_test_folds(folds, records, fold_count=5):
    if set(folds) != set(range(fold_count)):
        raise ValueError("All five folds are required before exporting a five-fold submission.")
    mappings = [validate_test_rows(folds[f], records, f) for f in range(fold_count)]
    averaged = []
    for source in records:
        key = int(source["image_id"])
        averaged.append(
            dict(
                image_id=key,
                width=int(source["width"]),
                height=int(source["height"]),
                Fovea_X=statistics.mean(float(m[key]["Fovea_X"]) for m in mappings),
                Fovea_Y=statistics.mean(float(m[key]["Fovea_Y"]) for m in mappings),
            )
        )
    return averaged


def write_submission(template_path, predictions, target):
    lookup = {}
    for row in predictions:
        key = int(row["image_id"])
        for axis, dimension in [("X", "width"), ("Y", "height")]:
            value = float(row[f"Fovea_{axis}"])
            if not math.isfinite(value) or not 0 <= value < int(row[dimension]):
                raise ValueError("Invalid original-pixel coordinate.")
            image_id = f"{key}_Fovea_{axis}"
            if image_id in lookup:
                raise ValueError("Repeated submission ID.")
            lookup[image_id] = value
    template = read_rows(template_path)
    ids = [row["ImageID"] for row in template]
    if len(ids) != len(set(ids)) or set(ids) != set(lookup):
        raise ValueError("Submission IDs differ from template.")
    rows = [dict(ImageID=key, value=f"{lookup[key]:.6f}") for key in ids]
    atomic_csv(target, rows, ["ImageID", "value"])
    check = read_rows(target)
    if [r["ImageID"] for r in check] != ids or any(
        not math.isfinite(float(r["value"])) for r in check
    ):
        raise AssertionError("Submission round trip failed.")


def scaled_heatmap_settings(size):
    if size not in (320, 512):
        raise ValueError("Only the predeclared sizes 320 and 512 are supported.")
    return dict(sigma=1.5 * size / 320, decode_radius=max(1, int(round(3 * size / 320))))


def source_fingerprints():
    if "NOTEBOOK_SOURCE_SHA256" in globals():
        return dict(NOTEBOOK_SOURCE_SHA256)
    base = Path(__file__).resolve().parent
    return {
        name: hashlib.sha256((base / name).read_bytes()).hexdigest()
        for name in ["task3_pipeline.py", "task3_kaggle_cv.py"]
    }


def resolve_weights(output, explicit=None):
    candidates = [Path(explicit)] if explicit else []
    candidates += [Path(torch.hub.get_dir()) / "checkpoints" / WEIGHT_NAME]
    if "__file__" in globals():
        candidates += [
            Path(__file__).resolve().parents[i] / "task2/.torch/hub/checkpoints" / WEIGHT_NAME
            for i in (1, 2)
        ]
    if Path("/kaggle/input").exists():
        candidates += sorted(Path("/kaggle/input").rglob(WEIGHT_NAME))
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        path = output / "_pretrained" / WEIGHT_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            print("Downloading official ResNet34 ImageNet weights (about 87 MB).", flush=True)
            torch.hub.download_url_to_file(
                WEIGHT_URL, str(path), hash_prefix="b627a593", progress=True
            )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not digest.startswith("b627a593"):
        raise ValueError(f"ImageNet weight hash mismatch: {path}")
    save_json(
        output / "pretrained_source.json", dict(path=str(path), url=WEIGHT_URL, sha256=digest)
    )
    return path


class ResNet34CV(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.backbone = models.resnet34(weights=None)
        if weights is not None:
            self.backbone.load_state_dict(
                torch.load(weights, map_location="cpu", weights_only=True), strict=True
            )
        self.backbone.fc = nn.Identity()
        self.d3 = DecoderBlock(512, 256, 128)
        self.d2 = DecoderBlock(128, 128, 64)
        self.d1 = DecoderBlock(64, 64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, images):
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(images))))
        c1 = b.layer1(x)
        c2 = b.layer2(c1)
        c3 = b.layer3(c2)
        c4 = b.layer4(c3)
        return self.head(self.d1(self.d2(self.d3(c4, c3), c2), c1))[:, 0]


@torch.inference_mode()
def predict_cv(model, loader, device, radius, flip=False):
    model.eval()
    result = []
    for images, _, _ in loader:
        images = images.to(device)
        with torch.autocast(device_type="cuda", enabled=True):
            coordinates = decode_heatmap(model(images), radius=radius)
            if flip:
                mirrored = decode_heatmap(model(images.flip(-1)), radius=radius)
                mirrored[:, 0] = 1 - mirrored[:, 0]
                coordinates = (coordinates + mirrored) / 2
        result.extend(coordinates.float().cpu().tolist())
    return result


def atomic_torch_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def read_checkpoint(path, signature):
    # Only this experiment's own files/backups are loaded with weights_only=False.
    item = torch.load(path, map_location="cpu", weights_only=False)
    if item.get("signature") != signature:
        raise ValueError(f"Checkpoint protocol mismatch: {path}")
    return item


def current_eta(output, size, fold, epoch, epochs, wall_times, completed_jobs):
    if len(wall_times) < 3:
        return
    rate = statistics.median(wall_times[-8:])
    timing_path = output / "timing_by_size.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    timing[str(size)] = dict(seconds_per_epoch=rate, samples=len(wall_times))
    save_json(timing_path, timing)
    estimate = (epochs - epoch) * rate
    estimated_jobs = []
    for candidate_size in [320, 512]:
        known = timing.get(str(candidate_size), {}).get("seconds_per_epoch")
        candidate_rate = known if known is not None else rate * (candidate_size / size) ** 2
        for candidate_fold in range(5):
            key = f"{candidate_size}/{candidate_fold}"
            if key not in completed_jobs and (candidate_size, candidate_fold) != (size, fold):
                estimate += epochs * candidate_rate
                estimated_jobs.append(key)
    information = dict(
        current_size=size,
        current_fold=fold,
        epoch=epoch,
        median_epoch_seconds=rate,
        approximate_training_minutes_remaining=estimate / 60,
        calibration="Measured completed epochs; unmeasured resolution estimated by pixel count.",
        excludes="Future cold starts, prediction/plotting, downloads and restore overhead.",
    )
    save_json(output / "eta.json", information)
    if epoch == 3 or epoch % 10 == 0:
        print(
            f"ETA: about {estimate/60:.1f} training minutes remaining for the full two-size plan.",
            flush=True,
        )


def train_cv_fold(
    data, development, valid, folder, config, weights, device, output, completed_jobs
):
    folder.mkdir(parents=True, exist_ok=True)
    signature = stable_hash(config)
    config_path = folder / "train_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError(f"Configuration differs: {folder}")
    save_json(config_path, config)
    selected = config["selection_metric"]
    complete_path = folder / "training_complete.json"
    if complete_path.exists():
        done = json.loads(complete_path.read_text())
        if done["signature"] != signature:
            raise ValueError("Completed fold protocol differs.")
        checkpoint = read_checkpoint(folder / f"best_{selected}.pt", signature)
        model = ResNet34CV().to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        return model, done
    seed_everything(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    train_set = FoveaDataset(data, development, config["image_size"], augment=True)
    valid_set = FoveaDataset(data, valid, config["image_size"])
    loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    valid_loader = DataLoader(
        valid_set, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )
    last_path = folder / "last.pt"
    model = ResNet34CV(weights=None if last_path.exists() else weights).to(device)
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    optimizer = torch.optim.AdamW(
        [
            dict(params=model.backbone.parameters(), lr=config["lr_backbone"]),
            dict(
                params=[p for p in model.parameters() if id(p) not in backbone_ids],
                lr=config["lr_decoder"],
            ),
        ],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=config["lr_backbone"] * 0.05
    )
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats(device=0)
    history, start, wall_times = [], 0, []
    if last_path.exists():
        state = read_checkpoint(last_path, signature)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history, start = state["history"], int(state["epoch"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], device=0)
        del state
        print(f'Resume {config["image_size"]}, fold {config["fold"]}, epoch {start}.', flush=True)
    tracked = ("rmse_coordinate_px", "mean_distance_px")
    best = {key: dict(score=float("inf"), epoch=0) for key in tracked}
    # Best files can be newer than the optimizer checkpoint after an interruption.
    # Retain these already evaluated, signature-matched candidates rather than mixing scores and weights.
    for key in tracked:
        path = folder / f"best_{key}.pt"
        if path.exists():
            item = read_checkpoint(path, signature)
            best[key] = dict(score=float(item["metrics"][key]), epoch=int(item["epoch"]))
            del item
    for epoch in range(start + 1, config["epochs"] + 1):
        begin = time.perf_counter()
        model.train()
        total, count, skipped_steps = 0.0, 0, 0
        for images, points, _ in loader:
            images, points = images.to(device), points.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=True):
                logits = model(images)
                if logits.shape != (
                    len(images),
                    config["image_size"] // 4,
                    config["image_size"] // 4,
                ):
                    raise RuntimeError("Unexpected heatmap dimensions.")
                loss = heatmap_loss(logits, points, sigma=config["sigma"])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite heatmap loss.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped_steps += int(scaler.get_scale() < previous_scale)
            total += float(loss.detach()) * len(images)
            count += len(images)
        if (
            skipped_steps == len(loader)
            and history
            and history[-1].get("amp_skipped_steps") == len(loader)
        ):
            raise RuntimeError(
                "AMP skipped every optimizer step for two consecutive epochs; stop for diagnosis."
            )
        validation = prediction_rows(
            valid.to_dict("records"),
            predict_cv(model, valid_loader, device, config["decode_radius"]),
            config["fold"],
            True,
        )
        metrics = metrics_from_rows(validation)
        for key in tracked:
            if metrics[key] < best[key]["score"]:
                best[key] = dict(score=metrics[key], epoch=epoch)
                atomic_torch_save(
                    folder / f"best_{key}.pt",
                    dict(
                        model=model.state_dict(),
                        epoch=epoch,
                        signature=signature,
                        train_config=config,
                        metrics=metrics,
                    ),
                )
        applied_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        row = dict(
            epoch=epoch,
            train_heatmap_ce=total / count,
            lr_backbone=applied_lr,
            **metrics,
            amp_skipped_steps=skipped_steps,
            gpu_peak_mib=torch.cuda.max_memory_allocated(device=0) / 1024**2,
            compute_seconds=time.perf_counter() - begin,
        )
        history.append(row)
        if epoch % config["checkpoint_every"] == 0 or epoch == config["epochs"]:
            atomic_torch_save(
                last_path,
                dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    scaler=scaler.state_dict(),
                    epoch=epoch,
                    history=history,
                    signature=signature,
                    python_rng=random.getstate(),
                    numpy_rng=np.random.get_state(),
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(device=0),
                ),
            )
        wall = time.perf_counter() - begin
        wall_times.append(wall)
        atomic_csv(folder / "history.csv", history)
        save_json(
            folder / "status.json",
            dict(
                state="training",
                epoch=epoch,
                epochs=config["epochs"],
                best=best,
                updated_unix=time.time(),
            ),
        )
        current_eta(
            output,
            config["image_size"],
            config["fold"],
            epoch,
            config["epochs"],
            wall_times,
            completed_jobs,
        )
        print(
            f'{config["image_size"]} fold{config["fold"]} {epoch:03}/{config["epochs"]}'
            f' CE={total/count:.4f} RMSE={metrics["rmse_coordinate_px"]:.2f}px'
            f' MRE={metrics["mean_distance_px"]:.2f}px best={best[selected]["score"]:.2f}px'
            f' AMPskip={skipped_steps} GPUpeak={row["gpu_peak_mib"]:.0f}MiB'
            f" {wall:.1f}s",
            flush=True,
        )
    done = dict(signature=signature, best=best, selection_metric=selected, epochs=config["epochs"])
    save_json(complete_path, done)
    checkpoint = read_checkpoint(folder / f"best_{selected}.pt", signature)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, done


def fold_config(protocol, size, fold):
    return dict(
        cv_version=CV_VERSION,
        image_size=size,
        fold=fold,
        seed=protocol["seed"] + fold,
        split_seed=protocol["seed"],
        epochs=protocol["epochs"],
        batch_size=protocol["batch_size"],
        lr_backbone=1e-4,
        lr_decoder=1e-3,
        weight_decay=1e-4,
        cpu_threads=protocol["cpu_threads"],
        checkpoint_every=protocol["checkpoint_every"],
        selection_metric=protocol["selection_metric"],
        data_fingerprint=protocol["data_fingerprint"],
        source_sha256=protocol["source_sha256"],
        **scaled_heatmap_settings(size),
    )


def run_one_fold(
    data, train, test, size, fold, output, protocol, weights, completed_jobs, keep_last
):
    folder = output / f"resnet34_{size}" / f"fold_{fold}"
    config = fold_config(protocol, size, fold)
    signature = stable_hash(config)
    if (folder / "fold_complete.json").exists():
        done = json.loads((folder / "fold_complete.json").read_text())
        if done["signature"] != signature:
            raise ValueError("Cached fold signature differs.")
        for metric in ["rmse_coordinate_px", "mean_distance_px"]:
            if not (folder / f"best_{metric}.pt").is_file():
                raise ValueError("Completed fold is missing a best-model checkpoint.")
        valid = train[train.fold == fold]
        validate_oof(read_rows(folder / "validation_raw.csv"), valid.to_dict("records"))
        validate_test_rows(read_rows(folder / "test_raw.csv"), test.to_dict("records"), fold)
        if size == 320:
            validate_oof(read_rows(folder / "validation_flip.csv"), valid.to_dict("records"))
            validate_test_rows(read_rows(folder / "test_flip.csv"), test.to_dict("records"), fold)
        print(f"Skip completed {size}, fold{fold}.", flush=True)
        return
    save_json(
        output / "status.json",
        dict(state="running", image_size=size, fold=fold, updated_unix=time.time()),
    )
    model, training = train_cv_fold(
        data,
        train[train.fold != fold],
        train[train.fold == fold],
        folder,
        config,
        weights,
        torch.device("cuda:0"),
        output,
        completed_jobs,
    )
    valid = train[train.fold == fold]
    valid_loader = DataLoader(
        FoveaDataset(data, valid, size),
        batch_size=protocol["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        FoveaDataset(data, test, size),
        batch_size=protocol["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    metrics = []
    for mode in (["raw", "flip"] if size == 320 else ["raw"]):
        val_points = predict_cv(
            model, valid_loader, torch.device("cuda:0"), config["decode_radius"], mode == "flip"
        )
        val_rows = prediction_rows(valid.to_dict("records"), val_points, fold, True)
        atomic_csv(folder / f"validation_{mode}.csv", val_rows)
        test_points = predict_cv(
            model, test_loader, torch.device("cuda:0"), config["decode_radius"], mode == "flip"
        )
        atomic_csv(
            folder / f"test_{mode}.csv",
            prediction_rows(test.to_dict("records"), test_points, fold, False),
        )
        metrics.append(
            dict(
                arm=f"resnet34_{size}" + ("_flip" if mode == "flip" else ""),
                fold=fold,
                best_epoch=training["best"][protocol["selection_metric"]]["epoch"],
                **metrics_from_rows(val_rows),
            )
        )
    atomic_csv(folder / "fold_metrics.csv", metrics)
    save_json(
        folder / "fold_complete.json",
        dict(
            signature=signature,
            training=training,
            predictions_complete=True,
            updated_unix=time.time(),
        ),
    )
    # Remove only this completed run's own optimizer checkpoint, keeping both best model files.
    if not keep_last and (folder / "last.pt").exists():
        (folder / "last.pt").unlink()
    del model, valid_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()


def build_reports(data, train, test, output, protocol):
    comparisons, fold_metrics, available, epochs = [], [], [], {}
    for arm in ARMS:
        size = 512 if arm.endswith("512") else 320
        mode = "flip" if arm.endswith("flip") else "raw"
        folders = [output / f"resnet34_{size}" / f"fold_{fold}" for fold in range(5)]
        if not all((folder / "fold_complete.json").exists() for folder in folders):
            continue
        oof, test_folds = [], {}
        epochs[arm] = []
        for fold, folder in enumerate(folders):
            done = json.loads((folder / "fold_complete.json").read_text())
            if done["signature"] != stable_hash(fold_config(protocol, size, fold)):
                raise ValueError("Report found mismatched completed fold.")
            oof.extend(read_rows(folder / f"validation_{mode}.csv"))
            test_folds[fold] = read_rows(folder / f"test_{mode}.csv")
            for row in read_rows(folder / "fold_metrics.csv"):
                if row["arm"] == arm:
                    fold_metrics.append(row)
                    epochs[arm].append(int(row["best_epoch"]))
        oof = validate_oof(oof, train.to_dict("records"))
        atomic_csv(output / f"oof_{arm}.csv", oof)
        averaged = average_test_folds(test_folds, test.to_dict("records"))
        atomic_csv(output / f"test_{arm}_5fold.csv", averaged)
        write_submission(
            data / "sample_submission.csv", averaged, output / f"submission_{arm}_5fold.csv"
        )
        comparisons.append(dict(arm=arm, **metrics_from_rows(oof)))
        available.append(arm)
        worst_ids = [
            int(r["image_id"])
            for r in sorted(oof, key=lambda r: float(r["distance_px"]), reverse=True)[:20]
        ]
        ordered = train.set_index("image_id").loc[worst_ids].reset_index()
        lookup = {int(r["image_id"]): r for r in oof}
        points = [
            [
                float(lookup[key]["Fovea_X"]) / float(lookup[key]["width"]),
                float(lookup[key]["Fovea_Y"]) / float(lookup[key]["height"]),
            ]
            for key in worst_ids
        ]
        contact_sheet(data, ordered, output / f"worst_examples_{arm}.png", points)
    if comparisons:
        comparisons.sort(key=lambda r: r[protocol["selection_metric"]])
        atomic_csv(output / "comparison.csv", comparisons)
        atomic_csv(output / "fold_metrics.csv", fold_metrics)
        print("\nOOF comparison (development scores):", flush=True)
        print(pd.DataFrame(comparisons).to_string(index=False), flush=True)
    complete = set(available) == set(ARMS)
    selection = dict(
        complete=complete,
        available_arms=available,
        metric=protocol["selection_metric"],
        official_metric_confirmed=False,
        fold_epochs=epochs,
        limitation="Fold stopping and arm choice use validation labels; OOF is development evaluation, not an independent test score.",
        submission_models="Equal-weight ensemble of five 64-image-training fold checkpoints; no full-data refit.",
        submitted_to_competition=False,
    )
    if complete:
        selection["selected_arm"] = comparisons[0]["arm"]
        source = output / f'submission_{comparisons[0]["arm"]}_5fold.csv'
        shutil.copy2(source, output / "submission_selected_5fold.csv")
    save_json(output / "selection.json", selection)
    return complete


def restore_backup(source, output, protocol):
    if source is None:
        return
    source = Path(source).resolve()
    if source == output.resolve():
        return
    candidate = source / "protocol.json"
    if not candidate.is_file() or json.loads(candidate.read_text()) != protocol:
        raise ValueError(
            "Backup protocol/data/code differs. Choose the correct prior output folder."
        )
    if (output / "protocol.json").exists():
        print(
            "Current output already has progress; resuming it without merging backup files.",
            flush=True,
        )
        return
    for path in source.rglob("*"):
        if not path.is_file() or path.is_symlink() or path.suffix in {".tmp", ".zip"}:
            continue
        if not path.resolve().is_relative_to(source):
            raise ValueError("Backup path escapes its source directory.")
        destination = output / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def light_archive(output):
    target = output / "task3_cv_results_light.zip"
    temporary = output / "task3_cv_results_light.zip.tmp"
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in output.rglob("*"):
            if path.is_file() and path.suffix in {".json", ".csv", ".png", ".md"}:
                archive.write(path, path.relative_to(output))
    temporary.replace(target)


def run_cv(
    data_dir=None,
    output_dir="/kaggle/working/task3_resnet34_cv",
    resume_dir=None,
    weights=None,
    epochs=80,
    batch_size=8,
    seed=42,
    selection_metric="rmse_coordinate_px",
    checkpoint_every=5,
    cpu_threads=2,
    keep_last=False,
    run_sizes=(320, 512),
    run_folds=(0, 1, 2, 3, 4),
    allow_local=False,
):
    if not Path("/kaggle/working").exists() and not allow_local:
        raise RuntimeError("Use this Notebook on Kaggle; local training is disabled by default.")
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is not enabled; stop before CPU training.")
    if selection_metric not in ("rmse_coordinate_px", "mean_distance_px"):
        raise ValueError("Unsupported selection metric.")
    if epochs < 1 or batch_size < 1 or checkpoint_every < 1:
        raise ValueError("Epoch, batch, checkpoint interval must be positive.")
    if not run_sizes or not set(run_sizes) <= {320, 512} or len(set(run_sizes)) != len(run_sizes):
        raise ValueError("Invalid resolution schedule.")
    if (
        not run_folds
        or not set(run_folds) <= set(range(5))
        or len(set(run_folds)) != len(run_folds)
    ):
        raise ValueError("Invalid fold schedule.")
    data, output = find_data(data_dir), Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 3 * 1024**3:
        raise RuntimeError(
            "Need at least 3 GiB free for weights, temporary checkpoint replacement and results."
        )
    manifest = load_manifest(data)
    train = assign_folds(manifest[manifest.subset == "train"], seed, 5)
    test = manifest[manifest.subset == "test"].copy()
    if len(train) != 80 or len(test) != 20:
        raise ValueError("Unexpected Task3 image counts.")
    protocol = dict(
        version=CV_VERSION,
        seed=seed,
        folds=5,
        epochs=epochs,
        batch_size=batch_size,
        cpu_threads=cpu_threads,
        checkpoint_every=checkpoint_every,
        selection_metric=selection_metric,
        data_fingerprint=hashlib.sha256(train.to_csv(index=False).encode()).hexdigest(),
        test_fingerprint=hashlib.sha256(test.to_csv(index=False).encode()).hexdigest(),
        source_sha256=source_fingerprints(),
        plan_sizes=[320, 512],
        plan_arms=list(ARMS),
        gaussian_policy="sigma=1.5*(size/320); radius=round(3*size/320)",
        tta_policy="Horizontal flip, inverse x, equal coordinate average; same raw-selected checkpoint.",
        augmentation="Same FoveaDataset geometric/color augmentations and ImageNet normalization as prior baseline.",
    )
    restore_backup(resume_dir, output, protocol)
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing run configuration differs. Use a new output directory.")
    save_json(protocol_path, protocol)
    train.to_csv(output / "split.csv", index=False)
    manifest.to_csv(output / "manifest.csv", index=False)
    save_json(
        output / "environment.json",
        dict(
            torch=torch.__version__,
            numpy=np.__version__,
            gpu=torch.cuda.get_device_name(0),
            visible_gpu_count=torch.cuda.device_count(),
            device_policy="cuda:0, folds sequential, no DataParallel",
            cpu_threads=cpu_threads,
            workers=0,
        ),
    )
    print(
        f"{len(train)} labeled images, {train.pixel_sha256.nunique()} pixel groups, 5 folds.",
        flush=True,
    )
    print(
        "GPU:", torch.cuda.get_device_name(0), "| using cuda:0 only | outputs:", output, flush=True
    )
    weights = resolve_weights(output, weights)
    completed_jobs = {
        f"{size}/{fold}"
        for size in (320, 512)
        for fold in range(5)
        if (output / f"resnet34_{size}/fold_{fold}/fold_complete.json").exists()
    }
    with training_lock(output):
        try:
            for size in run_sizes:
                for fold in run_folds:
                    run_one_fold(
                        data,
                        train,
                        test,
                        size,
                        fold,
                        output,
                        protocol,
                        weights,
                        completed_jobs,
                        keep_last,
                    )
                    completed_jobs.add(f"{size}/{fold}")
                build_reports(data, train, test, output, protocol)
                light_archive(output)
            complete = build_reports(data, train, test, output, protocol)
            save_json(
                output / "status.json",
                dict(
                    state="completed" if complete else "partial_completed",
                    completed_jobs=sorted(completed_jobs),
                    updated_unix=time.time(),
                ),
            )
            light_archive(output)
        except BaseException as error:
            save_json(
                output / "status.json",
                dict(
                    state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    error=str(error),
                    updated_unix=time.time(),
                ),
            )
            light_archive(output)
            raise
    return output


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir", default="/kaggle/working/task3_resnet34_cv")
    parser.add_argument("--resume-dir")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--allow-local", action="store_true")
    args = parser.parse_args()
    run_cv(**vars(args))


if __name__ == "__main__":
    cli()
