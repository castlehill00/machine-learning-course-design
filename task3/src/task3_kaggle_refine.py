"""Kaggle-only five-fold local refinement using cached 512 coarse predictions."""

from __future__ import annotations

import csv
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import statistics
import sys
import time
import zipfile

if __name__ == "__main__" and not Path("/kaggle/working").is_dir():
    raise SystemExit("Use the generated Kaggle Notebook. No local training was started.")

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from task3_pipeline import (
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
from task3_kaggle_cv import (
    ResNet34CV,
    atomic_csv,
    atomic_torch_save,
    average_test_folds,
    metrics_from_rows,
    read_rows,
    resolve_weights,
    restore_backup,
    stable_hash,
    validate_oof,
    validate_test_rows,
    write_submission,
)

REFINE_VERSION = "resnet34-refine-v1"
REFINE_ARMS = ("coarse512", "refine384", "coarse_refine_equal")


class CoarseBundle:
    """Read only metadata/predictions from a directory or ZIP; never extract/load weights."""

    def __init__(self, path):
        self.path, self.prefix, self.archive = Path(path).resolve(), "", None
        if self.path.is_dir():
            return
        if not self.path.is_file() or self.path.suffix.lower() != ".zip":
            raise ValueError(f"Coarse source is not a directory or ZIP: {path}")
        self.archive = zipfile.ZipFile(self.path)
        try:
            candidates = []
            for entry in self.archive.infolist():
                pure = PurePosixPath(entry.filename)
                if pure.is_absolute() or ".." in pure.parts or "\\" in entry.filename:
                    raise ValueError("Unsafe archive member path.")
                if pure.name == "protocol.json" and entry.file_size < 1000000:
                    value = json.loads(self.archive.read(entry).decode("utf-8-sig"))
                    if value.get("version") == "resnet34-cv-v1":
                        candidates.append(entry.filename[: -len("protocol.json")])
            if len(candidates) != 1:
                raise ValueError("ZIP must contain exactly one original CV protocol.")
            self.prefix = candidates[0]
        except BaseException:
            self.close()
            raise

    def text(self, relative):
        if self.archive is None:
            path = (self.path / relative).resolve()
            if not path.is_relative_to(self.path):
                raise ValueError("Coarse file escapes its source folder.")
            if path.stat().st_size > 5000000:
                raise ValueError("Unexpectedly large metadata file.")
            return path.read_text(encoding="utf-8-sig")
        name = self.prefix + relative
        if self.archive.getinfo(name).file_size > 5000000:
            raise ValueError("Unexpectedly large metadata member.")
        return self.archive.read(name).decode("utf-8-sig")

    def rows(self, name):
        return list(csv.DictReader(io.StringIO(self.text(name), newline="")))

    def json(self, name):
        return json.loads(self.text(name))

    def close(self):
        if self.archive is not None:
            self.archive.close()
            self.archive = None


def find_coarse_source(explicit=None, working="/kaggle/working", input_root="/kaggle/input"):
    if explicit is not None:
        return Path(explicit).resolve()
    current = Path(working) / "task3_resnet34_cv"
    if (current / "protocol.json").is_file():
        return current.resolve()
    candidates = []
    root = Path(input_root)
    if root.exists():
        for path in root.rglob("protocol.json"):
            try:
                if (
                    json.loads(path.read_text(encoding="utf-8-sig")).get("version")
                    == "resnet34-cv-v1"
                ):
                    candidates.append(path.parent.resolve())
            except (OSError, ValueError):
                continue
        for name in ("task3_cv_results_light.zip", "task3_cv_full_backup.zip"):
            candidates += [p.resolve() for p in root.rglob(name)]
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise ValueError(
            f"Attach the previous light results ZIP as an Input, or set COARSE_SOURCE. Found: {candidates}"
        )
    return candidates[0]


def validate_coarse_bundle(source, train_records, test_records):
    bundle = CoarseBundle(source)
    try:
        protocol = bundle.json("protocol.json")
        if protocol.get("version") != "resnet34-cv-v1" or protocol.get("folds") != 5:
            raise ValueError("Expected the previous five-fold CV experiment.")
        if protocol.get("selection_metric") != "rmse_coordinate_px":
            raise ValueError(
                "This predeclared experiment reuses the RMSE-selected 512 coarse checkpoints."
            )
        old_split = bundle.rows("split.csv")
        if len(old_split) != len(train_records):
            raise ValueError("Previous split is incomplete.")
        prior = {int(r["image_id"]): r for r in old_split}
        if len(prior) != len(train_records):
            raise ValueError("Duplicate IDs in previous split.")
        for record in train_records:
            old = prior[int(record["image_id"])]
            for field in ("fold", "width", "height"):
                if int(old[field]) != int(record[field]):
                    raise ValueError("Coarse split or dimensions differ.")
            if old["pixel_sha256"] != record["pixel_sha256"]:
                raise ValueError("Coarse image hash differs.")
            if any(abs(float(old[k]) - float(record[k])) > 1e-6 for k in ("x", "y")):
                raise ValueError("Coarse labels differ.")
        validation, testing, metadata, coverage = {}, {}, [], []
        for fold in range(5):
            prefix = f"resnet34_512/fold_{fold}/"
            config = bundle.json(prefix + "train_config.json")
            done = bundle.json(prefix + "fold_complete.json")
            signature = stable_hash(config)
            if (
                int(config["fold"]) != fold
                or int(config["image_size"]) != 512
                or config["data_fingerprint"] != protocol["data_fingerprint"]
                or config["source_sha256"] != protocol["source_sha256"]
                or done.get("signature") != signature
                or not done.get("predictions_complete")
                or done["training"]["signature"] != signature
            ):
                raise ValueError("Coarse fold metadata is inconsistent.")
            held = [r for r in train_records if int(r["fold"]) == fold]
            rows = validate_oof(bundle.rows(prefix + "validation_raw.csv"), held)
            # Verify finite in-bounds coordinates as well as OOF membership/labels.
            validate_test_rows(rows, held, fold)
            best = done["training"]["best"]["rmse_coordinate_px"]
            if not math.isclose(
                metrics_from_rows(rows)["rmse_coordinate_px"], float(best["score"]), abs_tol=1e-6
            ):
                raise ValueError("Coarse predictions do not match the selected checkpoint metrics.")
            validation[fold] = {int(r["image_id"]): r for r in rows}
            testing[fold] = validate_test_rows(
                bundle.rows(prefix + "test_raw.csv"), test_records, fold
            )
            metadata.append(dict(fold=fold, config=config, selected=best))
            coverage.extend(rows)
        fingerprint = stable_hash(
            dict(
                protocol=protocol,
                split=old_split,
                folds=metadata,
                validation=validation,
                test=testing,
            )
        )
        return dict(
            validation=validation,
            test=testing,
            fingerprint=fingerprint,
            protocol=protocol,
            metadata=metadata,
            baseline_metrics=metrics_from_rows(coverage),
        )
    finally:
        bundle.close()


def crop_geometry(center, side):
    if side < 1 or int(side) != side or not all(math.isfinite(float(v)) for v in center):
        raise ValueError("Invalid crop geometry.")
    left, top = [math.floor(float(v) - side / 2) for v in center]
    return (left, top, left + int(side), top + int(side))


def point_to_patch(point, box):
    return (
        (float(point[0]) - box[0]) / (box[2] - box[0]),
        (float(point[1]) - box[1]) / (box[3] - box[1]),
    )


def patch_to_original(point, box):
    return (
        box[0] + float(point[0]) * (box[2] - box[0]),
        box[1] + float(point[1]) * (box[3] - box[1]),
    )


def make_refinement_row(record, point, coarse, box, fold, max_shift):
    """Map patch coordinates back before scoring; a fixed fallback needs no labels."""
    width, height = int(record["width"]), int(record["height"])
    cx, cy = float(coarse["Fovea_X"]), float(coarse["Fovea_Y"])
    if not (0 <= cx < width and 0 <= cy < height):
        raise ValueError("Coarse coordinate outside original image.")
    raw_x, raw_y = patch_to_original(point, box)
    shift = math.hypot(raw_x - cx, raw_y - cy)
    reason = ""
    if not all(math.isfinite(v) for v in (raw_x, raw_y)):
        raise ValueError("Non-finite refinement coordinate.")
    if not (0 <= raw_x < width and 0 <= raw_y < height):
        reason = "outside_image"
    elif shift > max_shift:
        reason = "shift_gt_fixed_limit"
    x, y = (cx, cy) if reason else (raw_x, raw_y)
    row = dict(
        image_id=int(record["image_id"]),
        fold=int(fold),
        width=width,
        height=height,
        pixel_sha256=str(record["pixel_sha256"]),
        nx=x / width,
        ny=y / height,
        Fovea_X=x,
        Fovea_Y=y,
        coarse_x=cx,
        coarse_y=cy,
        raw_refined_x=raw_x,
        raw_refined_y=raw_y,
        shift_px=shift,
        fallback=int(bool(reason)),
        fallback_reason=reason,
        crop_left=box[0],
        crop_top=box[1],
        crop_side=box[2] - box[0],
    )
    if record["subset"] == "train":
        dx, dy = x - float(record["x"]), y - float(record["y"])
        row.update(
            gt_x=float(record["x"]),
            gt_y=float(record["y"]),
            dx=dx,
            dy=dy,
            distance_px=math.hypot(dx, dy),
            squared_error=dx * dx + dy * dy,
        )
    return row


def blend_prediction_rows(coarse_rows, refine_rows, labeled):
    coarse = {int(r["image_id"]): r for r in coarse_rows}
    refined = {int(r["image_id"]): r for r in refine_rows}
    if (
        len(coarse) != len(coarse_rows)
        or len(refined) != len(refine_rows)
        or set(coarse) != set(refined)
    ):
        raise ValueError("Blend IDs differ or repeat.")
    result = []
    for key in sorted(coarse):
        a, b = coarse[key], refined[key]
        for field in ("fold", "width", "height"):
            if int(a[field]) != int(b[field]):
                raise ValueError("Blend fold or dimensions differ.")
        if a["pixel_sha256"] != b["pixel_sha256"]:
            raise ValueError("Blend pixels differ.")
        row = {name: a[name] for name in ("image_id", "fold", "width", "height", "pixel_sha256")}
        x, y = [(float(a[f"Fovea_{axis}"]) + float(b[f"Fovea_{axis}"])) / 2 for axis in ("X", "Y")]
        row.update(Fovea_X=x, Fovea_Y=y, nx=x / int(a["width"]), ny=y / int(a["height"]))
        if labeled:
            if any(abs(float(a[k]) - float(b[k])) > 1e-6 for k in ("gt_x", "gt_y")):
                raise ValueError("Blend ground truth differs.")
            dx, dy = x - float(a["gt_x"]), y - float(a["gt_y"])
            row.update(
                gt_x=float(a["gt_x"]),
                gt_y=float(a["gt_y"]),
                dx=dx,
                dy=dy,
                distance_px=math.hypot(dx, dy),
                squared_error=dx * dx + dy * dy,
            )
        result.append(row)
    return result


class OriginalImageCache:
    def __init__(self, data):
        self.data, self.images = Path(data), {}

    def get(self, row):
        key = int(row["image_id"])
        if key not in self.images:
            with Image.open(self.data / row["subset"] / row["filename"]) as image:
                self.images[key] = image.convert("RGB")
        image = self.images[key]
        if image.size != (int(row["width"]), int(row["height"])):
            raise ValueError("Original image dimensions changed.")
        return image


class RefinementDataset(Dataset):
    def __init__(self, records, cache, config, fold, mode, coarse=None):
        self.records, self.cache, self.config = list(records), cache, config
        self.mode, self.coarse = mode, coarse
        if mode not in ("train", "validation", "test"):
            raise ValueError("Invalid dataset mode.")
        for r in self.records:
            if mode == "train" and (r["subset"] != "train" or int(r["fold"]) == fold):
                raise ValueError("Held-out labels must not enter refinement training.")
            if mode == "validation" and (r["subset"] != "train" or int(r["fold"]) != fold):
                raise ValueError("Validation fold differs.")
            if mode == "test" and r["subset"] != "test":
                raise ValueError("Test dataset contains labels.")
        if mode != "train" and (
            coarse is None or {int(r["image_id"]) for r in self.records} != set(coarse)
        ):
            raise ValueError("Inference requires matching cached coarse predictions.")
        self.tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.jitter = transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.08, hue=0.01
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row, cfg = self.records[index], self.config
        if self.mode == "train":
            center = (
                float(row["x"]) + random.uniform(-cfg["center_jitter"], cfg["center_jitter"]),
                float(row["y"]) + random.uniform(-cfg["center_jitter"], cfg["center_jitter"]),
            )
        else:
            # No label fields are consulted for validation/test crop geometry.
            prior = self.coarse[int(row["image_id"])]
            center = (float(prior["Fovea_X"]), float(prior["Fovea_Y"]))
        box = crop_geometry(center, cfg["crop_side"])
        image = (
            self.cache.get(row)
            .crop(box)
            .resize((cfg["image_size"], cfg["image_size"]), Image.Resampling.BILINEAR)
        )
        point = np.array(
            point_to_patch((row["x"], row["y"]), box) if self.mode == "train" else (0.5, 0.5),
            dtype=np.float32,
        )
        if self.mode == "train":
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                point[0] = 1 - point[0]
            angle, scale = math.radians(random.uniform(-8, 8)), random.uniform(0.95, 1.05)
            a, b = scale * math.cos(angle), -scale * math.sin(angle)
            size, middle = cfg["image_size"], cfg["image_size"] / 2
            tx, ty = random.uniform(-0.02, 0.02) * size, random.uniform(-0.02, 0.02) * size
            matrix = np.array(
                [
                    [a, b, middle + tx - a * middle - b * middle],
                    [-b, a, middle + ty + b * middle - a * middle],
                    [0, 0, 1],
                ]
            )
            point = ((matrix @ [point[0] * size, point[1] * size, 1])[:2] / size).astype(np.float32)
            if not np.isfinite(point).all() or not ((point > 0).all() and (point < 1).all()):
                raise ValueError("Augmentation moved the label out of the patch.")
            image = image.transform(
                (size, size),
                Image.Transform.AFFINE,
                np.linalg.inv(matrix)[:2].reshape(-1),
                resample=Image.Resampling.BILINEAR,
                fillcolor=(0, 0, 0),
            )
            image = self.jitter(image)
        return self.tensor(image), torch.from_numpy(point), torch.tensor(box, dtype=torch.int64)


@torch.inference_mode()
def predict_refinement(model, loader, device, config, coarse, fold):
    model.eval()
    result, offset = [], 0
    for images, _, boxes in loader:
        with torch.autocast(device_type="cuda", enabled=True):
            logits = model(images.to(device))
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite inference heatmap.")
        points = decode_heatmap(logits, radius=config["decode_radius"]).cpu().tolist()
        for point, box in zip(points, boxes.tolist()):
            row = loader.dataset.records[offset]
            result.append(
                make_refinement_row(
                    row, point, coarse[int(row["image_id"])], box, fold, config["max_shift"]
                )
            )
            offset += 1
    if offset != len(loader.dataset):
        raise ValueError("Refinement inference row count differs.")
    return result


def refinement_sources():
    if "REFINE_SOURCE_SHA256" in globals():
        return dict(REFINE_SOURCE_SHA256)
    base = Path(__file__).resolve().parent
    return {
        name: hashlib.sha256((base / name).read_bytes()).hexdigest()
        for name in ("task3_pipeline.py", "task3_kaggle_cv.py", "task3_kaggle_refine.py")
    }


def load_refine_checkpoint(path, signature, optimizer=False):
    item = torch.load(path, map_location="cpu", weights_only=not optimizer)
    if item.get("signature") != signature:
        raise ValueError("Refinement checkpoint protocol differs.")
    return item


def validate_refine_config(config):
    for key in (
        "epochs",
        "batch_size",
        "checkpoint_every",
        "cpu_threads",
        "image_size",
        "crop_side",
    ):
        if int(config[key]) != config[key] or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    if config["image_size"] != 384 or config["crop_side"] != 768:
        raise ValueError("This experiment predeclares a 768-pixel native crop resized to 384.")
    if (
        not 0 < config["center_jitter"] <= config["crop_side"] / 4
        or not 0 < config["max_shift"] <= config["crop_side"] / 2
    ):
        raise ValueError("Invalid crop jitter/fallback limit.")
    if config["selection_metric"] not in ("rmse_coordinate_px", "mean_distance_px"):
        raise ValueError("Unsupported metric.")


def refinement_eta(output, epoch, times, config):
    if len(times) < 3:
        return
    pending = sum(
        not (output / f"fold_{fold}/fold_complete.json").exists()
        for fold in range(5)
        if fold != config["fold"]
    )
    rate = statistics.median(times[-8:])
    remaining = (config["epochs"] - epoch + pending * config["epochs"]) * rate / 60
    save_json(
        output / "eta.json",
        dict(
            fold=config["fold"],
            epoch=epoch,
            seconds_per_epoch=rate,
            approximate_training_minutes_remaining=remaining,
            excludes="Image loading, future cold starts, checkpoint restore and final reporting; partial folds estimated as full.",
        ),
    )
    if epoch == 3 or epoch % 10 == 0:
        print(f"ETA about {remaining:.1f} training minutes for all five folds.", flush=True)


def train_refine_fold(development, held, cache, coarse, folder, config, weights, output):
    device, fold, selected = torch.device("cuda:0"), config["fold"], config["selection_metric"]
    folder.mkdir(parents=True, exist_ok=True)
    signature = stable_hash(config)
    config_file = folder / "train_config.json"
    if config_file.exists() and json.loads(config_file.read_text()) != config:
        raise ValueError("Refinement fold configuration differs.")
    save_json(config_file, config)
    valid_set = RefinementDataset(
        held, cache, config, fold, "validation", coarse["validation"][fold]
    )
    valid_loader = DataLoader(
        valid_set, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )
    complete_file = folder / "training_complete.json"
    if complete_file.exists():
        done = json.loads(complete_file.read_text())
        if done["signature"] != signature:
            raise ValueError("Completed refinement fold differs.")
        model = ResNet34CV().to(device)
        checkpoint = load_refine_checkpoint(folder / f"best_{selected}.pt", signature)
        model.load_state_dict(checkpoint["model"], strict=True)
        return model, done, valid_loader
    seed_everything(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    train_set = RefinementDataset(development, cache, config, fold, "train")
    loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    last = folder / "last.pt"
    model = ResNet34CV(None if last.exists() else weights).to(device)
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    optimizer = torch.optim.AdamW(
        [
            dict(params=model.backbone.parameters(), lr=config["lr_backbone"]),
            dict(
                params=[p for p in model.parameters() if id(p) not in backbone_ids],
                lr=config["lr_decoder"],
            ),
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 0.05 + 0.95 * (1 + math.cos(math.pi * step / config["epochs"])) / 2,
    )
    scaler = torch.amp.GradScaler("cuda")
    history, start, times = [], 0, []
    if last.exists():
        state = load_refine_checkpoint(last, signature, optimizer=True)
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
        print(f"Resume refinement fold {fold} after epoch {start}.", flush=True)
    tracked = ("rmse_coordinate_px", "mean_distance_px")
    best = {key: dict(score=float("inf"), epoch=0) for key in tracked}
    for key in tracked:
        if (folder / f"best_{key}.pt").exists():
            item = load_refine_checkpoint(folder / f"best_{key}.pt", signature)
            best[key] = dict(score=float(item["metrics"][key]), epoch=int(item["epoch"]))
            del item
    torch.cuda.reset_peak_memory_stats(device=0)
    for epoch in range(start + 1, config["epochs"] + 1):
        begin = time.perf_counter()
        model.train()
        total, count, skipped = 0.0, 0, 0
        for images, points, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            images, points = images.to(device), points.to(device)
            with torch.autocast(device_type="cuda", enabled=True):
                logits = model(images)
                if logits.shape != (len(images), 96, 96):
                    raise RuntimeError("Expected a 96x96 refinement heatmap.")
                loss = heatmap_loss(logits, points, sigma=config["sigma"])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite refinement loss.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            previous = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < previous)
            total += float(loss.detach()) * len(images)
            count += len(images)
        if skipped == len(loader) and history and history[-1]["amp_skipped_steps"] == len(loader):
            raise RuntimeError("Every optimizer step skipped for two epochs.")
        validation = predict_refinement(
            model, valid_loader, device, config, coarse["validation"][fold], fold
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
                        metrics=metrics,
                        train_config=config,
                    ),
                )
        applied_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        row = dict(
            epoch=epoch,
            train_heatmap_ce=total / count,
            lr_backbone=applied_lr,
            **metrics,
            validation_fallbacks=sum(int(r["fallback"]) for r in validation),
            amp_skipped_steps=skipped,
            gpu_peak_mib=torch.cuda.max_memory_allocated(device=0) / 1024**2,
            compute_seconds=time.perf_counter() - begin,
        )
        history.append(row)
        if epoch % config["checkpoint_every"] == 0 or epoch == config["epochs"]:
            atomic_torch_save(
                last,
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
        times.append(time.perf_counter() - begin)
        atomic_csv(folder / "history.csv", history)
        save_json(
            folder / "status.json",
            dict(state="training", epoch=epoch, best=best, updated_unix=time.time()),
        )
        refinement_eta(output, epoch, times, config)
        print(
            f'Refine fold{fold} {epoch:03}/{config["epochs"]} CE={total/count:.4f}'
            f' RMSE={metrics["rmse_coordinate_px"]:.2f}px MRE={metrics["mean_distance_px"]:.2f}px'
            f' best={best[selected]["score"]:.2f}px fallback={row["validation_fallbacks"]}'
            f" AMPskip={skipped} {times[-1]:.1f}s",
            flush=True,
        )
    done = dict(signature=signature, best=best, selection_metric=selected, epochs=config["epochs"])
    save_json(complete_file, done)
    model.load_state_dict(
        load_refine_checkpoint(folder / f"best_{selected}.pt", signature)["model"], strict=True
    )
    return model, done, valid_loader


def refine_fold_config(protocol, fold):
    return dict(protocol, fold=fold, seed=protocol["split_seed"] + 100 + fold)


def refinement_reports(data, train, test, coarse, output, protocol, render=True):
    records, tests = train.to_dict("records"), test.to_dict("records")
    val_folds = {f: [] for f in range(5)}
    test_folds, fold_metrics = {}, []
    available = []
    for fold in range(5):
        folder = output / f"fold_{fold}"
        if not (folder / "fold_complete.json").exists():
            continue
        expected = stable_hash(refine_fold_config(protocol, fold))
        done = json.loads((folder / "fold_complete.json").read_text())
        if done["signature"] != expected or not done["predictions_complete"]:
            raise ValueError("Refinement fold completion metadata differs.")
        rows = read_rows(folder / "validation_refine.csv")
        held = [r for r in records if int(r["fold"]) == fold]
        val_folds[fold] = validate_oof(rows, held)
        validate_test_rows(rows, held, fold)
        test_folds[fold] = read_rows(folder / "test_refine.csv")
        validate_test_rows(test_folds[fold], tests, fold)
        available.append(fold)
        fold_metrics.append(
            dict(
                fold=fold,
                best_epoch=done["training"]["best"][protocol["selection_metric"]]["epoch"],
                **metrics_from_rows(rows),
            )
        )
    if fold_metrics:
        atomic_csv(output / "fold_metrics.csv", fold_metrics)
    complete = set(available) == set(range(5))
    selection = dict(
        complete=complete,
        completed_folds=available,
        metric=protocol["selection_metric"],
        official_metric_confirmed=False,
        submitted_to_competition=False,
        coarse_source="Cached original 512 per-fold predictions; second stage initialized from ImageNet.",
        limitation="Both stages selected epochs using the same development folds; OOF is not an independent test.",
    )
    if not complete:
        save_json(output / "selection.json", selection)
        return False
    refined = validate_oof([r for f in range(5) for r in val_folds[f]], records)
    coarse_oof = validate_oof(
        [r for f in range(5) for r in coarse["validation"][f].values()], records
    )
    arms = dict(
        coarse512=coarse_oof,
        refine384=refined,
        coarse_refine_equal=blend_prediction_rows(coarse_oof, refined, True),
    )
    arm_tests = dict(
        coarse512={f: list(coarse["test"][f].values()) for f in range(5)},
        refine384=test_folds,
        coarse_refine_equal={
            f: blend_prediction_rows(list(coarse["test"][f].values()), test_folds[f], False)
            for f in range(5)
        },
    )
    comparison = []
    for arm in REFINE_ARMS:
        rows = validate_oof(arms[arm], records)
        atomic_csv(output / f"oof_{arm}.csv", rows)
        averages = average_test_folds(arm_tests[arm], tests)
        atomic_csv(output / f"test_{arm}_5fold.csv", averages)
        write_submission(
            data / "sample_submission.csv", averages, output / f"submission_{arm}_5fold.csv"
        )
        comparison.append(dict(arm=arm, **metrics_from_rows(rows)))
        if render:
            worst = sorted(rows, key=lambda r: float(r["distance_px"]), reverse=True)[:20]
            ordered = (
                train.set_index("image_id").loc[[int(r["image_id"]) for r in worst]].reset_index()
            )
            points = [
                [float(r["Fovea_X"]) / int(r["width"]), float(r["Fovea_Y"]) / int(r["height"])]
                for r in worst
            ]
            contact_sheet(data, ordered, output / f"worst_examples_{arm}.png", points)
    comparison.sort(key=lambda r: r[protocol["selection_metric"]])
    atomic_csv(output / "comparison.csv", comparison)
    selection.update(
        selected_arm=comparison[0]["arm"],
        validation_fallbacks=sum(int(r["fallback"]) for r in refined),
        test_model_fallbacks=sum(int(r["fallback"]) for f in range(5) for r in test_folds[f]),
        public_reference_score=12.31624,
        public_reference_file="submission_resnet34_fulltrain.csv",
    )
    shutil.copy2(
        output / f'submission_{selection["selected_arm"]}_5fold.csv',
        output / "submission_selected_refine_5fold.csv",
    )
    save_json(output / "selection.json", selection)
    print("\nFive-fold development comparison:", flush=True)
    print(pd.DataFrame(comparison).to_string(index=False), flush=True)
    return True


def refine_light_archive(output):
    target = output / "task3_refine_results_light.zip"
    temporary = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in output.rglob("*"):
            if path.is_file() and path.suffix in {".csv", ".json", ".png", ".md"}:
                archive.write(path, path.relative_to(output))
    temporary.replace(target)


def prepare_refinement(
    data_dir=None,
    coarse_source=None,
    output_dir="/kaggle/working/task3_refine_cv",
    resume_dir=None,
    epochs=60,
    batch_size=8,
    checkpoint_every=5,
    cpu_threads=2,
    selection_metric="rmse_coordinate_px",
):
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("Run this Notebook on Kaggle. No local training was started.")
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU before starting; CPU training is disabled.")
    cfg = dict(
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_every=checkpoint_every,
        cpu_threads=cpu_threads,
        selection_metric=selection_metric,
        image_size=384,
        crop_side=768,
        center_jitter=160.0,
        max_shift=256.0,
        sigma=1.5,
        decode_radius=3,
        lr_backbone=5e-5,
        lr_decoder=5e-4,
    )
    validate_refine_config(cfg)
    data, source, output = (
        find_data(data_dir),
        find_coarse_source(coarse_source),
        Path(output_dir).resolve(),
    )
    if source.is_dir() and (
        output == source or output.is_relative_to(source) or source.is_relative_to(output)
    ):
        raise ValueError("Keep refinement and coarse output directories separate.")
    manifest = load_manifest(data)
    # Read the previous split seed; matching is then checked against all current pixels/labels.
    bundle = CoarseBundle(source)
    try:
        split_seed = int(bundle.json("protocol.json")["seed"])
    finally:
        bundle.close()
    train = assign_folds(manifest[manifest.subset == "train"], split_seed, 5)
    test = manifest[manifest.subset == "test"].copy()
    if len(train) != 80 or len(test) != 20:
        raise ValueError("Unexpected image counts.")
    coarse = validate_coarse_bundle(source, train.to_dict("records"), test.to_dict("records"))
    protocol = dict(
        version=REFINE_VERSION,
        split_seed=split_seed,
        folds=5,
        **cfg,
        source_sha256=refinement_sources(),
        coarse_fingerprint=coarse["fingerprint"],
        initialization="Official ImageNet ResNet34; no coarse-model weight warm start",
        train_crop="Ground-truth center plus independent uniform +/-160 native pixel jitter per axis",
        inference_crop="Matching held-out/test coarse512 prediction; zero GT geometry access",
        fallback="Coarse point if refined point outside image or Euclidean shift >256 native pixels",
        augmentation="hflip0.5 affine8deg scale0.95-1.05 translate0.02 colorjitter; no flip TTA",
    )
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 3 * 1024**3:
        raise RuntimeError(
            "Need at least 3 GiB free for models, optimizer and atomic checkpoint writes."
        )
    restore_backup(resume_dir, output, protocol)
    if (output / "protocol.json").exists() and json.loads(
        (output / "protocol.json").read_text()
    ) != protocol:
        raise ValueError("Existing refinement protocol differs. Use a new OUTPUT_DIR.")
    save_json(output / "protocol.json", protocol)
    save_json(
        output / "coarse_source.json",
        dict(
            path=str(source),
            fingerprint=coarse["fingerprint"],
            protocol=coarse["protocol"],
            baseline_metrics=coarse["baseline_metrics"],
        ),
    )
    train.to_csv(output / "split.csv", index=False)
    manifest.to_csv(output / "manifest.csv", index=False)
    coverage = []
    for f in range(5):
        for row in coarse["validation"][f].values():
            box = crop_geometry((float(row["Fovea_X"]), float(row["Fovea_Y"])), cfg["crop_side"])
            x, y = point_to_patch((float(row["gt_x"]), float(row["gt_y"])), box)
            coverage.append(
                dict(
                    image_id=int(row["image_id"]),
                    fold=f,
                    gt_inside_crop=int(0 <= x < 1 and 0 <= y < 1),
                )
            )
    atomic_csv(output / "coarse_crop_coverage.csv", coverage)
    save_json(
        output / "environment.json",
        dict(
            torch=torch.__version__,
            numpy=np.__version__,
            gpu=torch.cuda.get_device_name(0),
            visible_gpu_count=torch.cuda.device_count(),
            device_policy="cuda:0 sequential folds; workers=0; original RGB cache up to about 2 GiB",
        ),
    )
    print("Previous 512 coarse OOF:", coarse["baseline_metrics"], flush=True)
    print(
        "Held-out GT inside fixed crops:",
        sum(r["gt_inside_crop"] for r in coverage),
        "/80; diagnostic only, not used to move crops.",
        flush=True,
    )
    return dict(
        data=data,
        source=source,
        output=output,
        manifest=manifest,
        train=train,
        test=test,
        coarse=coarse,
        protocol=protocol,
    )


def run_refinement(prepared, run_folds=(0, 1, 2, 3, 4), weights=None, keep_last=False):
    if not Path("/kaggle/working").is_dir() or not torch.cuda.is_available():
        raise RuntimeError("Use Kaggle GPU; local/CPU training is disabled.")
    if (
        not run_folds
        or len(set(run_folds)) != len(run_folds)
        or not set(run_folds) <= set(range(5))
    ):
        raise ValueError("Invalid fold schedule.")
    data, output, train, test, coarse, protocol = [
        prepared[k] for k in ("data", "output", "train", "test", "coarse", "protocol")
    ]
    pending = [f for f in run_folds if not (output / f"fold_{f}/fold_complete.json").exists()]
    if pending:
        cached = prepared["source"] / "_pretrained/resnet34-b627a593.pth"
        weights = resolve_weights(output, weights or (cached if cached.is_file() else None))
    cache = OriginalImageCache(data)
    with training_lock(output):
        try:
            for fold in run_folds:
                config = refine_fold_config(protocol, fold)
                signature = stable_hash(config)
                folder = output / f"fold_{fold}"
                if (folder / "fold_complete.json").exists():
                    done = json.loads((folder / "fold_complete.json").read_text())
                    if done["signature"] != signature or not done["predictions_complete"]:
                        raise ValueError("Completed refinement protocol differs.")
                    if any(
                        not (folder / f"best_{key}.pt").is_file()
                        for key in ("rmse_coordinate_px", "mean_distance_px")
                    ):
                        raise ValueError(
                            "Completed fold lacks weights. Mount the FULL refinement output, not its light ZIP."
                        )
                    print(f"Reuse completed refinement fold {fold}.", flush=True)
                    continue
                development = train[train.fold != fold].to_dict("records")
                held = train[train.fold == fold].to_dict("records")
                save_json(
                    output / "status.json",
                    dict(state="running", fold=fold, updated_unix=time.time()),
                )
                model, done, valid_loader = train_refine_fold(
                    development, held, cache, coarse, folder, config, weights, output
                )
                validation = predict_refinement(
                    model,
                    valid_loader,
                    torch.device("cuda:0"),
                    config,
                    coarse["validation"][fold],
                    fold,
                )
                test_loader = DataLoader(
                    RefinementDataset(
                        test.to_dict("records"), cache, config, fold, "test", coarse["test"][fold]
                    ),
                    batch_size=config["batch_size"],
                    shuffle=False,
                    num_workers=0,
                )
                predictions = predict_refinement(
                    model, test_loader, torch.device("cuda:0"), config, coarse["test"][fold], fold
                )
                atomic_csv(folder / "validation_refine.csv", validation)
                atomic_csv(folder / "test_refine.csv", predictions)
                save_json(
                    folder / "fold_complete.json",
                    dict(
                        signature=signature,
                        training=done,
                        predictions_complete=True,
                        updated_unix=time.time(),
                    ),
                )
                save_json(
                    folder / "status.json",
                    dict(state="completed", best=done["best"], updated_unix=time.time()),
                )
                if not keep_last and (folder / "last.pt").exists():
                    (folder / "last.pt").unlink()
                del model, valid_loader, test_loader
                gc.collect()
                torch.cuda.empty_cache()
                refinement_reports(data, train, test, coarse, output, protocol, render=False)
                refine_light_archive(output)
            complete = refinement_reports(data, train, test, coarse, output, protocol)
            save_json(
                output / "status.json",
                dict(
                    state="completed" if complete else "partial_completed", updated_unix=time.time()
                ),
            )
            refine_light_archive(output)
        except BaseException as error:
            save_json(
                output / "status.json",
                dict(
                    state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    error=str(error),
                    updated_unix=time.time(),
                ),
            )
            refine_light_archive(output)
            raise
    return output
