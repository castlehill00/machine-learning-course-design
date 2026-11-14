"""Conditional refinement ablations: inner-only stopping and fixed coarse outputs."""

from __future__ import annotations
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import sys
import time
import zipfile

if __name__ == "__main__" and not Path("/kaggle/working").is_dir():
    raise SystemExit("Use the Kaggle Notebook. No local training was started.")

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from task3_pipeline import (
    assign_folds,
    decode_heatmap,
    find_data,
    heatmap_loss,
    load_manifest,
    save_json,
    seed_everything,
    train_model,
    training_lock,
    FoveaDataset,
)
from task3_kaggle_cv import (
    ResNet34CV,
    atomic_csv,
    atomic_torch_save,
    average_test_folds,
    metrics_from_rows,
    predict_cv,
    prediction_rows,
    read_rows,
    resolve_weights,
    restore_backup,
    stable_hash,
    validate_oof,
    validate_test_rows,
    write_submission,
)
from task3_kaggle_refine import (
    CoarseBundle,
    OriginalImageCache,
    blend_prediction_rows,
    crop_geometry,
    make_refinement_row,
    point_to_patch,
    validate_coarse_bundle,
)

DIAGNOSTIC_VERSION = "refine-diagnostic-v3"
VARIANTS = ("wide", "small", "small_prior")


def diagnostic_sources():
    if "DIAGNOSTIC_SOURCE_SHA256" in globals():
        return dict(DIAGNOSTIC_SOURCE_SHA256)
    base = Path(__file__).resolve().parent
    return {
        name: hashlib.sha256((base / name).read_bytes()).hexdigest()
        for name in (
            "task3_pipeline.py",
            "task3_kaggle_cv.py",
            "task3_kaggle_refine.py",
            "task3_refine_diagnostic_v3.py",
        )
    }


def inner_partition(train, outer_fold):
    development = train[train.fold != outer_fold].copy()
    outer = train[train.fold == outer_fold].copy()
    assigned = assign_folds(development, 7000 + outer_fold, 5)
    validation_ids = set(int(x) for x in assigned.loc[assigned.fold == 0, "image_id"])
    fit = development[~development.image_id.isin(validation_ids)].copy()
    validation = development[development.image_id.isin(validation_ids)].copy()
    sets = [set(frame.pixel_sha256) for frame in (fit, validation, outer)]
    if (
        not len(fit)
        or not len(validation)
        or any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2)))
    ):
        raise ValueError("Inner/outer image groups overlap or are empty.")
    if set(fit.image_id) | set(validation.image_id) != set(development.image_id):
        raise ValueError("Inner split does not cover the outer training images.")
    return development, fit, validation, outer


def center_offset(arm, uniforms):
    u, v, tail = [float(x) for x in uniforms]
    if not all(0 < x < 1 for x in (u, v, tail)):
        raise ValueError("Uniforms must be strictly between0 and1.")
    if arm == "wide":
        return ((2 * u - 1) * 160, (2 * v - 1) * 160)
    if arm != "small":
        raise ValueError("Unknown training arm.")
    if tail < 0.1:
        return ((2 * u - 1) * 64, (2 * v - 1) * 64)
    normal = statistics.NormalDist()
    return (
        max(-48.0, min(48.0, normal.inv_cdf(u) * 16)),
        max(-48.0, min(48.0, normal.inv_cdf(v) * 16)),
    )


def selected_inner_epoch(best, baseline, min_gain=0.05):
    score = float(best["score"])
    base = float(baseline)
    if not math.isfinite(score + base) or base < 0 or int(best["epoch"]) < 1:
        raise ValueError("Invalid inner selection metrics.")
    enabled = base > 0 and score <= (1 - min_gain) * base
    return dict(
        enabled=enabled,
        epoch=int(best["epoch"]) if enabled else 0,
        inner_best_epoch=int(best["epoch"]),
        inner_score=score,
        inner_baseline=base,
        min_relative_gain=min_gain,
        reason="inner_gain_passed" if enabled else "inner_gain_below_fixed_margin",
    )


def validate_phase_config(config):
    fit = set(config["train_ids"])
    held = set(config["evaluation_ids"])
    forbidden = set(config["forbidden_ids"])
    if (
        len(fit) != len(config["train_ids"])
        or len(held) != len(config["evaluation_ids"])
        or len(forbidden) != len(config["forbidden_ids"])
        or not fit
        or fit & held
        or fit & forbidden
        or held & forbidden
    ):
        raise ValueError("Phase IDs overlap or repeat.")
    if config["role"] not in ("inner", "outer") or config["training_arm"] not in ("wide", "small"):
        raise ValueError("Unknown training role/arm.")
    if config["role"] == "inner" and not held:
        raise ValueError("Inner phase needs independent held-out images.")
    if config["role"] == "outer" and held:
        raise ValueError("Outer refit cannot use outer labels for stopping.")
    if config["epochs"] < 1 or config["image_size"] != 384 or config["crop_side"] != 768:
        raise ValueError("Invalid fine phase geometry/epochs.")


def validate_phase_request(config, has_evaluator, snapshots):
    validate_phase_config(config)
    if (config["role"] == "inner") != bool(has_evaluator):
        raise ValueError("Only inner fitting may evaluate a held-out set for epoch selection.")
    if config["role"] == "outer" and (
        not snapshots or max(snapshots) != config["epochs"] or min(snapshots) < 1
    ):
        raise ValueError(
            "Outer refit needs exactly preselected epoch snapshots, including its final epoch."
        )
    if config["role"] == "inner" and snapshots:
        raise ValueError("Inner checkpoints must be selected by inner metrics.")


class DiagnosticPatchDataset(Dataset):
    def __init__(self, records, cache, config, training, centers=None):
        self.records, self.cache, self.config = list(records), cache, config
        self.training, self.centers = training, centers
        ids = {int(r["image_id"]) for r in records}
        if len(ids) != len(records):
            raise ValueError("Repeated image in patch dataset.")
        if training:
            if ids != set(config["train_ids"]) or ids & set(config["forbidden_ids"]):
                raise ValueError("Training dataset differs from allowed phase images.")
        elif centers is None or ids != set(centers):
            raise ValueError("Prediction centers/IDs differ.")
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
        if self.training:
            # Both training arms consume the same3 draws before identical augmentations.
            draws = [max(1e-10, min(1 - 1e-10, random.random())) for _ in range(3)]
            dx, dy = center_offset(cfg["training_arm"], draws)
            center = (float(row["x"]) + dx, float(row["y"]) + dy)
        else:
            center = (
                float(self.centers[int(row["image_id"])]["Fovea_X"]),
                float(self.centers[int(row["image_id"])]["Fovea_Y"]),
            )
        box = crop_geometry(center, 768)
        image = self.cache.get(row).crop(box).resize((384, 384), Image.Resampling.BILINEAR)
        point = np.array(
            point_to_patch((row["x"], row["y"]), box) if self.training else (0.5, 0.5),
            dtype=np.float32,
        )
        if self.training:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                point[0] = 1 - point[0]
            angle, scale = math.radians(random.uniform(-8, 8)), random.uniform(0.95, 1.05)
            a, b = scale * math.cos(angle), -scale * math.sin(angle)
            tx, ty = random.uniform(-0.02, 0.02) * 384, random.uniform(-0.02, 0.02) * 384
            matrix = np.array(
                [
                    [a, b, 192 + tx - a * 192 - b * 192],
                    [-b, a, 192 + ty + b * 192 - a * 192],
                    [0, 0, 1],
                ]
            )
            point = ((matrix @ [point[0] * 384, point[1] * 384, 1])[:2] / 384).astype(np.float32)
            if not np.isfinite(point).all() or not ((point > 0).all() and (point < 1).all()):
                raise ValueError("Augmentation moved the target out of patch.")
            image = image.transform(
                (384, 384),
                Image.Transform.AFFINE,
                np.linalg.inv(matrix)[:2].reshape(-1),
                resample=Image.Resampling.BILINEAR,
                fillcolor=(0, 0, 0),
            )
            image = self.jitter(image)
        return self.tensor(image), torch.from_numpy(point), torch.tensor(box, dtype=torch.int64)


def coarse_log_prior(logits, boxes, coarse_points, sigma_native=48.0):
    if sigma_native <= 0:
        raise ValueError("Prior width must be positive.")
    height, width = logits.shape[-2:]
    # Pixel centers use the same convention as decode_heatmap and PIL resize.
    box = boxes.to(logits.device, dtype=torch.float32)
    points = torch.as_tensor(coarse_points, device=logits.device, dtype=torch.float32)
    xs = box[:, 0, None, None] + (
        torch.arange(width, device=logits.device)[None, None, :] + 0.5
    ) * (768 / width)
    ys = box[:, 1, None, None] + (
        torch.arange(height, device=logits.device)[None, :, None] + 0.5
    ) * (768 / height)
    distance = (xs - points[:, 0, None, None]) ** 2 + (ys - points[:, 1, None, None]) ** 2
    return logits.float() - distance / (2 * sigma_native**2)


@torch.inference_mode()
def diagnostic_predict(model, loader, config, centers, fold, modes):
    model.eval()
    outputs = {mode: [] for mode in modes}
    offset = 0
    for images, _, boxes in loader:
        with torch.autocast(device_type="cuda", enabled=True):
            logits = model(images.to("cuda:0"))
        if logits.shape != (len(images), 96, 96) or not torch.isfinite(logits).all():
            raise RuntimeError("Invalid diagnostic heatmap.")
        records = loader.dataset.records[offset : offset + len(images)]
        prior_points = [
            [
                float(centers[int(r["image_id"])]["Fovea_X"]),
                float(centers[int(r["image_id"])]["Fovea_Y"]),
            ]
            for r in records
        ]
        for mode in modes:
            value = (
                logits
                if mode == "plain"
                else coarse_log_prior(logits, boxes, prior_points, config["prior_sigma"])
            )
            points = decode_heatmap(value, radius=3).cpu().tolist()
            for r, point, box in zip(records, points, boxes.tolist()):
                row = make_refinement_row(r, point, centers[int(r["image_id"])], box, fold, 256.0)
                row.update(decoder=mode, refine_applied=1)
                outputs[mode].append(row)
        offset += len(images)
    if offset != len(loader.dataset):
        raise ValueError("Inference count differs.")
    return outputs


def raw_prediction_rows(rows):
    result = []
    for source in rows:
        row = dict(source)
        x, y = float(row["raw_refined_x"]), float(row["raw_refined_y"])
        row.update(Fovea_X=x, Fovea_Y=y, nx=x / int(row["width"]), ny=y / int(row["height"]))
        if "gt_x" in row:
            dx, dy = x - float(row["gt_x"]), y - float(row["gt_y"])
            row.update(
                dx=dx, dy=dy, distance_px=math.hypot(dx, dy), squared_error=dx * dx + dy * dy
            )
        result.append(row)
    return result


def inner_mode_metrics(refined, centers):
    coarse = [centers[int(r["image_id"])] for r in refined]
    equal = blend_prediction_rows(coarse, refined, True)
    raw = raw_prediction_rows(refined)
    return dict(
        score=metrics_from_rows(equal)["rmse_coordinate_px"],
        refined_rmse=metrics_from_rows(refined)["rmse_coordinate_px"],
        raw_rmse=metrics_from_rows(raw)["rmse_coordinate_px"],
        mean_distance=metrics_from_rows(equal)["mean_distance_px"],
        fallbacks=sum(int(r["fallback"]) for r in refined),
    )


def phase_checkpoint(path, signature):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("signature") != signature:
        raise ValueError("Fine checkpoint signature differs.")
    return state


def train_diagnostic_fine(
    data, records, cache, config, folder, weights, evaluator=None, snapshots=()
):
    validate_phase_request(config, evaluator is not None, snapshots)
    folder.mkdir(parents=True, exist_ok=True)
    signature = stable_hash(config)
    if (folder / "config.json").exists() and json.loads(
        (folder / "config.json").read_text()
    ) != config:
        raise ValueError("Fine phase configuration differs.")
    save_json(folder / "config.json", config)
    done_path = folder / "phase_complete.json"
    if done_path.exists():
        done = json.loads(done_path.read_text())
        if done["signature"] != signature:
            raise ValueError("Completed phase signature differs.")
        expected = (
            [f"best_{mode}.pt" for mode in config["modes"]]
            if evaluator
            else [f"epoch_{e}.pt" for e in snapshots]
        )
        if any(not (folder / name).exists() for name in expected):
            raise ValueError("Full phase weights missing; do not restore from light ZIP.")
        return done
    seed_everything(config["seed"])
    torch.set_num_threads(2)
    dataset = DiagnosticPatchDataset(records, cache, config, True)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    last = folder / "last.pt"
    model = ResNet34CV(None if last.exists() else weights).to("cuda:0")
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    optimizer = torch.optim.AdamW(
        [
            dict(params=model.backbone.parameters(), lr=5e-5),
            dict(params=[p for p in model.parameters() if id(p) not in backbone_ids], lr=5e-4),
        ],
        weight_decay=1e-4,
    )
    # Keep the60-epoch learning-rate policy identical for inner fitting and outer refit.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 0.05
        + 0.95 * (1 + math.cos(math.pi * step / config["schedule_epochs"])) / 2,
    )
    scaler = torch.amp.GradScaler("cuda")
    history = []
    start = 0
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["signature"] != signature:
            raise ValueError("Optimizer signature differs.")
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
    best = {mode: dict(score=float("inf"), epoch=0) for mode in config["modes"]}
    if evaluator:
        for mode in best:
            if (folder / f"best_{mode}.pt").exists():
                saved = phase_checkpoint(folder / f"best_{mode}.pt", signature)
                best[mode] = dict(saved["selection"])
                del saved
    for epoch in range(start + 1, config["epochs"] + 1):
        begin = time.perf_counter()
        model.train()
        total = count = skipped = 0
        for images, points, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=True):
                logits = model(images.to("cuda:0"))
                if logits.shape != (len(images), 96, 96):
                    raise RuntimeError("Unexpected fine heatmap shape.")
                loss = heatmap_loss(logits, points.to("cuda:0"), 1.5)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite fine loss.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < old_scale)
            total += float(loss.detach()) * len(images)
            count += len(images)
        if skipped == len(loader) and history and history[-1]["amp_skips"] == len(loader):
            raise RuntimeError("Every optimizer step skipped in two consecutive epochs.")
        row = dict(
            epoch=epoch,
            train_ce=total / count,
            lr_backbone=optimizer.param_groups[0]["lr"],
            amp_skips=skipped,
        )
        if evaluator:
            results = evaluator(model)
            for mode, values in results.items():
                row.update({f"{mode}_{k}": v for k, v in values.items()})
                if values["score"] < best[mode]["score"]:
                    best[mode] = dict(values, epoch=epoch)
                    atomic_torch_save(
                        folder / f"best_{mode}.pt",
                        dict(
                            model=model.state_dict(),
                            signature=signature,
                            epoch=epoch,
                            selection=best[mode],
                            config=config,
                        ),
                    )
        elif epoch in snapshots:
            atomic_torch_save(
                folder / f"epoch_{epoch}.pt",
                dict(model=model.state_dict(), signature=signature, epoch=epoch, config=config),
            )
        scheduler.step()
        row["seconds"] = time.perf_counter() - begin
        history.append(row)
        if epoch % 5 == 0 or epoch == config["epochs"]:
            atomic_torch_save(
                last,
                dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    scaler=scaler.state_dict(),
                    signature=signature,
                    epoch=epoch,
                    history=history,
                    python_rng=random.getstate(),
                    numpy_rng=np.random.get_state(),
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(device=0),
                ),
            )
        atomic_csv(folder / "history.csv", history)
        save_json(folder / "status.json", dict(state="training", epoch=epoch, signature=signature))
        scores = " ".join(
            f'{mode}={values["score"]:.2f}' for mode, values in best.items() if values["epoch"]
        )
        print(
            f'Outer{config["outer_fold"]} {config["role"]}/{config["training_arm"]} {epoch:02}/{config["epochs"]}'
            f' CE={total/count:.4f} {scores} {row["seconds"]:.1f}s',
            flush=True,
        )
    done = dict(signature=signature, best=best if evaluator else {}, epochs=config["epochs"])
    save_json(done_path, done)
    if last.exists():
        last.unlink()
    del model, loader, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return done


def validate_inner_centers(rows, records):
    mapping = validate_test_rows(rows, records, -1)
    for r in records:
        p = mapping[int(r["image_id"])]
        if (
            abs(float(p["gt_x"]) - float(r["x"])) > 1e-6
            or abs(float(p["gt_y"]) - float(r["y"])) > 1e-6
        ):
            raise ValueError("Inner center labels differ.")
    return mapping


def make_inner_coarse(data, fit, held, outer_ids, fold, folder, protocol, weights):
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(
        model_name="resnet34",
        seed=31000 + fold,
        epochs=protocol["inner_coarse_epochs"],
        schedule_epochs=80,
        image_size=512,
        batch_size=8,
        lr_backbone=1e-4,
        lr_decoder=1e-3,
        weight_decay=1e-4,
        sigma=2.4,
        weights="resnet34-b627a593.pth",
        pretrained=True,
        train_ids=[int(x) for x in fit.image_id],
        inner_validation_ids=[int(x) for x in held.image_id],
        forbidden_outer_ids=outer_ids,
        source_sha256=protocol["source_sha256"],
        coarse_fingerprint=protocol["coarse_fingerprint"],
    )
    if set(config["train_ids"]) & set(config["inner_validation_ids"]) or (
        set(config["train_ids"]) | set(config["inner_validation_ids"])
    ) & set(outer_ids):
        raise ValueError("Inner coarse image leakage.")
    signature = stable_hash(config)
    done_path = folder / "coarse_complete.json"
    pred_path = folder / "inner_validation.csv"
    if done_path.exists():
        done = json.loads(done_path.read_text())
        if done["signature"] != signature or not (folder / "checkpoints/best.pt").is_file():
            raise ValueError("Completed inner coarse output differs or lacks model.")
        return validate_inner_centers(read_rows(pred_path), held.to_dict("records"))
    last = folder / "checkpoints/last.pt"
    if (folder / "checkpoints/best.pt").exists() and not last.exists():
        raise ValueError(
            "Unfinished inner coarse model lacks its optimizer checkpoint; restore full output."
        )
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        if len(state["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("Resume auxiliary coarse training with the same visible GPU count.")
        del state
    # No held-out images or metrics are passed to the auxiliary coarse trainer.
    model, epoch = train_model(
        data,
        fit,
        None,
        folder / "checkpoints",
        config,
        torch.device("cuda:0"),
        model_builder=lambda cfg: ResNet34CV(weights),
        dataset_builder=FoveaDataset,
    )
    if epoch != protocol["inner_coarse_epochs"]:
        raise ValueError("Inner coarse incomplete.")
    loader = DataLoader(FoveaDataset(data, held, 512), batch_size=8, shuffle=False, num_workers=0)
    points = predict_cv(model, loader, torch.device("cuda:0"), 5, False)
    rows = prediction_rows(held.to_dict("records"), points, -1, True)
    atomic_csv(pred_path, rows)
    save_json(
        done_path,
        dict(
            signature=signature,
            epoch=epoch,
            training_ids=config["train_ids"],
            validation_ids=config["inner_validation_ids"],
            validation_used_for_training=False,
            baseline_metrics=metrics_from_rows(rows),
        ),
    )
    last = folder / "checkpoints/last.pt"
    if last.exists():
        last.unlink()
    del model, loader
    gc.collect()
    torch.cuda.empty_cache()
    return validate_inner_centers(rows, held.to_dict("records"))


def fine_config(
    protocol, fold, arm, role, train_ids, held_ids, forbidden_ids, epochs, targets=None
):
    config = dict(
        version=DIAGNOSTIC_VERSION,
        source_sha256=protocol["source_sha256"],
        coarse_fingerprint=protocol["coarse_fingerprint"],
        outer_fold=fold,
        training_arm=arm,
        role=role,
        seed=(32000 if role == "inner" else 33000) + fold,
        epochs=epochs,
        schedule_epochs=protocol["inner_epochs"],
        image_size=384,
        crop_side=768,
        prior_sigma=protocol["prior_sigma"],
        train_ids=list(train_ids),
        evaluation_ids=list(held_ids),
        forbidden_ids=list(forbidden_ids),
        modes=["plain"] if arm == "wide" else ["plain", "prior"],
        targets=targets or {},
    )
    validate_phase_config(config)
    return config


def identity_rows(records, centers, fold, selection):
    rows = []
    for r in records:
        c = centers[int(r["image_id"])]
        point = (float(c["Fovea_X"]), float(c["Fovea_Y"]))
        box = crop_geometry(point, 768)
        row = make_refinement_row(r, point_to_patch(point, box), c, box, fold, 256.0)
        row.update(decoder="identity", refine_applied=0, selection_reason=selection["reason"])
        rows.append(row)
    return rows


def diagnostic_fold(prepared, fold, weights, cache):
    data, train, test, coarse, output, protocol = [
        prepared[k] for k in ("data", "train", "test", "coarse", "output", "protocol")
    ]
    development, fit, inner_val, outer = inner_partition(train, fold)
    job = output / f"outer_{fold}"
    job.mkdir(parents=True, exist_ok=True)
    outer_ids = [int(x) for x in outer.image_id]
    design = dict(
        protocol=protocol,
        fold=fold,
        fit_ids=[int(x) for x in fit.image_id],
        inner_val_ids=[int(x) for x in inner_val.image_id],
        outer_ids=outer_ids,
    )
    signature = stable_hash(design)
    if (job / "fold_complete.json").exists():
        done = json.loads((job / "fold_complete.json").read_text())
        if done["signature"] != signature:
            raise ValueError("Completed outer fold differs.")
        if any(not (job / name).is_file() for name in done["required_models"]):
            raise ValueError("Restore full output, not light results.")
        return
    split = []
    for role, frame in [
        ("inner_train", fit),
        ("inner_validation", inner_val),
        ("outer_validation", outer),
    ]:
        split.extend(
            dict(image_id=int(r.image_id), pixel_sha256=r.pixel_sha256, role=role)
            for r in frame.itertuples()
        )
    atomic_csv(job / "split_roles.csv", split)
    save_json(job / "design.json", design)
    centers = make_inner_coarse(
        data, fit, inner_val, outer_ids, fold, job / "inner_coarse", protocol, weights
    )
    baseline = metrics_from_rows(list(centers.values()))["rmse_coordinate_px"]
    decisions = {}
    inner_configs = {}
    for arm in ("wide", "small"):
        config = fine_config(
            protocol,
            fold,
            arm,
            "inner",
            [int(x) for x in fit.image_id],
            [int(x) for x in inner_val.image_id],
            outer_ids,
            protocol["inner_epochs"],
        )
        inner_configs[arm] = config
        loader = DataLoader(
            DiagnosticPatchDataset(inner_val.to_dict("records"), cache, config, False, centers),
            batch_size=8,
            shuffle=False,
            num_workers=0,
        )

        def evaluate(model):
            predictions = diagnostic_predict(model, loader, config, centers, -1, config["modes"])
            return {mode: inner_mode_metrics(rows, centers) for mode, rows in predictions.items()}

        done = train_diagnostic_fine(
            data, fit.to_dict("records"), cache, config, job / f"inner_{arm}", weights, evaluate
        )
        for mode, best in done["best"].items():
            name = arm if mode == "plain" else "small_prior"
            decisions[name] = selected_inner_epoch(best, baseline, protocol["min_inner_gain"])
        del loader
    save_json(job / "inner_decisions.json", decisions)
    for arm in ("wide", "small"):
        modes = {"plain": arm} if arm == "wide" else {"plain": "small", "prior": "small_prior"}
        enabled = {
            mode: decisions[name]["epoch"]
            for mode, name in modes.items()
            if decisions[name]["enabled"]
        }
        model = None
        config = None
        if enabled:
            config = fine_config(
                protocol,
                fold,
                arm,
                "outer",
                [int(x) for x in development.image_id],
                [],
                outer_ids,
                max(enabled.values()),
                enabled,
            )
            phase = job / f"outer_{arm}"
            train_diagnostic_fine(
                data,
                development.to_dict("records"),
                cache,
                config,
                phase,
                weights,
                snapshots=sorted(set(enabled.values())),
            )
            model = ResNet34CV().to("cuda:0")
        for mode, name in modes.items():
            decision = decisions[name]
            for subset, frame, prior in [
                ("validation", outer, coarse["validation"][fold]),
                ("test", test, coarse["test"][fold]),
            ]:
                records = frame.to_dict("records")
                if decision["enabled"]:
                    state = phase_checkpoint(
                        job / f'outer_{arm}/epoch_{decision["epoch"]}.pt', stable_hash(config)
                    )
                    model.load_state_dict(state["model"], strict=True)
                    del state
                    loader = DataLoader(
                        DiagnosticPatchDataset(records, cache, config, False, prior),
                        batch_size=8,
                        shuffle=False,
                        num_workers=0,
                    )
                    rows = diagnostic_predict(model, loader, config, prior, fold, [mode])[mode]
                    for row in rows:
                        row["selection_reason"] = decision["reason"]
                    del loader
                else:
                    rows = identity_rows(records, prior, fold, decision)
                atomic_csv(job / f"{subset}_{name}.csv", rows)
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
    required = [p.relative_to(job).as_posix() for p in job.rglob("*.pt")]
    save_json(
        job / "fold_complete.json",
        dict(
            signature=signature,
            decisions=decisions,
            required_models=required,
            outer_labels_used_for_epoch_selection=False,
            predictions_complete=True,
        ),
    )


def diagnostic_reports(prepared):
    data, train, test, coarse, output = [
        prepared[k] for k in ("data", "train", "test", "coarse", "output")
    ]
    available = [f for f in range(5) if (output / f"outer_{f}/fold_complete.json").exists()]
    selection = dict(
        complete=len(available) == 5,
        completed_folds=available,
        candidate_rule="Predeclared small-prior equal blend; per-fold inner5% gain gate",
        selected_by_outer_scores=False,
        official_metric_confirmed=False,
        agent_submitted_to_competition=False,
        limitation="Outer fine labels never choose epochs/gates. Legacy coarse checkpoints already used these outer labels for stopping; comparison is conditional development evaluation, not unbiased validation of the entire cascade.",
    )
    selection["smoke_test"] = bool(prepared["protocol"]["smoke_test"])
    if len(available) != 5 or selection["smoke_test"]:
        selection["complete"] = False
        save_json(output / "selection.json", selection)
        return False
    records, test_records = train.to_dict("records"), test.to_dict("records")
    coarse_oof = validate_oof(
        [r for f in range(5) for r in coarse["validation"][f].values()], records
    )
    atomic_csv(output / "oof_coarse512.csv", coarse_oof)
    baseline_test = average_test_folds(
        {f: list(coarse["test"][f].values()) for f in range(5)}, test_records
    )
    atomic_csv(output / "test_coarse512.csv", baseline_test)
    write_submission(
        data / "sample_submission.csv", baseline_test, output / "submission_coarse512.csv"
    )
    comparisons = [dict(arm="coarse512", **metrics_from_rows(coarse_oof))]
    decision_rows = []
    for variant in VARIANTS:
        refined = []
        test_folds = {}
        active = 0
        for f in range(5):
            job = output / f"outer_{f}"
            done = json.loads((job / "fold_complete.json").read_text())
            development, fit, inner_val, outer = inner_partition(train, f)
            design = dict(
                protocol=prepared["protocol"],
                fold=f,
                fit_ids=[int(x) for x in fit.image_id],
                inner_val_ids=[int(x) for x in inner_val.image_id],
                outer_ids=[int(x) for x in outer.image_id],
            )
            if (
                done["signature"] != stable_hash(design)
                or not done["predictions_complete"]
                or done["outer_labels_used_for_epoch_selection"]
            ):
                raise ValueError("Outer completion provenance differs.")
            decisions = done["decisions"][variant]
            recomputed = selected_inner_epoch(
                dict(score=decisions["inner_score"], epoch=decisions["inner_best_epoch"]),
                decisions["inner_baseline"],
                prepared["protocol"]["min_inner_gain"],
            )
            if decisions != recomputed:
                raise ValueError("Inner gate/epoch rule differs from predeclared selection.")
            decision_rows.append(dict(variant=variant, outer_fold=f, **decisions))
            active += int(decisions["enabled"])
            held = train[train.fold == f].to_dict("records")
            rows = validate_oof(read_rows(job / f"validation_{variant}.csv"), held)
            validate_test_rows(rows, held, f)
            refined.extend(rows)
            test_folds[f] = read_rows(job / f"test_{variant}.csv")
            validate_test_rows(test_folds[f], test_records, f)
        refined = validate_oof(refined, records)
        raw = raw_prediction_rows(refined)
        for suffix, rows, preds in [
            ("refined", refined, test_folds),
            (
                "equal",
                blend_prediction_rows(coarse_oof, refined, True),
                {
                    f: blend_prediction_rows(list(coarse["test"][f].values()), test_folds[f], False)
                    for f in range(5)
                },
            ),
        ]:
            name = f"{variant}_{suffix}"
            atomic_csv(output / f"oof_{name}.csv", rows)
            mean = average_test_folds(preds, test_records)
            atomic_csv(output / f"test_{name}.csv", mean)
            write_submission(
                data / "sample_submission.csv", mean, output / f"submission_{name}.csv"
            )
            comparisons.append(dict(arm=name, active_folds=active, **metrics_from_rows(rows)))
        atomic_csv(output / f"oof_{variant}_raw.csv", raw)
        comparisons.append(
            dict(arm=f"{variant}_raw_diagnostic", active_folds=active, **metrics_from_rows(raw))
        )
    atomic_csv(
        output / "comparison.csv",
        comparisons,
        fields=[
            "arm",
            "active_folds",
            "n",
            "mean_distance_px",
            "median_distance_px",
            "p90_distance_px",
            "max_distance_px",
            "mae_x_px",
            "mae_y_px",
            "rmse_coordinate_px",
        ],
    )
    atomic_csv(output / "inner_decisions.csv", decision_rows)
    candidate = output / "submission_small_prior_equal.csv"
    shutil.copy2(candidate, output / "submission_refine_v3_candidate.csv")
    selection.update(
        candidate="submission_refine_v3_candidate.csv",
        public_coarse_reference=14.16538,
        public_best_reference=12.31624,
        candidate_public_score=None,
    )
    save_json(output / "selection.json", selection)
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)
    return True


def diagnostic_archive(output, full=False):
    target = (
        output.parent / "task3_refine_v3_full_backup.zip"
        if full
        else output / "task3_refine_v3_results_light.zip"
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


def prepare_diagnostic(
    coarse_source,
    data_dir=None,
    output_dir="/kaggle/working/task3_refine_diagnostic_v3",
    resume_dir=None,
    smoke=False,
):
    if not Path("/kaggle/working").is_dir() or not torch.cuda.is_available():
        raise RuntimeError("Use Kaggle GPU; no local/CPU training was started.")
    data, output = find_data(data_dir), Path(output_dir).resolve()
    manifest = load_manifest(data)
    train = assign_folds(manifest[manifest.subset == "train"], 42, 5)
    test = manifest[manifest.subset == "test"].copy()
    if len(train) != 80 or len(test) != 20:
        raise ValueError("Unexpected image counts.")
    coarse = validate_coarse_bundle(
        coarse_source, train.to_dict("records"), test.to_dict("records")
    )
    protocol = dict(
        version=DIAGNOSTIC_VERSION,
        source_sha256=diagnostic_sources(),
        coarse_fingerprint=coarse["fingerprint"],
        inner_coarse_epochs=2 if smoke else 43,
        inner_epochs=2 if smoke else 60,
        prior_sigma=48.0,
        min_inner_gain=0.05,
        inner_split_seed_base=7000,
        training_arms=["wide", "small"],
        candidate_variant="small_prior_equal",
        outer_epochs_selected_by_inner=True,
        legacy_coarse_outer_selection_bias_remains=True,
        smoke_test=bool(smoke),
    )
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 9 * 1024**3:
        raise RuntimeError("Need9 GiB free for nested phases, atomic saves and full backup.")
    restore_backup(resume_dir, output, protocol)
    if (output / "protocol.json").exists() and json.loads(
        (output / "protocol.json").read_text()
    ) != protocol:
        raise ValueError("Existing diagnostic protocol differs; use a new output directory.")
    save_json(output / "protocol.json", protocol)
    train.to_csv(output / "split.csv", index=False)
    manifest.to_csv(output / "manifest.csv", index=False)
    save_json(
        output / "coarse_source.json",
        dict(fingerprint=coarse["fingerprint"], legacy_protocol=coarse["protocol"]),
    )
    save_json(
        output / "environment.json",
        dict(
            torch=torch.__version__,
            numpy=np.__version__,
            gpu=torch.cuda.get_device_name(0),
            visible_gpu_count=torch.cuda.device_count(),
            device_policy="cuda:0 sequential phases; original image RGB cache up to about2 GiB",
        ),
    )
    print("Fixed coarse512 OOF:", coarse["baseline_metrics"], flush=True)
    print(
        "Auxiliary inner coarse + wide/small inner fits + at most2 outer refits per fold; up to25 training phases.",
        flush=True,
    )
    return dict(data=data, output=output, train=train, test=test, coarse=coarse, protocol=protocol)


def run_diagnostic(prepared, run_folds=(0, 1, 2, 3, 4), weights=None, full_backup=True):
    if not Path("/kaggle/working").is_dir() or not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU required.")
    if (
        not run_folds
        or len(set(run_folds)) != len(run_folds)
        or not set(run_folds) <= set(range(5))
    ):
        raise ValueError("Invalid outer fold schedule.")
    output = prepared["output"]
    cache = OriginalImageCache(prepared["data"])
    pending = any(not (output / f"outer_{f}/fold_complete.json").exists() for f in run_folds)
    weights = resolve_weights(output, weights) if pending else None
    with training_lock(output):
        try:
            for fold in run_folds:
                save_json(
                    output / "status.json",
                    dict(state="running", outer_fold=fold, updated_unix=time.time()),
                )
                diagnostic_fold(prepared, fold, weights, cache)
                diagnostic_reports(prepared)
                diagnostic_archive(output)
            complete = diagnostic_reports(prepared)
            save_json(
                output / "status.json",
                dict(
                    state="completed" if complete else "partial_completed", updated_unix=time.time()
                ),
            )
            diagnostic_archive(output)
            if complete and full_backup:
                backup = diagnostic_archive(output, True)
                with zipfile.ZipFile(backup) as archive:
                    for f in range(5):
                        done = json.loads((output / f"outer_{f}/fold_complete.json").read_text())
                        if any(
                            f"outer_{f}/{name}" not in archive.namelist()
                            for name in done["required_models"]
                        ):
                            raise RuntimeError("Full backup lacks a required model.")
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
            diagnostic_archive(output)
            raise
    return output
