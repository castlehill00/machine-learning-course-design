"""Fixed-epoch five-fold ResNet34 U-Net; native pixels, EMA, paired D4 evaluation.

The notebook embeds this file AND task4_pipeline.py, so no extra source upload is needed.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import socket
import time
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet34_Weights, resnet34

import task4_pipeline as base


@dataclass
class ConfigV2:
    seed: int = 42
    folds: int = 5
    fold: int = 0  # audit helper only; run_cv uses all requested folds
    epochs: int = 80
    patch: int = 256
    patches_per_epoch: int = 256
    batch_size: int = 8
    encoder_lr: float = 0.0001
    decoder_lr: float = 0.0007
    weight_decay: float = 0.0001
    warmup_epochs: int = 3
    ema_decay: float = 0.99
    monitor_every: int = 10
    threads: int = 4
    amp: bool = True
    pretrained: bool = True
    tta_gain_required: float = 0.001


def audit_v2(root, output, cfg):
    """Freeze the original ID folds; JPEG decoder changes must not reshuffle folds."""
    train, test, labels, _, ids, summary = base.audit(root, output, cfg)
    frozen = [
        ["2", "6", "9", "10"],
        ["3", "14", "17", "20"],
        ["1", "4", "5", "11"],
        ["15", "16", "18", "19"],
        ["7", "8", "12", "13"],
    ]
    assignment = {i: fold for fold, group in enumerate(frozen) for i in group}
    if set(assignment) != {p.stem for p in train}:
        raise ValueError("This frozen task-4 split expects train IDs 1 through 20.")
    manifest = output / "manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    groups = {}
    for row in rows:
        if row["split"] == "train":
            row["fold"] = assignment[row["id"]]
            groups.setdefault(row["pixel_sha256"], set()).add(row["fold"])
    if any(len(folds) > 1 for folds in groups.values()):
        raise ValueError(
            "A decoded duplicate crosses frozen folds; inspect dataset before training."
        )
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    # Raw file hashes are stable across JPEG decoder/library versions, unlike pixel hashes.
    stable = [
        {
            k: row[k]
            for k in ["split", "id", "width", "height", "image_sha256", "label_sha256", "fold"]
        }
        for row in rows
    ]
    summary.update(
        data_fingerprint=hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest(),
        validation_ids=frozen[0],
        folds=[4] * 5,
        split_protocol="frozen original seed42 ID folds; training seed may vary",
    )
    base.save_json(output / "audit_summary.json", summary)
    return train, test, labels, assignment, ids, summary


def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB")).copy()


def read_mask(path):
    return (np.asarray(Image.open(path).convert("L")) > 127).astype(np.float32)


def resize_probability(prob, shape=(512, 512)):
    # PIL receives a float image: interpolate probabilities, then threshold once.
    h, w = shape
    return np.clip(
        np.asarray(
            Image.fromarray(np.asarray(prob, np.float32)).resize((w, h), Image.Resampling.BILINEAR)
        ),
        0,
        1,
    ).copy()


class NativePatches(Dataset):
    def __init__(self, paths, labels, cfg):
        self.images = [read_rgb(p) for p in paths]
        self.masks = [read_mask(labels[p.stem]) for p in paths]
        self.positives = [np.argwhere(m > 0) for m in self.masks]
        self.cfg = cfg

    def __len__(self):
        return self.cfg.patches_per_epoch

    def __getitem__(self, index):
        i = random.randrange(len(self.images))
        image, mask = self.images[i], self.masks[i]
        h, w = mask.shape
        patch = self.cfg.patch
        crop = random.randint(int(patch * 0.9), min(int(patch * 1.1), h, w))
        if random.random() < 0.3 and len(self.positives[i]):
            y, x = self.positives[i][random.randrange(len(self.positives[i]))]
            top, left = int(np.clip(y - crop // 2, 0, h - crop)), int(
                np.clip(x - crop // 2, 0, w - crop)
            )
        else:
            top, left = random.randint(0, h - crop), random.randint(0, w - crop)
        image = cv2.resize(
            image[top : top + crop, left : left + crop],
            (patch, patch),
            interpolation=cv2.INTER_LINEAR,
        )
        mask = cv2.resize(
            mask[top : top + crop, left : left + crop],
            (patch, patch),
            interpolation=cv2.INTER_NEAREST,
        )
        image, mask = base.paired_d4(image, mask, random.randrange(4), random.random() < 0.5)
        image = image.astype(np.float32) / 255
        image = np.clip(
            (image - 0.5) * random.uniform(0.85, 1.15) + 0.5 + random.uniform(-0.04, 0.04), 0, 1
        )
        image = image ** random.uniform(0.85, 1.15)
        image = np.clip(
            image * np.array([random.uniform(0.95, 1.05) for _ in range(3)], np.float32), 0, 1
        )
        return torch.from_numpy(image.transpose(2, 0, 1).copy()), torch.from_numpy(
            mask[None].copy()
        )


class ResNet34UNet(nn.Module):
    def __init__(self, initial_state=None):
        super().__init__()
        self.encoder = resnet34(weights=None)
        if initial_state is not None:
            self.encoder.load_state_dict(initial_state, strict=True)
        self.encoder.fc = nn.Identity()
        self.shallow = base.ConvBlock(3, 32)
        self.up4 = base.ConvBlock(512 + 256, 256)
        self.up3 = base.ConvBlock(256 + 128, 128)
        self.up2 = base.ConvBlock(128 + 64, 64)
        self.up1 = base.ConvBlock(64 + 64, 48)
        self.up0 = base.ConvBlock(48 + 32, 32)
        self.head = nn.Conv2d(32, 1, 1)
        self.aux2 = nn.Conv2d(48, 1, 1)
        self.aux4 = nn.Conv2d(64, 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        # Small batches: keep ImageNet running statistics; affine parameters still train.
        for module in self.encoder.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    @staticmethod
    def join(x, skip, block):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return block(torch.cat([x, skip], 1))

    def forward(self, image):
        x = (image - self.mean) / self.std
        s0 = self.shallow(x)
        s1 = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))
        s2 = self.encoder.layer1(self.encoder.maxpool(s1))
        s3 = self.encoder.layer2(s2)
        s4 = self.encoder.layer3(s3)
        z = self.encoder.layer4(s4)
        z = self.join(z, s4, self.up4)
        z = self.join(z, s3, self.up3)
        z4 = self.join(z, s2, self.up2)
        z2 = self.join(z4, s1, self.up1)
        z = self.join(z2, s0, self.up0)
        return self.head(z), self.aux2(z2), self.aux4(z4)


def objective(outputs, target):
    losses = []
    for logits in outputs:
        truth = (
            target
            if logits.shape[-2:] == target.shape[-2:]
            else F.interpolate(target, size=logits.shape[-2:], mode="area")
        )
        losses.append(base.segmentation_loss(logits, truth))
    return (losses[0] + 0.1 * losses[1] + 0.2 * losses[2]) / 1.3


@torch.no_grad()
def update_ema(ema, model, decay):
    for target, source in zip(ema.parameters(), model.parameters()):
        target.lerp_(source, 1 - decay)
    for target, source in zip(ema.buffers(), model.buffers()):
        target.copy_(source)


def transform_d4(x, rotation, flip):
    x = torch.rot90(x, rotation, (-2, -1))
    return torch.flip(x, (-1,)) if flip else x


def inverse_d4(x, rotation, flip):
    if flip:
        x = torch.flip(x, (-1,))
    return torch.rot90(x, -rotation, (-2, -1))


def window_starts(length, patch, stride):
    if length < patch:
        raise ValueError("Pad before calculating window starts")
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


@torch.inference_mode()
def predict_pair(model, image, cfg, device, with_tta=True):
    """Patchwise D4; return native probabilities for none and D4 using identical windows."""
    model.eval()
    if isinstance(image, (str, Path)):
        image = read_rgb(image)
    h, w = image.shape[:2]
    pad_h, pad_w = max(0, cfg.patch - h), max(0, cfg.patch - w)
    image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    height, width = image.shape[:2]
    coords = [
        (y, x)
        for y in window_starts(height, cfg.patch, cfg.patch // 2)
        for x in window_starts(width, cfg.patch, cfg.patch // 2)
    ]
    weight = np.maximum(np.outer(np.hanning(cfg.patch), np.hanning(cfg.patch)), 0.05).astype(
        np.float32
    )
    accum = np.zeros((2, height, width), np.float32)
    weights = np.zeros((height, width), np.float32)
    transforms = [(0, False)] + (
        [(r, f) for r in range(4) for f in [False, True] if (r, f) != (0, False)]
        if with_tta
        else []
    )
    for start in range(0, len(coords), cfg.batch_size):
        positions = coords[start : start + cfg.batch_size]
        batch = np.stack(
            [image[y : y + cfg.patch, x : x + cfg.patch].transpose(2, 0, 1) for y, x in positions]
        )
        tensor = torch.from_numpy(batch.copy()).to(device).float() / 255
        plain, total = None, None
        for rotation, flip in transforms:
            transformed = transform_d4(tensor, rotation, flip)
            with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
                prediction = model(transformed)[0].float().sigmoid()
            prediction = inverse_d4(prediction, rotation, flip)
            if plain is None:
                plain = prediction
                total = prediction.clone()
            else:
                total.add_(prediction)
        pair = torch.stack([plain, total / len(transforms)])[:, :, 0].cpu().numpy()
        for i, (y, x) in enumerate(positions):
            accum[:, y : y + cfg.patch, x : x + cfg.patch] += pair[:, i] * weight
            weights[y : y + cfg.patch, x : x + cfg.patch] += weight
    result = accum / weights[None]
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite inference output")
    return {"none": result[0, :h, :w].copy(), "d4": result[1, :h, :w].copy()}


def resolve_weights(cfg, output, weights_path=None):
    if not cfg.pretrained:
        return None, dict(source="random initialization; debug/ablation only", sha256=None)
    enum = ResNet34_Weights.IMAGENET1K_V1
    target = (
        Path(weights_path) if weights_path else output / "pretrained_cache" / Path(enum.url).name
    )
    if not target.is_file():
        if weights_path:
            raise FileNotFoundError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading official ImageNet encoder weights:", enum.url, flush=True)
        torch.hub.download_url_to_file(enum.url, str(target), hash_prefix="b627a593", progress=True)
    digest = base.file_sha(target)
    if not digest.startswith("b627a593"):
        raise ValueError(
            "Expected official torchvision resnet34-b627a593.pth, not task-finetuned weights."
        )
    state = torch.load(target, map_location="cpu", weights_only=True)
    probe = resnet34(weights=None)
    probe.load_state_dict(state, strict=True)
    del probe
    cached = output / "pretrained_cache" / Path(enum.url).name
    if target.resolve() != cached.resolve():
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(target, cached)
    return state, dict(
        source=enum.url,
        sha256=digest,
        supervision="generic ImageNet classification, not vessel segmentation",
    )


def set_lrs(optimizer, step, steps_per_epoch, cfg):
    total = cfg.epochs * steps_per_epoch
    warm = min(cfg.warmup_epochs * steps_per_epoch, max(0, total - 1))
    if warm and step < warm:
        factor = (step + 1) / warm
    else:
        progress = (step - warm) / max(1, total - warm - 1)
        factor = 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1, progress)))
    for group, initial in zip(optimizer.param_groups, [cfg.encoder_lr, cfg.decoder_lr]):
        group["lr"] = initial * factor


def train_fold(paths, val_paths, labels, cfg, fold, folder, initial_state, device):
    folder.mkdir(parents=True, exist_ok=True)
    base.seed_all(cfg.seed + 1009 * fold)
    dataset = NativePatches(paths, labels, cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    model = ResNet34UNet(initial_state).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    encoder = list(model.encoder.parameters())
    decoder = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [{"params": encoder, "lr": cfg.encoder_lr}, {"params": decoder, "lr": cfg.decoder_lr}],
        weight_decay=cfg.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    history, first, step = [], 1, 0
    train_ids, val_ids = [p.stem for p in paths], [p.stem for p in val_paths]
    last = folder / "last.pt"
    if last.exists():
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        if ckpt["train_ids"] != train_ids or ckpt["val_ids"] != val_ids:
            raise ValueError("Checkpoint split mismatch")
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        history, first, step = ckpt["history"], ckpt["epoch"] + 1, ckpt["step"]
        base.restore_rng(ckpt["rng"])
        del ckpt
        print(f"fold{fold}: resume at {first}/{cfg.epochs}", flush=True)
    for epoch in range(first, cfg.epochs + 1):
        begin = time.perf_counter()
        model.train()
        losses = []
        for image, target in loader:
            set_lrs(optimizer, step, len(loader), cfg)
            image, target = image.to(device, non_blocking=True), target.to(
                device, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
                loss = objective(model(image), target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                update_ema(ema, model, cfg.ema_decay)
            losses.append(float(loss.detach()))
            step += 1
        score = None
        if epoch % cfg.monitor_every == 0 or epoch == cfg.epochs:
            score = float(
                np.mean(
                    [
                        base.dice_score(
                            resize_probability(predict_pair(ema, p, cfg, device, False)["none"])
                            >= 0.5,
                            base.strict_label(labels[p.stem]),
                        )
                        for p in val_paths
                    ]
                )
            )
        row = dict(
            epoch=epoch,
            loss=float(np.mean(losses)),
            monitor_dice_05=score,
            encoder_lr=optimizer.param_groups[0]["lr"],
            decoder_lr=optimizer.param_groups[1]["lr"],
            seconds=round(time.perf_counter() - begin, 3),
        )
        history.append(row)
        if shutil.disk_usage(folder).free < 1200 * 1024**2:
            raise OSError(
                "Less than 1.2 GiB free: preserve old checkpoint and free space before resuming."
            )
        base.atomic_checkpoint(
            last,
            dict(
                model=model.state_dict(),
                ema=ema.state_dict(),
                optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(),
                history=history,
                epoch=epoch,
                step=step,
                rng=base.rng_state(),
                train_ids=train_ids,
                val_ids=val_ids,
            ),
        )
        with (folder / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, list(row))
            writer.writeheader()
            writer.writerows(history)
        print(
            f'fold{fold} {epoch:03d}/{cfg.epochs} loss={row["loss"]:.4f} '
            + (f"monitor_dice@0.5={score:.5f} " if score is not None else "")
            + f'seconds={row["seconds"]}',
            flush=True,
        )
        if epoch == 1:
            print(
                f"fold{fold}: measured estimate for this fold training/monitoring: "
                f'{row["seconds"]*cfg.epochs/60:.1f} min (startup may inflate this; D4 export extra)',
                flush=True,
            )
    # Final EMA is fixed in advance. Validation never selects epochs/checkpoints.
    base.atomic_checkpoint(
        folder / "final_ema.pt",
        dict(
            model=ema.state_dict(),
            epoch=cfg.epochs,
            train_ids=train_ids,
            val_ids=val_ids,
            selection="predeclared final EMA",
        ),
    )
    del model, optimizer, scaler, dataset, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ema


def infer_fold(model, val_paths, test, labels, cfg, folder, device):
    val_probs, test_probs = {}, {}
    for p in val_paths:
        pair = predict_pair(model, p, cfg, device)
        for mode, native in pair.items():
            val_probs[f"{mode}_{p.stem}"] = resize_probability(native)
            val_probs[f"native_{mode}_{p.stem}"] = native
    np.savez_compressed(folder / "validation_probabilities.npz", **val_probs)
    for p in test:
        pair = predict_pair(model, p, cfg, device)
        for mode, native in pair.items():
            test_probs[f"{mode}_{p.stem}"] = resize_probability(native)
    np.savez_compressed(folder / "test_probabilities.npz", **test_probs)
    clean = {p.stem: base.strict_label(labels[p.stem]) for p in val_paths}
    summary = dict(
        epoch=cfg.epochs,
        val_ids=[p.stem for p in val_paths],
        none_dice_05=float(
            np.mean([base.dice_score(val_probs[f"none_{i}"] >= 0.5, clean[i]) for i in clean])
        ),
        d4_dice_05=float(
            np.mean([base.dice_score(val_probs[f"d4_{i}"] >= 0.5, clean[i]) for i in clean])
        ),
    )
    files = [
        "history.csv",
        "last.pt",
        "final_ema.pt",
        "validation_probabilities.npz",
        "test_probabilities.npz",
    ]
    summary["sha256"] = {name: base.file_sha(folder / name) for name in files}
    base.save_json(folder / "completed.json", summary)
    print(
        "Fold completed:",
        summary["val_ids"],
        summary["none_dice_05"],
        summary["d4_dice_05"],
        flush=True,
    )


def threshold_sweep(probs, targets):
    thresholds = np.round(np.arange(0.1, 0.901, 0.025), 3)
    rows = [
        dict(
            threshold=float(t),
            mean_dice=float(np.mean([base.dice_score(probs[i] >= t, targets[i]) for i in probs])),
        )
        for t in thresholds
    ]
    best = max(rows, key=lambda r: (r["mean_dice"], -abs(r["threshold"] - 0.5)))
    return best, rows


def export_csv(probs, ids, threshold, name, root, output):
    folder = output / "masks_512_white_vessels" / name
    folder.mkdir(parents=True, exist_ok=True)
    filenames, encoded = [], []
    for i in ids:
        mask = probs[i] >= threshold
        path = folder / f"{i}.png"
        Image.fromarray((mask * 255).astype(np.uint8)).save(path)
        value = base.rle_encode(mask)
        if not np.array_equal(base.rle_decode(value), mask):
            raise AssertionError("RLE mismatch")
        filenames.append(str(path))
        encoded.append(value)
    official = base.load_converter(root).turn_to_str(filenames)
    if official != encoded:
        raise AssertionError("Official converter mismatch")
    path = output / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Predicted"])
        writer.writerows(zip(ids, official))
    return dict(
        file=path.name,
        threshold=threshold,
        sha256=base.file_sha(path),
        rows=len(ids),
        official_converter_exact_match=True,
        rle_roundtrip=True,
    )


def aggregate(root, output, cfg, train, test, labels, assignment, ids):
    folds = [output / f"fold{f}" for f in range(cfg.folds)]
    if not all((folder / "completed.json").exists() for folder in folds):
        finished = [f for f, folder in enumerate(folds) if (folder / "completed.json").exists()]
        base.save_json(
            output / "partial_cv_summary.json", dict(completed_folds=finished, complete_oof=False)
        )
        print(
            "Partial CV only; no full-OOF score or ensemble submission. Completed:",
            finished,
            flush=True,
        )
        return None
    oof = {mode: {} for mode in ["none", "d4"]}
    native = {mode: {} for mode in ["none", "d4"]}
    test_mean = {mode: {i: np.zeros((512, 512), np.float64) for i in ids} for mode in oof}
    for fold, folder in enumerate(folds):
        metadata = json.loads((folder / "completed.json").read_text(encoding="utf-8"))
        for name, digest in metadata["sha256"].items():
            if base.file_sha(folder / name) != digest:
                raise ValueError(f"Artifact changed: {folder/name}")
        expected = {p.stem for p in train if assignment[p.stem] == fold}
        if set(metadata["val_ids"]) != expected:
            raise ValueError("Validation IDs mismatch")
        with np.load(folder / "validation_probabilities.npz") as values:
            for mode in oof:
                for i in expected:
                    if i in oof[mode]:
                        raise AssertionError("Repeated OOF ID")
                    oof[mode][i] = values[f"{mode}_{i}"].copy()
                    native[mode][i] = values[f"native_{mode}_{i}"].copy()
        with np.load(folder / "test_probabilities.npz") as values:
            for mode in oof:
                for i in ids:
                    test_mean[mode][i] += values[f"{mode}_{i}"] / cfg.folds
    train_ids = {p.stem for p in train}
    if any(set(probs) != train_ids for probs in oof.values()):
        raise AssertionError("Incomplete OOF coverage")
    clean = {i: base.strict_label(labels[i]) for i in train_ids}
    proxy = {i: base.converter_label(labels[i]) for i in train_ids}
    rows, mode_summary, exported = [], {}, {}
    for mode in oof:
        calibrated, sweep = threshold_sweep(oof[mode], clean)
        proxy_calibrated, proxy_sweep = threshold_sweep(oof[mode], proxy)
        threshold = calibrated["threshold"]
        cross_thresholds = {}
        for fold in range(cfg.folds):
            other = {j: oof[mode][j] for j in train_ids if assignment[j] != fold}
            cross_thresholds[fold] = threshold_sweep(other, {j: clean[j] for j in other})[0][
                "threshold"
            ]
        with (output / f"threshold_sweep_{mode}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, ["threshold", "clean_dice", "converter_proxy_dice"])
            writer.writeheader()
            writer.writerows(
                dict(
                    threshold=a["threshold"],
                    clean_dice=a["mean_dice"],
                    converter_proxy_dice=b["mean_dice"],
                )
                for a, b in zip(sweep, proxy_sweep)
            )
        per = []
        for i in sorted(train_ids, key=int):
            # Threshold calibration excluding this outer fold: diagnostic, not nested CV.
            cross_t = cross_thresholds[assignment[i]]
            gt_native = read_mask(labels[i]) > 0
            row = dict(
                id=i,
                fold=assignment[i],
                mode=mode,
                dice_05=base.dice_score(oof[mode][i] >= 0.5, clean[i]),
                dice_global_threshold=base.dice_score(oof[mode][i] >= threshold, clean[i]),
                dice_other_fold_threshold=base.dice_score(oof[mode][i] >= cross_t, clean[i]),
                other_fold_threshold=cross_t,
                native_dice_05=base.dice_score(native[mode][i] >= 0.5, gt_native),
                converter_proxy_dice=base.dice_score(oof[mode][i] >= threshold, proxy[i]),
            )
            rows.append(row)
            per.append(row)
        mode_summary[mode] = dict(
            mean_dice_05=float(np.mean([r["dice_05"] for r in per])),
            global_threshold=threshold,
            mean_dice_global_threshold=calibrated["mean_dice"],
            mean_dice_other_fold_threshold=float(
                np.mean([r["dice_other_fold_threshold"] for r in per])
            ),
            mean_native_dice_05=float(np.mean([r["native_dice_05"] for r in per])),
            mean_converter_proxy_dice=float(np.mean([r["converter_proxy_dice"] for r in per])),
            converter_proxy_best_threshold=proxy_calibrated["threshold"],
            converter_proxy_best_dice=proxy_calibrated["mean_dice"],
        )
        exported[mode] = export_csv(
            test_mean[mode], ids, threshold, f"submission_{mode}", root, output
        )
    chosen = (
        "d4"
        if mode_summary["d4"]["mean_dice_05"]
        >= mode_summary["none"]["mean_dice_05"] + cfg.tta_gain_required
        else "none"
    )
    source = output / exported[chosen]["file"]
    shutil.copyfile(source, output / "submission.csv")
    # Explicitly named diagnostic candidate; never silently substitute it for clean vessels.
    proxy_file = export_csv(
        test_mean[chosen],
        ids,
        mode_summary[chosen]["converter_proxy_best_threshold"],
        "submission_converter_proxy",
        root,
        output,
    )
    threshold = mode_summary[chosen]["global_threshold"]
    base.preview(
        test,
        {p.stem: test_mean[chosen][p.stem] for p in test},
        None,
        threshold,
        output / "test_preview.png",
    )
    fold0 = [p for p in train if assignment[p.stem] == 0]
    base.preview(fold0, oof[chosen], labels, threshold, output / "oof_fold0_preview.png")
    black = output / "masks_native_black_vessels"
    black.mkdir(exist_ok=True)
    for p in test:
        width, height = Image.open(p).size
        prob = resize_probability(test_mean[chosen][p.stem], (height, width))
        Image.fromarray(np.where(prob >= threshold, 0, 255).astype(np.uint8)).save(
            black / f"{p.stem}.png"
        )
    with (output / "oof_per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output / "oof_probabilities.npz",
        **{f"{mode}_{i}": p for mode, probs in oof.items() for i, p in probs.items()},
    )
    np.savez_compressed(
        output / "test_ensemble_probabilities.npz",
        **{
            f"{mode}_{i}": p.astype(np.float32)
            for mode, probs in test_mean.items()
            for i, p in probs.items()
        },
    )
    summary = dict(
        complete_oof=True,
        unique_oof_images=len(train_ids),
        epochs=cfg.epochs,
        checkpoint_selection="fixed final EMA; no outer-validation epoch selection",
        modes=mode_summary,
        selected_mode=chosen,
        tta_gain_required=cfg.tta_gain_required,
        test_ensemble="equal average of five fold models; each trained on 16 images",
        baseline_reference=dict(
            ids=["2", "6", "9", "10"],
            baseline_dice_05=0.7839949354578135,
            v2_same_ids_none_dice_05=float(
                np.mean(
                    [
                        r["dice_05"]
                        for r in rows
                        if r["id"] in {"2", "6", "9", "10"} and r["mode"] == "none"
                    ]
                )
            ),
            note="Matched subset only; baseline did not produce full five-fold OOF.",
        ),
        selection_caveat="TTA mode and global threshold use OOF labels. Tuned scores are development estimates. "
        "Other-fold calibration is diagnostic, not a nested independent test.",
        private_target_preprocessing_verified=False,
        submitted_to_competition=False,
    )
    base.save_json(output / "cv_summary.json", summary)
    base.save_json(
        output / "submission_provenance.json",
        dict(
            chosen_mode=chosen,
            primary_csv_sha256=base.file_sha(output / "submission.csv"),
            alternatives=exported,
            converter_proxy_candidate=proxy_file,
            fold_models=[base.file_sha(p / "final_ema.pt") for p in folds],
            calibrated_on="20 OOF clean masks; proxy candidate separately calibrated on JPEG converter targets",
            image_polarity="CSV white vessels; separate native-size PNGs have vessel=0, background=255",
            official_converter_exact_match=True,
            external_vessel_training_data=False,
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def make_backups(output, full=True):
    """Portable analysis archive; optional full archive includes model and resume states."""
    output = Path(output)
    files = [
        p
        for p in output.rglob("*")
        if p.is_file() and p.suffix not in {".zip", ".tmp"} and p.name != "training.lock"
    ]
    light = [
        p
        for p in files
        if p.suffix in {".json", ".csv", ".png", ".py", ".md"}
        and "pretrained_cache" not in p.parts
        and not (p.suffix == ".png" and "masks_512_white_vessels" in p.parts)
    ]
    for name, paths in [("analysis_results.zip", light)] + (
        [("full_backup.zip", files)] if full else []
    ):
        target = output / name
        tmp = output / (name + ".tmp")
        estimated = sum(p.stat().st_size for p in paths)
        if shutil.disk_usage(output).free < estimated + 256 * 1024**2:
            raise OSError(
                f"Not enough free space to create {name}; completed results remain intact."
            )
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            for path in paths:
                z.write(path, path.relative_to(output).as_posix())
        with zipfile.ZipFile(tmp) as z:
            bad = z.testzip()
            if bad:
                raise IOError(f"Archive CRC failed: {bad}")
        os.replace(tmp, target)
        print("Backup saved:", target, f"{target.stat().st_size/1024**2:.1f} MiB", flush=True)


def restore_output(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        return
    if output.exists() and any(output.iterdir()):
        # Existing working output is the authoritative continuation; do not overlay backup.
        if (output / "run_config.json").exists():
            return
        raise ValueError("Restore target is not empty")
    output.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(
            source,
            output,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("*.zip", "*.tmp", "training.lock"),
        )
    elif source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as z:
            for item in z.infolist():
                resolved = (output / item.filename).resolve()
                if not resolved.is_relative_to(output):
                    raise ValueError("Unsafe archive path")
            z.extractall(output)
        (output / "training.lock").unlink(missing_ok=True)
    else:
        raise ValueError("RESUME_FROM must be an existing output folder or full_backup.zip")


@contextmanager
def run_lock(output):
    lock = output / "training.lock"
    if lock.exists():
        info = json.loads(lock.read_text())
        alive = False
        if info.get("host") == socket.gethostname():
            if os.name == "nt":
                import ctypes
                from ctypes import wintypes

                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
                kernel.OpenProcess.restype = wintypes.HANDLE
                kernel.GetExitCodeProcess.argtypes = [
                    wintypes.HANDLE,
                    ctypes.POINTER(wintypes.DWORD),
                ]
                kernel.CloseHandle.argtypes = [wintypes.HANDLE]
                handle = kernel.OpenProcess(0x1000, False, int(info["pid"]))
                if handle:
                    code = wintypes.DWORD()
                    alive = (
                        bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code)))
                        and code.value == 259
                    )
                    kernel.CloseHandle(handle)
            else:
                try:
                    os.kill(info["pid"], 0)
                    alive = True
                except OSError:
                    pass
        if alive:
            raise RuntimeError(f'Output is in use by PID {info["pid"]}')
        lock.unlink()
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, json.dumps(dict(pid=os.getpid(), host=socket.gethostname())).encode())
    os.close(descriptor)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def run_cv(
    data_dir=None,
    output_dir="task4_v2_results",
    cfg=None,
    folds_to_run=(0, 1, 2, 3, 4),
    weights_path=None,
    resume_from=None,
    backups=True,
    full_backup=True,
):
    cfg = cfg or ConfigV2()
    if cfg.folds != 5 or cfg.fold != 0 or cfg.patch < 64 or cfg.patch % 32 or cfg.patch > 512:
        raise ValueError("Require 5 folds, audit fold=0, patch multiple of 32 in [64,512].")
    if cfg.epochs < 1 or cfg.batch_size < 1 or cfg.patches_per_epoch < 1 or cfg.monitor_every < 1:
        raise ValueError("Invalid training configuration")
    if (
        not 0 < cfg.ema_decay < 1
        or len(set(folds_to_run)) != len(folds_to_run)
        or not set(folds_to_run) <= set(range(5))
    ):
        raise ValueError("Invalid EMA decay/fold list")
    torch.set_num_threads(cfg.threads)
    cv2.setNumThreads(0)
    output = Path(output_dir).resolve()
    if resume_from:
        restore_output(resume_from, output)
    output.mkdir(parents=True, exist_ok=True)
    root = base.find_data(data_dir)
    identity = dict(
        config=asdict(cfg),
        source_sha256=base.file_sha(__file__),
        baseline_helpers_sha256=base.file_sha(base.__file__),
    )
    config_path = output / "run_config.json"
    previous = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else None
    if previous and any(previous.get(k) != v for k, v in identity.items()):
        raise ValueError("Changed config/source: use a NEW OUTPUT_DIR to preserve prior results.")
    with run_lock(output):
        train, test, labels, assignment, ids, audit = audit_v2(root, output, cfg)
        identity["data_fingerprint"] = audit["data_fingerprint"]
        if previous and previous["data_fingerprint"] != identity["data_fingerprint"]:
            raise ValueError("Dataset changed: use a new output directory")
        initial_state, provenance = resolve_weights(cfg, output, weights_path)
        identity["pretrained_sha256"] = provenance["sha256"]
        if previous and previous != identity:
            raise ValueError("Pretrained weights differ from saved run")
        base.save_json(config_path, identity)
        base.save_json(output / "pretrained_source.json", provenance)
        base.save_json(
            output / "environment.json",
            dict(
                python=platform.python_version(),
                torch=torch.__version__,
                torchvision=torchvision.__version__,
                opencv=cv2.__version__,
                numpy=np.__version__,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                training_device_count=1,
                torch_cuda=torch.version.cuda,
            ),
        )
        code_dir = output / "runtime_source"
        code_dir.mkdir(exist_ok=True)
        for p in [Path(__file__), Path(base.__file__)]:
            if p.resolve() != (code_dir / p.name).resolve():
                shutil.copyfile(p, code_dir / p.name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            "Device:",
            device,
            "fixed epochs:",
            cfg.epochs,
            "folds requested:",
            folds_to_run,
            flush=True,
        )
        if device.type != "cuda":
            print("CPU detected; formal five-fold training will be slow. Enable GPU.", flush=True)
        for fold in folds_to_run:
            folder = output / f"fold{fold}"
            if (folder / "completed.json").exists():
                print(f"fold{fold}: completed, reusing verified artifacts", flush=True)
                continue
            development = [p for p in train if assignment[p.stem] != fold]
            validation = [p for p in train if assignment[p.stem] == fold]
            model = train_fold(
                development, validation, labels, cfg, fold, folder, initial_state, device
            )
            infer_fold(model, validation, test, labels, cfg, folder, device)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        summary = aggregate(root, output, cfg, train, test, labels, assignment, ids)
        if backups:
            make_backups(output, full=full_backup)
    return output, summary
