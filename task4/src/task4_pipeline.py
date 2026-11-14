"""Task 4: small-data retinal vessel segmentation, audit through submission.

Only the supplied training images/labels are used. No downloaded weights.
The original segmentation_to_csv.py is kept unchanged and checked on export.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class Config:
    seed: int = 42
    epochs: int = 80
    size: int = 512
    patch: int = 256
    patches_per_epoch: int = 128
    batch_size: int = 8
    base_channels: int = 24
    lr: float = 0.001
    weight_decay: float = 0.0001
    folds: int = 5
    fold: int = 0
    threads: int = 4
    amp: bool = True


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_data(data_dir=None):
    if data_dir:
        candidates = [Path(data_dir)]
    else:
        base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
        candidates = [
            base / "2026task-4/segmentation",
            base.parent / "2026task-4/segmentation",
            Path.cwd() / "2026task-4/segmentation",
            Path.cwd() / "task4/2026task-4/segmentation",
        ]
        if Path("/kaggle/input").exists():
            candidates += [p.parent for p in Path("/kaggle/input").rglob("sample_submission.csv")]
    found = {
        p.resolve()
        for p in candidates
        if (p / "train/label").is_dir()
        and (p / "test/image").is_dir()
        and (p / "sample_submission.csv").is_file()
    }
    if len(found) != 1:
        raise ValueError(
            f"Set data_dir to the segmentation folder. Found: {sorted(map(str, found))}"
        )
    return found.pop()


def list_images(directory):
    files = sorted(Path(directory).glob("*.jpg"), key=lambda p: int(p.stem))
    if not files:
        raise ValueError(f"No jpg images: {directory}")
    return files


def strict_label(path, size=512):
    # JPEG labels contain ringing; >0 would thicken the foreground substantially.
    mask = (np.asarray(Image.open(path).convert("L")) > 127).astype(np.uint8)
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)


def converter_label(path):
    # Exact target IF hidden ground truth was processed by the supplied converter.
    # This is a diagnostic proxy, not an assertion about the private scoring code.
    return (
        np.asarray(Image.open(path).convert("L").resize((512, 512), Image.Resampling.BILINEAR)) > 0
    )


def make_features(path, size=512):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    green = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(rgb[:, :, 1])
    return np.concatenate([rgb, green[:, :, None]], axis=2)


def audit(root, output, cfg):
    output.mkdir(parents=True, exist_ok=True)
    train, test = list_images(root / "train/image"), list_images(root / "test/image")
    labels = {p.stem: p for p in list_images(root / "train/label")}
    if len(train) != 20 or len(test) != 20 or set(labels) != {p.stem for p in train}:
        raise ValueError("Expected 20 train image/label pairs and 20 test images.")
    with (root / "sample_submission.csv").open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["Id", "Predicted"]:
            raise ValueError(f"Unexpected template columns: {reader.fieldnames}")
        ids = [r["Id"] for r in reader]
    if len(ids) != len(set(ids)) or set(ids) != {p.stem for p in test}:
        raise ValueError("Template IDs do not match test images.")
    rows, groups, test_hashes = [], {}, set()
    for split, paths in [("train", train), ("test", test)]:
        for p in paths:
            im = Image.open(p).convert("RGB")
            digest = hashlib.sha256(np.asarray(im).tobytes()).hexdigest()
            row = dict(
                split=split,
                id=p.stem,
                width=im.width,
                height=im.height,
                image_sha256=file_sha(p),
                pixel_sha256=digest,
            )
            if split == "train":
                if Image.open(labels[p.stem]).size != im.size:
                    raise ValueError(f"Image/label size mismatch: {p.stem}")
                row.update(
                    label_sha256=file_sha(labels[p.stem]),
                    vessel_fraction=float(strict_label(labels[p.stem]).mean()),
                    converter_fraction=float(converter_label(labels[p.stem]).mean()),
                )
                groups.setdefault(digest, []).append(p.stem)
            else:
                test_hashes.add(digest)
            rows.append(row)
    if set(groups) & test_hashes:
        raise ValueError("Exact decoded train/test duplicate detected; inspect before training.")
    ordered = sorted(groups)
    random.Random(cfg.seed).shuffle(ordered)
    assignments = {}
    sizes = [0] * cfg.folds
    for digest in ordered:
        fold = min(range(cfg.folds), key=lambda k: sizes[k])
        for image_id in groups[digest]:
            assignments[image_id] = fold
        sizes[fold] += len(groups[digest])
    for row in rows:
        row["fold"] = assignments.get(row["id"], -1) if row["split"] == "train" else -1
    fields = sorted({key for row in rows for key in row})
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest_fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    summary = dict(
        train_images=len(train),
        test_images=len(test),
        unique_train_groups=len(groups),
        folds=sizes,
        validation_ids=[p.stem for p in train if assignments[p.stem] == cfg.fold],
        decoded_train_test_overlap=0,
        patient_ids_available=False,
        data_fingerprint=manifest_fingerprint,
        observed_label_polarity="white vessels / black background",
        screenshot_label_polarity="black vessels / white background",
        converter=dict(
            size=[512, 512],
            resize="PIL bilinear",
            foreground="nonzero",
            flatten="row-major C order",
            start_index=1,
            source_sha256=file_sha(root / "segmentation_to_csv.py"),
        ),
        strict_label="native grayscale >127, then nearest resize",
        mean_clean_foreground=float(
            np.mean([r["vessel_fraction"] for r in rows if r["split"] == "train"])
        ),
        mean_converter_foreground=float(
            np.mean([r["converter_fraction"] for r in rows if r["split"] == "train"])
        ),
    )
    save_json(output / "audit_summary.json", summary)
    return train, test, labels, assignments, ids, summary


def paired_d4(image, mask, rotation, flip):
    image, mask = np.rot90(image, rotation), np.rot90(mask, rotation)
    if flip:
        image, mask = np.flip(image, 1), np.flip(mask, 1)
    return image.copy(), mask.copy()


class PatchDataset(Dataset):
    def __init__(self, paths, labels, cfg):
        self.cfg = cfg
        self.images = [make_features(p, cfg.size) for p in paths]
        self.masks = [strict_label(labels[p.stem], cfg.size).astype(np.float32) for p in paths]
        self.positive = [np.argwhere(m > 0) for m in self.masks]

    def __len__(self):
        return self.cfg.patches_per_epoch

    def __getitem__(self, index):
        idx = random.randrange(len(self.images))
        image, mask = self.images[idx], self.masks[idx]
        size, patch = self.cfg.size, self.cfg.patch
        crop = random.randint(patch, min(size, int(patch * 1.25)))
        if random.random() < 0.3 and len(self.positive[idx]):
            y, x = self.positive[idx][random.randrange(len(self.positive[idx]))]
            top, left = int(np.clip(y - crop // 2, 0, size - crop)), int(
                np.clip(x - crop // 2, 0, size - crop)
            )
        else:
            top, left = random.randint(0, size - crop), random.randint(0, size - crop)
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
        image, mask = paired_d4(image, mask, random.randrange(4), random.random() < 0.5)
        image = image.astype(np.float32) / 255.0
        image = np.clip(image * random.uniform(0.9, 1.1) + random.uniform(-0.04, 0.04), 0, 1)
        image = image ** random.uniform(0.85, 1.15)
        return torch.from_numpy(image.transpose(2, 0, 1).copy()) * 2 - 1, torch.from_numpy(
            mask[None].copy()
        )


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
        )


class VesselUNet(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        ch = [base * (2**i) for i in range(5)]
        self.enc = nn.ModuleList([ConvBlock(4 if i == 0 else ch[i - 1], ch[i]) for i in range(5)])
        self.dec = nn.ModuleList([ConvBlock(ch[i + 1] + ch[i], ch[i]) for i in range(3, -1, -1)])
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for block, skip in zip(self.dec, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], 1))
        return self.head(x)


def segmentation_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits.float(), target.float())
    prob = logits.float().sigmoid()
    dice = (2 * (prob * target).sum((1, 2, 3)) + 1) / (
        prob.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1
    )
    return 0.5 * bce + 0.5 * (1 - dice.mean())


def dice_score(pred, target):
    pred, target = np.asarray(pred, bool), np.asarray(target, bool)
    return float(
        (2 * np.logical_and(pred, target).sum() + 1e-7) / (pred.sum() + target.sum() + 1e-7)
    )


@torch.inference_mode()
def predict(model, path, cfg, device):
    model.eval()
    image = make_features(path, cfg.size)
    step = cfg.patch // 2
    starts = list(range(0, cfg.size - cfg.patch + 1, step))
    if starts[-1] != cfg.size - cfg.patch:
        starts.append(cfg.size - cfg.patch)
    coords = [(y, x) for y in starts for x in starts]
    weight = np.maximum(np.outer(np.hanning(cfg.patch), np.hanning(cfg.patch)), 0.05).astype(
        np.float32
    )
    sums, weights = np.zeros((cfg.size, cfg.size), np.float32), np.zeros(
        (cfg.size, cfg.size), np.float32
    )
    for start in range(0, len(coords), cfg.batch_size):
        positions = coords[start : start + cfg.batch_size]
        patches = np.stack(
            [image[y : y + cfg.patch, x : x + cfg.patch].transpose(2, 0, 1) for y, x in positions]
        )
        tensor = torch.from_numpy(patches.copy()).to(device).float() / 127.5 - 1
        with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
            probs = model(tensor).float().sigmoid()[:, 0].cpu().numpy()
        for (y, x), prob in zip(positions, probs):
            sums[y : y + cfg.patch, x : x + cfg.patch] += prob * weight
            weights[y : y + cfg.patch, x : x + cfg.patch] += weight
    return sums / weights


def atomic_checkpoint(path, value):
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def train_stage(paths, val_paths, labels, cfg, folder, device, epochs):
    folder.mkdir(parents=True, exist_ok=True)
    seed_all(cfg.seed)
    dataset = PatchDataset(paths, labels, cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    model = VesselUNet(cfg.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Full retraining retains the original 80-epoch cosine horizon.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.02
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    best, history, first = -1.0, [], 1
    last = folder / "last.pt"
    if last.exists():
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        if ckpt["train_ids"] != [p.stem for p in paths] or ckpt["target_epochs"] != epochs:
            raise ValueError("Stage mismatch. Use a new output directory.")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        first, best, history = ckpt["epoch"] + 1, ckpt["best"], ckpt["history"]
        restore_rng(ckpt["rng"])
        print(f"{folder.name}: resume at epoch {first}/{epochs}", flush=True)
    targets = {p.stem: strict_label(labels[p.stem], cfg.size) for p in val_paths}
    for epoch in range(first, epochs + 1):
        begin = time.perf_counter()
        model.train()
        losses = []
        lr = optimizer.param_groups[0]["lr"]
        for image, mask in loader:
            image, mask = image.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
                loss = segmentation_loss(model(image), mask)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        score = (
            float(
                np.mean(
                    [
                        dice_score(predict(model, p, cfg, device) >= 0.5, targets[p.stem])
                        for p in val_paths
                    ]
                )
            )
            if val_paths
            else None
        )
        improved = score is not None and score > best
        if improved:
            best = score
            atomic_checkpoint(
                folder / "best.pt", dict(model=model.state_dict(), epoch=epoch, dice_05=score)
            )
        row = dict(
            epoch=epoch,
            loss=float(np.mean(losses)),
            dice_05=score,
            lr=lr,
            seconds=round(time.perf_counter() - begin, 3),
        )
        history.append(row)
        atomic_checkpoint(
            last,
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),
                epoch=epoch,
                best=best,
                history=history,
                rng=rng_state(),
                train_ids=[p.stem for p in paths],
                target_epochs=epochs,
            ),
        )
        with (folder / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, list(row))
            writer.writeheader()
            writer.writerows(history)
        print(
            f'{folder.name} {epoch:03d}/{epochs} loss={row["loss"]:.4f}'
            + (f" val_dice@0.5={score:.5f} best={best:.5f}" if score is not None else "")
            + f' seconds={row["seconds"]}',
            flush=True,
        )
    selected = torch.load(
        folder / ("best.pt" if val_paths else "last.pt"), map_location="cpu", weights_only=False
    )
    model.load_state_dict(selected["model"])
    return model, int(selected["epoch"])


def rle_encode(mask):
    flat = np.asarray(mask, dtype=np.uint8).ravel(order="C")
    changes = np.flatnonzero(np.diff(np.r_[np.uint8(0), flat, np.uint8(0)])) + 1
    changes[1::2] -= changes[::2]
    return " ".join(map(str, changes))


def rle_decode(encoded, shape=(512, 512)):
    values = np.array([int(s) for s in encoded.split()], dtype=np.int64)
    if len(values) % 2:
        raise ValueError("Odd RLE length")
    mask = np.zeros(int(np.prod(shape)), dtype=bool)
    previous_end = 0
    for start, length in values.reshape(-1, 2):
        if start < 1 or length < 1 or start <= previous_end or start + length - 1 > mask.size:
            raise ValueError("RLE bounds/order/overlap violation")
        mask[start - 1 : start - 1 + length] = True
        previous_end = start + length - 1
    return mask.reshape(shape)


def load_converter(root):
    spec = importlib.util.spec_from_file_location(
        "task4_official_converter", root / "segmentation_to_csv.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preview(paths, probs, labels, threshold, output):
    tile = 256
    canvas = Image.new("RGB", (tile * 3, (tile + 24) * min(4, len(paths))), "white")
    draw = ImageDraw.Draw(canvas)
    for row, p in enumerate(paths[:4]):
        rgb = Image.open(p).convert("RGB").resize((tile, tile))
        target = strict_label(labels[p.stem], tile) if labels else None
        pred = cv2.resize(probs[p.stem], (tile, tile), interpolation=cv2.INTER_LINEAR) >= threshold
        y = row * (tile + 24)
        canvas.paste(rgb, (0, y + 24))
        draw.text((4, y + 4), f"Image {p.stem}", fill="black")
        if target is not None:
            canvas.paste(
                Image.fromarray((target * 255).astype(np.uint8)).convert("RGB"), (tile, y + 24)
            )
            draw.text((tile + 4, y + 4), "Label: white vessels", fill="black")
        else:
            overlay = np.asarray(rgb).copy()
            overlay[pred] = (0.4 * overlay[pred] + 0.6 * np.array([0, 255, 100])).astype(np.uint8)
            canvas.paste(Image.fromarray(overlay), (tile, y + 24))
            draw.text((tile + 4, y + 4), "Vessel overlay", fill="black")
        canvas.paste(
            Image.fromarray((pred * 255).astype(np.uint8)).convert("RGB"), (tile * 2, y + 24)
        )
        draw.text((tile * 2 + 4, y + 4), f"Prediction t={threshold:.3f}", fill="black")
    canvas.save(output)


def validation_report(model, paths, labels, cfg, device, output, epoch):
    probs = {p.stem: predict(model, p, cfg, device) for p in paths}
    np.savez_compressed(output / "validation_probabilities.npz", **probs)
    thresholds = [round(float(t), 3) for t in np.arange(0.2, 0.801, 0.025)]
    rows = []
    for threshold in thresholds:
        strict = [
            dice_score(probs[p.stem] >= threshold, strict_label(labels[p.stem], cfg.size))
            for p in paths
        ]
        proxy = [
            dice_score(probs[p.stem] >= threshold, converter_label(labels[p.stem])) for p in paths
        ]
        rows.append(
            dict(
                threshold=threshold,
                mean_vessel_dice=float(np.mean(strict)),
                converter_target_proxy_dice=float(np.mean(proxy)),
            )
        )
    best = max(rows, key=lambda r: (r["mean_vessel_dice"], -abs(r["threshold"] - 0.5)))
    threshold = best["threshold"]
    with (output / "threshold_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    per_image = []
    for p in paths:
        pred = probs[p.stem] >= threshold
        gt = strict_label(labels[p.stem], cfg.size)
        native = np.asarray(Image.open(labels[p.stem]).convert("L")) > 127
        native_pred = (
            cv2.resize(
                probs[p.stem], (native.shape[1], native.shape[0]), interpolation=cv2.INTER_LINEAR
            )
            >= threshold
        )
        per_image.append(
            dict(
                id=p.stem,
                dice_05=dice_score(probs[p.stem] >= 0.5, gt),
                dice=dice_score(pred, gt),
                native_dice=dice_score(native_pred, native),
                converter_target_proxy_dice=dice_score(pred, converter_label(labels[p.stem])),
                predicted_fraction=float(pred.mean()),
                target_fraction=float(gt.mean()),
            )
        )
    with (output / "validation_per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    report = dict(
        best_epoch=epoch,
        threshold=threshold,
        validation_ids=[p.stem for p in paths],
        mean_vessel_dice_05=float(np.mean([r["dice_05"] for r in per_image])),
        mean_vessel_dice=best["mean_vessel_dice"],
        mean_native_vessel_dice=float(np.mean([r["native_dice"] for r in per_image])),
        converter_target_proxy_dice=best["converter_target_proxy_dice"],
        evaluation="single held-out fold (4 images), not full five-fold OOF",
        limitation="Epoch and threshold chosen on the same small validation set; development estimate only.",
        private_ground_truth_encoding_verified=False,
    )
    save_json(output / "validation_metrics.json", report)
    preview(paths, probs, labels, threshold, output / "validation_preview.png")
    return report


def export_submission(model, test, template_ids, cfg, device, root, output, threshold, epoch):
    white_dir = output / "masks_512_white_vessels"
    black_dir = output / "masks_native_black_vessels"
    white_dir.mkdir(exist_ok=True)
    black_dir.mkdir(exist_ok=True)
    probs = {p.stem: predict(model, p, cfg, device) for p in test}
    paths = {p.stem: p for p in test}
    white_paths = []
    encoded = []
    for image_id in template_ids:
        prob = probs[image_id]
        mask = prob >= threshold
        white_path = white_dir / f"{image_id}.png"
        Image.fromarray((mask * 255).astype(np.uint8)).save(white_path)
        white_paths.append(str(white_path))
        width, height = Image.open(paths[image_id]).size
        native = cv2.resize(prob, (width, height), interpolation=cv2.INTER_LINEAR) >= threshold
        Image.fromarray(np.where(native, 0, 255).astype(np.uint8)).save(
            black_dir / f"{image_id}.png"
        )
        value = rle_encode(mask)
        if not np.array_equal(rle_decode(value), mask):
            raise AssertionError("RLE roundtrip mismatch")
        encoded.append(value)
    # Execute the supplied conversion function on the actual PNG predictions.
    official = load_converter(root).turn_to_str(white_paths)
    if encoded != official:
        raise AssertionError("Vectorized encoding differs from the official converter")
    target = output / "submission.csv"
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Predicted"])
        writer.writerows(zip(template_ids, official))
    with target.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if [r["Id"] for r in rows] != template_ids or len(rows) != 20:
        raise AssertionError("Saved submission IDs/count mismatch")
    for row in rows:
        rle_decode(row["Predicted"])
    np.savez_compressed(output / "test_probabilities.npz", **probs)
    preview(test, probs, None, threshold, output / "test_preview.png")
    save_json(
        output / "submission_provenance.json",
        dict(
            csv_sha256=file_sha(target),
            model_sha256=file_sha(output / "fulltrain/last.pt"),
            train_images=20,
            epochs=epoch,
            threshold=threshold,
            external_training_data=False,
            pretrained=False,
            csv_polarity="white vessels, following observed training labels and supplied converter",
            image_output_polarity="masks_native_black_vessels: vessel=0, background=255",
            official_converter_exact_match=True,
            rle_roundtrip_all_20=True,
            template_order_preserved=True,
            submitted_to_competition=False,
        ),
    )
    print(f"Submission saved: {target}", flush=True)
    return target


def run(phase="full", data_dir=None, output_dir=None, config=None):
    cfg = config or Config()
    if (
        cfg.size != 512
        or cfg.patch < 16
        or cfg.patch % 16
        or cfg.patch > cfg.size
        or cfg.base_channels < 8
        or cfg.base_channels % 8
    ):
        raise ValueError(
            "Require size=512, patch divisible by 16 <=512, base_channels divisible by 8."
        )
    if (
        cfg.epochs < 1
        or cfg.patches_per_epoch < 1
        or cfg.batch_size < 1
        or cfg.threads < 1
        or not 2 <= cfg.folds <= 20
        or not 0 <= cfg.fold < cfg.folds
    ):
        raise ValueError("Invalid training configuration")
    torch.set_num_threads(cfg.threads)
    cv2.setNumThreads(0)
    root = find_data(data_dir)
    output = Path(output_dir or "task4_baseline_results").resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_sha = file_sha(Path(__file__)) if "__file__" in globals() else PIPELINE_SHA256
    config_path = output / "run_config.json"
    previous = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else None
    if previous and (
        previous["config"] != asdict(cfg) or previous["pipeline_sha256"] != source_sha
    ):
        raise ValueError("Existing output belongs to another config/source. Use a new directory.")
    train, test, labels, assignment, ids, summary = audit(root, output, cfg)
    identity = dict(
        config=asdict(cfg), data_fingerprint=summary["data_fingerprint"], pipeline_sha256=source_sha
    )
    if previous and previous != identity:
        raise ValueError(
            "Existing output belongs to another config/data/source. Use a new directory."
        )
    save_json(config_path, identity)
    if phase == "audit":
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return output
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; train=16, validation=4; no pretrained weights", flush=True)
    val = [p for p in train if assignment[p.stem] == cfg.fold]
    development = [p for p in train if assignment[p.stem] != cfg.fold]
    # A single writer protects checkpoints from two simultaneous runs.
    lock = output / "training.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Run lock exists: {lock}. Check its PID before removing a stale lock.")
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    try:
        model, epoch = train_stage(
            development, val, labels, cfg, output / "holdout", device, cfg.epochs
        )
        metrics = validation_report(model, val, labels, cfg, device, output, epoch)
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
        if phase == "full":
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            model, full_epoch = train_stage(
                train, [], labels, cfg, output / "fulltrain", device, epoch
            )
            export_submission(
                model, test, ids, cfg, device, root, output, metrics["threshold"], full_epoch
            )
    finally:
        lock.unlink(missing_ok=True)
    return output


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["audit", "train", "full"], default="full")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir", default="task4_baseline_results")
    for key, value in asdict(Config()).items():
        if isinstance(value, bool):
            continue
        parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    args = vars(parser.parse_args())
    phase, data_dir, output = args.pop("phase"), args.pop("data_dir"), args.pop("output_dir")
    run(phase, data_dir, output, Config(**args))


if __name__ == "__main__":
    cli()
