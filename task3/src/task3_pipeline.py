"""Reproducible fovea-localization baseline; also embedded in task3.ipynb."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def find_data(explicit=None):
    if explicit:
        candidates = [Path(explicit)]
    else:
        base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
        candidates = [
            base / "2026task3/detection",
            base.parent / "2026task3/detection",
            Path.cwd() / "2026task3/detection",
            Path.cwd() / "task3/2026task3/detection",
        ]
        if Path("/kaggle/input").exists():
            candidates += [
                p.parent for p in Path("/kaggle/input").rglob("fovea_localization_train_GT.csv")
            ]
    valid = sorted(
        {
            p.resolve()
            for p in candidates
            if (p / "fovea_localization_train_GT.csv").is_file()
            and (p / "sample_submission.csv").is_file()
            and (p / "train").is_dir()
            and (p / "test").is_dir()
        }
    )
    if len(valid) != 1:
        raise ValueError(f"Expected one dataset; set data_dir explicitly. Found: {valid}")
    return valid[0]


def load_manifest(data):
    labels = pd.read_csv(data / "fovea_localization_train_GT.csv")
    if list(labels.columns) != ["data", "Fovea_X", "Fovea_Y"] or labels.isna().any().any():
        raise ValueError("Invalid training CSV columns or missing values.")
    if labels["data"].duplicated().any():
        raise ValueError("Duplicate training IDs.")
    lookup = labels.set_index("data")
    records = []
    for subset in ["train", "test"]:
        files = sorted(
            p for p in (data / subset).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        for p in files:
            image_id = int(p.stem)
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.load()
                w, h = im.size
                # Hash decoded pixels, not JPEG metadata. Size participates in the hash.
                digest = hashlib.sha256(f"{w},{h}:".encode() + im.tobytes()).hexdigest()
            record = dict(
                image_id=image_id,
                subset=subset,
                filename=p.name,
                width=w,
                height=h,
                pixel_sha256=digest,
            )
            if subset == "train":
                if image_id not in lookup.index:
                    raise ValueError(f"Missing label: {p.name}")
                x, y = lookup.loc[image_id, ["Fovea_X", "Fovea_Y"]].astype(float)
                if not np.isfinite([x, y]).all() or not (0 <= x < w and 0 <= y < h):
                    raise ValueError(f"Coordinate outside image: {p.name}")
                record.update(x=float(x), y=float(y), nx=float(x / w), ny=float(y / h))
                xml_path = data / "train_location" / f"{image_id:04}.xml"
                if not xml_path.is_file():
                    raise ValueError(f"Missing XML: {xml_path}")
                xml = ET.parse(xml_path).getroot()
                if (int(xml.findtext("size/width")), int(xml.findtext("size/height"))) != (w, h):
                    raise ValueError(f"XML size mismatch: {p.name}")
                objects = [
                    v for v in xml.findall("object") if v.findtext("name").lower() == "fovea"
                ]
                if len(objects) != 1:
                    raise ValueError(f"Expected one Fovea box: {xml_path}")
                box = objects[0].find("bndbox")
                xmin, ymin, xmax, ymax = [
                    float(box.findtext(k)) for k in ["xmin", "ymin", "xmax", "ymax"]
                ]
                if not (xmin <= x <= xmax and ymin <= y <= ymax):
                    raise ValueError(f"CSV coordinate outside XML box: {p.name}")
                record.update(
                    xml_center_dx=float(x - (xmin + xmax) / 2),
                    xml_center_dy=float(y - (ymin + ymax) / 2),
                )
            records.append(record)
    manifest = pd.DataFrame(records)
    if set(manifest.loc[manifest.subset == "train", "image_id"]) != set(labels["data"]):
        raise ValueError("Training images and CSV IDs differ.")
    if manifest.image_id.duplicated().any():
        raise ValueError("Image IDs overlap between train/test.")
    cross = manifest.groupby("pixel_sha256").subset.nunique()
    if (cross > 1).any():
        raise ValueError(
            "Identical pixels occur in both train and test; inspect before continuing."
        )
    return manifest


def assign_folds(train, seed=42, folds=5):
    """Balance exact-duplicate groups across image resolutions; no test labels used."""
    if folds < 2:
        raise ValueError("Need at least two folds.")
    train = train.copy()
    groups = list(train.groupby("pixel_sha256", sort=True))
    rng = np.random.default_rng(seed)
    buckets = defaultdict(list)
    for digest, group in groups:
        row = group.iloc[0]
        buckets[(int(row.width), int(row.height))].append((digest, len(group)))
    total = np.zeros(folds, dtype=int)
    mapping = {}
    for key in sorted(buckets):
        entries = buckets[key]
        rng.shuffle(entries)
        entries.sort(key=lambda pair: -pair[1])
        sizes = np.zeros(folds, dtype=int)
        for digest, count in entries:
            tie = rng.random(folds)
            fold = min(range(folds), key=lambda f: (sizes[f], total[f], tie[f]))
            mapping[digest] = fold
            sizes[fold] += count
            total[fold] += count
    train["fold"] = train.pixel_sha256.map(mapping)
    if train.groupby("pixel_sha256").fold.nunique().max() != 1:
        raise AssertionError("Duplicate group split across folds.")
    return train


def error_metrics(predicted, truth, sizes):
    delta = (np.asarray(predicted, dtype=np.float64) - np.asarray(truth, dtype=np.float64)) * sizes
    distance = np.linalg.norm(delta, axis=1)
    return dict(
        n=len(distance),
        mean_distance_px=float(distance.mean()),
        median_distance_px=float(np.median(distance)),
        p90_distance_px=float(np.quantile(distance, 0.9)),
        mae_x_px=float(np.abs(delta[:, 0]).mean()),
        mae_y_px=float(np.abs(delta[:, 1]).mean()),
        rmse_coordinate_px=float(np.sqrt(np.mean(delta**2))),
    )


def coordinate_baselines(train, valid):
    truth = valid[["nx", "ny"]].to_numpy()
    sizes = valid[["width", "height"]].to_numpy()
    result = []
    predictions = {}
    for name, center in [
        ("normalized_mean", train[["nx", "ny"]].mean().to_numpy()),
        ("normalized_median", train[["nx", "ny"]].median().to_numpy()),
    ]:
        prediction = np.tile(center, (len(valid), 1))
        result.append(dict(method=name, **error_metrics(prediction, truth, sizes)))
        predictions[name] = prediction
    # Resolution-specific priors expose how much information the camera geometry provides.
    centers = train.groupby(["width", "height"])[["nx", "ny"]].mean()
    fallback = train[["nx", "ny"]].mean().to_numpy()
    prediction = np.array(
        [
            (
                centers.loc[(int(row.width), int(row.height))].to_numpy()
                if (int(row.width), int(row.height)) in centers.index
                else fallback
            )
            for row in valid.itertuples()
        ]
    )
    result.append(dict(method="resolution_mean", **error_metrics(prediction, truth, sizes)))
    predictions["resolution_mean"] = prediction
    return result, predictions


def contact_sheet(data, records, path, predictions=None):
    rows = records.reset_index(drop=True)
    n = min(len(rows), 20)
    canvas = Image.new("RGB", (1120, math.ceil(n / 4) * 232), "#f3f5f7")
    draw = ImageDraw.Draw(canvas)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = (
        ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
    )
    for i, row in rows.iloc[:n].iterrows():
        with Image.open(data / row["subset"] / row.filename) as im:
            im = im.convert("RGB")
            im.thumbnail((276, 195))
            ox, oy = (i % 4) * 280 + (280 - im.width) // 2, (i // 4) * 232
            canvas.paste(im, (ox, oy))
            points = []
            if "nx" in row and pd.notna(row.nx):
                points.append((row.nx, row.ny, "#00ffff"))
            if predictions is not None:
                points.append((*predictions[i], "#ffff00"))
            for x, y, color in points:
                cx, cy = ox + x * im.width, oy + y * im.height
                draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), outline=color, width=2)
                draw.line((cx - 8, cy, cx + 8, cy), fill=color)
                draw.line((cx, cy - 8, cx, cy + 8), fill=color)
        draw.text(
            ((i % 4) * 280 + 7, (i // 4) * 232 + 199),
            f"{int(row.image_id):04} | {int(row.width)} x {int(row.height)}",
            font=font,
            fill="#15202b",
        )
    canvas.save(path)


def audit(data, out, seed=42, folds=5, fold=0):
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(data)
    train = assign_folds(manifest[manifest.subset == "train"], seed, folds)
    train.to_csv(out / "split.csv", index=False)
    manifest.to_csv(out / "manifest.csv", index=False)
    if not 0 <= fold < folds:
        raise ValueError("fold outside [0, folds).")
    development = train[train.fold != fold]
    valid = train[train.fold == fold]
    baselines, predictions = coordinate_baselines(development, valid)
    pd.DataFrame(baselines).to_csv(out / "coordinate_baselines.csv", index=False)
    per_image = valid[["image_id", "width", "height", "x", "y"]].copy()
    for name, pred in predictions.items():
        xy = pred * valid[["width", "height"]].to_numpy()
        per_image[f"{name}_x"] = xy[:, 0]
        per_image[f"{name}_y"] = xy[:, 1]
        per_image[f"{name}_distance_px"] = np.linalg.norm(xy - valid[["x", "y"]].to_numpy(), axis=1)
    per_image.to_csv(out / "baseline_validation_predictions.csv", index=False)
    duplicates = [
        g.image_id.astype(int).tolist() for _, g in manifest.groupby("pixel_sha256") if len(g) > 1
    ]
    summary = dict(
        train_count=len(train),
        test_count=int((manifest.subset == "test").sum()),
        unique_train_images=int(train.pixel_sha256.nunique()),
        exact_duplicate_groups=duplicates,
        resolution_counts=[
            dict(width=int(w), height=int(h), count=len(g))
            for (w, h), g in train.groupby(["width", "height"])
        ],
        fold_sizes={str(k): len(g) for k, g in train.groupby("fold")},
        validation_fold=fold,
        xml_max_abs_offset_px=train[["xml_center_dx", "xml_center_dy"]].abs().max().to_dict(),
        label_source="fovea_localization_train_GT.csv; XML used only for consistency checks",
        split_limit="Exact duplicates grouped; patient identities unavailable, so patient-level independence is unverified.",
        metric_status="Official evaluation formula not provided; pixel Euclidean distance is provisional.",
        coordinate_baselines=baselines,
    )
    save_json(out / "audit_summary.json", summary)
    # Select examples from across the ID range instead of displaying only the first camera type.
    picks = np.linspace(0, len(train) - 1, min(20, len(train))).astype(int)
    contact_sheet(data, train.iloc[picks], out / "training_samples.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return manifest, train


class FoveaDataset(Dataset):
    def __init__(self, data, records, image_size, augment=False):
        self.records = records.reset_index(drop=True)
        self.image_size = image_size
        self.augment = augment
        self.images = []
        self.jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.12, hue=0.02
        )
        self.tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        for row in self.records.itertuples():
            with Image.open(data / row.subset / row.filename) as im:
                self.images.append(
                    im.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
                )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records.iloc[index]
        im = self.images[index].copy()
        point = (
            np.array([row.nx, row.ny], dtype=np.float32)
            if row["subset"] == "train"
            else np.array([0.5, 0.5], dtype=np.float32)
        )
        if self.augment:
            if random.random() < 0.5:
                im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                point[0] = 1 - point[0]
            angle, scale = math.radians(random.uniform(-10, 10)), random.uniform(0.92, 1.08)
            a, b = scale * math.cos(angle), -scale * math.sin(angle)
            size, center = self.image_size, self.image_size / 2
            tx, ty = random.uniform(-0.035, 0.035) * size, random.uniform(-0.035, 0.035) * size
            matrix = np.array(
                [
                    [a, b, center + tx - a * center - b * center],
                    [-b, a, center + ty + b * center - a * center],
                    [0, 0, 1],
                ]
            )
            point = ((matrix @ np.array([point[0] * size, point[1] * size, 1]))[:2] / size).astype(
                np.float32
            )
            if not (point >= 0).all() or not (point <= 1).all():
                raise ValueError("Augmentation moved label outside image.")
            inverse = np.linalg.inv(matrix)[:2].reshape(-1)
            im = im.transform(
                (size, size),
                Image.Transform.AFFINE,
                inverse,
                resample=Image.Resampling.BILINEAR,
                fillcolor=(0, 0, 0),
            )
            im = self.jitter(im)
        return (
            self.tensor(im),
            torch.from_numpy(point),
            torch.tensor([row.width, row.height], dtype=torch.float32),
        )


def pretrained_backbone(pretrained=True, explicit=None):
    model = models.resnet18(weights=None)
    if not pretrained:
        return model
    base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    candidates = [Path(explicit)] if explicit else []
    candidates += [
        base.parent / "task2/.torch/hub/checkpoints/resnet18-f37072fd.pth",
        base.parent.parent / "task2/.torch/hub/checkpoints/resnet18-f37072fd.pth",
        Path.cwd().parent / "task2/.torch/hub/checkpoints/resnet18-f37072fd.pth",
        Path.cwd() / "task2/.torch/hub/checkpoints/resnet18-f37072fd.pth",
        Path(torch.hub.get_dir()) / "checkpoints/resnet18-f37072fd.pth",
    ]
    for path in candidates:
        if path.is_file():
            if not hashlib.sha256(path.read_bytes()).hexdigest().startswith("f37072fd"):
                raise ValueError(f"ImageNet checkpoint hash mismatch: {path}")
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            print(f"ImageNet weights: {path}", flush=True)
            return model
    # Official torchvision ImageNet weights only. No eye-dataset or Task2 fine-tuned weights.
    state = models.ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
    model.load_state_dict(state)
    return model


class DecoderBlock(nn.Module):
    def __init__(self, incoming, skip, outgoing):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(incoming + skip, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(8, outgoing),
            nn.ReLU(inplace=True),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(8, outgoing),
            nn.ReLU(inplace=True),
        )

    def forward(self, value, skip):
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.layers(torch.cat([value, skip], dim=1))


class FoveaHeatmap(nn.Module):
    def __init__(self, pretrained=True, weights=None):
        super().__init__()
        self.backbone = pretrained_backbone(pretrained, weights)
        self.backbone.fc = nn.Identity()
        self.d3 = DecoderBlock(512, 256, 128)
        self.d2 = DecoderBlock(128, 128, 64)
        self.d1 = DecoderBlock(64, 64, 32)
        self.head = nn.Conv2d(32, 1, 1)
        self.freeze_bn = pretrained

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_bn:
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, image):
        b = self.backbone
        value = b.maxpool(b.relu(b.bn1(b.conv1(image))))
        c1 = b.layer1(value)
        c2 = b.layer2(c1)
        c3 = b.layer3(c2)
        c4 = b.layer4(c3)
        return self.head(self.d1(self.d2(self.d3(c4, c3), c2), c1))[:, 0]


def heatmap_loss(logits, points, sigma=1.5):
    height, width = logits.shape[-2:]
    yy = torch.arange(height, device=logits.device, dtype=torch.float32) + 0.5
    xx = torch.arange(width, device=logits.device, dtype=torch.float32) + 0.5
    distance = (xx[None, None, :] - points[:, 0, None, None] * width) ** 2
    distance = distance + (yy[None, :, None] - points[:, 1, None, None] * height) ** 2
    target = torch.exp(-distance / (2 * sigma**2))
    target = target / target.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return -(target.flatten(1) * F.log_softmax(logits.float().flatten(1), dim=1)).sum(1).mean()


def decode_heatmap(logits, radius=3):
    """Local expectation around the peak; cell centers match resize align_corners=False."""
    batch, height, width = logits.shape
    peak = logits.flatten(1).argmax(1)
    py, px = peak // width, peak % width
    yy = torch.arange(height, device=logits.device)[None, :, None]
    xx = torch.arange(width, device=logits.device)[None, None, :]
    mask = (yy - py[:, None, None]).abs() <= radius
    mask = mask & ((xx - px[:, None, None]).abs() <= radius)
    weights = (
        logits.float()
        .masked_fill(~mask, -torch.inf)
        .flatten(1)
        .softmax(1)
        .view(batch, height, width)
    )
    x = (weights * ((xx + 0.5) / width)).sum((-2, -1))
    y = (weights * ((yy + 0.5) / height)).sum((-2, -1))
    return torch.stack([x, y], dim=1)


@torch.inference_mode()
def predict(model, loader, device, flip_tta=False):
    model.eval()
    predictions = []
    for images, _, _ in loader:
        images = images.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            points = decode_heatmap(model(images))
            if flip_tta:
                other = decode_heatmap(model(images.flip(-1)))
                other[:, 0] = 1 - other[:, 0]
                points = (points + other) / 2
        predictions.append(points.cpu().numpy())
    return np.concatenate(predictions)


@contextmanager
def training_lock(out):
    handle = (out / "training.lock").open("a+b")
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        handle.close()


def train_model(
    data, training, validation, out, config, device, model_builder=None, dataset_builder=None
):
    out.mkdir(parents=True, exist_ok=True)
    with training_lock(out):
        # Atomic checkpoint replacement temporarily keeps both the old and new files.
        if shutil.disk_usage(out).free < 600 * 1024**2:
            raise RuntimeError(
                f"Need at least 600 MiB free for atomic checkpoints; choose another output directory: {out}"
            )
        config_path = out / "config.json"
        if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError(f"Configuration changed. Use a new output directory: {out}")
        save_json(config_path, config)
        seed_everything(config["seed"])
        dataset_builder = dataset_builder or FoveaDataset
        train_set = dataset_builder(data, training, config["image_size"], augment=True)
        loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=0)
        valid_loader = (
            DataLoader(
                dataset_builder(data, validation, config["image_size"]),
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=0,
            )
            if validation is not None
            else None
        )
        model = (
            model_builder(config)
            if model_builder
            else FoveaHeatmap(config["pretrained"], config["weights"])
        ).to(device)
        backbone_ids = {id(v) for v in model.backbone.parameters()}
        decoder_params = [v for v in model.parameters() if id(v) not in backbone_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": config["lr_backbone"]},
                {"params": decoder_params, "lr": config["lr_decoder"]},
            ],
            weight_decay=config["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get("schedule_epochs", config["epochs"]),
            eta_min=config["lr_backbone"] * 0.05,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        start, best, best_epoch, history = 0, float("inf"), 0, []
        last_path, best_path = out / "last.pt", out / "best.pt"
        if last_path.exists():
            checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            start, best, best_epoch, history = (
                checkpoint[k] for k in ["epoch", "best", "best_epoch", "history"]
            )
            random.setstate(checkpoint["python_rng"])
            np.random.set_state(checkpoint["numpy_rng"])
            torch.set_rng_state(checkpoint["torch_rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
            print(f"Resume {out.name}: epoch {start}", flush=True)
        for epoch in range(start, config["epochs"]):
            begin = time.perf_counter()
            model.train()
            total, seen = 0.0, 0
            for images, points, _ in loader:
                images, points = images.to(device), points.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    loss = heatmap_loss(model(images), points, config["sigma"])
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss.")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                total += float(loss.detach()) * len(images)
                seen += len(images)
            row = dict(
                epoch=epoch + 1,
                train_heatmap_ce=total / seen,
                lr_backbone=optimizer.param_groups[0]["lr"],
            )
            if valid_loader is not None:
                prediction = predict(model, valid_loader, device)
                row.update(
                    error_metrics(
                        prediction,
                        validation[["nx", "ny"]].to_numpy(),
                        validation[["width", "height"]].to_numpy(),
                    )
                )
                score = row["mean_distance_px"]
            else:
                score = 0.0
            if score < best or valid_loader is None:
                best, best_epoch = score, epoch + 1
                torch.save(
                    dict(model=model.state_dict(), epoch=best_epoch, config=config),
                    out / "best.tmp",
                )
                (out / "best.tmp").replace(best_path)
            scheduler.step()
            row["seconds"] = time.perf_counter() - begin
            history.append(row)
            checkpoint = dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),
                epoch=epoch + 1,
                best=best,
                best_epoch=best_epoch,
                history=history,
                python_rng=random.getstate(),
                numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),
            )
            torch.save(checkpoint, out / "last.tmp")
            (out / "last.tmp").replace(last_path)
            pd.DataFrame(history).to_csv(out / "history.csv", index=False)
            save_json(
                out / "status.json",
                dict(
                    state="running",
                    epoch=epoch + 1,
                    epochs=config["epochs"],
                    best_epoch=best_epoch,
                    best_mean_distance_px=best,
                    pid=os.getpid(),
                    updated_unix=time.time(),
                ),
            )
            print(
                f'{out.name} {epoch+1:03}/{config["epochs"]} ce={row["train_heatmap_ce"]:.4f}'
                + (f" val={score:.2f}px best={best:.2f}px@{best_epoch}" if valid_loader else "")
                + f' {row["seconds"]:.1f}s',
                flush=True,
            )
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        save_json(
            out / "status.json",
            dict(
                state="completed",
                best_epoch=best_epoch,
                best_mean_distance_px=best if validation is not None else None,
                updated_unix=time.time(),
            ),
        )
        return model, best_epoch


def export_submission(data, test, prediction, out):
    sample = pd.read_csv(data / "sample_submission.csv")
    if list(sample.columns) != ["ImageID", "value"] or sample.ImageID.duplicated().any():
        raise ValueError("Unexpected submission template.")
    sizes = test[["width", "height"]].to_numpy()
    if prediction.shape != (len(test), 2) or not np.isfinite(prediction).all():
        raise ValueError("Invalid predictions.")
    xy = np.clip(prediction, 0, 1) * sizes
    xy = np.minimum(xy, sizes - 1)
    lookup = {
        f"{int(image_id)}_Fovea_{axis}": float(coord)
        for image_id, coords in zip(test.image_id, xy)
        for axis, coord in zip(["X", "Y"], coords)
    }
    if set(sample.ImageID) != set(lookup) or len(sample) != 2 * len(test):
        raise ValueError("Template IDs do not exactly match test images.")
    sample["value"] = sample.ImageID.map(lookup)
    if sample.value.isna().any():
        raise ValueError("Missing submission coordinates.")
    sample.to_csv(out / "submission.csv", index=False, float_format="%.6f")
    check = pd.read_csv(out / "submission.csv")
    if not check.ImageID.equals(sample.ImageID) or not np.isfinite(check.value).all():
        raise AssertionError("Submission round trip failed.")
    raw = test[["image_id", "width", "height"]].copy()
    raw["Fovea_X"], raw["Fovea_Y"] = xy[:, 0], xy[:, 1]
    raw.to_csv(out / "test_predictions.csv", index=False)
    contact_sheet(data, test, out / "test_predictions.png", prediction)


def run(
    data_dir=None,
    output_dir="task3_baseline_seed42",
    phase="validate",
    seed=42,
    folds=5,
    fold=0,
    epochs=80,
    image_size=320,
    batch_size=8,
    pretrained=True,
    weights=None,
    allow_cpu=False,
    lr_backbone=1e-4,
    lr_decoder=1e-3,
    weight_decay=1e-4,
    sigma=1.5,
):
    data, out = find_data(data_dir), Path(output_dir).resolve()
    if phase not in {"audit", "validate", "full"}:
        raise ValueError(f"Unknown phase: {phase}")
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        seed=seed,
        folds=folds,
        fold=fold,
        epochs=epochs,
        image_size=image_size,
        batch_size=batch_size,
        pretrained=pretrained,
        weights=str(weights) if weights else None,
        lr_backbone=lr_backbone,
        lr_decoder=lr_decoder,
        weight_decay=weight_decay,
        sigma=sigma,
    )
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError(
            "Existing output uses a different protocol; choose a new output directory."
        )
    save_json(protocol_path, protocol)
    manifest, train = audit(data, out, seed, folds, fold)
    if phase == "audit":
        return out
    if image_size < 64 or image_size % 32 or epochs < 1:
        raise ValueError("image_size must be a positive multiple of 32 >=64; epochs >=1.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not allow_cpu:
        raise RuntimeError(
            "CUDA unavailable; explicitly set allow_cpu=True for a small smoke check."
        )
    development, valid = train[train.fold != fold], train[train.fold == fold]
    fingerprint = hashlib.sha256(train.to_csv(index=False).encode()).hexdigest()
    config = dict(
        seed=seed,
        folds=folds,
        fold=fold,
        epochs=epochs,
        image_size=image_size,
        batch_size=batch_size,
        pretrained=pretrained,
        weights=str(weights) if weights else None,
        lr_backbone=lr_backbone,
        lr_decoder=lr_decoder,
        weight_decay=weight_decay,
        sigma=sigma,
        dataset_fingerprint=fingerprint,
        selection_metric="mean_distance_px",
        architecture="resnet18_unet_stride4",
        decoder="peak_local_expectation_radius3",
        augmentation="hflip_p0.5_affine10deg_scale0.92-1.08_translate0.035_colorjitter",
        metric_status="provisional_until_official_evaluation_is_provided",
    )
    save_json(
        out / "environment.json",
        dict(
            torch=torch.__version__,
            numpy=np.__version__,
            device=str(device),
            gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
        ),
    )
    validation_dir = out / f"fold{fold}"
    model, best_epoch = train_model(data, development, valid, validation_dir, config, device)
    development_loader = DataLoader(
        FoveaDataset(data, development, image_size), batch_size=batch_size, shuffle=False
    )
    development_prediction = predict(model, development_loader, device)
    development_metrics = error_metrics(
        development_prediction,
        development[["nx", "ny"]].to_numpy(),
        development[["width", "height"]].to_numpy(),
    )
    save_json(
        out / "development_train_metrics.json",
        dict(
            **development_metrics,
            evaluation="eval mode, no augmentation, no TTA, same selected checkpoint as validation",
        ),
    )
    loader = DataLoader(FoveaDataset(data, valid, image_size), batch_size=batch_size, shuffle=False)
    scores, predictions = coordinate_baselines(development, valid)
    for name, use_tta in [("heatmap", False), ("heatmap_flip_tta", True)]:
        prediction = predict(model, loader, device, use_tta)
        scores.append(
            dict(
                method=name,
                **error_metrics(
                    prediction,
                    valid[["nx", "ny"]].to_numpy(),
                    valid[["width", "height"]].to_numpy(),
                ),
            )
        )
        predictions[name] = prediction
    table = pd.DataFrame(scores)
    table.to_csv(out / "validation_comparison.csv", index=False)
    chosen = min(
        ["heatmap", "heatmap_flip_tta"],
        key=lambda name: float(table.set_index("method").loc[name, "mean_distance_px"]),
    )
    selected = predictions[chosen]
    per_image = valid[["image_id", "width", "height", "x", "y"]].copy()
    for name, pred in predictions.items():
        xy = pred * valid[["width", "height"]].to_numpy()
        per_image[f"{name}_x"], per_image[f"{name}_y"] = xy[:, 0], xy[:, 1]
        per_image[f"{name}_distance_px"] = np.linalg.norm(xy - valid[["x", "y"]].to_numpy(), axis=1)
    per_image.to_csv(out / "validation_predictions.csv", index=False)
    contact_sheet(data, valid, out / "validation_predictions.png", selected)
    selection = dict(
        best_epoch=best_epoch,
        prediction=chosen,
        validation_n=len(valid),
        selection_metric="mean_distance_px",
        limitation="One holdout used for epoch and TTA selection; these are development scores, not independent test scores.",
        official_metric_status="not_provided",
    )
    save_json(out / "selection.json", selection)
    print(table.to_string(index=False), flush=True)
    if phase == "full":
        # Reinitialize from ImageNet and train on all 80 images for the holdout-selected epoch count.
        full_config = dict(
            config,
            epochs=best_epoch,
            schedule_epochs=config["epochs"],
            fold="full",
            selected_from_fold=fold,
        )
        model, _ = train_model(data, train, None, out / "fulltrain", full_config, device)
        test = manifest[manifest.subset == "test"].copy()
        loader = DataLoader(
            FoveaDataset(data, test, image_size), batch_size=batch_size, shuffle=False
        )
        prediction = predict(model, loader, device, chosen.endswith("tta"))
        export_submission(data, test, prediction, out)
        save_json(
            out / "submission_provenance.json",
            dict(
                **selection,
                fulltrain_images=len(train),
                fulltrain_epochs=best_epoch,
                test_images=len(test),
                cosine_schedule_epochs=config["epochs"],
                submitted_to_competition=False,
            ),
        )
    return out


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir", default="task3_baseline_seed42")
    parser.add_argument("--phase", choices=["audit", "validate", "full"], default="validate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--weights")
    parser.add_argument("--allow-cpu", action="store_true")
    args = vars(parser.parse_args())
    args["pretrained"] = not args.pop("no_pretrained")
    run(**args)


if __name__ == "__main__":
    cli()
