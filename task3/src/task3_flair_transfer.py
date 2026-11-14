"""FLAIR visual initialization versus ImageNet: independent full-image localization."""

from __future__ import annotations
import argparse
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
import urllib.request
import zipfile

if __name__ == "__main__" and not any(x in sys.argv for x in ("--train", "--download-only")):
    raise SystemExit(
        "Use --train to explicitly run training, or --download-only to fetch the two source checkpoints."
    )

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from task3_pipeline import (
    DecoderBlock,
    assign_folds,
    decode_heatmap,
    find_data,
    heatmap_loss,
    load_manifest,
    save_json,
    seed_everything,
    training_lock,
)
from task3_kaggle_cv import (
    atomic_csv,
    atomic_torch_save,
    average_test_folds,
    metrics_from_rows,
    prediction_rows,
    read_rows,
    restore_backup,
    stable_hash,
    validate_oof,
    validate_test_rows,
    write_submission,
)

TRANSFER_VERSION = "flair-transfer-v1"
RECIPES = ("imagenet", "flair")
SOURCE_NAMES = ("task3_pipeline.py", "task3_kaggle_cv.py", "task3_flair_transfer.py")
WEIGHT_SPECS = {
    "imagenet": dict(
        filename="resnet50-0676ba61.pth",
        url="https://download.pytorch.org/models/resnet50-0676ba61.pth",
        sha256_prefix="0676ba61",
        normalization="imagenet",
        pretraining="ImageNet-1K IMAGENET1K_V1; no fovea localization checkpoint",
    ),
    "flair": dict(
        filename="flair_model.safetensors",
        url="https://huggingface.co/jusiro2/FLAIR/resolve/main/model.safetensors",
        sha256_prefix="050334cd934fa3126435c202c41f493ca222fc3fe3ea9c50cf21f67c792a1440",
        normalization="zero_one",
        pretraining="Author FLAIR base image-text/categorical pretraining; visual encoder only; no fovea-specific decoder",
    ),
}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_weights(output, explicit=None):
    folder = Path(output) / "_pretrained"
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    sources = {}
    for recipe, spec in WEIGHT_SPECS.items():
        path = (
            Path(explicit[recipe])
            if explicit and explicit.get(recipe)
            else folder / spec["filename"]
        )
        if not path.is_file():
            if explicit and explicit.get(recipe):
                raise FileNotFoundError(path)
            temporary = path.with_suffix(path.suffix + ".part")
            print(f'Downloading {recipe}: {spec["url"]}', flush=True)
            request = urllib.request.Request(
                spec["url"], headers={"User-Agent": "Task3 research weight downloader"}
            )
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                temporary.open("wb") as stream,
            ):
                if "html" in response.headers.get("Content-Type", "").lower():
                    raise ValueError("Weight endpoint returned HTML.")
                count = 0
                for chunk in iter(lambda: response.read(8 * 1024**2), b""):
                    stream.write(chunk)
                    count += len(chunk)
                print(f"{recipe}: downloaded {count/1024**2:.1f} MiB", flush=True)
            if not file_sha(temporary).startswith(spec["sha256_prefix"]):
                raise ValueError("Downloaded checkpoint checksum differs.")
            temporary.replace(path)
        digest = file_sha(path)
        if not digest.startswith(spec["sha256_prefix"]):
            raise ValueError(f"Checkpoint checksum differs: {path}")
        paths[recipe] = path
        sources[recipe] = dict(
            spec, path=str(path.resolve()), sha256=digest, size_bytes=path.stat().st_size
        )
    save_json(Path(output) / "pretraining_sources.json", sources)
    return paths


def require_safe_open():
    try:
        from safetensors import safe_open
    except ImportError:
        base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
        for folder in (
            base.parent / "task2/.model_deps",
            base.parent.parent / "task2/.model_deps",
            base / ".model_deps",
            base.parent / ".model_deps",
        ):
            if (folder / "safetensors").is_dir():
                sys.path.insert(0, str(folder))
                break
        from safetensors import safe_open
    return safe_open


def backbone_state(recipe, path):
    if recipe == "imagenet":
        state = torch.load(path, map_location="cpu", weights_only=True)
        if set(k for k in state if k.startswith("fc.")) != {"fc.weight", "fc.bias"}:
            raise ValueError("Unexpected ImageNet classifier keys.")
        return {k: v for k, v in state.items() if not k.startswith("fc.")}
    if recipe != "flair":
        raise ValueError("Unknown recipe.")
    safe_open = require_safe_open()
    prefix = "vision_model.model."
    with safe_open(path, framework="pt", device="cpu") as tensors:
        state = {
            k[len(prefix) :]: tensors.get_tensor(k) for k in tensors.keys() if k.startswith(prefix)
        }
    if len(state) != 318 or "conv1.weight" not in state or any(k.startswith("fc.") for k in state):
        raise ValueError("Unexpected FLAIR visual encoder key layout.")
    return state


class TransferHeatmap(nn.Module):
    def __init__(self, recipe=None, weights=None):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Identity()
        if weights is not None:
            self.backbone.load_state_dict(backbone_state(recipe, weights), strict=True)
        self.d3 = DecoderBlock(2048, 1024, 128)
        self.d2 = DecoderBlock(128, 512, 64)
        self.d1 = DecoderBlock(64, 256, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def train(self, mode=True):
        super().train(mode)
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, image):
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(image))))
        c1 = b.layer1(x)
        c2 = b.layer2(c1)
        c3 = b.layer3(c2)
        c4 = b.layer4(c3)
        return self.head(self.d1(self.d2(self.d3(c4, c3), c2), c1))[:, 0]


def set_trainable(model, epoch, warm_epochs=5):
    for name, p in model.backbone.named_parameters():
        p.requires_grad_(epoch > warm_epochs and name.startswith(("layer3.", "layer4.")))


def canvas_size(width, height, size=512):
    if min(width, height, size) <= 0:
        raise ValueError("Nonpositive image dimension.")
    scale = size / max(width, height)
    return max(1, int(width * scale)), max(1, int(height * scale))


def to_canvas(point, width, height, size=512):
    nw, nh = canvas_size(width, height, size)
    return float(point[0]) / width * nw / size, float(point[1]) / height * nh / size


def from_canvas(point, width, height, size=512):
    """Undo canvas padding and return original-image NORMALIZED coordinates."""
    nw, nh = canvas_size(width, height, size)
    return float(point[0]) * size / nw, float(point[1]) * size / nh


class CanvasCache:
    def __init__(self, data):
        self.data = Path(data)
        self.images = {}

    def get(self, row):
        key = int(row["image_id"])
        if key not in self.images:
            with Image.open(self.data / row["subset"] / row["filename"]) as im:
                if im.size != (int(row["width"]), int(row["height"])):
                    raise ValueError("Image dimensions changed.")
                self.images[key] = im.convert("RGB").resize(
                    canvas_size(*im.size), Image.Resampling.BILINEAR
                )
        return self.images[key].copy()


class CanvasDataset(Dataset):
    def __init__(self, records, cache, cfg, training):
        self.records, self.cache, self.cfg, self.training = list(records), cache, cfg, training
        ids = {int(r["image_id"]) for r in self.records}
        if len(ids) != len(self.records):
            raise ValueError("Repeated image.")
        if training and (ids != set(cfg["train_ids"]) or ids & set(cfg["forbidden_ids"])):
            raise ValueError("Training images differ from phase.")
        operations = [transforms.ToTensor()]
        if cfg["recipe"] == "imagenet":
            operations.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        self.tensor = transforms.Compose(operations)
        self.jitter = transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.08, hue=0.01
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        image = self.cache.get(row)
        nw, nh = image.size
        point = np.array(
            (
                to_canvas((row["x"], row["y"]), int(row["width"]), int(row["height"]))
                if self.training
                else (nw / 1024, nh / 1024)
            ),
            dtype=np.float32,
        )
        if self.training:
            point *= 512
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                point[0] = nw - point[0]
            angle, scale = math.radians(random.uniform(-8, 8)), random.uniform(0.95, 1.05)
            a, b = scale * math.cos(angle), -scale * math.sin(angle)
            tx, ty = random.uniform(-0.02, 0.02) * nw, random.uniform(-0.02, 0.02) * nh
            matrix = np.array(
                [
                    [a, b, nw / 2 + tx - a * nw / 2 - b * nh / 2],
                    [-b, a, nh / 2 + ty + b * nw / 2 - a * nh / 2],
                    [0.0, 0.0, 1.0],
                ]
            )
            point = ((matrix @ [point[0], point[1], 1.0])[:2] / 512).astype(np.float32)
            if not (0 < point[0] < nw / 512 and 0 < point[1] < nh / 512):
                raise ValueError("Augmentation moved target outside content.")
            image = image.transform(
                (nw, nh),
                Image.Transform.AFFINE,
                np.linalg.inv(matrix)[:2].reshape(-1),
                resample=Image.Resampling.BILINEAR,
                fillcolor=(0, 0, 0),
            )
            image = self.jitter(image)
        canvas = Image.new("RGB", (512, 512))
        canvas.paste(image, (0, 0))
        return (
            self.tensor(canvas),
            torch.from_numpy(point),
            torch.tensor([nw, nh], dtype=torch.float32),
        )


def mask_canvas(logits, content):
    h, w = logits.shape[-2:]
    bounds = content.to(logits.device)
    xx = (torch.arange(w, device=logits.device) + 0.5) * (512 / w)
    yy = (torch.arange(h, device=logits.device) + 0.5) * (512 / h)
    valid = (xx[None, None, :] < bounds[:, 0, None, None]) & (
        yy[None, :, None] < bounds[:, 1, None, None]
    )
    return logits.float().masked_fill(~valid, -1e4)


def domain_partition(train, fold):
    dev = train[train.fold != fold].copy()
    outer = train[train.fold == fold].copy()
    assigned = assign_folds(dev, 7000 + fold, 5)
    held_ids = set(assigned.loc[assigned.fold == 0, "image_id"])
    fit = dev[~dev.image_id.isin(held_ids)].copy()
    held = dev[dev.image_id.isin(held_ids)].copy()
    groups = [set(f.pixel_sha256) for f in (fit, held, outer)]
    if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))) or len(fit) + len(
        held
    ) != len(dev):
        raise ValueError("Group split differs.")
    return dev, fit, held, outer


def phase_config(protocol, fold, recipe, role, training, held, outer_ids, epochs):
    cfg = dict(
        version=TRANSFER_VERSION,
        source_sha256=protocol["source_sha256"],
        data_fingerprint=protocol["data_fingerprint"],
        recipe=recipe,
        role=role,
        outer_fold=fold,
        seed=(52000 if role == "inner" else 53000) + fold,
        train_ids=list(map(int, training.image_id)),
        evaluation_ids=list(map(int, held.image_id)) if held is not None else [],
        forbidden_ids=list(outer_ids),
        epochs=int(epochs),
        schedule_epochs=protocol["inner_epochs"],
        warm_epochs=5,
        image_size=512,
        batch_size=protocol["batch_size"],
    )
    validate_phase(cfg)
    return cfg


def validate_phase(cfg):
    sets = [set(cfg[k]) for k in ("train_ids", "evaluation_ids", "forbidden_ids")]
    if (
        not sets[0]
        or any(
            len(s) != len(cfg[k])
            for s, k in zip(sets, ("train_ids", "evaluation_ids", "forbidden_ids"))
        )
        or any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2)))
    ):
        raise ValueError("Phase IDs overlap.")
    if (
        cfg["recipe"] not in RECIPES
        or cfg["role"] not in ("inner", "outer")
        or bool(sets[1]) != (cfg["role"] == "inner")
    ):
        raise ValueError("Phase role differs.")
    if (
        not 1 <= cfg["epochs"] <= cfg["schedule_epochs"]
        or cfg["image_size"] != 512
        or cfg["warm_epochs"] != 5
    ):
        raise ValueError("Phase settings differ.")


@torch.inference_mode()
def predict_transfer(model, loader, fold):
    model.eval()
    points = []
    for images, _, content in loader:
        with torch.autocast(device_type="cuda"):
            logits = model(images.to("cuda:0"))
        if logits.shape != (len(images), 128, 128) or not torch.isfinite(logits).all():
            raise RuntimeError("Invalid heatmap.")
        decoded = decode_heatmap(mask_canvas(logits, content), radius=3).cpu().tolist()
        records = loader.dataset.records[len(points) : len(points) + len(images)]
        points.extend(
            from_canvas(p, int(r["width"]), int(r["height"])) for p, r in zip(decoded, records)
        )
    if len(points) != len(loader.dataset):
        raise ValueError("Prediction count differs.")
    records = loader.dataset.records
    labeled = all(r["subset"] == "train" for r in records)
    return prediction_rows(records, points, fold, labeled)


def head_sha(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if not name.startswith("backbone."):
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def gpu_preflight(paths, output):
    results = {}
    head_hashes = []
    for recipe in RECIPES:
        seed_everything(54000)
        model = TransferHeatmap(recipe, paths[recipe]).to("cuda:0")
        head_hashes.append(head_sha(model))
        image = torch.randn(1, 3, 512, 512, device="cuda:0")
        target = torch.tensor([[0.5, 0.3]], device="cuda:0")
        content = torch.tensor([[512.0, 342.0]], device="cuda:0")
        checks = {}
        for epoch in (1, 6):
            set_trainable(model, epoch)
            model.train()
            model.zero_grad(set_to_none=True)
            logits = model(image)
            loss = heatmap_loss(mask_canvas(logits, content), target, 2.4)
            loss.backward()
            if logits.shape != (1, 128, 128) or not torch.isfinite(loss):
                raise RuntimeError("GPU preflight shape/loss failure.")
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("GPU preflight non-finite gradient.")
            if epoch == 1 and any(p.grad is not None for p in model.backbone.parameters()):
                raise RuntimeError("Warm phase updated backbone.")
            if epoch == 6:
                for name in ("layer3.0.conv1.weight", "layer4.0.conv1.weight"):
                    p = dict(model.backbone.named_parameters())[name]
                    if p.grad is None or not p.grad.abs().sum() > 0:
                        raise RuntimeError("Partial fine-tune has no gradient.")
            if any(m.training for m in model.backbone.modules() if isinstance(m, nn.BatchNorm2d)):
                raise RuntimeError("Backbone BN is not frozen.")
            checks[str(epoch)] = float(loss.detach())
        results[recipe] = checks
        del model, image, target, content, logits, loss
        gc.collect()
        torch.cuda.empty_cache()
    if len(set(head_hashes)) != 1:
        raise RuntimeError("The paired random localization heads differ.")
    save_json(
        Path(output) / "gpu_preflight.json",
        dict(passed=True, head_sha256=head_hashes[0], losses=results),
    )


def train_phase(records, cache, cfg, folder, weight_path, evaluator=None):
    validate_phase(cfg)
    if (evaluator is not None) != (cfg["role"] == "inner"):
        raise ValueError("Outer evaluator is forbidden.")
    folder.mkdir(parents=True, exist_ok=True)
    signature = stable_hash(cfg)
    if (folder / "config.json").exists() and json.loads(
        (folder / "config.json").read_text()
    ) != cfg:
        raise ValueError("Configuration differs.")
    save_json(folder / "config.json", cfg)
    checkpoint = folder / ("best.pt" if evaluator else "final.pt")
    if (folder / "phase_complete.json").exists():
        done = json.loads((folder / "phase_complete.json").read_text())
        if done["signature"] != signature or not checkpoint.is_file():
            raise ValueError("Completed phase requires full weights.")
        return done
    seed_everything(cfg["seed"])
    last = folder / "last.pt"
    model = TransferHeatmap(cfg["recipe"], None if last.exists() else weight_path).to("cuda:0")
    save_json(folder / "initial_head.json", dict(seed=cfg["seed"], sha256=head_sha(model)))
    ids = {id(p) for p in model.backbone.parameters()}
    optimizer = torch.optim.AdamW(
        [
            dict(params=model.backbone.parameters(), lr=1e-5),
            dict(params=[p for p in model.parameters() if id(p) not in ids], lr=5e-4),
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.05 + 0.95 * (1 + math.cos(math.pi * step / cfg["schedule_epochs"])) / 2,
    )
    scaler = torch.amp.GradScaler("cuda")
    start = 0
    history = []
    best = dict(score=float("inf"), epoch=0)
    best_model = None
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["signature"] != signature:
            raise ValueError("Resume signature differs.")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start, history, best, best_model = (
            state["epoch"],
            state["history"],
            state["best"],
            state["best_model"],
        )
        del state
        if evaluator and best["epoch"]:
            atomic_torch_save(
                checkpoint,
                dict(
                    model=best_model,
                    signature=signature,
                    epoch=best["epoch"],
                    config=cfg,
                    selection=best,
                ),
            )
    loader = DataLoader(
        CanvasDataset(records, cache, cfg, True),
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    for epoch in range(start + 1, cfg["epochs"] + 1):
        seed_everything(cfg["seed"] + 1000 * epoch)
        torch.set_num_threads(2)
        set_trainable(model, epoch)
        model.train()
        begin = time.perf_counter()
        total = count = skips = 0
        for images, points, content in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda"):
                logits = model(images.to("cuda:0"))
                loss = heatmap_loss(mask_canvas(logits, content), points.to("cuda:0"), 2.4)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skips += int(scaler.get_scale() < old_scale)
            total += float(loss.detach()) * len(images)
            count += len(images)
        if skips == len(loader) and history and history[-1]["amp_skips"] == len(loader):
            raise RuntimeError("All optimizer updates skipped in consecutive epochs.")
        row = dict(
            epoch=epoch,
            train_ce=total / count,
            amp_skips=skips,
            backbone_stage="head_only" if epoch <= 5 else "layer3_layer4",
            lr_backbone=optimizer.param_groups[0]["lr"],
        )
        if evaluator:
            metrics = evaluator(model)
            score = metrics["rmse_coordinate_px"]
            row.update(metrics)
            if not math.isfinite(score):
                raise RuntimeError("Non-finite validation RMSE.")
            if score < best["score"]:
                best = dict(score=score, epoch=epoch)
                best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                atomic_torch_save(
                    checkpoint,
                    dict(
                        model=best_model,
                        signature=signature,
                        epoch=epoch,
                        config=cfg,
                        selection=best,
                    ),
                )
        scheduler.step()
        row["seconds"] = time.perf_counter() - begin
        history.append(row)
        if epoch % 5 == 0 or epoch == cfg["epochs"]:
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
                    best=best,
                    best_model=best_model,
                ),
            )
        atomic_csv(folder / "history.csv", history)
        save_json(folder / "status.json", dict(state="training", epoch=epoch, signature=signature))
        print(
            f'Fold{cfg["outer_fold"]} {cfg["role"]}/{cfg["recipe"]} {epoch:02}/{cfg["epochs"]} CE={total/count:.4f}'
            + (f' RMSE={score:.3f} best={best["score"]:.3f}' if evaluator else "")
            + f' {row["seconds"]:.1f}s',
            flush=True,
        )
    if not evaluator:
        atomic_torch_save(
            checkpoint,
            dict(model=model.state_dict(), signature=signature, epoch=cfg["epochs"], config=cfg),
        )
    done = dict(signature=signature, epochs=cfg["epochs"], best=best if evaluator else {})
    save_json(folder / "phase_complete.json", done)
    if last.exists():
        last.unlink()
    del model, optimizer, scaler, loader, best_model
    gc.collect()
    torch.cuda.empty_cache()
    return done


def fold_design(prepared, fold):
    dev, fit, held, outer = domain_partition(prepared["train"], fold)
    design = dict(
        protocol=prepared["protocol"],
        fold=fold,
        fit_ids=list(map(int, fit.image_id)),
        inner_val_ids=list(map(int, held.image_id)),
        outer_ids=list(map(int, outer.image_id)),
    )
    return dev, fit, held, outer, design


def run_fold(prepared, fold, paths, cache):
    dev, fit, held, outer, design = fold_design(prepared, fold)
    job = prepared["output"] / f"fold_{fold}"
    job.mkdir(parents=True, exist_ok=True)
    signature = stable_hash(design)
    if (job / "fold_complete.json").exists():
        done = json.loads((job / "fold_complete.json").read_text())
        if done["signature"] != signature or any(
            not (job / p).is_file() for p in done["required_models"]
        ):
            raise ValueError("Completed fold needs full matching weights.")
        return
    save_json(job / "design.json", design)
    decisions = {}
    atomic_csv(
        job / "split_roles.csv",
        [
            dict(image_id=int(r["image_id"]), pixel_sha256=r["pixel_sha256"], role=role)
            for role, frame in [
                ("inner_train", fit),
                ("inner_validation", held),
                ("outer_validation", outer),
            ]
            for r in frame.to_dict("records")
        ],
    )
    for recipe in RECIPES:
        cfg = phase_config(
            prepared["protocol"],
            fold,
            recipe,
            "inner",
            fit,
            held,
            design["outer_ids"],
            prepared["protocol"]["inner_epochs"],
        )
        validation = DataLoader(
            CanvasDataset(held.to_dict("records"), cache, cfg, False),
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=0,
        )

        def evaluate(model):
            return metrics_from_rows(predict_transfer(model, validation, -1))

        done = train_phase(
            fit.to_dict("records"), cache, cfg, job / f"inner_{recipe}", paths[recipe], evaluate
        )
        decisions[recipe] = done["best"]
        del validation
        cfg = phase_config(
            prepared["protocol"],
            fold,
            recipe,
            "outer",
            dev,
            None,
            design["outer_ids"],
            done["best"]["epoch"],
        )
        folder = job / f"outer_{recipe}"
        train_phase(dev.to_dict("records"), cache, cfg, folder, paths[recipe])
        state = torch.load(folder / "final.pt", map_location="cpu", weights_only=True)
        if state["signature"] != stable_hash(cfg):
            raise ValueError("Inference signature differs.")
        model = TransferHeatmap().to("cuda:0")
        model.load_state_dict(state["model"], strict=True)
        del state
        for subset, frame in [("validation", outer), ("test", prepared["test"])]:
            loader = DataLoader(
                CanvasDataset(frame.to_dict("records"), cache, cfg, False),
                batch_size=cfg["batch_size"],
                shuffle=False,
                num_workers=0,
            )
            atomic_csv(job / f"{subset}_{recipe}.csv", predict_transfer(model, loader, fold))
            del loader
        del model
        gc.collect()
        torch.cuda.empty_cache()
    required = sorted(p.relative_to(job).as_posix() for p in job.rglob("*.pt"))
    save_json(
        job / "fold_complete.json",
        dict(
            signature=signature,
            decisions=decisions,
            required_models=required,
            predictions_complete=True,
            outer_labels_used_for_epoch_selection=False,
        ),
    )


def reports(prepared):
    output = prepared["output"]
    folds = [f for f in range(5) if (output / f"fold_{f}/fold_complete.json").exists()]
    selection = dict(
        complete=len(folds) == 5 and not prepared["protocol"]["smoke_test"],
        completed_folds=folds,
        smoke_test=prepared["protocol"]["smoke_test"],
        metric="rmse_coordinate_px",
        candidate_rule="Predeclared FLAIR single-backbone five-fold average",
        selected_by_outer_scores=False,
        candidate_public_score=None,
        limitation="Pretraining image overlap not fully audited. Initialization recipes also use their source-specific normalization. The same80 images have previously been used for development.",
    )
    if not selection["complete"]:
        save_json(output / "selection.json", selection)
        return False
    records, testing = prepared["train"].to_dict("records"), prepared["test"].to_dict("records")
    oof = {}
    tests = {}
    comparisons = []
    for recipe in RECIPES:
        preds = []
        test_folds = {}
        for f in range(5):
            job = output / f"fold_{f}"
            done = json.loads((job / "fold_complete.json").read_text())
            design = fold_design(prepared, f)[-1]
            if (
                done["signature"] != stable_hash(design)
                or not done["predictions_complete"]
                or done["outer_labels_used_for_epoch_selection"]
            ):
                raise ValueError("Fold provenance differs.")
            outer = prepared["train"][prepared["train"].fold == f].to_dict("records")
            val = validate_oof(read_rows(job / f"validation_{recipe}.csv"), outer)
            validate_test_rows(val, outer, f)
            preds.extend(val)
            test_folds[f] = read_rows(job / f"test_{recipe}.csv")
            validate_test_rows(test_folds[f], testing, f)
        oof[recipe] = validate_oof(preds, records)
        tests[recipe] = average_test_folds(test_folds, testing)
        atomic_csv(output / f"oof_{recipe}.csv", oof[recipe])
        atomic_csv(output / f"test_{recipe}.csv", tests[recipe])
        write_submission(
            prepared["data"] / "sample_submission.csv",
            tests[recipe],
            output / f"submission_{recipe}_5fold.csv",
        )
        comparisons.append(dict(model=recipe, **metrics_from_rows(oof[recipe])))
    # A fixed two-model equal blend is diagnostic; it never replaces the declared FLAIR candidate automatically.
    for labeled, groups, target in [(True, oof, "oof_equal.csv"), (False, tests, "test_equal.csv")]:
        maps = {r: {int(x["image_id"]): x for x in group} for r, group in groups.items()}
        blended = []
        for key in sorted(maps["flair"]):
            row = dict(maps["flair"][key])
            x, y = [
                statistics.mean(float(maps[r][key][f"Fovea_{a}"]) for r in RECIPES)
                for a in ("X", "Y")
            ]
            row.update(Fovea_X=x, Fovea_Y=y, nx=x / int(row["width"]), ny=y / int(row["height"]))
            if labeled:
                dx, dy = x - float(row["gt_x"]), y - float(row["gt_y"])
                row.update(
                    dx=dx, dy=dy, distance_px=math.hypot(dx, dy), squared_error=dx * dx + dy * dy
                )
            blended.append(row)
        atomic_csv(output / target, blended)
        if labeled:
            comparisons.append(dict(model="fixed_equal", **metrics_from_rows(blended)))
        else:
            write_submission(
                prepared["data"] / "sample_submission.csv",
                blended,
                output / "submission_equal_5fold.csv",
            )
    atomic_csv(output / "comparison.csv", comparisons)
    shutil.copy2(output / "submission_flair_5fold.csv", output / "submission_flair_candidate.csv")
    selection.update(candidate="submission_flair_candidate.csv", public_best_reference=12.31624)
    save_json(output / "selection.json", selection)
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)
    return True


def archive_results(output, full=False):
    target = (
        output.parent / "task3_flair_transfer_full_backup.zip"
        if full
        else output / "task3_flair_transfer_results_light.zip"
    )
    temporary = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True
    ) as archive:
        for p in sorted(output.rglob("*")):
            if (
                not p.is_file()
                or p.is_symlink()
                or "_pretrained" in p.relative_to(output).parts
                or p.suffix in (".zip", ".tmp", ".part")
            ):
                continue
            if full or p.suffix in (".csv", ".json", ".md"):
                archive.write(p, p.relative_to(output))
    temporary.replace(target)
    if full:
        with zipfile.ZipFile(target) as archive:
            names = set(archive.namelist())
            for f in range(5):
                done = json.loads((output / f"fold_{f}/fold_complete.json").read_text())
                if any(f"fold_{f}/{name}" not in names for name in done["required_models"]):
                    raise RuntimeError("Backup missing model.")
    return target


def prepare(data_dir=None, output_dir=None, resume_dir=None, smoke=False, batch_size=4):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; no CPU training fallback.")
    data, output = find_data(data_dir), Path(output_dir).resolve()
    manifest = load_manifest(data)
    train = assign_folds(manifest[manifest.subset == "train"], 42, 5)
    test = manifest[manifest.subset == "test"].copy()
    if len(train) != 80 or len(test) != 20:
        raise ValueError("Expected80 training and20 test images.")
    fingerprint = []
    for row in manifest.to_dict("records"):
        r = {
            k: row[k] for k in ("image_id", "subset", "filename", "width", "height", "pixel_sha256")
        }
        if row["subset"] == "train":
            r.update(x=float(row["x"]), y=float(row["y"]))
        fingerprint.append(r)
    hashes = (
        dict(TRANSFER_SOURCE_SHA256)
        if "TRANSFER_SOURCE_SHA256" in globals()
        else {name: file_sha(Path(__file__).resolve().parent / name) for name in SOURCE_NAMES}
    )
    protocol = dict(
        version=TRANSFER_VERSION,
        source_sha256=hashes,
        data_fingerprint=stable_hash(fingerprint),
        weight_specs=WEIGHT_SPECS,
        inner_epochs=2 if smoke else 40,
        warm_epochs=5,
        batch_size=int(batch_size),
        image_size=512,
        geometry="aspect-preserving top-left canvas; padded heatmap cells masked",
        adaptation="head warmup5; then layer3/layer4 + decoder; backbone BN frozen",
        candidate="flair",
        smoke_test=bool(smoke),
        task_specific_pretraining=False,
        pretraining_image_overlap_audited=False,
    )
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 9 * 1024**3:
        raise RuntimeError("Need9 GiB free for checkpoints and backup.")
    restore_backup(resume_dir, output, protocol)
    if (output / "protocol.json").exists() and json.loads(
        (output / "protocol.json").read_text()
    ) != protocol:
        raise ValueError("Output protocol differs; choose a new directory.")
    save_json(output / "protocol.json", protocol)
    train.to_csv(output / "split.csv", index=False)
    manifest.to_csv(output / "manifest.csv", index=False)
    save_json(
        output / "environment.json",
        dict(
            torch=str(torch.__version__),
            numpy=np.__version__,
            gpu=torch.cuda.get_device_name(0),
            device="cuda:0",
            cpu_threads=2,
        ),
    )
    return dict(data=data, output=output, train=train, test=test, protocol=protocol)


def run(prepared, run_folds=(0, 1, 2, 3, 4), weight_paths=None, full_backup=True):
    if (
        not run_folds
        or len(set(run_folds)) != len(run_folds)
        or not set(run_folds) <= set(range(5))
    ):
        raise ValueError("Invalid fold schedule.")
    output = prepared["output"]
    torch.set_num_threads(2)
    cache = CanvasCache(prepared["data"])
    pending = any(not (output / f"fold_{f}/fold_complete.json").exists() for f in run_folds)
    with training_lock(output):
        try:
            paths = get_weights(output, weight_paths) if pending else {}
            if pending:
                gpu_preflight(paths, output)
            for f in run_folds:
                save_json(
                    output / "status.json", dict(state="running", fold=f, updated_unix=time.time())
                )
                run_fold(prepared, f, paths, cache)
                reports(prepared)
                archive_results(output)
            complete = reports(prepared)
            save_json(
                output / "status.json",
                dict(
                    state="completed" if complete else "partial_completed", updated_unix=time.time()
                ),
            )
            archive_results(output)
            if complete and full_backup:
                print("Full backup:", archive_results(output, True), flush=True)
        except BaseException as error:
            save_json(
                output / "status.json",
                dict(
                    state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    error=str(error),
                    updated_unix=time.time(),
                ),
            )
            archive_results(output)
            raise
    return output


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--resume-dir")
    parser.add_argument("--weights-dir")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--no-full-backup", action="store_true")
    args = parser.parse_args()
    if args.download_only:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        get_weights(out)
        return
    prepared = prepare(args.data_dir, args.output_dir, args.resume_dir, args.smoke, args.batch_size)
    paths = (
        {r: Path(args.weights_dir) / s["filename"] for r, s in WEIGHT_SPECS.items()}
        if args.weights_dir
        else None
    )
    run(
        prepared,
        [0] if args.smoke else args.folds,
        weight_paths=paths,
        full_backup=not args.no_full_backup,
    )


if __name__ == "__main__":
    cli()
