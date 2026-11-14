"""Paired local/global-context refinement, using frozen coarse-prediction caches."""

from __future__ import annotations
import base64
import csv
import gc
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import random
import shutil
import statistics
import time
import zipfile

if __name__ == "__main__" and not Path("/kaggle/working").is_dir():
    raise SystemExit("Use the Kaggle Notebook. No local training was started.")

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
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
from task3_kaggle_refine import (
    OriginalImageCache,
    blend_prediction_rows,
    crop_geometry,
    make_refinement_row,
    point_to_patch,
    validate_coarse_bundle,
)
from task3_refine_diagnostic_v3 import (
    center_offset,
    coarse_log_prior,
    identity_rows,
    inner_mode_metrics,
    inner_partition,
    raw_prediction_rows,
    selected_inner_epoch,
    validate_inner_centers,
)

CONTEXT_VERSION = "context-refine-v4"
CONTEXT_ARMS = ("local", "context")
V3_SOURCE_SHA = "72387a1d8c0636767aa3881f636c4241136ce52367d49ebd8301fabeb9b0cf26"


def unpack_context_inputs(encoded, digest, output):
    payload = base64.b64decode(encoded, validate=True)
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("Embedded cache checksum differs.")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Repeated cache member.")
        for item in archive.infolist():
            path = PurePosixPath(item.filename)
            target = (output / item.filename).resolve()
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in item.filename
                or not target.is_relative_to(output)
            ):
                raise ValueError("Cache path escapes output.")
            if item.file_size > 5000000 or path.suffix not in (".json", ".csv"):
                raise ValueError("Unexpected cache member.")
        for item in archive.infolist():
            target = output / item.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            data = archive.read(item)
            if target.exists() and target.read_bytes() != data:
                raise ValueError("Existing cache differs; use a clean cache directory.")
            if not target.exists():
                target.write_bytes(data)
    return output


def validate_context_centers(source, train, coarse_fingerprint):
    source = Path(source)
    old_protocol = json.loads((source / "inner/protocol.json").read_text())
    if (
        old_protocol["version"] != "refine-diagnostic-v3"
        or old_protocol["smoke_test"]
        or old_protocol["inner_coarse_epochs"] != 43
        or old_protocol["coarse_fingerprint"] != coarse_fingerprint
        or old_protocol["source_sha256"]["task3_refine_diagnostic_v3.py"] != V3_SOURCE_SHA
    ):
        raise ValueError("Expected the completed v3 auxiliary coarse predictions.")
    centers, provenance = {}, []
    for fold in range(5):
        dev, fit, held, outer = inner_partition(train, fold)
        folder = source / f"inner/outer_{fold}"
        design = json.loads((folder / "design.json").read_text())
        expected = dict(
            protocol=old_protocol,
            fold=fold,
            fit_ids=list(map(int, fit.image_id)),
            inner_val_ids=list(map(int, held.image_id)),
            outer_ids=list(map(int, outer.image_id)),
        )
        if design != expected:
            raise ValueError("Cached inner split differs.")
        cfg = json.loads((folder / "config.json").read_text())
        done = json.loads((folder / "coarse_complete.json").read_text())
        if (
            cfg["train_ids"] != expected["fit_ids"]
            or cfg["inner_validation_ids"] != expected["inner_val_ids"]
            or cfg["forbidden_outer_ids"] != expected["outer_ids"]
            or cfg["epochs"] != 43
            or cfg["schedule_epochs"] != 80
            or cfg["source_sha256"] != old_protocol["source_sha256"]
            or cfg["coarse_fingerprint"] != coarse_fingerprint
            or done["signature"] != stable_hash(cfg)
            or done["epoch"] != 43
            or done["validation_used_for_training"]
        ):
            raise ValueError("Cached auxiliary model provenance differs.")
        predictions = read_rows(folder / "inner_validation.csv")
        centers[fold] = validate_inner_centers(predictions, held.to_dict("records"))
        actual = metrics_from_rows(predictions)
        if not math.isclose(
            actual["rmse_coordinate_px"],
            done["baseline_metrics"]["rmse_coordinate_px"],
            abs_tol=1e-6,
        ):
            raise ValueError("Cached inner baseline metric differs.")
        provenance.append(dict(fold=fold, config=cfg, complete=done, predictions=predictions))
    return centers, stable_hash(provenance)


def context_grid(boxes, sizes, height, width):
    """Feature-cell centers mapped from the local crop into normalized full-image coordinates."""
    b, size = boxes.float(), sizes.float()
    xs = b[:, 0, None, None] + (torch.arange(width, device=b.device)[None, None, :] + 0.5) * (
        768 / width
    )
    ys = b[:, 1, None, None] + (torch.arange(height, device=b.device)[None, :, None] + 0.5) * (
        768 / height
    )
    gx = (2 * xs / size[:, 0, None, None] - 1).expand(-1, height, -1)
    gy = (2 * ys / size[:, 1, None, None] - 1).expand(-1, -1, width)
    return torch.stack((gx, gy), dim=-1)


class ContextRefiner(ResNet34CV):
    def __init__(self, weights=None, context=True):
        super().__init__(weights)
        self.use_context = bool(context)
        if self.use_context:
            self.context3 = nn.Conv2d(256, 256, 1)
            self.context4 = nn.Conv2d(1024, 512, 1)
            for layer in (self.context3, self.context4):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def encode(self, image):
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(image))))
        c1 = b.layer1(x)
        c2 = b.layer2(c1)
        c3 = b.layer3(c2)
        c4 = b.layer4(c3)
        return c1, c2, c3, c4

    def aligned(self, feature, boxes, sizes, target):
        grid = context_grid(boxes, sizes, *target.shape[-2:])
        # Explicit fp32 sampling avoids mixed-dtype grid/input surprises under AMP.
        with torch.autocast(device_type=feature.device.type, enabled=False):
            value = F.grid_sample(
                feature.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )
        return value.to(target.dtype)

    def forward(self, local, global_view, boxes, sizes):
        c1, c2, c3, c4 = self.encode(local)
        if self.use_context:
            _, _, g3, g4 = self.encode(global_view)
            c3 = c3 + self.context3(self.aligned(g3, boxes, sizes, c3))
            aligned = self.aligned(g4, boxes, sizes, c4)
            pooled = g4.mean(dim=(-2, -1), keepdim=True).expand(-1, -1, *c4.shape[-2:])
            c4 = c4 + self.context4(torch.cat((aligned, pooled), dim=1))
        return self.head(self.d1(self.d2(self.d3(c4, c3), c2), c1))[:, 0]


def native_augmentation(box, flip, affine):
    # The local384 transformation and global256 transformation describe the same native frame.
    native_from_patch = np.array([[2.0, 0.0, box[0]], [0.0, 2.0, box[1]], [0.0, 0.0, 1.0]])
    reflection = (
        np.array([[-1.0, 0.0, 384.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) if flip else np.eye(3)
    )
    return native_from_patch @ affine @ reflection @ np.linalg.inv(native_from_patch)


class ContextPatchDataset(Dataset):
    def __init__(self, records, cache, config, training, centers=None):
        self.records, self.cache, self.config = list(records), cache, config
        self.training, self.centers = training, centers
        ids = {int(r["image_id"]) for r in records}
        if len(ids) != len(records):
            raise ValueError("Repeated patch ID.")
        if training:
            if ids != set(config["train_ids"]) or ids & set(config["forbidden_ids"]):
                raise ValueError("Training IDs differ.")
        elif centers is None or ids != set(centers):
            raise ValueError("Prediction centers/IDs differ.")
        if not hasattr(cache, "context_views"):
            cache.context_views = {}
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
        row = self.records[index]
        key = int(row["image_id"])
        image = self.cache.get(row)
        if key not in self.cache.context_views:
            self.cache.context_views[key] = image.resize((256, 256), Image.Resampling.BILINEAR)
        global_view = self.cache.context_views[key].copy()
        if self.training:
            dx, dy = center_offset(
                "small", [max(1e-10, min(1 - 1e-10, random.random())) for _ in range(3)]
            )
            center = (float(row["x"]) + dx, float(row["y"]) + dy)
        else:
            c = self.centers[key]
            center = (float(c["Fovea_X"]), float(c["Fovea_Y"]))
        box = crop_geometry(center, 768)
        local = image.crop(box).resize((384, 384), Image.Resampling.BILINEAR)
        point = np.array(
            point_to_patch((row["x"], row["y"]), box) if self.training else (0.5, 0.5),
            dtype=np.float32,
        )
        if self.training:
            flip = random.random() < 0.5
            if flip:
                local = local.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                point[0] = 1 - point[0]
            angle, scale = math.radians(random.uniform(-8, 8)), random.uniform(0.95, 1.05)
            a, b = scale * math.cos(angle), -scale * math.sin(angle)
            tx, ty = random.uniform(-0.02, 0.02) * 384, random.uniform(-0.02, 0.02) * 384
            affine = np.array(
                [
                    [a, b, 192 + tx - a * 192 - b * 192],
                    [-b, a, 192 + ty + b * 192 - a * 192],
                    [0.0, 0.0, 1.0],
                ]
            )
            point = ((affine @ [point[0] * 384, point[1] * 384, 1.0])[:2] / 384).astype(np.float32)
            if not np.isfinite(point).all() or not ((point > 0).all() and (point < 1).all()):
                raise ValueError("Target left patch.")
            local = local.transform(
                (384, 384),
                Image.Transform.AFFINE,
                np.linalg.inv(affine)[:2].reshape(-1),
                resample=Image.Resampling.BILINEAR,
                fillcolor=(0, 0, 0),
            )
            scaled = np.diag([256 / int(row["width"]), 256 / int(row["height"]), 1.0])
            global_forward = scaled @ native_augmentation(box, flip, affine) @ np.linalg.inv(scaled)
            global_view = global_view.transform(
                (256, 256),
                Image.Transform.AFFINE,
                np.linalg.inv(global_forward)[:2].reshape(-1),
                resample=Image.Resampling.BILINEAR,
                fillcolor=(0, 0, 0),
            )
            # torchvision ColorJitter draws use CPU Torch RNG; consume one draw sequence for both views.
            before = torch.get_rng_state()
            local = self.jitter(local)
            after = torch.get_rng_state()
            torch.set_rng_state(before)
            global_view = self.jitter(global_view)
            torch.set_rng_state(after)
        return (
            self.tensor(local),
            self.tensor(global_view),
            torch.from_numpy(point),
            torch.tensor(box, dtype=torch.int64),
            torch.tensor([row["width"], row["height"]], dtype=torch.float32),
        )


def context_config(protocol, fold, arm, role, train_ids, evaluation_ids, forbidden_ids, epochs):
    cfg = dict(
        version=CONTEXT_VERSION,
        source_sha256=protocol["source_sha256"],
        coarse_fingerprint=protocol["coarse_fingerprint"],
        inner_fingerprint=protocol["inner_fingerprint"],
        outer_fold=fold,
        arm=arm,
        role=role,
        seed=(42000 if role == "inner" else 43000) + fold,
        train_ids=list(train_ids),
        evaluation_ids=list(evaluation_ids),
        forbidden_ids=list(forbidden_ids),
        epochs=int(epochs),
        schedule_epochs=protocol["inner_epochs"],
        image_size=384,
        global_size=256,
        crop_side=768,
        prior_sigma=48.0,
        max_shift=256.0,
    )
    validate_context_config(cfg)
    return cfg


def validate_context_config(cfg):
    sets = [set(cfg[k]) for k in ("train_ids", "evaluation_ids", "forbidden_ids")]
    if (
        any(
            len(s) != len(cfg[k])
            for s, k in zip(sets, ("train_ids", "evaluation_ids", "forbidden_ids"))
        )
        or not sets[0]
        or any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2)))
    ):
        raise ValueError("Overlapping/repeated phase IDs.")
    if (
        cfg["arm"] not in CONTEXT_ARMS
        or cfg["role"] not in ("inner", "outer")
        or bool(sets[1]) != (cfg["role"] == "inner")
    ):
        raise ValueError("Invalid phase role.")
    if not 1 <= cfg["epochs"] <= cfg["schedule_epochs"] or (
        cfg["image_size"],
        cfg["global_size"],
        cfg["crop_side"],
    ) != (384, 256, 768):
        raise ValueError("Invalid epoch/geometry.")
    if cfg["prior_sigma"] != 48.0 or cfg["max_shift"] != 256.0:
        raise ValueError("This comparison fixes the decoder prior and fallback threshold.")


def load_context_checkpoint(path, signature):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["signature"] != signature:
        raise ValueError("Checkpoint signature differs.")
    return state


def context_prediction_fold(cfg, centers, outer_fold):
    if cfg["role"] not in ("inner", "outer") or int(cfg["outer_fold"]) != int(outer_fold):
        raise ValueError("Prediction role or outer fold differs from the phase.")
    expected = -1 if cfg["role"] == "inner" else int(outer_fold)
    if not centers or any(int(row["fold"]) != expected for row in centers.values()):
        raise ValueError("Coarse center fold differs from the prediction phase.")
    return expected


@torch.inference_mode()
def predict_context(model, loader, cfg, centers, fold):
    prediction_fold = context_prediction_fold(cfg, centers, fold)
    model.eval()
    rows = []
    offset = 0
    for local, global_view, _, boxes, sizes in loader:
        with torch.autocast(device_type="cuda", enabled=True):
            logits = model(
                local.to("cuda:0"), global_view.to("cuda:0"), boxes.to("cuda:0"), sizes.to("cuda:0")
            )
        if logits.shape != (len(local), 96, 96) or not torch.isfinite(logits).all():
            raise RuntimeError("Invalid context heatmap.")
        records = loader.dataset.records[offset : offset + len(local)]
        points = [
            [
                float(centers[int(r["image_id"])]["Fovea_X"]),
                float(centers[int(r["image_id"])]["Fovea_Y"]),
            ]
            for r in records
        ]
        decoded = (
            decode_heatmap(coarse_log_prior(logits, boxes, points, 48.0), radius=3).cpu().tolist()
        )
        for record, point, box in zip(records, decoded, boxes.tolist()):
            row = make_refinement_row(
                record, point, centers[int(record["image_id"])], box, prediction_fold, 256.0
            )
            row.update(decoder="prior", refine_applied=1, arm=cfg["arm"])
            rows.append(row)
        offset += len(local)
    if offset != len(loader.dataset):
        raise ValueError("Inference coverage differs.")
    return rows


def gpu_context_preflight(weights, output):
    seed_everything(44000)
    local = ContextRefiner(weights, False).to("cuda:0").eval()
    seed_everything(44000)
    context = ContextRefiner(weights, True).to("cuda:0").eval()
    x = torch.randn(1, 3, 384, 384, device="cuda:0")
    g = torch.randn(1, 3, 256, 256, device="cuda:0")
    box = torch.tensor([[1100, 600, 1868, 1368]], device="cuda:0")
    size = torch.tensor([[2992, 2000]], device="cuda:0")
    with torch.no_grad(), torch.autocast(device_type="cuda"):
        expected = local(x, g, box, size)
        actual = context(x, g, box, size)
    difference = float((actual - expected).abs().max())
    if difference > 1e-3 or actual.shape != (1, 96, 96):
        raise RuntimeError("Zero-initialized context changed the local starting function.")
    del local, expected, actual
    context.train()
    context.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda"):
        logits = context(x, g, box, size)
    loss = heatmap_loss(logits, torch.tensor([[0.5, 0.5]], device="cuda:0"), 1.5)
    loss.backward()
    gradients = {
        name: bool(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
        for name, p in context.named_parameters()
        if name in ("context3.weight", "context4.weight")
    }
    if not math.isfinite(float(loss.detach())) or not all(gradients.values()):
        raise RuntimeError("Global fusion gradient check failed.")
    if any(m.training for m in context.backbone.modules() if isinstance(m, nn.BatchNorm2d)):
        raise RuntimeError("Backbone BatchNorm not frozen.")
    save_json(
        output / "gpu_preflight.json",
        dict(
            passed=True,
            zero_init_max_abs_difference=difference,
            gradients=gradients,
            loss=float(loss.detach()),
        ),
    )
    del context, x, g, box, size, logits, loss
    gc.collect()
    torch.cuda.empty_cache()


def train_context_phase(records, cache, cfg, folder, weights, evaluator=None):
    validate_context_config(cfg)
    if (evaluator is not None) != (cfg["role"] == "inner"):
        raise ValueError("Outer training must not have an evaluation callback.")
    folder.mkdir(parents=True, exist_ok=True)
    signature = stable_hash(cfg)
    if (folder / "config.json").exists() and json.loads(
        (folder / "config.json").read_text()
    ) != cfg:
        raise ValueError("Phase config differs.")
    save_json(folder / "config.json", cfg)
    if (folder / "phase_complete.json").exists():
        done = json.loads((folder / "phase_complete.json").read_text())
        checkpoint = folder / ("best.pt" if evaluator else "final.pt")
        if done["signature"] != signature or not checkpoint.is_file():
            raise ValueError("Restore the full v4 Output, not a light ZIP.")
        return done
    seed_everything(cfg["seed"])
    torch.set_num_threads(2)
    dataset = ContextPatchDataset(records, cache, cfg, True)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    last = folder / "last.pt"
    model = ContextRefiner(None if last.exists() else weights, cfg["arm"] == "context").to("cuda:0")
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    optimizer = torch.optim.AdamW(
        [
            dict(params=model.backbone.parameters(), lr=5e-5),
            dict(params=[p for p in model.parameters() if id(p) not in backbone_ids], lr=5e-4),
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
    best_weights = None
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["signature"] != signature:
            raise ValueError("Optimizer signature differs.")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start, history, best = state["epoch"], state["history"], state["best"]
        best_weights = state["best_model"]
        del state
        # Restore the best model belonging to this committed optimizer/history prefix.
        # A crash after a later best.pt write must not mix an uncommitted future best with this prefix.
        if evaluator and best["epoch"]:
            atomic_torch_save(
                folder / "best.pt",
                dict(
                    model=best_weights,
                    signature=signature,
                    epoch=best["epoch"],
                    selection=best,
                    config=cfg,
                ),
            )
    for epoch in range(start + 1, cfg["epochs"] + 1):
        # Additional global layers must not change the paired minibatches or augmentation draws.
        seed_everything(cfg["seed"] + 1000 * epoch)
        begin = time.perf_counter()
        model.train()
        total = count = skipped = 0
        for local, global_view, points, boxes, sizes in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda"):
                logits = model(
                    local.to("cuda:0"),
                    global_view.to("cuda:0"),
                    boxes.to("cuda:0"),
                    sizes.to("cuda:0"),
                )
                if logits.shape != (len(local), 96, 96):
                    raise RuntimeError("Fine shape differs.")
                loss = heatmap_loss(logits, points.to("cuda:0"), 1.5)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < old_scale)
            total += float(loss.detach()) * len(local)
            count += len(local)
        if skipped == len(loader) and history and history[-1]["amp_skips"] == len(loader):
            raise RuntimeError("All optimizer updates skipped in consecutive epochs.")
        row = dict(
            epoch=epoch,
            train_ce=total / count,
            amp_skips=skipped,
            lr_backbone=optimizer.param_groups[0]["lr"],
        )
        if evaluator:
            values = evaluator(model)
            row.update(values)
            if values["score"] < best["score"]:
                best = dict(values, epoch=epoch)
                best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                atomic_torch_save(
                    folder / "best.pt",
                    dict(
                        model=best_weights,
                        signature=signature,
                        epoch=epoch,
                        selection=best,
                        config=cfg,
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
                    best_model=best_weights,
                ),
            )
        atomic_csv(folder / "history.csv", history)
        save_json(folder / "status.json", dict(state="training", epoch=epoch, signature=signature))
        print(
            f'Outer{cfg["outer_fold"]} {cfg["role"]}/{cfg["arm"]} {epoch:02}/{cfg["epochs"]} CE={total/count:.4f} {row["seconds"]:.1f}s',
            flush=True,
        )
    if not evaluator:
        atomic_torch_save(
            folder / "final.pt",
            dict(model=model.state_dict(), signature=signature, epoch=cfg["epochs"], config=cfg),
        )
    done = dict(signature=signature, best=best if evaluator else {}, epochs=cfg["epochs"])
    save_json(folder / "phase_complete.json", done)
    if last.exists():
        last.unlink()
    del model, loader, dataset, best_weights
    gc.collect()
    torch.cuda.empty_cache()
    return done


def context_design(prepared, fold):
    dev, fit, held, outer = inner_partition(prepared["train"], fold)
    design = dict(
        protocol=prepared["protocol"],
        fold=fold,
        fit_ids=list(map(int, fit.image_id)),
        inner_val_ids=list(map(int, held.image_id)),
        outer_ids=list(map(int, outer.image_id)),
    )
    return dev, fit, held, outer, design


def run_context_fold(prepared, fold, weights, cache):
    dev, fit, held, outer, design = context_design(prepared, fold)
    job = prepared["output"] / f"outer_{fold}"
    job.mkdir(parents=True, exist_ok=True)
    signature = stable_hash(design)
    if (job / "fold_complete.json").exists():
        done = json.loads((job / "fold_complete.json").read_text())
        if done["signature"] != signature or any(
            not (job / p).is_file() for p in done["required_models"]
        ):
            raise ValueError("Completed fold lacks full weights or differs.")
        return
    save_json(job / "design.json", design)
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
    centers = prepared["inner"][fold]
    baseline = metrics_from_rows(list(centers.values()))["rmse_coordinate_px"]
    decisions = {}
    for arm in CONTEXT_ARMS:
        cfg = context_config(
            prepared["protocol"],
            fold,
            arm,
            "inner",
            design["fit_ids"],
            design["inner_val_ids"],
            design["outer_ids"],
            prepared["protocol"]["inner_epochs"],
        )
        validation = DataLoader(
            ContextPatchDataset(held.to_dict("records"), cache, cfg, False, centers),
            batch_size=8,
            shuffle=False,
            num_workers=0,
        )

        def evaluate(model):
            return inner_mode_metrics(
                predict_context(model, validation, cfg, centers, fold), centers
            )

        done = train_context_phase(
            fit.to_dict("records"), cache, cfg, job / f"inner_{arm}", weights, evaluate
        )
        decision = selected_inner_epoch(done["best"], baseline, 0.05)
        decisions[arm] = decision
        del validation
        if decision["enabled"]:
            cfg = context_config(
                prepared["protocol"],
                fold,
                arm,
                "outer",
                list(map(int, dev.image_id)),
                [],
                design["outer_ids"],
                decision["epoch"],
            )
            folder = job / f"outer_{arm}"
            train_context_phase(dev.to_dict("records"), cache, cfg, folder, weights)
            model = ContextRefiner(None, arm == "context").to("cuda:0")
            state = load_context_checkpoint(folder / "final.pt", stable_hash(cfg))
            model.load_state_dict(state["model"], strict=True)
            del state
        for subset, frame, prior in [
            ("validation", outer, prepared["coarse"]["validation"][fold]),
            ("test", prepared["test"], prepared["coarse"]["test"][fold]),
        ]:
            records = frame.to_dict("records")
            if decision["enabled"]:
                loader = DataLoader(
                    ContextPatchDataset(records, cache, cfg, False, prior),
                    batch_size=8,
                    shuffle=False,
                    num_workers=0,
                )
                predictions = predict_context(model, loader, cfg, prior, fold)
                del loader
                for row in predictions:
                    row["selection_reason"] = decision["reason"]
            else:
                predictions = identity_rows(records, prior, fold, decision)
            for row in predictions:
                row["arm"] = arm
            atomic_csv(job / f"{subset}_{arm}.csv", predictions)
        if decision["enabled"]:
            del model
        gc.collect()
        torch.cuda.empty_cache()
    save_json(job / "inner_decisions.json", decisions)
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


def context_reports(prepared):
    output = prepared["output"]
    available = [f for f in range(5) if (output / f"outer_{f}/fold_complete.json").exists()]
    selection = dict(
        complete=len(available) == 5 and not prepared["protocol"]["smoke_test"],
        completed_folds=available,
        smoke_test=prepared["protocol"]["smoke_test"],
        official_metric_confirmed=True,
        metric="rmse_coordinate_px",
        selected_by_outer_scores=False,
        candidate_rule="Predeclared context_equal; inner 5% gain gate",
        candidate_public_score=None,
        agent_submitted_to_competition=False,
        limitation="Legacy fixed coarse checkpoints used outer labels for stopping. This is conditional development evaluation, not fully independent evaluation of the complete pipeline.",
    )
    if not selection["complete"]:
        save_json(output / "selection.json", selection)
        return False
    records, testing = prepared["train"].to_dict("records"), prepared["test"].to_dict("records")
    coarse = prepared["coarse"]
    baseline = validate_oof(
        [r for f in range(5) for r in coarse["validation"][f].values()], records
    )
    atomic_csv(output / "oof_coarse512.csv", baseline)
    write_submission(
        prepared["data"] / "sample_submission.csv",
        average_test_folds({f: list(coarse["test"][f].values()) for f in range(5)}, testing),
        output / "submission_coarse512.csv",
    )
    comparisons = [dict(arm="coarse512", **metrics_from_rows(baseline))]
    decision_rows = []
    for arm in CONTEXT_ARMS:
        predictions = []
        test_folds = {}
        active = 0
        for f in range(5):
            job = output / f"outer_{f}"
            done = json.loads((job / "fold_complete.json").read_text())
            design = context_design(prepared, f)[-1]
            if (
                done["signature"] != stable_hash(design)
                or not done["predictions_complete"]
                or done["outer_labels_used_for_epoch_selection"]
            ):
                raise ValueError("Fold provenance differs.")
            decision = done["decisions"][arm]
            expected = selected_inner_epoch(
                dict(score=decision["inner_score"], epoch=decision["inner_best_epoch"]),
                decision["inner_baseline"],
                0.05,
            )
            if decision != expected:
                raise ValueError("Inner selection differs.")
            active += int(decision["enabled"])
            decision_rows.append(dict(arm=arm, outer_fold=f, **decision))
            held = prepared["train"][prepared["train"].fold == f].to_dict("records")
            val = validate_oof(read_rows(job / f"validation_{arm}.csv"), held)
            validate_test_rows(val, held, f)
            predictions.extend(val)
            test_folds[f] = read_rows(job / f"test_{arm}.csv")
            validate_test_rows(test_folds[f], testing, f)
        predictions = validate_oof(predictions, records)
        for suffix, preds, folds in [
            ("refined", predictions, test_folds),
            (
                "equal",
                blend_prediction_rows(baseline, predictions, True),
                {
                    f: blend_prediction_rows(list(coarse["test"][f].values()), test_folds[f], False)
                    for f in range(5)
                },
            ),
        ]:
            name = f"{arm}_{suffix}"
            atomic_csv(output / f"oof_{name}.csv", preds)
            mean = average_test_folds(folds, testing)
            atomic_csv(output / f"test_{name}.csv", mean)
            write_submission(
                prepared["data"] / "sample_submission.csv", mean, output / f"submission_{name}.csv"
            )
            comparisons.append(dict(arm=name, active_folds=active, **metrics_from_rows(preds)))
        raw = raw_prediction_rows(predictions)
        atomic_csv(output / f"oof_{arm}_raw.csv", raw)
        comparisons.append(
            dict(arm=f"{arm}_raw_diagnostic", active_folds=active, **metrics_from_rows(raw))
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
    shutil.copy2(
        output / "submission_context_equal.csv", output / "submission_context_v4_candidate.csv"
    )
    selection.update(
        candidate="submission_context_v4_candidate.csv",
        public_best_reference=12.31624,
        public_coarse_reference=14.16538,
    )
    save_json(output / "selection.json", selection)
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)
    return True


def context_archive(output, full=False):
    target = (
        output.parent / "task3_context_v4_full_backup.zip"
        if full
        else output / "task3_context_v4_results_light.zip"
    )
    temporary = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True
    ) as archive:
        for p in sorted(output.rglob("*")):
            if (
                p.is_file()
                and not p.is_symlink()
                and p.suffix not in (".tmp", ".zip")
                and (full or p.suffix in (".csv", ".json", ".md"))
            ):
                archive.write(p, p.relative_to(output))
    temporary.replace(target)
    if full:
        with zipfile.ZipFile(target) as archive:
            names = set(archive.namelist())
            for f in range(5):
                done = json.loads((output / f"outer_{f}/fold_complete.json").read_text())
                if any(f"outer_{f}/{name}" not in names for name in done["required_models"]):
                    raise RuntimeError("Full backup lacks required weights.")
    return target


def prepare_context(
    source,
    data_dir=None,
    output_dir="/kaggle/working/task3_context_refine_v4",
    resume_dir=None,
    smoke=False,
):
    if not Path("/kaggle/working").is_dir() or not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU required. No local training was started.")
    data, output = find_data(data_dir), Path(output_dir).resolve()
    source = Path(source)
    manifest = load_manifest(data)
    train = assign_folds(manifest[manifest.subset == "train"], 42, 5)
    test = manifest[manifest.subset == "test"].copy()
    if len(train) != 80 or len(test) != 20:
        raise ValueError("Expected80/20 images.")
    coarse = validate_coarse_bundle(
        source / "coarse", train.to_dict("records"), test.to_dict("records")
    )
    centers, inner_fingerprint = validate_context_centers(source, train, coarse["fingerprint"])
    protocol = dict(
        version=CONTEXT_VERSION,
        source_sha256=dict(CONTEXT_SOURCE_SHA256),
        coarse_fingerprint=coarse["fingerprint"],
        inner_fingerprint=inner_fingerprint,
        inner_epochs=2 if smoke else 60,
        arms=list(CONTEXT_ARMS),
        prior_sigma=48.0,
        max_shift=256.0,
        min_inner_gain=0.05,
        candidate_variant="context_equal",
        smoke_test=bool(smoke),
        official_metric="rmse_coordinate_px",
        augmentation="v3 small jitter + paired native affine/global view + shared ColorJitter; epoch-keyed RNG",
        global_size=256,
        context_initialization="zero additive projections",
        legacy_coarse_outer_selection_bias_remains=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 9 * 1024**3:
        raise RuntimeError("Need9 GiB free for weights, atomic saves and full backup.")
    restore_backup(resume_dir, output, protocol)
    if (output / "protocol.json").exists() and json.loads(
        (output / "protocol.json").read_text()
    ) != protocol:
        raise ValueError("Output protocol differs; use a new directory.")
    save_json(output / "protocol.json", protocol)
    train.to_csv(output / "split.csv", index=False)
    manifest.to_csv(output / "manifest.csv", index=False)
    save_json(
        output / "environment.json",
        dict(
            torch=torch.__version__,
            numpy=np.__version__,
            gpu=torch.cuda.get_device_name(0),
            visible_gpu_count=torch.cuda.device_count(),
            device="cuda:0 sequential",
            grid_sample_backward="CUDA backward may be nondeterministic; bitwise reruns are not promised",
        ),
    )
    for f in range(5):
        folder = output / f"inner_cache/fold_{f}"
        folder.mkdir(parents=True, exist_ok=True)
        atomic_csv(folder / "inner_validation.csv", list(centers[f].values()))
    save_json(
        output / "cache_provenance.json",
        dict(
            coarse_fingerprint=coarse["fingerprint"],
            inner_fingerprint=inner_fingerprint,
            input_payload_sha256=CONTEXT_CACHE_SHA256,
        ),
    )
    return dict(
        data=data,
        output=output,
        train=train,
        test=test,
        coarse=coarse,
        inner=centers,
        protocol=protocol,
    )


def run_context(prepared, run_folds=(0, 1, 2, 3, 4), weights=None, full_backup=True):
    if not Path("/kaggle/working").is_dir() or not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU required.")
    if (
        not run_folds
        or len(set(run_folds)) != len(run_folds)
        or not set(run_folds) <= set(range(5))
    ):
        raise ValueError("Invalid fold schedule.")
    output = prepared["output"]
    cache = OriginalImageCache(prepared["data"])
    pending = any(not (output / f"outer_{f}/fold_complete.json").exists() for f in run_folds)
    weights = resolve_weights(output, weights) if pending else None
    with training_lock(output):
        try:
            if pending:
                gpu_context_preflight(weights, output)
            for f in run_folds:
                save_json(
                    output / "status.json",
                    dict(state="running", outer_fold=f, updated_unix=time.time()),
                )
                run_context_fold(prepared, f, weights, cache)
                context_reports(prepared)
                context_archive(output)
            complete = context_reports(prepared)
            save_json(
                output / "status.json",
                dict(
                    state="completed" if complete else "partial_completed", updated_unix=time.time()
                ),
            )
            context_archive(output)
            if complete and full_backup:
                print("Full backup:", context_archive(output, True), flush=True)
        except BaseException as error:
            save_json(
                output / "status.json",
                dict(
                    state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    error=str(error),
                    updated_unix=time.time(),
                ),
            )
            context_archive(output)
            raise
    return output
