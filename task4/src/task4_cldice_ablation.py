"""One controlled extension: L2 + 0.1 soft-clDice on the full-resolution head.

The three previous modules stay unchanged. Original L2 OOF files are read-only;
only the new five-fold arm trains. No test inference or competition submissions.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import platform
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import task4_controlled_ablation as abl

base, v2 = abl.base, abl.v2
REFERENCE = "L2_reference"
METHOD = "T1_L2_cldice"


@dataclass
class StructureConfig(abl.AblationConfig):
    cldice_weight: float = 0.1
    skeleton_iterations: int = 10
    skeleton_smooth: float = 1.0


def environment():
    import PIL
    import skimage

    return dict(
        python=platform.python_version(),
        torch=str(torch.__version__),
        torchvision=str(v2.torchvision.__version__),
        numpy=np.__version__,
        opencv=cv2.__version__,
        pillow=PIL.__version__,
        skimage=skimage.__version__,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    )


def soft_erode(image):
    return torch.minimum(
        -F.max_pool2d(-image, (3, 1), 1, (1, 0)), -F.max_pool2d(-image, (1, 3), 1, (0, 1))
    )


def soft_open(image):
    return F.max_pool2d(soft_erode(image), 3, 1, 1)


def soft_skeleton(image, iterations=10):
    """2D min/max-pool skeleton as in the authors' differentiable formulation."""
    skeleton = F.relu(image - soft_open(image))
    for _ in range(iterations):
        image = soft_erode(image)
        delta = F.relu(image - soft_open(image))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(logits, clean, iterations=10, smooth=1.0):
    # Per-image reduction matches the reference's per-image Dice convention.
    # Float32 morphology avoids half-precision reductions under mixed precision.
    with torch.autocast(device_type=logits.device.type, enabled=False):
        prob, clean = logits.float().sigmoid(), clean.float()
        pred_skeleton = soft_skeleton(prob, iterations)
        with torch.no_grad():
            true_skeleton = soft_skeleton(clean, iterations)
        axes = (1, 2, 3)
        precision = ((pred_skeleton * clean).sum(axes) + smooth) / (
            pred_skeleton.sum(axes) + smooth
        )
        recall = ((true_skeleton * prob).sum(axes) + smooth) / (true_skeleton.sum(axes) + smooth)
        return (1 - 2 * precision * recall / (precision + recall)).mean()


def objective(outputs, pair, cfg):
    reference = abl.supervised_loss(outputs, pair, "L2_soft", cfg.soft_boundary)
    if cfg.cldice_weight == 0:
        return reference, reference, reference.detach() * 0
    topology = soft_cldice_loss(
        outputs[0], pair[:, :1], cfg.skeleton_iterations, cfg.skeleton_smooth
    )
    return reference + cfg.cldice_weight * topology, reference, topology


def valid_probability(path, val_ids):
    values = {}
    with np.load(path, allow_pickle=False) as arrays:
        expected = {f"{mode}_{i}" for mode in ["none", "d4"] for i in val_ids}
        if set(arrays.files) != expected:
            raise ValueError(f"Unexpected OOF probability keys: {path}")
        for key in expected:
            value = arrays[key]
            if (
                value.shape != (512, 512)
                or not np.isfinite(value).all()
                or value.min() < 0
                or value.max() > 1
            ):
                raise ValueError(f"Invalid probability: {path}:{key}")
            values[key] = value.copy()
    return values


def safe_members(archive):
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError("Duplicate archive paths")
    for item in archive.infolist():
        path = PurePosixPath(item.filename)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in item.filename
            or ":" in item.filename
            or (item.external_attr >> 16) & 0o170000 == 0o120000
        ):
            raise ValueError("Unsafe archive path")
    if archive.testzip() is not None:
        raise ValueError("Archive CRC failed")


def materialize_reference(source, output):
    """Find a full label output; extract only JSON, L2 history/OOF and initializer."""
    if source is None:
        candidates = []
        local = Path("/kaggle/working/task4_label_supervision_ablation")
        if (local / "L2_soft/fold0/validation_probabilities.npz").exists():
            candidates.append(local)
        if Path("/kaggle/input").exists():
            candidates += [
                p.parent
                for p in Path("/kaggle/input").rglob("ablation_config.json")
                if all(
                    (p.parent / f"L2_soft/fold{f}/validation_probabilities.npz").exists()
                    for f in range(5)
                )
            ]
        candidates = list(dict.fromkeys(p.resolve() for p in candidates))
        if len(candidates) != 1:
            raise ValueError(
                f"Found {len(candidates)} full L2 outputs. Set BASELINE_RUN to the label-stage output folder. "
                "The lightweight ablation_analysis_results ZIP has no OOF probabilities."
            )
        return candidates[0]
    source = Path(source).resolve()
    if source.is_dir():
        return source
    if not source.is_file() or source.suffix.lower() != ".zip":
        raise ValueError("BASELINE_RUN must be the full label-stage folder or a full backup ZIP")
    cache = Path(output) / "reference_cache"
    required = {"ablation_config.json", "execution_environment.json", "manifest.csv"}
    required |= {
        f"L2_soft/fold{f}/{name}"
        for f in range(5)
        for name in ["completed.json", "history.csv", "validation_probabilities.npz"]
    }
    optional = {"pretrained_cache/resnet34-b627a593.pth"}
    with zipfile.ZipFile(source) as z:
        safe_members(z)
        if not required <= set(z.namelist()):
            raise ValueError(
                "Missing full L2 probabilities/records. A lightweight analysis ZIP cannot serve as baseline."
            )
        for name in sorted(required | (optional & set(z.namelist()))):
            target = cache / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return cache


def verify_reference(source, expected, expected_config, assignment):
    source = Path(source)
    config = json.loads((source / "ablation_config.json").read_text())
    if config != expected_config:
        raise ValueError(
            "Baseline config/source/dataset differs from the returned formal L2 experiment"
        )
    for fold in range(5):
        folder = source / "L2_soft" / f"fold{fold}"
        done = json.loads((folder / "completed.json").read_text())
        if done != expected[str(fold)] or done["epoch"] != expected_config["config"]["epochs"]:
            raise ValueError(f"Baseline completed record differs: fold{fold}")
        if set(done["val_ids"]) != {i for i, f in assignment.items() if f == fold}:
            raise ValueError("Baseline uses different folds")
        for name in ["history.csv", "validation_probabilities.npz"]:
            if base.file_sha(folder / name) != done["sha256"][name]:
                raise ValueError(f"Baseline artifact hash mismatch: fold{fold}/{name}")
        history = read_table(folder / "history.csv")
        if [int(r["epoch"]) for r in history] != list(range(1, done["epoch"] + 1)):
            raise ValueError("Incomplete baseline history")
        if [r["plan_sha256"] for r in history] != done["plan_hashes"]:
            raise ValueError("Baseline sampling history mismatch")
        valid_probability(folder / "validation_probabilities.npz", done["val_ids"])
    return config


def read_table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def train_fold(paths, validation, labels, cfg, fold, folder, initial, device, expected):
    """Same L2 loop/seed/scheduler; the only optimization change is objective()."""
    folder.mkdir(parents=True, exist_ok=True)
    base.seed_all(cfg.seed + 1009 * fold)
    dataset = abl.PairedPatches(paths, labels, cfg, fold)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = v2.ResNet34UNet(initial).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    initial_hash = abl.digest_json(
        {
            k: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
            for k, v in model.state_dict().items()
        }
    )
    train_ids, val_ids = [p.stem for p in paths], [p.stem for p in validation]
    if (
        initial_hash != expected["initial_model_hash"]
        or train_ids != expected["train_ids"]
        or val_ids != expected["val_ids"]
    ):
        raise ValueError(
            "Initial model/split differs from the paired L2 reference; no training started"
        )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": cfg.encoder_lr},
            {
                "params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")],
                "lr": cfg.decoder_lr,
            },
        ],
        weight_decay=cfg.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    history, first, step = [], 1, 0
    last = folder / "last.pt"
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if (
            saved["method"] != METHOD
            or saved["config"] != asdict(cfg)
            or saved["initial_model_hash"] != initial_hash
            or saved["train_ids"] != train_ids
            or saved["val_ids"] != val_ids
        ):
            raise ValueError("Structural checkpoint identity mismatch")
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        first, step, history = saved["epoch"] + 1, saved["step"], saved["history"]
        base.restore_rng(saved["rng"])
        del saved
        print(f"{METHOD}/fold{fold}: resume at {first}/{cfg.epochs}", flush=True)
    for epoch in range(first, cfg.epochs + 1):
        begin = time.perf_counter()
        model.train()
        plan_hash = dataset.set_epoch(epoch)
        if plan_hash != expected["plan_hashes"][epoch - 1]:
            raise ValueError(f"RGB sampling/augmentation plan differs: fold{fold}, epoch{epoch}")
        losses, reference_losses, topology_losses = [], [], []
        for image, pair in loader:
            v2.set_lrs(optimizer, step, len(loader), cfg)
            image, pair = image.to(device, non_blocking=True), pair.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
                loss, reference, topology = objective(model(image), pair, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite structural loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                v2.update_ema(ema, model, cfg.ema_decay)
            losses.append(float(loss.detach()))
            reference_losses.append(float(reference.detach()))
            topology_losses.append(float(topology.detach()))
            step += 1
        monitor_c, monitor_p = None, None
        if epoch % cfg.monitor_every == 0 or epoch == cfg.epochs:
            scores = []
            for path in validation:
                image = cv2.resize(v2.read_rgb(path), (512, 512), interpolation=cv2.INTER_AREA)
                prob = v2.predict_pair(ema, image, cfg, device, False)["none"]
                scores.append(
                    (
                        base.dice_score(prob >= 0.5, base.strict_label(labels[path.stem])),
                        base.dice_score(prob >= 0.5, base.converter_label(labels[path.stem])),
                    )
                )
            monitor_c, monitor_p = map(float, np.mean(scores, axis=0))
        row = dict(
            epoch=epoch,
            loss=float(np.mean(losses)),
            reference_loss=float(np.mean(reference_losses)),
            soft_cldice_loss=float(np.mean(topology_losses)),
            monitor_clean_05=monitor_c,
            monitor_proxy_05=monitor_p,
            plan_sha256=plan_hash,
            seconds=round(time.perf_counter() - begin, 3),
        )
        history.append(row)
        if shutil.disk_usage(folder).free < 1200 * 1024**2:
            raise OSError(
                "Less than 1.2 GiB free; preserve checkpoint and free space before resuming"
            )
        base.atomic_checkpoint(
            last,
            dict(
                model=model.state_dict(),
                ema=ema.state_dict(),
                optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(),
                epoch=epoch,
                step=step,
                history=history,
                rng=base.rng_state(),
                train_ids=train_ids,
                val_ids=val_ids,
                method=METHOD,
                config=asdict(cfg),
                initial_model_hash=initial_hash,
            ),
        )
        abl.write_table(folder / "history.csv", history)
        print(
            f"{METHOD}/fold{fold} {epoch:03d}/{cfg.epochs} "
            f'loss={row["loss"]:.4f} cl={row["soft_cldice_loss"]:.4f} seconds={row["seconds"]}',
            flush=True,
        )
    base.atomic_checkpoint(
        folder / "final_ema.pt",
        dict(
            model=ema.state_dict(),
            epoch=cfg.epochs,
            method=METHOD,
            config=asdict(cfg),
            train_ids=train_ids,
            val_ids=val_ids,
            initial_model_hash=initial_hash,
        ),
    )
    values = {}
    for path in validation:
        image = cv2.resize(v2.read_rgb(path), (512, 512), interpolation=cv2.INTER_AREA)
        for mode, prob in v2.predict_pair(ema, image, cfg, device).items():
            values[f"{mode}_{path.stem}"] = prob
    np.savez_compressed(folder / "validation_probabilities.npz", **values)
    names = ["last.pt", "final_ema.pt", "history.csv", "validation_probabilities.npz"]
    base.save_json(
        folder / "completed.json",
        dict(
            epoch=cfg.epochs,
            method=METHOD,
            config=asdict(cfg),
            train_ids=train_ids,
            val_ids=val_ids,
            initial_model_hash=initial_hash,
            plan_hashes=[r["plan_sha256"] for r in history],
            sha256={n: base.file_sha(folder / n) for n in names},
        ),
    )
    del model, ema, optimizer, scaler, dataset, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()


def verify_new_fold(folder, cfg, expected):
    done = json.loads((folder / "completed.json").read_text())
    if (
        done["method"] != METHOD
        or done["epoch"] != cfg.epochs
        or done["config"] != asdict(cfg)
        or any(
            done[k] != expected[k]
            for k in ["initial_model_hash", "train_ids", "val_ids", "plan_hashes"]
        )
    ):
        raise ValueError(f"New method fold identity/pairing mismatch: {folder}")
    for name, digest in done["sha256"].items():
        if base.file_sha(folder / name) != digest:
            raise ValueError(f"New method artifact changed: {folder/name}")
    valid_probability(folder / "validation_probabilities.npz", done["val_ids"])
    return done


def evaluate_probabilities(name, probabilities, truth, assignment):
    rows = []
    grid = []
    calibration = []
    for mode, values in probabilities.items():
        for threshold in [0.1, 0.4, 0.5]:
            rows += abl.evaluate_masks(
                f"{name}_{mode}_t{threshold:g}",
                {i: p >= threshold for i, p in values.items()},
                truth,
                assignment,
            )
        for target in ["clean", "proxy"]:
            group = []
            for threshold in abl.CALIBRATION_GRID:
                scores = [
                    float(
                        np.mean(
                            [
                                base.dice_score(p >= threshold, truth[i][target])
                                for i, p in values.items()
                                if assignment[i] == fold
                            ]
                        )
                    )
                    for fold in range(5)
                ]
                r = dict(
                    method=name,
                    mode=mode,
                    target=target,
                    threshold=threshold,
                    mean_dice=float(np.mean(scores)),
                    worst_fold_dice=min(scores),
                    **{f"fold{f}": v for f, v in enumerate(scores)},
                )
                group.append(r)
            grid += group
            best = max(group, key=lambda r: (r["mean_dice"], -abs(r["threshold"] - 0.5)))
            calibration.append(
                dict(
                    **best,
                    selection_caveat="OOF-tuned development estimate; not nested or independent validation",
                )
            )
    return rows, grid, calibration


def reference_probabilities(source, expected):
    probabilities = {mode: {} for mode in ["none", "d4"]}
    for fold in range(5):
        values = valid_probability(
            Path(source) / "L2_soft" / f"fold{fold}" / "validation_probabilities.npz",
            expected[str(fold)]["val_ids"],
        )
        for mode in probabilities:
            for i in expected[str(fold)]["val_ids"]:
                probabilities[mode][i] = values[f"{mode}_{i}"]
    return probabilities


def compare_reference_metrics(rows, expected_rows):
    lookup = {
        (r["experiment"].replace(REFERENCE, "L2_soft", 1), r["target"], r["id"]): r for r in rows
    }
    metrics = [
        "dice",
        "precision",
        "recall",
        "centerline_precision",
        "centerline_recall",
        "cldice",
        "foreground_fraction",
        "thin_clean_centerline_recall",
    ]
    if len(expected_rows) != 240 or len(lookup) != 240:
        raise ValueError("Need the full 240-row frozen L2 baseline metric table")
    drift = {k: 0.0 for k in metrics}
    for expected in expected_rows:
        actual = lookup[expected["experiment"], expected["target"], expected["id"]]
        for k in metrics:
            drift[k] = max(drift[k], abs(float(actual[k]) - float(expected[k])))
    if any(v > 1e-10 for v in drift.values()):
        raise ValueError(f"Baseline metric replay differs from returned L2 tables: {drift}")
    return drift


def previews(output, paths, truth, probabilities):
    folder = output / "oof_error_previews"
    folder.mkdir(exist_ok=True)
    images = {p.stem: p for p in paths}
    for i in sorted(images, key=int):
        c, p = truth[i]["clean"], truth[i]["proxy"]
        a = probabilities[REFERENCE]["d4"][i] >= 0.5
        b = probabilities[METHOD]["d4"][i] >= 0.5
        rgb = cv2.resize(v2.read_rgb(images[i]), (512, 512), interpolation=cv2.INTER_AREA)

        def mask(m):
            return np.repeat((m.astype(np.uint8) * 255)[..., None], 3, axis=2)

        def errors(m):
            view = mask(m & c)
            view[m & ~c] = [235, 65, 65]
            view[c & ~m] = [70, 125, 245]
            return view

        change = mask(a & b)
        change[b & ~a] = [70, 200, 100]
        change[a & ~b] = [230, 110, 70]
        panels = [rgb, mask(c), mask(p), change, mask(a), mask(b), errors(a), errors(b)]
        names = [
            f"OOF {i} RGB",
            "Clean C",
            "Converter proxy P",
            "Green added / orange removed",
            "L2 reference D4 >=0.5",
            "L2 + clDice D4 >=0.5",
            "Ref vs C: red FP / blue FN",
            "New vs C: red FP / blue FN",
        ]
        canvas = Image.new("RGB", (2048, 1080), "white")
        draw = ImageDraw.Draw(canvas)
        for j, (panel, name) in enumerate(zip(panels, names)):
            x, y = (j % 4) * 512, (j // 4) * 540
            draw.text((x + 6, y + 7), name, fill="black")
            canvas.paste(Image.fromarray(panel), (x, y + 28))
        canvas.save(folder / f"oof_{int(i):02d}.png")


def aggregate(output, source, cfg, labels, paths, assignment, expected, make_previews=True):
    complete = []
    for fold in range(5):
        folder = output / METHOD / f"fold{fold}"
        if (folder / "completed.json").exists():
            verify_new_fold(folder, cfg, expected[str(fold)])
            complete.append(fold)
    if complete != list(range(5)):
        return [], dict(
            complete_folds=complete,
            complete_20_image_oof=False,
            partial_results_are_not_full_oof=True,
            paired_initialization_and_plans=True,
        )
    truth = abl.targets(labels)
    probabilities = {
        REFERENCE: reference_probabilities(source, expected),
        METHOD: {mode: {} for mode in ["none", "d4"]},
    }
    for fold in range(5):
        values = valid_probability(
            output / METHOD / f"fold{fold}" / "validation_probabilities.npz",
            expected[str(fold)]["val_ids"],
        )
        for mode in ["none", "d4"]:
            for i in expected[str(fold)]["val_ids"]:
                probabilities[METHOD][mode][i] = values[f"{mode}_{i}"]
    rows, grid, calibration = [], [], []
    for method, probs in probabilities.items():
        if any(set(values) != set(assignment) for values in probs.values()):
            raise ValueError("Incomplete OOF coverage")
        r, g, c = evaluate_probabilities(method, probs, truth, assignment)
        rows += r
        grid += g
        calibration += c
    summary, folds = abl.summarize_rows(rows)
    abl.write_table(output / "structure_per_image.csv", rows)
    abl.write_table(output / "structure_per_fold.csv", folds)
    abl.write_table(output / "structure_summary.csv", summary)
    abl.write_table(output / "structure_threshold_grid.csv", grid)
    abl.write_table(output / "structure_calibration_development.csv", calibration)
    lookup = {(r["experiment"], r["target"], r["id"]): r for r in rows}
    paired, paired_summary, paired_folds = [], [], []
    metrics = [
        "dice",
        "precision",
        "recall",
        "centerline_precision",
        "centerline_recall",
        "cldice",
        "foreground_fraction",
        "thin_clean_centerline_recall",
    ]
    for mode in ["none", "d4"]:
        for threshold in [0.1, 0.4, 0.5]:
            for target in ["clean", "proxy"]:
                for key in metrics:
                    ds = []
                    fs = []
                    for i in sorted(assignment, key=int):
                        new = lookup[f"{METHOD}_{mode}_t{threshold:g}", target, i][key]
                        old = lookup[f"{REFERENCE}_{mode}_t{threshold:g}", target, i][key]
                        delta = float(new - old)
                        ds.append((i, delta))
                        paired.append(
                            dict(
                                mode=mode,
                                threshold=threshold,
                                target=target,
                                metric=key,
                                id=i,
                                fold=assignment[i],
                                delta=delta,
                            )
                        )
                    for fold in range(5):
                        delta = float(np.mean([d for i, d in ds if assignment[i] == fold]))
                        fs.append(delta)
                        paired_folds.append(
                            dict(
                                mode=mode,
                                threshold=threshold,
                                target=target,
                                metric=key,
                                fold=fold,
                                delta=delta,
                            )
                        )
                    paired_summary.append(
                        dict(
                            mode=mode,
                            threshold=threshold,
                            target=target,
                            metric=key,
                            mean_delta=float(np.mean([d for _, d in ds])),
                            positive_images=sum(d > 0 for _, d in ds),
                            negative_images=sum(d < 0 for _, d in ds),
                            positive_folds=sum(d > 0 for d in fs),
                            negative_folds=sum(d < 0 for d in fs),
                            min_fold_delta=min(fs),
                            max_fold_delta=max(fs),
                        )
                    )
    abl.write_table(output / "structure_paired_per_image.csv", paired)
    abl.write_table(output / "structure_paired_per_fold.csv", paired_folds)
    abl.write_table(output / "structure_paired_summary.csv", paired_summary)
    if make_previews:
        previews(output, paths, truth, probabilities)
    return summary, dict(
        complete_folds=complete,
        complete_20_image_oof=True,
        paired_initialization_and_plans=True,
        reference_source="verified original L2 OOF probabilities",
        comparison="new fixed final80 EMA versus old fixed final80 EMA; no outer-fold checkpoint selection",
        automatic_model_selection=False,
        public_scores_used=False,
        competition_csv_generated=False,
    )


def analysis_archive(output):
    path = output / "structure_ablation_analysis_results.zip"
    temporary = path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in output.rglob("*"):
            if (
                p.is_file()
                and p.suffix in {".json", ".csv", ".png", ".md", ".py"}
                and "reference_cache" not in p.relative_to(output).parts
                and p.name != "training.lock"
            ):
                z.write(p, p.relative_to(output).as_posix())
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None:
            raise IOError("Analysis ZIP CRC failed")
    os.replace(temporary, path)
    return path


def restore_output(source, output):
    source = Path(source).resolve()
    if source == output:
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Resume requires a new empty OUTPUT_DIR, or use the existing working output directly"
        )
    if output.is_relative_to(source):
        raise ValueError("Resume destination cannot be inside its source")
    if source.is_dir():
        if not (source / "structure_config.json").exists():
            raise ValueError("Not a structure-stage resume folder")
        shutil.copytree(
            source,
            output,
            ignore=shutil.ignore_patterns("*.zip", "*.tmp", "training.lock", "reference_cache"),
        )
    elif source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as z:
            safe_members(z)
            if "structure_config.json" not in z.namelist() or not any(
                n.endswith("/last.pt") for n in z.namelist()
            ):
                raise ValueError(
                    "Resume needs full structure output with checkpoints, not the lightweight analysis ZIP"
                )
            output.mkdir(parents=True, exist_ok=True)
            for item in z.infolist():
                if item.filename.endswith(("training.lock", ".tmp", ".zip")):
                    continue
                z.extract(item, output)
    else:
        raise ValueError("RESUME_FROM must be a full structure-stage folder/backup")


def run_structure(
    data_dir,
    output_dir,
    cfg,
    baseline_run,
    expected,
    expected_config,
    expected_rows=None,
    folds_to_run=(0, 1, 2, 3, 4),
    weights_path=None,
    resume_from=None,
    engineering=False,
):
    if cfg.cldice_weight != 0.1 or cfg.skeleton_iterations != 10 or cfg.skeleton_smooth != 1.0:
        raise ValueError(
            "This experiment fixes clDice weight=0.1, iterations=10, smooth=1; no loss-weight search"
        )
    if any(asdict(cfg)[k] != v for k, v in expected_config["config"].items()):
        raise ValueError("Training recipe must match paired L2 reference exactly")
    if not engineering and (
        cfg.epochs != 80
        or cfg.patches_per_epoch != 256
        or cfg.batch_size != 8
        or expected_rows is None
    ):
        raise ValueError(
            "Formal run requires the frozen 5-fold80 recipe and frozen reference metrics"
        )
    if len(set(folds_to_run)) != len(folds_to_run) or not set(folds_to_run) <= set(range(5)):
        raise ValueError("Invalid requested folds")
    output = Path(output_dir).resolve()
    if resume_from and not (output / "structure_config.json").exists():
        restore_output(resume_from, output)
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(
        config=asdict(cfg),
        source_sha256=base.file_sha(__file__),
        reference_source_sha256=base.file_sha(abl.__file__),
        v2_source_sha256=base.file_sha(v2.__file__),
        base_source_sha256=base.file_sha(base.__file__),
        reference_expected_records_hash=abl.digest_json(expected),
        reference_expected_config=expected_config,
        engineering_only=engineering,
    )
    marker = output / "structure_config.json"
    previous = json.loads(marker.read_text()) if marker.exists() else None
    if previous and any(previous.get(k) != v for k, v in identity.items()):
        raise ValueError("Changed source/config/baseline: use a new output directory")
    for key, name in [
        ("source_sha256", abl.__file__),
        ("v2_source_sha256", v2.__file__),
        ("base_source_sha256", base.__file__),
    ]:
        if expected_config[key] != base.file_sha(name):
            raise ValueError("Original L2 helper source changed")
    torch.set_num_threads(cfg.threads)
    cv2.setNumThreads(0)
    with v2.run_lock(output):
        train, _, labels, assignment, _, audit = v2.audit_v2(base.find_data(data_dir), output, cfg)
        identity["data_fingerprint"] = audit["data_fingerprint"]
        if identity["data_fingerprint"] != expected_config["data_fingerprint"]:
            raise ValueError("Dataset fingerprint differs from the frozen L2 reference")
        if previous and previous != identity:
            raise ValueError("Dataset identity changed")
        source = materialize_reference(baseline_run, output)
        if source == output or output.is_relative_to(source):
            raise ValueError("Keep the new output outside the read-only baseline output")
        verify_reference(source, expected, expected_config, assignment)
        if expected_rows is not None:
            truth = abl.targets(labels)
            replay_rows, _, _ = evaluate_probabilities(
                REFERENCE, reference_probabilities(source, expected), truth, assignment
            )
            drift = compare_reference_metrics(replay_rows, expected_rows)
        else:
            drift = None
        reference_env = (
            json.loads((source / "execution_environment.json").read_text())
            if (source / "execution_environment.json").exists()
            else {}
        )
        current_env = environment()
        differences = {
            k: dict(reference=reference_env.get(k), current=current_env.get(k))
            for k in set(reference_env) | set(current_env)
            if reference_env.get(k) != current_env.get(k)
        }
        base.save_json(marker, identity)
        base.save_json(
            output / "reference_audit.json",
            dict(
                source=str(source),
                all_five_probability_hashes_verified=True,
                reference_records=expected,
                baseline_metric_max_absolute_drift=drift,
                environment_differences=differences,
                reference_environment=reference_env,
            ),
        )
        base.save_json(output / "execution_environment.json", current_env)
        runtime = output / "runtime_source"
        runtime.mkdir(exist_ok=True)
        for path in [__file__, abl.__file__, v2.__file__, base.__file__]:
            shutil.copyfile(path, runtime / Path(path).name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("Enable GPU for the new structural-loss training arm")
        if weights_path is None and (source / "pretrained_cache/resnet34-b627a593.pth").exists():
            weights_path = source / "pretrained_cache/resnet34-b627a593.pth"
        initial, provenance = v2.resolve_weights(cfg, output, weights_path)
        if (
            provenance["sha256"]
            != "b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f"
        ):
            raise ValueError("Expected the same official generic ImageNet initializer")
        base.save_json(output / "pretrained_source.json", provenance)
        for fold in folds_to_run:
            folder = output / METHOD / f"fold{fold}"
            if (folder / "completed.json").exists():
                verify_new_fold(folder, cfg, expected[str(fold)])
                print(f"{METHOD}/fold{fold}: completed, reusing", flush=True)
                continue
            development = [p for p in train if assignment[p.stem] != fold]
            validation = [p for p in train if assignment[p.stem] == fold]
            train_fold(
                development,
                validation,
                labels,
                cfg,
                fold,
                folder,
                initial,
                device,
                expected[str(fold)],
            )
        summary, status = aggregate(output, source, cfg, labels, train, assignment, expected)
        status.update(
            requested_folds=list(folds_to_run),
            new_training_arms=1,
            reference_retrained=False,
            engineering_only=engineering,
            automatic_model_selection=False,
            public_scores_used=False,
            competition_csv_generated=False,
            test_predictions_generated=False,
            environment_differences=differences,
            summary=summary,
        )
        base.save_json(output / "structure_status.json", status)
        print("OOF-only analysis:", analysis_archive(output), flush=True)
    return output
