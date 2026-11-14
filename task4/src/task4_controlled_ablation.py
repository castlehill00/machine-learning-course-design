"""OOF-only boundary/EMA replay and paired label-supervision experiments.

No test-set predictions, competition submissions, or public-score-based selection.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import task4_pipeline as base
import task4_kaggle_v2 as v2

VARIANTS = ("L0_clean", "L1_proxy", "L2_soft")
CALIBRATION_GRID = (0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass
class AblationConfig(v2.ConfigV2):
    soft_boundary: float = 0.5


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_table(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def skeleton(mask):
    try:
        from skimage.morphology import skeletonize
    except ImportError as exc:
        raise RuntimeError(
            "Install scikit-image for the preregistered centerline metrics."
        ) from exc
    return skeletonize(np.asarray(mask, bool))


def targets(labels):
    result = {}
    for i, path in labels.items():
        c, p = base.strict_label(path), base.converter_label(path)
        if np.any(c & ~p):
            raise ValueError("Expected clean mask to be a subset of converter proxy")
        sc, sp = skeleton(c), skeleton(p)
        radius = cv2.distanceTransform(c.astype(np.uint8), cv2.DIST_L2, 5)
        result[i] = dict(
            clean=c,
            proxy=p,
            clean_skeleton=sc,
            proxy_skeleton=sp,
            thin_clean_centerline=sc & (radius <= 1.5),
        )
    return result


def segmentation_metrics(pred, truth, truth_skeleton, pred_skeleton):
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    tp = int((pred & truth).sum())
    n_pred = int(pred.sum())
    n_true = int(truth.sum())
    precision = tp / n_pred if n_pred else (1.0 if not n_true else 0.0)
    recall = tp / n_true if n_true else 1.0
    top_precision = (
        float((pred_skeleton & truth).sum() / pred_skeleton.sum())
        if pred_skeleton.any()
        else (1.0 if not n_true else 0.0)
    )
    top_recall = (
        float((truth_skeleton & pred).sum() / truth_skeleton.sum()) if truth_skeleton.any() else 1.0
    )
    cldice = (
        2 * top_precision * top_recall / (top_precision + top_recall)
        if top_precision + top_recall
        else 0.0
    )
    return dict(
        dice=base.dice_score(pred, truth),
        precision=precision,
        recall=recall,
        centerline_precision=top_precision,
        centerline_recall=top_recall,
        cldice=cldice,
        foreground_fraction=n_pred / pred.size,
    )


def evaluate_masks(experiment, masks, truth, assignments):
    rows = []
    for i in sorted(masks, key=int):
        mask = np.asarray(masks[i], bool)
        sp = skeleton(mask)
        thin = truth[i]["thin_clean_centerline"]
        thin_recall = float((mask & thin).sum() / thin.sum()) if thin.any() else None
        for target in ["clean", "proxy"]:
            values = segmentation_metrics(
                mask, truth[i][target], truth[i][target + "_skeleton"], sp
            )
            rows.append(
                dict(
                    experiment=experiment,
                    id=i,
                    fold=assignments[i],
                    target=target,
                    **values,
                    thin_clean_centerline_recall=thin_recall,
                )
            )
    return rows


def summarize_rows(rows):
    numeric = [
        "dice",
        "precision",
        "recall",
        "centerline_precision",
        "centerline_recall",
        "cldice",
        "foreground_fraction",
        "thin_clean_centerline_recall",
    ]
    summaries, folds = [], []
    for experiment, target in sorted({(r["experiment"], r["target"]) for r in rows}):
        group = [r for r in rows if r["experiment"] == experiment and r["target"] == target]
        assert len({r["id"] for r in group}) == len(group)
        values = {
            key: (
                float(np.mean([r[key] for r in group if r[key] is not None]))
                if any(r[key] is not None for r in group)
                else None
            )
            for key in numeric
        }
        fold_values = []
        for fold in sorted({r["fold"] for r in group}):
            subset = [r for r in group if r["fold"] == fold]
            score = float(np.mean([r["dice"] for r in subset]))
            fold_values.append(score)
            folds.append(
                dict(
                    experiment=experiment,
                    target=target,
                    fold=fold,
                    n_images=len(subset),
                    **{
                        key: (
                            float(np.mean([r[key] for r in subset if r[key] is not None]))
                            if any(r[key] is not None for r in subset)
                            else None
                        )
                        for key in numeric
                    },
                )
            )
        summaries.append(
            dict(
                experiment=experiment,
                target=target,
                n_images=len(group),
                complete_20_image_oof=len(group) == 20,
                **values,
                worst_fold_dice=min(fold_values),
                fold_dice_std=float(np.std(fold_values)),
            )
        )
    return summaries, folds


def probability_preview(images, truth, probs, output):
    ids = sorted(probs, key=int)[:4]
    tile = 220
    canvas = Image.new("RGB", (tile * 4, (tile + 24) * len(ids)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, i in enumerate(ids):
        y = row * (tile + 24)
        p = probs[i]
        hard = p >= 0.4
        low = p >= 0.1
        dil = (
            cv2.dilate(hard.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            > 0
        )
        panels = [Image.open(images[i]).convert("RGB")] + [
            Image.fromarray((m * 255).astype(np.uint8)).convert("RGB") for m in [hard, low, dil]
        ]
        for col, (panel, label) in enumerate(
            zip(panels, [f"OOF {i}", "p >= 0.4", "p >= 0.1", "dilate(p >= 0.4)"])
        ):
            draw.text((col * tile + 3, y + 4), label, fill="black")
            canvas.paste(panel.resize((tile, tile)), (col * tile, y + 24))
    canvas.save(output)


def materialize_previous(source, output):
    """Use a mounted folder read-only, or extract only the artifacts used by replay."""
    if source is None:
        preferred = Path("/kaggle/working/task4_v2_results")
        roots = [preferred] if (preferred / "run_config.json").exists() else []
        if not roots and Path("/kaggle/input").exists():
            for p in Path("/kaggle/input").rglob("run_config.json"):
                try:
                    cfg = json.loads(p.read_text())
                except (ValueError, OSError):
                    continue
                if cfg.get("source_sha256") == base.file_sha(v2.__file__) and all(
                    (p.parent / f"fold{f}/validation_probabilities.npz").exists() for f in range(5)
                ):
                    roots.append(p.parent)
        roots = sorted(set(p.resolve() for p in roots))
        if len(roots) == 1:
            return roots[0]
        zips = (
            list(Path("/kaggle/input").rglob("full_backup.zip"))
            if Path("/kaggle/input").exists()
            else []
        )
        if not roots and len(zips) == 1:
            source = zips[0]
        else:
            raise ValueError(
                f"Set PREVIOUS_RUN to the previous complete output folder or full_backup.zip. Folders found: {roots}"
            )
    source = Path(source).resolve()
    if source.is_dir():
        return source
    if not source.is_file() or source.suffix.lower() != ".zip":
        raise ValueError("PREVIOUS_RUN must be a folder or full_backup.zip")
    cached = output / "previous_run_cache"
    cached.mkdir(parents=True, exist_ok=True)
    archive_sha = base.file_sha(source)
    marker = cached / "archive_identity.json"
    if marker.exists():
        if json.loads(marker.read_text())["archive_sha256"] != archive_sha:
            raise ValueError("Different previous archive: use a new OUTPUT_DIR")
        return cached
    with zipfile.ZipFile(source) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP members")
        wanted = []
        for name in names:
            dest = (cached / name).resolve()
            if not dest.is_relative_to(cached):
                raise ValueError("Unsafe archive path")
            parts = Path(name).parts
            if (
                name in {"run_config.json", "pretrained_source.json"}
                or (
                    len(parts) == 2
                    and parts[0] in {f"fold{f}" for f in range(5)}
                    and parts[1]
                    in {"completed.json", "last.pt", "validation_probabilities.npz", "history.csv"}
                )
                or (
                    len(parts) == 2
                    and parts[0] == "pretrained_cache"
                    and parts[1] == "resnet34-b627a593.pth"
                )
            ):
                wanted.append(name)
        required_members = {"run_config.json"} | {
            f"fold{f}/{name}"
            for f in range(5)
            for name in ["completed.json", "validation_probabilities.npz"]
        }
        if not required_members <= set(wanted):
            raise ValueError(
                "The analysis-only ZIP cannot support these ablations; use the full backup with all five OOF probability files."
            )
        required = sum(z.getinfo(n).file_size for n in wanted)
        if shutil.disk_usage(output).free < required + 512 * 1024**2:
            raise OSError("Not enough disk space to extract the required prior artifacts")
        for name in wanted:
            z.extract(name, cached)
    base.save_json(marker, dict(archive_sha256=archive_sha, extracted_files=wanted))
    return cached


def validate_previous(
    previous, assignment, fingerprint, expected=None, require_last=True, expected_epochs=80
):
    identity = json.loads((previous / "run_config.json").read_text())
    if identity["source_sha256"] != base.file_sha(v2.__file__) or identity[
        "baseline_helpers_sha256"
    ] != base.file_sha(base.__file__):
        raise ValueError("Previous source differs from the pinned v2 implementation")
    if identity["data_fingerprint"] != fingerprint:
        raise ValueError("Previous run uses different training data/split")
    cfg = v2.ConfigV2(**identity["config"])
    if cfg.epochs != expected_epochs:
        raise ValueError(f"Expected a completed {expected_epochs}-epoch run, found {cfg.epochs}")
    records = {}
    for fold in range(5):
        folder = previous / f"fold{fold}"
        done = json.loads((folder / "completed.json").read_text())
        if done["epoch"] != cfg.epochs or set(done["val_ids"]) != {
            i for i, f in assignment.items() if f == fold
        }:
            raise ValueError(f"Incomplete/wrong split: fold{fold}")
        wanted = ["validation_probabilities.npz"] + (["last.pt"] if require_last else [])
        for name in wanted:
            if not (folder / name).is_file():
                raise ValueError(
                    f"Missing fold{fold}/{name}; use the complete prior output/full backup, not the analysis-only ZIP."
                )
            actual = base.file_sha(folder / name)
            if actual != done["sha256"][name]:
                raise ValueError(f"Changed previous artifact: fold{fold}/{name}")
            if expected is not None and actual != expected[str(fold)]["sha256"][name]:
                raise ValueError(
                    f"Previous artifact does not match the returned analysis archive: fold{fold}/{name}"
                )
        records[str(fold)] = done
    return cfg, records


def replay_fold(previous, fold, paths, cfg, device, cache):
    checkpoint = previous / f"fold{fold}/last.pt"
    key = dict(
        checkpoint_sha256=base.file_sha(checkpoint),
        config=asdict(cfg),
        gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        torch=torch.__version__,
    )
    marker = cache / f"fold{fold}_identity.json"
    npz = cache / f"fold{fold}_replay.npz"
    if marker.exists() and npz.exists():
        old = json.loads(marker.read_text())
        if old["identity"] != key or old["npz_sha256"] != base.file_sha(npz):
            raise ValueError("Replay cache changed; use a new output directory")
        with np.load(npz) as arrays:
            return {k: arrays[k].copy() for k in arrays.files}
    # Hashes were checked against the known previous run before loading its own resume file.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_ids = {p.stem for p in paths}
    if (
        state["epoch"] != cfg.epochs
        or set(state["val_ids"]) != expected_ids
        or set(state["train_ids"]) & expected_ids
    ):
        raise ValueError("OOF leakage/epoch check failed")
    model = v2.ResNet34UNet().to(device)
    values = {}
    for name, key_name in [("replay_ema", "ema"), ("raw", "model")]:
        model.load_state_dict(state[key_name])
        model.eval()
        for p in paths:
            pair = v2.predict_pair(model, p, cfg, device)
            for mode, prob in pair.items():
                values[f"{name}_{mode}_{p.stem}"] = v2.resize_probability(prob)
    del model, state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, **values)
    base.save_json(marker, dict(identity=key, npz_sha256=base.file_sha(npz)))
    return values


def run_replay(
    previous, output, train, labels, assignment, fingerprint, expected=None, raw_replay=True
):
    cfg, records = validate_previous(previous, assignment, fingerprint, expected, raw_replay)
    truth = targets(labels)
    cached = {}
    replayed = {"replay_ema": {}, "raw": {}}
    drift = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw_replay and device.type != "cuda":
        raise RuntimeError("Enable GPU for paired EMA/raw replay, or turn RUN_RAW_REPLAY off")
    for fold in range(5):
        val = [p for p in train if assignment[p.stem] == fold]
        with np.load(previous / f"fold{fold}/validation_probabilities.npz") as arrays:
            for p in val:
                prob = arrays[f"d4_{p.stem}"].copy()
                if (
                    prob.shape != (512, 512)
                    or not np.isfinite(prob).all()
                    or prob.min() < 0
                    or prob.max() > 1
                ):
                    raise ValueError("Invalid cached OOF probability")
                cached[p.stem] = prob
        if raw_replay:
            values = replay_fold(previous, fold, val, cfg, device, output / "replay_cache")
            for p in val:
                for name in replayed:
                    replayed[name][p.stem] = values[f"{name}_d4_{p.stem}"]
                delta = np.abs(replayed["replay_ema"][p.stem] - cached[p.stem])
                drift.append(
                    dict(
                        id=p.stem,
                        fold=fold,
                        mean_absolute_probability_drift=float(delta.mean()),
                        max_absolute_probability_drift=float(delta.max()),
                    )
                )
        print(f"Replay fold{fold} completed; no training", flush=True)
    if set(cached) != set(assignment):
        raise AssertionError("Incomplete OOF coverage")
    experiments = {
        "E1_ema_t04": {i: p >= 0.4 for i, p in cached.items()},
        "E1_ema_t01": {i: p >= 0.1 for i, p in cached.items()},
        "E1_ema_t04_dilate1": {
            i: cv2.dilate(
                (p >= 0.4).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            )
            > 0
            for i, p in cached.items()
        },
    }
    if raw_replay:
        for name, probs in replayed.items():
            for t, label in [(0.1, "t01"), (0.4, "t04")]:
                experiments[f"E2_{name}_{label}"] = {i: p >= t for i, p in probs.items()}
    rows = [
        row
        for name, masks in experiments.items()
        for row in evaluate_masks(name, masks, truth, assignment)
    ]
    summary, folds = summarize_rows(rows)
    write_table(output / "replay_per_image.csv", rows)
    write_table(output / "replay_summary.csv", summary)
    write_table(output / "replay_per_fold.csv", folds)
    write_table(output / "ema_replay_drift.csv", drift)
    probability_preview(
        {p.stem: p for p in train}, truth, cached, output / "boundary_ablation_preview.png"
    )
    base.save_json(
        output / "replay_audit.json",
        dict(
            complete_20_image_oof=True,
            fold_records=records,
            raw_and_ema_replayed_in_same_environment=raw_replay,
            previous_probability_hashes_verified=True,
            training_started=False,
            competition_csv_generated=False,
            interpretation="Morphology and centerline changes are paired diagnostics; no winner selected from public scores.",
        ),
    )
    return summary


class PairedPatches(Dataset):
    """All label variants receive exactly the same RGB patches and augmentations."""

    def __init__(self, paths, labels, cfg, fold):
        self.ids = [p.stem for p in paths]
        self.cfg = cfg
        self.fold = fold
        self.images = [
            cv2.resize(v2.read_rgb(p), (512, 512), interpolation=cv2.INTER_AREA) for p in paths
        ]
        self.clean = [base.strict_label(labels[i]).astype(np.float32) for i in self.ids]
        self.proxy = [base.converter_label(labels[i]).astype(np.float32) for i in self.ids]
        self.positive = [np.argwhere(c > 0) for c in self.clean]
        self.plans = []

    def set_epoch(self, epoch):
        self.plans = []
        for index in range(self.cfg.patches_per_epoch):
            rng = random.Random((self.cfg.seed + 1009 * self.fold) * 10**10 + epoch * 10**6 + index)
            i = rng.randrange(len(self.ids))
            patch = self.cfg.patch
            crop = rng.randint(int(patch * 0.9), min(int(patch * 1.1), 512))
            if rng.random() < 0.3 and len(self.positive[i]):
                y, x = self.positive[i][rng.randrange(len(self.positive[i]))]
                top, left = int(np.clip(y - crop // 2, 0, 512 - crop)), int(
                    np.clip(x - crop // 2, 0, 512 - crop)
                )
            else:
                top, left = rng.randint(0, 512 - crop), rng.randint(0, 512 - crop)
            self.plans.append(
                (
                    i,
                    top,
                    left,
                    crop,
                    rng.randrange(4),
                    rng.random() < 0.5,
                    rng.uniform(0.85, 1.15),
                    rng.uniform(-0.04, 0.04),
                    rng.uniform(0.85, 1.15),
                    [rng.uniform(0.95, 1.05) for _ in range(3)],
                )
            )
        return digest_json(self.plans)

    def __len__(self):
        return self.cfg.patches_per_epoch

    def __getitem__(self, index):
        i, top, left, crop, rotation, flip, contrast, brightness, gamma, gains = self.plans[index]
        patch = self.cfg.patch
        image = cv2.resize(
            self.images[i][top : top + crop, left : left + crop],
            (patch, patch),
            interpolation=cv2.INTER_LINEAR,
        )
        pair = np.stack([self.clean[i], self.proxy[i]], axis=-1)
        pair = cv2.resize(
            pair[top : top + crop, left : left + crop],
            (patch, patch),
            interpolation=cv2.INTER_NEAREST,
        )
        image, pair = base.paired_d4(image, pair, rotation, flip)
        image = (
            np.clip((image.astype(np.float32) / 255 - 0.5) * contrast + 0.5 + brightness, 0, 1)
            ** gamma
        )
        image = np.clip(image * np.array(gains, np.float32), 0, 1)
        return torch.from_numpy(image.transpose(2, 0, 1).copy()), torch.from_numpy(
            pair.transpose(2, 0, 1).copy()
        )


def supervised_loss(outputs, pair, variant, alpha=0.5):
    c, p = pair[:, 0:1], pair[:, 1:2]
    if variant == "L0_clean":
        bce_target, dice_target = c, c
    elif variant == "L1_proxy":
        bce_target, dice_target = p, p
    elif variant == "L2_soft":
        bce_target, dice_target = c + alpha * (p - c), c
    else:
        raise ValueError(variant)
    losses = []
    for logits in outputs:
        bt = F.interpolate(bce_target, size=logits.shape[-2:], mode="area")
        dt = F.interpolate(dice_target, size=logits.shape[-2:], mode="area")
        prob = logits.float().sigmoid()
        dice = (2 * (prob * dt).sum((1, 2, 3)) + 1) / (prob.sum((1, 2, 3)) + dt.sum((1, 2, 3)) + 1)
        losses.append(
            0.5 * F.binary_cross_entropy_with_logits(logits.float(), bt.float())
            + 0.5 * (1 - dice.mean())
        )
    return (losses[0] + 0.1 * losses[1] + 0.2 * losses[2]) / 1.3


def train_label_fold(paths, validation, labels, cfg, fold, variant, folder, initial, device):
    folder.mkdir(parents=True, exist_ok=True)
    base.seed_all(cfg.seed + 1009 * fold)
    dataset = PairedPatches(paths, labels, cfg, fold)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = v2.ResNet34UNet(initial).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    initial_model_hash = digest_json(
        {
            k: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
            for k, v in model.state_dict().items()
        }
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
    train_ids, val_ids = [p.stem for p in paths], [p.stem for p in validation]
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if (
            saved["variant"] != variant
            or saved["train_ids"] != train_ids
            or saved["val_ids"] != val_ids
            or saved["initial_model_hash"] != initial_model_hash
        ):
            raise ValueError("Label experiment checkpoint mismatch")
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        first, step, history = saved["epoch"] + 1, saved["step"], saved["history"]
        base.restore_rng(saved["rng"])
        del saved
        print(f"{variant}/fold{fold}: resume at {first}/{cfg.epochs}", flush=True)
    for epoch in range(first, cfg.epochs + 1):
        begin = time.perf_counter()
        model.train()
        plan_hash = dataset.set_epoch(epoch)
        losses = []
        for image, pair in loader:
            v2.set_lrs(optimizer, step, len(loader), cfg)
            image, pair = image.to(device, non_blocking=True), pair.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
                loss = supervised_loss(model(image), pair, variant, cfg.soft_boundary)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite label experiment loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                v2.update_ema(ema, model, cfg.ema_decay)
            losses.append(float(loss.detach()))
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
            monitor_clean_05=monitor_c,
            monitor_proxy_05=monitor_p,
            plan_sha256=plan_hash,
            seconds=round(time.perf_counter() - begin, 3),
        )
        history.append(row)
        if shutil.disk_usage(folder).free < 1200 * 1024**2:
            raise OSError(
                "Less than 1.2 GiB free; preserve checkpoint and resume after freeing space"
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
                variant=variant,
                initial_model_hash=initial_model_hash,
            ),
        )
        write_table(folder / "history.csv", history)
        print(
            f'{variant}/fold{fold} {epoch:03d}/{cfg.epochs} loss={row["loss"]:.4f} seconds={row["seconds"]}',
            flush=True,
        )
    base.atomic_checkpoint(
        folder / "final_ema.pt",
        dict(
            model=ema.state_dict(),
            epoch=cfg.epochs,
            variant=variant,
            train_ids=train_ids,
            val_ids=val_ids,
            initial_model_hash=initial_model_hash,
        ),
    )
    values = {}
    for path in validation:
        image = cv2.resize(v2.read_rgb(path), (512, 512), interpolation=cv2.INTER_AREA)
        for mode, prob in v2.predict_pair(ema, image, cfg, device).items():
            values[f"{mode}_{path.stem}"] = prob
    np.savez_compressed(folder / "validation_probabilities.npz", **values)
    files = ["last.pt", "final_ema.pt", "history.csv", "validation_probabilities.npz"]
    base.save_json(
        folder / "completed.json",
        dict(
            epoch=cfg.epochs,
            variant=variant,
            train_ids=train_ids,
            val_ids=val_ids,
            initial_model_hash=initial_model_hash,
            plan_hashes=[r["plan_sha256"] for r in history],
            sha256={name: base.file_sha(folder / name) for name in files},
        ),
    )
    del model, ema, optimizer, scaler, dataset, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()


def aggregate_labels(output, variants, cfg, labels, assignment):
    complete = [
        variant
        for variant in variants
        if all((output / variant / f"fold{fold}/completed.json").exists() for fold in range(5))
    ]
    truth = targets(labels)
    all_rows = []
    calibrated = []
    sweeps = []
    for variant in complete:
        probs = {mode: {} for mode in ["none", "d4"]}
        for fold in range(5):
            folder = output / variant / f"fold{fold}"
            done = json.loads((folder / "completed.json").read_text())
            if done["epoch"] != cfg.epochs or set(done["val_ids"]) != {
                i for i, f in assignment.items() if f == fold
            }:
                raise ValueError("Wrong label OOF fold")
            for name, digest in done["sha256"].items():
                if base.file_sha(folder / name) != digest:
                    raise ValueError(f"Changed label experiment artifact: {folder/name}")
            with np.load(folder / "validation_probabilities.npz") as arrays:
                for mode in probs:
                    for i in done["val_ids"]:
                        probs[mode][i] = arrays[f"{mode}_{i}"].copy()
        if any(set(p) != set(assignment) for p in probs.values()):
            raise AssertionError("Incomplete label OOF")
        for mode, pred in probs.items():
            for t in [0.1, 0.4, 0.5]:
                all_rows += evaluate_masks(
                    f"{variant}_{mode}_t{t:g}",
                    {i: p >= t for i, p in pred.items()},
                    truth,
                    assignment,
                )
            for target in ["clean", "proxy"]:
                grid = []
                for t in CALIBRATION_GRID:
                    fold_scores = {
                        f: float(
                            np.mean(
                                [
                                    base.dice_score(pred[i] >= t, truth[i][target])
                                    for i in pred
                                    if assignment[i] == f
                                ]
                            )
                        )
                        for f in range(5)
                    }
                    row = dict(
                        variant=variant,
                        mode=mode,
                        target=target,
                        threshold=t,
                        mean_dice=float(np.mean(list(fold_scores.values()))),
                        worst_fold_dice=min(fold_scores.values()),
                        **{f"fold{f}": v for f, v in fold_scores.items()},
                    )
                    grid.append(row)
                    sweeps.append(row)
                best = max(grid, key=lambda r: (r["mean_dice"], -abs(r["threshold"] - 0.5)))
                calibrated.append(
                    dict(
                        **best,
                        selection_caveat="OOF-tuned development estimate; not nested or independent validation",
                    )
                )
    # Pairing is verified even when only some folds/variants are ready.
    pairing = []
    for fold in range(5):
        records = []
        for variant in variants:
            path = output / variant / f"fold{fold}/completed.json"
            if path.exists():
                records.append((variant, json.loads(path.read_text())))
        if len(records) > 1:
            first = records[0][1]
            for variant, done in records[1:]:
                if (
                    done["initial_model_hash"] != first["initial_model_hash"]
                    or done["plan_hashes"] != first["plan_hashes"]
                ):
                    raise AssertionError(
                        "Ablation pairing failed: initial weights or sampled RGB patches differ"
                    )
            pairing.append(
                dict(
                    fold=fold,
                    variants=[v for v, _ in records],
                    same_initial_weights=True,
                    same_rgb_plan=True,
                )
            )
    summary, fold_rows = summarize_rows(all_rows)
    write_table(output / "label_per_image.csv", all_rows)
    write_table(output / "label_summary.csv", summary)
    write_table(output / "label_per_fold.csv", fold_rows)
    write_table(output / "label_threshold_grid.csv", sweeps)
    write_table(output / "label_calibration_development.csv", calibrated)
    base.save_json(
        output / "label_audit.json",
        dict(
            complete_variants=complete,
            requested_variants=list(variants),
            pairing=pairing,
            complete_all_requested=len(complete) == len(variants),
            no_automatic_winner=True,
            competition_csv_generated=False,
            public_scores_used=False,
            calibration_grid=list(CALIBRATION_GRID),
            geometry="RGB/targets both 512; matched across L0/L1/L2",
        ),
    )
    return summary


def analysis_archive(output):
    path = output / "ablation_analysis_results.zip"
    tmp = output / "ablation_analysis_results.zip.tmp"
    files = [
        p
        for p in output.rglob("*")
        if p.is_file()
        and p.suffix in {".json", ".csv", ".png", ".md", ".py"}
        and "previous_run_cache" not in p.parts
        and p.name != "training.lock"
    ]
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(output).as_posix())
    with zipfile.ZipFile(tmp) as z:
        if z.testzip():
            raise IOError("Analysis ZIP CRC failed")
    os.replace(tmp, path)
    return path


def run_ablation(
    data_dir=None,
    output_dir="task4_ablation_results",
    cfg=None,
    previous_run=None,
    run_inference=True,
    raw_replay=True,
    run_training=False,
    variants=VARIANTS,
    folds_to_run=(0, 1, 2, 3, 4),
    weights_path=None,
    expected_previous=None,
    resume_from=None,
):
    cfg = cfg or AblationConfig()
    if (
        cfg.patch < 64
        or cfg.patch % 32
        or cfg.patch > 512
        or cfg.soft_boundary != 0.5
        or cfg.folds != 5
    ):
        raise ValueError(
            "Require patch multiple of 32 in [64,512], five folds, fixed soft band=0.5"
        )
    if (
        cfg.epochs < 1
        or cfg.batch_size < 1
        or cfg.patches_per_epoch < 1
        or cfg.monitor_every < 1
        or cfg.threads < 1
    ):
        raise ValueError("Invalid experiment settings")
    if not set(variants) <= set(VARIANTS) or len(set(variants)) != len(variants):
        raise ValueError("Invalid label variants")
    if not set(folds_to_run) <= set(range(5)) or len(set(folds_to_run)) != len(folds_to_run):
        raise ValueError("Invalid fold list")
    output = Path(output_dir).resolve()
    if resume_from and not (output / "ablation_config.json").exists():
        source = Path(resume_from)
        if source.is_dir() and not (source / "ablation_config.json").exists():
            raise ValueError(
                "RESUME_ABLATION_FROM must point to this ablation output, not the original v2 run"
            )
        if source.is_file():
            if source.name == "ablation_analysis_results.zip":
                raise ValueError("The lightweight analysis ZIP cannot resume training")
            with zipfile.ZipFile(source) as z:
                if "ablation_config.json" not in z.namelist():
                    raise ValueError("Not an ablation resume archive")
        v2.restore_output(source, output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg.threads)
    cv2.setNumThreads(0)
    root = base.find_data(data_dir)
    identity = dict(
        config=asdict(cfg),
        source_sha256=base.file_sha(__file__),
        v2_source_sha256=base.file_sha(v2.__file__),
        base_source_sha256=base.file_sha(base.__file__),
    )
    path = output / "ablation_config.json"
    previous_identity = json.loads(path.read_text()) if path.exists() else None
    if previous_identity and any(previous_identity.get(k) != v for k, v in identity.items()):
        raise ValueError("Changed recipe/source: use a new OUTPUT_DIR")
    with v2.run_lock(output):
        train, _, labels, assignment, _, audit = v2.audit_v2(root, output, cfg)
        identity["data_fingerprint"] = audit["data_fingerprint"]
        if previous_identity and previous_identity != identity:
            raise ValueError("Dataset changed")
        base.save_json(path, identity)
        sources = output / "runtime_source"
        sources.mkdir(exist_ok=True)
        for source in [Path(__file__), Path(v2.__file__), Path(base.__file__)]:
            shutil.copyfile(source, sources / source.name)
        previous = None
        inference_summary = []
        label_summary = []
        if run_inference:
            previous = materialize_previous(previous_run, output)
            inference_summary = run_replay(
                previous,
                output,
                train,
                labels,
                assignment,
                audit["data_fingerprint"],
                expected_previous,
                raw_replay,
            )
        if run_training:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type != "cuda":
                raise RuntimeError("Enable GPU for label-supervision training")
            if weights_path is None and previous is not None:
                candidate = previous / "pretrained_cache/resnet34-b627a593.pth"
                if candidate.exists():
                    weights_path = candidate
            initial, weight_info = v2.resolve_weights(cfg, output, weights_path)
            base.save_json(output / "pretrained_source.json", weight_info)
            for variant in variants:
                for fold in folds_to_run:
                    folder = output / variant / f"fold{fold}"
                    if (folder / "completed.json").exists():
                        print(f"{variant}/fold{fold}: completed, reusing", flush=True)
                        continue
                    development = [p for p in train if assignment[p.stem] != fold]
                    validation = [p for p in train if assignment[p.stem] == fold]
                    train_label_fold(
                        development, validation, labels, cfg, fold, variant, folder, initial, device
                    )
            label_summary = aggregate_labels(output, variants, cfg, labels, assignment)
        base.save_json(
            output / "experiment_status.json",
            dict(
                inference_done=run_inference,
                raw_replayed=run_inference and raw_replay,
                label_training_requested=run_training,
                requested_folds=list(folds_to_run),
                requested_variants=list(variants),
                competition_csv_generated=False,
                public_scores_used=False,
                automatic_model_selection=False,
                inference_summary=inference_summary,
                label_summary=label_summary,
            ),
        )
        archive = analysis_archive(output)
        print("OOF-only analysis saved:", archive, flush=True)
    return output
