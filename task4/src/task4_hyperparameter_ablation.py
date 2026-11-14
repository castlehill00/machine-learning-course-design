"""Paired, one-factor learning-rate and BCE/Dice sensitivity on the L1 recipe.

Four new five-fold arms plus a frozen historical reference. No public score is
used, and no interaction or independent generalization claim is made.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import shutil
import time
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import task4_controlled_ablation as abl

base, v2 = abl.base, abl.v2


# ---------- 数值稳定性：AMP 出现非有限损失时，同一批次用 FP32 重算一次 ----------
# 有限的 AMP 结果保持不变；重算前不裁剪损失、不改学习率、不跳过样本。
NUMERICAL_POLICY = dict(
    version=1,
    finite_amp_path="unchanged",
    nonfinite_amp_loss="retry same batch once in FP32",
    nonfinite_fp32_loss="record diagnostics and stop; never mask numerical failure",
    restore_before_retry="RNG and model buffers",
    changed_training_hyperparameters=False,
    effect="Fallback changes effective numerical precision and must be reported",
)


def nonfinite_paths(value, prefix="", limit=8):
    result = []

    def walk(v, path):
        if len(result) >= limit:
            return
        if torch.is_tensor(v):
            if (v.is_floating_point() or v.is_complex()) and not bool(torch.isfinite(v).all()):
                result.append(path)
        elif isinstance(v, dict):
            for k, x in v.items():
                walk(x, path + "." + str(k))
        elif isinstance(v, (tuple, list)):
            for i, x in enumerate(v):
                walk(x, path + "." + str(i))

    walk(value, prefix)
    return result


def record(path, event):
    path = Path(path)
    saved = (
        json.loads(path.read_text()) if path.exists() else dict(policy=NUMERICAL_POLICY, events=[])
    )
    if saved["policy"] != NUMERICAL_POLICY:
        raise ValueError("Numerical-repair policy changed")
    saved["events"].append(event)
    base.save_json(path, saved)


def tensor_stats(tensor):
    x = tensor.detach().float()
    finite = torch.isfinite(x)
    values = x[finite]
    return dict(
        shape=list(x.shape),
        dtype=str(tensor.dtype),
        nonfinite=int((~finite).sum()),
        finite_abs_max=float(values.abs().max()) if values.numel() else None,
    )


def restore_forward_state(model, rng, buffers):
    base.restore_rng(rng)
    with torch.no_grad():
        for n, b in model.named_buffers():
            b.copy_(buffers[n])


def loss_with_retry(model, image, pair, beta, amp, objective, event_path, location):
    if nonfinite_paths({"image": image, "target": pair}):
        record(
            event_path,
            dict(
                **location,
                event="invalid_input",
                image=tensor_stats(image),
                target=tensor_stats(pair),
            ),
        )
        raise FloatingPointError(f"Invalid input at {location}; see {event_path}")
    enabled = bool(amp and image.device.type == "cuda")
    rng = base.rng_state()
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    with torch.autocast(device_type=image.device.type, enabled=enabled):
        outputs = model(image)
        loss, terms = objective(outputs, pair, beta)
    if not nonfinite_paths((loss, terms)):
        return loss, terms
    diagnostics = dict(
        **location,
        event="amp_loss_nonfinite" if enabled else "fp32_loss_nonfinite",
        outputs=[tensor_stats(x) for x in outputs],
        loss=tensor_stats(loss),
        nonfinite_model_state=nonfinite_paths(model.state_dict()),
    )
    # Release the invalid graph before allocating a complete FP32 graph.
    del outputs, loss, terms
    if not enabled:
        record(event_path, diagnostics)
        raise FloatingPointError(f"FP32 loss is nonfinite at {location}; see {event_path}")
    restore_forward_state(model, rng, buffers)
    with torch.autocast(device_type=image.device.type, enabled=False):
        outputs = model(image.float())
        loss, terms = objective(outputs, pair.float(), beta)
    success = not nonfinite_paths((loss, terms))
    diagnostics.update(
        fp32_retry_succeeded=success,
        fp32_outputs=[tensor_stats(x) for x in outputs],
        fp32_loss=tensor_stats(loss),
    )
    record(event_path, diagnostics)
    if not success:
        raise FloatingPointError(
            f"AMP and FP32 loss are nonfinite at {location}; see {event_path}. Last complete epoch is preserved."
        )
    print(
        f"FP32 retry succeeded: {location}; original learning rate, samples and loss weights retained",
        flush=True,
    )
    return loss, terms


def predict_with_retry(
    model, image, cfg, device, predict_pair, event_path, location, with_tta=True
):
    values = predict_pair(model, image, cfg, device, with_tta)
    invalid = any(not np.isfinite(p).all() for p in values.values())
    if not invalid:
        return values
    if not cfg.amp:
        record(event_path, dict(**location, event="fp32_inference_nonfinite"))
        raise FloatingPointError(f"FP32 inference is nonfinite at {location}; see {event_path}")
    values = predict_pair(model, image, replace(cfg, amp=False), device, with_tta)
    success = all(np.isfinite(p).all() for p in values.values())
    record(event_path, dict(**location, event="inference_fp32_retry", fp32_retry_succeeded=success))
    if not success:
        raise FloatingPointError(f"FP32 inference retry failed at {location}; see {event_path}")
    return values


def summarize(output):
    rows = []
    for path in sorted(Path(output).glob("*/fold*/*precision_events.json")):
        saved = json.loads(path.read_text())
        events = saved["events"]
        rows.append(
            dict(
                file=path.relative_to(output).as_posix(),
                event_count=len(events),
                fp32_retries=sum(e.get("fp32_retry_succeeded") is True for e in events),
                failed_retries=sum(e.get("fp32_retry_succeeded") is False for e in events),
            )
        )
    return dict(
        policy=NUMERICAL_POLICY,
        per_file=rows,
        total_events=sum(r["event_count"] for r in rows),
        fp32_retries=sum(r["fp32_retries"] for r in rows),
        limitation="Completed legacy folds are preserved; fallback precision is an additional numerical intervention when activated.",
    )


REFERENCE = "H0_reference"
SETTINGS = {
    REFERENCE: dict(lr_multiplier=1.0, bce_weight=0.5, factor="reference"),
    "H1_lr_half": dict(lr_multiplier=0.5, bce_weight=0.5, factor="learning_rate"),
    "H2_lr_double": dict(lr_multiplier=2.0, bce_weight=0.5, factor="learning_rate"),
    "H3_bce_quarter": dict(lr_multiplier=1.0, bce_weight=0.25, factor="bce_weight"),
    "H4_bce_three_quarters": dict(lr_multiplier=1.0, bce_weight=0.75, factor="bce_weight"),
}
NEW_SETTINGS = tuple(s for s in SETTINGS if s != REFERENCE)
LEGACY_SOURCE_SHA = "70d55fd30dbc2447fec8cc1f74b3d90d8b6f474cc57f3e23324a0d42bb9af98a"
PREVIOUS_REPAIR_SHA = "899c57318f4f889507806f507b89324f3eb3ab7092c4583db066a5a8f31de666"
METRICS = (
    "dice",
    "precision",
    "recall",
    "centerline_precision",
    "centerline_recall",
    "cldice",
    "foreground_fraction",
    "thin_clean_centerline_recall",
)
POLICY = dict(
    primary_ablation="D4 fixed threshold0.5, proxy-GT mean image Dice; also report clean GT",
    secondary="same frozen 13-point threshold grid for each setting, with both targets reported",
    development_selection="highest calibrated D4 proxy-GT mean Dice; ties prefer baseline, then nearest baseline setting",
    threshold_tie="nearest0.5",
    final_winner_selection=False,
    public_scores_used=False,
    baseline="cached completed recovery L1; not retrained concurrently",
    checkpoint="fixed final80 EMA; validation never chooses training epoch",
    geometry="RGB and targets512, patch256, batch8, 256patches/epoch; positive centers from clean C",
    optimizer_retention="last.pt during unfinished fold; discard only after validated final EMA and OOF exist",
    limitation="Reused 20-image OOF development selection, not nested CV; one seed; no LR/loss interaction estimate",
)


def state_hash(model):
    return abl.digest_json(
        {
            k: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
            for k, v in model.state_dict().items()
        }
    )


def setting_config(cfg, name):
    s = SETTINGS[name]
    return replace(
        cfg,
        encoder_lr=cfg.encoder_lr * s["lr_multiplier"],
        decoder_lr=cfg.decoder_lr * s["lr_multiplier"],
    )


def objective(outputs, pair, bce_weight):
    if not 0 <= bce_weight <= 1:
        raise ValueError("BCE weight outside [0,1]")
    target = pair[:, 1:2]
    bces, dices = [], []
    for logits in outputs:
        gt = F.interpolate(target, size=logits.shape[-2:], mode="area").float()
        prob = logits.float().sigmoid()
        dice = (2 * (prob * gt).sum((1, 2, 3)) + 1) / (prob.sum((1, 2, 3)) + gt.sum((1, 2, 3)) + 1)
        bces.append(F.binary_cross_entropy_with_logits(logits.float(), gt))
        dices.append(1 - dice.mean())
    # Preserve the original per-scale arithmetic when bce_weight == 0.5.
    mixed = [bce_weight * b + (1 - bce_weight) * d for b, d in zip(bces, dices)]
    loss = (mixed[0] + 0.1 * mixed[1] + 0.2 * mixed[2]) / 1.3
    terms = [(v[0] + 0.1 * v[1] + 0.2 * v[2]) / 1.3 for v in (bces, dices)]
    return loss, terms


def rgb(path):
    return cv2.resize(v2.read_rgb(path), (512, 512), interpolation=cv2.INTER_AREA)


def train_fold(paths, validation, labels, cfg, name, fold, folder, initial, device, reference):
    actual = setting_config(cfg, name)
    folder.mkdir(parents=True, exist_ok=True)
    base.seed_all(cfg.seed + 1009 * fold)
    dataset = abl.PairedPatches(paths, labels, actual, fold)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = v2.ResNet34UNet(initial).to(device)
    initial_hash = state_hash(model)
    train_ids, val_ids = [p.stem for p in paths], [p.stem for p in validation]
    if (
        initial_hash != reference["initial_model_hash"]
        or train_ids != reference["train_ids"]
        or val_ids != reference["val_ids"]
    ):
        raise ValueError("Shared initialization or split differs from frozen L1 reference")
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": actual.encoder_lr},
            {
                "params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")],
                "lr": actual.decoder_lr,
            },
        ],
        weight_decay=actual.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    history, first, step = [], 1, 0
    last = folder / "last.pt"
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        for k, v in dict(
            setting=name,
            parameters=SETTINGS[name],
            config=asdict(actual),
            train_ids=train_ids,
            val_ids=val_ids,
            initial_model_hash=initial_hash,
        ).items():
            if saved.get(k) != v:
                raise ValueError("Checkpoint mismatch: " + k)
        invalid = nonfinite_paths({k: saved[k] for k in ["model", "ema", "optimizer"]})
        if invalid:
            record(
                folder / "precision_events.json",
                dict(event="checkpoint_nonfinite", paths=invalid, epoch=saved["epoch"]),
            )
            raise FloatingPointError(
                f"Nonfinite saved checkpoint in {folder}; original files preserved; see precision_events.json"
            )
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        history, first, step = saved["history"], saved["epoch"] + 1, saved["step"]
        base.restore_rng(saved["rng"])
        del saved
        print(f"{name}/fold{fold}: resume {first}/{cfg.epochs}", flush=True)
    for epoch in range(first, cfg.epochs + 1):
        begin = time.perf_counter()
        model.train()
        plan = dataset.set_epoch(epoch)
        if plan != reference["plan_hashes"][epoch - 1]:
            raise ValueError("RGB sampling/augmentation plan changed")
        values = []
        for batch_index, (image, pair) in enumerate(loader, 1):
            v2.set_lrs(optimizer, step, len(loader), actual)
            image, pair = image.to(device, non_blocking=True), pair.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = loss_with_retry(
                model,
                image,
                pair,
                SETTINGS[name]["bce_weight"],
                cfg.amp,
                objective,
                folder / "precision_events.json",
                dict(setting=name, fold=fold, epoch=epoch, batch=batch_index, stage="training"),
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                v2.update_ema(ema, model, cfg.ema_decay)
            values.append([float(loss.detach())] + [float(x.detach()) for x in terms])
            step += 1
        monitor_c, monitor_p = None, None
        if epoch % cfg.monitor_every == 0 or epoch == cfg.epochs:
            scores = []
            for path in validation:
                prob = predict_with_retry(
                    ema,
                    rgb(path),
                    actual,
                    device,
                    v2.predict_pair,
                    folder / "precision_events.json",
                    dict(setting=name, fold=fold, epoch=epoch, image=path.stem, stage="monitor"),
                    False,
                )["none"]
                scores.append(
                    (
                        base.dice_score(prob >= 0.5, base.strict_label(labels[path.stem])),
                        base.dice_score(prob >= 0.5, base.converter_label(labels[path.stem])),
                    )
                )
            monitor_c, monitor_p = map(float, np.mean(scores, axis=0))
        means = np.mean(values, axis=0)
        history.append(
            dict(
                epoch=epoch,
                loss=float(means[0]),
                bce_loss=float(means[1]),
                dice_loss=float(means[2]),
                monitor_clean_05=monitor_c,
                monitor_proxy_05=monitor_p,
                plan_sha256=plan,
                encoder_lr=float(optimizer.param_groups[0]["lr"]),
                decoder_lr=float(optimizer.param_groups[1]["lr"]),
                seconds=round(time.perf_counter() - begin, 3),
            )
        )
        if shutil.disk_usage(folder).free < 1200 * 1024**2:
            raise OSError("Insufficient disk space for next checkpoint")
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
                setting=name,
                parameters=SETTINGS[name],
                config=asdict(actual),
                train_ids=train_ids,
                val_ids=val_ids,
                initial_model_hash=initial_hash,
            ),
        )
        abl.write_table(folder / "history.csv", history)
        print(
            f'{name}/fold{fold} {epoch:03d}/{cfg.epochs} loss={means[0]:.4f} seconds={history[-1]["seconds"]}',
            flush=True,
        )
    base.atomic_checkpoint(
        folder / "final_ema.pt",
        dict(
            model=ema.state_dict(),
            epoch=cfg.epochs,
            setting=name,
            parameters=SETTINGS[name],
            config=asdict(actual),
            train_ids=train_ids,
            val_ids=val_ids,
            initial_model_hash=initial_hash,
        ),
    )
    arrays = {}
    for path in validation:
        values = predict_with_retry(
            ema,
            rgb(path),
            actual,
            device,
            v2.predict_pair,
            folder / "precision_events.json",
            dict(setting=name, fold=fold, epoch=cfg.epochs, image=path.stem, stage="oof"),
        )
        for mode, p in values.items():
            arrays[f"{mode}_{path.stem}"] = p
    np.savez_compressed(folder / "validation_probabilities.npz", **arrays)
    names = ["final_ema.pt", "history.csv", "validation_probabilities.npz"]
    if (folder / "precision_events.json").exists():
        names.append("precision_events.json")
    base.save_json(
        folder / "completed.json",
        dict(
            epoch=cfg.epochs,
            setting=name,
            parameters=SETTINGS[name],
            config=asdict(actual),
            train_ids=train_ids,
            val_ids=val_ids,
            initial_model_hash=initial_hash,
            plan_hashes=[r["plan_sha256"] for r in history],
            optimizer_checkpoint_retained=False,
            numerical_policy=NUMERICAL_POLICY,
            numerical_source_sha256=base.file_sha(__file__),
            sha256={n: base.file_sha(folder / n) for n in names},
        ),
    )
    verify_fold(folder, cfg, name, reference)
    # Only this run's finished optimizer checkpoint; the final EMA remains.
    last.unlink(missing_ok=True)
    del model, ema, optimizer, scaler, dataset, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()


def verify_fold(folder, cfg, name, reference, check_weights=True):
    done = json.loads((folder / "completed.json").read_text())
    expected = dict(
        epoch=cfg.epochs,
        setting=name,
        parameters=SETTINGS[name],
        config=asdict(setting_config(cfg, name)),
    )
    for k, v in expected.items():
        if done.get(k) != v:
            raise ValueError("Completed setting mismatch: " + k)
    for k in ["initial_model_hash", "train_ids", "val_ids", "plan_hashes"]:
        if done[k] != reference[k]:
            raise ValueError("Reference pairing mismatch: " + k)
    for n, h in done["sha256"].items():
        if not check_weights and n.endswith(".pt"):
            continue
        if base.file_sha(folder / n) != h:
            raise ValueError("Changed artifact: " + str(folder / n))
    with np.load(folder / "validation_probabilities.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != {f"{m}_{i}" for m in ("none", "d4") for i in done["val_ids"]}:
            raise ValueError("Wrong OOF probability keys")
        for p in arrays.values():
            if p.shape != (512, 512) or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
                raise ValueError("Invalid OOF probabilities")
    return done


def compare_reference(output, rows):
    lookup = {(r["experiment"], r["target"], r["id"]): r for r in rows}
    paired, summary = [], []
    for r in rows:
        if r["experiment"].startswith(REFERENCE + "_"):
            continue
        name = next(s for s in NEW_SETTINGS if r["experiment"].startswith(s + "_"))
        ref = lookup[REFERENCE + r["experiment"][len(name) :], r["target"], r["id"]]
        for k in METRICS:
            paired.append(
                dict(
                    setting=name,
                    experiment=r["experiment"],
                    target=r["target"],
                    id=r["id"],
                    fold=r["fold"],
                    metric=k,
                    delta=float(r[k]) - float(ref[k]),
                )
            )
    for e, t, k in sorted({(r["experiment"], r["target"], r["metric"]) for r in paired}):
        group = [r for r in paired if (r["experiment"], r["target"], r["metric"]) == (e, t, k)]
        ds = [r["delta"] for r in group]
        fs = [float(np.mean([r["delta"] for r in group if int(r["fold"]) == f])) for f in range(5)]
        summary.append(
            dict(
                setting=group[0]["setting"],
                experiment=e,
                target=t,
                metric=k,
                mean_delta=float(np.mean(ds)),
                positive_images=sum(d > 0 for d in ds),
                positive_folds=sum(d > 0 for d in fs),
                **{f"fold{f}_delta": d for f, d in enumerate(fs)},
            )
        )
    if paired:
        abl.write_table(output / "hyperparameter_paired_per_image.csv", paired)
        abl.write_table(output / "hyperparameter_paired_summary.csv", summary)
    return summary


def selection_key(row):
    name = row["setting"]
    s = SETTINGS[name]
    distance = abs(np.log2(s["lr_multiplier"])) + abs(s["bce_weight"] - 0.5)
    return float(row["mean_dice"]), name == REFERENCE, -distance, name


def analyze(output, cfg, labels, assignment, records, reference_rows, reference_grid):
    complete, counts = [], {}
    for name in NEW_SETTINGS:
        counts[name] = [f for f in range(5) if (output / name / f"fold{f}/completed.json").exists()]
        if counts[name] == list(range(5)):
            for f in range(5):
                verify_fold(output / name / f"fold{f}", cfg, name, records[str(f)], False)
            complete.append(name)
    truth = abl.targets(labels)
    rows = [
        dict(r, experiment=r["experiment"].replace("L1_proxy", REFERENCE, 1))
        for r in reference_rows
    ]
    grid = [dict(r, setting=REFERENCE) for r in reference_grid]
    probabilities = {}
    for name in complete:
        probs = {m: {} for m in ["none", "d4"]}
        for f in range(5):
            with np.load(
                output / name / f"fold{f}/validation_probabilities.npz", allow_pickle=False
            ) as z:
                for m in probs:
                    for i in records[str(f)]["val_ids"]:
                        probs[m][i] = z[f"{m}_{i}"].copy()
        if any(set(v) != set(assignment) for v in probs.values()):
            raise ValueError("Incomplete OOF")
        probabilities[name] = probs
        for m, values in probs.items():
            for t in [0.1, 0.4, 0.5]:
                rows += abl.evaluate_masks(
                    f"{name}_{m}_t{t:g}", {i: p >= t for i, p in values.items()}, truth, assignment
                )
            for target in ["clean", "proxy"]:
                for t in abl.CALIBRATION_GRID:
                    fs = [
                        float(
                            np.mean(
                                [
                                    base.dice_score(p >= t, truth[i][target])
                                    for i, p in values.items()
                                    if assignment[i] == f
                                ]
                            )
                        )
                        for f in range(5)
                    ]
                    grid.append(
                        dict(
                            mode=m,
                            target=target,
                            threshold=t,
                            mean_dice=float(np.mean(fs)),
                            worst_fold_dice=min(fs),
                            **{f"fold{f}": s for f, s in enumerate(fs)},
                            setting=name,
                        )
                    )
        for i, p in probs["d4"].items():
            folder = output / "oof_binary_masks" / name
            folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray(((p >= 0.5) * 255).astype(np.uint8)).save(folder / f"{i}.png")
    summaries, folds = abl.summarize_rows(rows)
    calibrated = []
    for name in [REFERENCE] + complete:
        for m in ["none", "d4"]:
            for target in ["clean", "proxy"]:
                group = [
                    r for r in grid if (r["setting"], r["mode"], r["target"]) == (name, m, target)
                ]
                best = max(
                    group, key=lambda r: (float(r["mean_dice"]), -abs(float(r["threshold"]) - 0.5))
                )
                calibrated.append(
                    dict(
                        **best,
                        caveat="OOF-tuned development estimate; not independent or nested validation",
                    )
                )
    for n, rs in [
        ("hyperparameter_per_image.csv", rows),
        ("hyperparameter_per_fold.csv", folds),
        ("hyperparameter_summary.csv", summaries),
        ("hyperparameter_threshold_grid.csv", grid),
        ("hyperparameter_calibration_development.csv", calibrated),
    ]:
        abl.write_table(output / n, rs)
    paired = compare_reference(output, rows)
    panel = []
    for factor, names in [
        ("learning_rate", ["H1_lr_half", REFERENCE, "H2_lr_double"]),
        ("bce_weight", ["H3_bce_quarter", REFERENCE, "H4_bce_three_quarters"]),
    ]:
        for name in names:
            if name not in [REFERENCE] + complete:
                continue
            for target in ["clean", "proxy"]:
                s = next(
                    r
                    for r in summaries
                    if r["experiment"] == f"{name}_d4_t0.5" and r["target"] == target
                )
                panel.append(
                    dict(
                        parameter=factor,
                        setting=name,
                        **SETTINGS[name],
                        target=target,
                        threshold=0.5,
                        dice=s["dice"],
                        worst_fold_dice=s["worst_fold_dice"],
                        fold_dice_std=s["fold_dice_std"],
                    )
                )
    abl.write_table(output / "hyperparameter_sensitivity.csv", panel)
    status = dict(
        complete_all_settings=complete == list(NEW_SETTINGS),
        completed_settings=complete,
        completed_folds=counts,
        reference_retrained=False,
        new_training_settings=4,
        policy=POLICY,
        public_scores_used=False,
        automatic_final_winner_selection=False,
        candidate_csv_generated=False,
        selected_setting=None,
        selected_threshold=None,
        submitted_to_competition=False,
        primary_fixed05_selection=None,
        model_weight_download_required=False,
    )
    if not status["complete_all_settings"]:
        return status
    fixed = [
        dict(
            setting=n,
            mean_dice=next(
                r["dice"]
                for r in summaries
                if r["experiment"] == f"{n}_d4_t0.5" and r["target"] == "proxy"
            ),
        )
        for n in SETTINGS
    ]
    chosen_fixed = max(fixed, key=selection_key)
    candidates = [r for r in calibrated if r["mode"] == "d4" and r["target"] == "proxy"]
    selected = max(candidates, key=selection_key)
    baseline = next(r for r in candidates if r["setting"] == REFERENCE)
    ranking = []
    for n in SETTINGS:
        c = next(r for r in candidates if r["setting"] == n)
        f = next(r for r in fixed if r["setting"] == n)
        deltas = [float(c[f"fold{i}"]) - float(baseline[f"fold{i}"]) for i in range(5)]
        ranking.append(
            dict(
                setting=n,
                **SETTINGS[n],
                fixed05_proxy_dice=float(f["mean_dice"]),
                calibrated_proxy_dice=float(c["mean_dice"]),
                selected_threshold=float(c["threshold"]),
                calibrated_worst_fold_dice=float(c["worst_fold_dice"]),
                calibrated_delta_vs_reference=float(c["mean_dice"]) - float(baseline["mean_dice"]),
                positive_folds=sum(d > 0 for d in deltas),
                **{f"fold{i}_delta": v for i, v in enumerate(deltas)},
            )
        )
    ranking.sort(
        key=lambda r: (-r["calibrated_proxy_dice"], r["setting"] != REFERENCE, r["setting"])
    )
    abl.write_table(output / "hyperparameter_choices_development.csv", ranking)
    status.update(
        selected_setting=selected["setting"],
        selected_threshold=float(selected["threshold"]),
        primary_fixed05_selection=chosen_fixed,
        development_ranking=ranking,
        caveat="Both parameter and threshold selection reuse the same OOF; not an unbiased performance estimate",
        test_inference_required=selected["setting"] != REFERENCE,
    )
    if selected["setting"] != REFERENCE:
        name, t = selected["setting"], float(selected["threshold"])
        masks = {i: p >= t for i, p in probabilities[name]["d4"].items()}
        candidate_rows = abl.evaluate_masks(
            "selected_development_candidate", masks, truth, assignment
        )
        s, f = abl.summarize_rows(candidate_rows)
        for n, rs in [
            ("candidate_oof_per_image.csv", candidate_rows),
            ("candidate_oof_per_fold.csv", f),
            ("candidate_oof_summary.csv", s),
        ]:
            abl.write_table(output / n, rs)
        for i, m in masks.items():
            folder = output / "oof_binary_masks" / "selected_candidate"
            folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray((m * 255).astype(np.uint8)).save(folder / f"{i}.png")
    return status


def infer_test_fold(folder, cfg, name, test, device):
    done = json.loads((folder / "completed.json").read_text())
    model_path = folder / "final_ema.pt"
    if base.file_sha(model_path) != done["sha256"]["final_ema.pt"]:
        raise ValueError("Selected final EMA changed")
    ids = [p.stem for p in test]
    identity = dict(
        setting=name,
        parameters=SETTINGS[name],
        final_ema_sha256=done["sha256"]["final_ema.pt"],
        test_image_sha256={p.stem: base.file_sha(p) for p in test},
        config=asdict(setting_config(cfg, name)),
    )
    marker, target = folder / "test_completed.json", folder / "test_probabilities.npz"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["identity"] != identity or base.file_sha(target) != saved["sha256"]:
            raise ValueError("Test cache identity changed")
    else:
        model = v2.ResNet34UNet(None).to(device)
        saved = torch.load(model_path, map_location="cpu", weights_only=True)
        if saved["setting"] != name or saved["epoch"] != cfg.epochs:
            raise ValueError("Wrong selected test model")
        model.load_state_dict(saved["model"])
        model.eval()
        values = {
            p.stem: predict_with_retry(
                model,
                rgb(p),
                cfg,
                device,
                v2.predict_pair,
                folder / "test_precision_events.json",
                dict(setting=name, fold=folder.name, image=p.stem, stage="test"),
            )["d4"]
            for p in test
        }
        np.savez_compressed(target, **values)
        base.save_json(marker, dict(identity=identity, sha256=base.file_sha(target)))
        del model, saved
        if device.type == "cuda":
            torch.cuda.empty_cache()
    with np.load(target, allow_pickle=False) as z:
        if set(z.files) != set(ids):
            raise ValueError("Wrong test cache IDs")
        result = {i: z[i].copy() for i in ids}
    for p in result.values():
        if p.shape != (512, 512) or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
            raise ValueError("Invalid test probabilities")
    return result


def export_selected(output, root, cfg, status, test, device):
    if not status["complete_all_settings"]:
        return status
    if status["selected_setting"] == REFERENCE:
        status.update(
            test_inference_complete=True,
            candidate_export_reason="Historical baseline wins development comparison; no new CSV needed",
        )
        return status
    name, threshold = status["selected_setting"], status["selected_threshold"]
    mean = {p.stem: np.zeros((512, 512), np.float64) for p in test}
    models = []
    for fold in range(5):
        folder = output / name / f"fold{fold}"
        values = infer_test_fold(folder, cfg, name, test, device)
        for i, p in values.items():
            mean[i] += p.astype(np.float64) / 5
        models.append(dict(fold=fold, **json.loads((folder / "test_completed.json").read_text())))
    exported = v2.export_csv(
        mean, [p.stem for p in test], threshold, "submission_hyperparameter_candidate", root, output
    )
    # Preserve the exact float64 mean used for thresholding and CSV export.
    np.savez_compressed(output / "selected_test_ensemble.npz", **mean)
    status.update(
        candidate_csv_generated=True,
        exported=exported,
        test_inference_complete=True,
        test_models=models,
        submitted_to_competition=False,
        candidate_export_reason="One OOF-selected development candidate; requires review, not automatic final selection",
    )
    return status


def package(output):
    files = [p for p in output.rglob("*") if p.is_file() and p.name != "training.lock"]
    paths = {}
    for kind in ["analysis", "oof"]:
        path = output / f"hyperparameter_{kind}.zip"
        temporary = path.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in files:
                include = p.suffix in {".json", ".csv", ".png", ".md", ".py"}
                if kind == "oof" and p.name == "validation_probabilities.npz":
                    include = True
                if include:
                    z.write(p, p.relative_to(output).as_posix())
        with zipfile.ZipFile(temporary) as z:
            if z.testzip() is not None:
                raise IOError("ZIP CRC failed")
        os.replace(temporary, path)
        paths[kind] = str(path)
    return paths


def restore(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        return
    if not source.is_dir() or not (source / "hyperparameter_config.json").is_file():
        raise ValueError(
            "Resume from this experiment full mounted Output directory; a lightweight ZIP cannot resume"
        )
    if output.is_relative_to(source):
        raise ValueError("Resume destination cannot be inside source")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty resume destination")
    if not any(source.glob("*/fold*/*.pt")):
        raise ValueError("No final weights or resumable checkpoints in source")
    shutil.copytree(
        source, output, ignore=shutil.ignore_patterns("*.zip", "*.tmp", "training.lock")
    )


def copy_source_file(source, destination):
    source, destination = Path(source), Path(destination)
    if (
        source.exists()
        and destination.exists()
        and (source.resolve() == destination.resolve() or source.samefile(destination))
    ):
        return destination
    return shutil.copyfile(source, destination)


def migrate_legacy_identity(output, previous, identity):
    """Only the delivered legacy source may migrate, with every recipe field equal."""
    previous_sha = previous.get("source_sha256")
    if previous_sha not in (LEGACY_SOURCE_SHA, PREVIOUS_REPAIR_SHA):
        raise ValueError("Not the supported pre-repair experiment")
    allowed = (
        {"source_sha256", "numerical_policy", "numerical_source_sha256"}
        if previous_sha == LEGACY_SOURCE_SHA
        else {"source_sha256"}
    )
    old = {k: v for k, v in previous.items() if k != "data_fingerprint"}
    changed = {k for k in set(old) | set(identity) if old.get(k) != identity.get(k)}
    if not changed <= allowed:
        raise ValueError(
            "Repair cannot change the recipe/reference: " + str(sorted(changed - allowed))
        )
    original = output / "runtime_source/task4_hyperparameter_ablation.py"
    if not original.exists() or base.file_sha(original) != previous_sha:
        raise ValueError("Original delivered source missing or changed; refusing silent migration")
    suffix = "precision_repair" if previous_sha == LEGACY_SOURCE_SHA else "source_copy_fix"
    archived = output / f"runtime_source/task4_hyperparameter_ablation_before_{suffix}.py"
    if archived.exists() and archived.read_bytes() != original.read_bytes():
        raise ValueError("Conflicting original-source archive")
    if not archived.exists():
        shutil.copyfile(original, archived)
    backup = output / f"hyperparameter_config_before_{suffix}.json"
    if backup.exists() and json.loads(backup.read_text()) != previous:
        raise ValueError("Conflicting original-config archive")
    if not backup.exists():
        base.save_json(backup, previous)
    completed = {
        p.parent.relative_to(output).as_posix(): base.file_sha(p)
        for p in sorted(output.glob("*/fold*/completed.json"))
    }
    incomplete = {
        p.parent.relative_to(output).as_posix(): base.file_sha(p)
        for p in sorted(output.glob("*/fold*/last.pt"))
        if not (p.parent / "completed.json").exists()
    }
    migration_name = (
        "precision_repair_migration.json"
        if previous_sha == LEGACY_SOURCE_SHA
        else "source_copy_fix_migration.json"
    )
    base.save_json(
        output / migration_name,
        dict(
            original_source_sha256=previous_sha,
            repaired_source_sha256=identity["source_sha256"],
            numerical_source_sha256=identity["numerical_source_sha256"],
            numerical_policy=NUMERICAL_POLICY,
            changed_identity_fields=sorted(changed),
            completed_folds_preserved=completed,
            incomplete_checkpoints_preserved=incomplete,
            all_training_hyperparameters_unchanged=True,
            resume_rule="Resume incomplete fold from its last complete epoch; verified completed folds are reused",
            limitation="Any activated FP32 retry is an additional numerical intervention, recorded explicitly",
        ),
    )
    print(
        f"Precision repair migration: preserve {len(completed)} completed folds and {len(incomplete)} unfinished checkpoints",
        flush=True,
    )
    return dict(identity, data_fingerprint=previous["data_fingerprint"])


def run(
    data_dir,
    output_dir,
    cfg,
    records,
    reference_rows,
    reference_grid,
    reference_info,
    weights_path=None,
    settings_to_run=NEW_SETTINGS,
    folds_to_run=(0, 1, 2, 3, 4),
    resume_from=None,
    on_package=None,
    engineering=False,
):
    if not engineering and asdict(cfg) != reference_info["config"]:
        raise ValueError("Keep the frozen seed42/fivefold/final80 recipe")
    if len(set(settings_to_run)) != len(settings_to_run) or not set(settings_to_run) <= set(
        NEW_SETTINGS
    ):
        raise ValueError("Invalid hyperparameter settings")
    if len(set(folds_to_run)) != len(folds_to_run) or not set(folds_to_run) <= set(range(5)):
        raise ValueError("Invalid folds")
    output = Path(output_dir).resolve()
    if resume_from and not (output / "hyperparameter_config.json").exists():
        restore(resume_from, output)
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(
        config=asdict(cfg),
        settings=SETTINGS,
        policy=POLICY,
        source_sha256=base.file_sha(__file__),
        helper_source_sha256={
            Path(p).name: base.file_sha(p) for p in [abl.__file__, v2.__file__, base.__file__]
        },
        reference_info=reference_info,
        reference_records_hash=abl.digest_json(records),
        reference_rows_hash=abl.digest_json(reference_rows),
        reference_grid_hash=abl.digest_json(reference_grid),
        engineering_only=engineering,
        numerical_policy=NUMERICAL_POLICY,
        numerical_source_sha256=base.file_sha(__file__),
    )
    marker = output / "hyperparameter_config.json"
    previous = json.loads(marker.read_text()) if marker.exists() else None
    legacy = previous is not None and previous.get("source_sha256") in (
        LEGACY_SOURCE_SHA,
        PREVIOUS_REPAIR_SHA,
    )
    if previous and not legacy and any(previous.get(k) != v for k, v in identity.items()):
        raise ValueError("Configuration/source/reference changed: use a new output directory")
    if previous is None and any(output.iterdir()):
        raise ValueError("New experiment requires an empty output directory")
    torch.set_num_threads(cfg.threads)
    cv2.setNumThreads(0)
    with v2.run_lock(output):
        if legacy:
            previous = migrate_legacy_identity(output, previous, identity)
        root = base.find_data(data_dir)
        train, test, labels, assignment, ids, audit = v2.audit_v2(root, output, cfg)
        if audit["data_fingerprint"] != reference_info["data_fingerprint"]:
            raise ValueError("Dataset differs from paired reference")
        identity["data_fingerprint"] = audit["data_fingerprint"]
        if previous and previous != identity:
            raise ValueError("Dataset identity changed")
        base.save_json(marker, identity)
        import PIL, skimage

        env = dict(
            python=platform.python_version(),
            torch=str(torch.__version__),
            torchvision=str(v2.torchvision.__version__),
            numpy=np.__version__,
            opencv=cv2.__version__,
            pillow=PIL.__version__,
            skimage=skimage.__version__,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        )
        differences = {
            k: dict(reference=v, current=env.get(k))
            for k, v in reference_info.get("environment", {}).items()
            if env.get(k) != v
        }
        base.save_json(output / "execution_environment.json", env)
        base.save_json(output / "reference_info.json", reference_info)
        base.save_json(output / "reference_records.json", records)
        abl.write_table(output / "reference_L1_per_image.csv", reference_rows)
        abl.write_table(output / "reference_L1_threshold_grid.csv", reference_grid)
        source_folder = output / "runtime_source"
        source_folder.mkdir(exist_ok=True)
        for path in [__file__, abl.__file__, v2.__file__, base.__file__]:
            copy_source_file(path, source_folder / Path(path).name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("Enable GPU for the paired training experiment")
        if weights_path is None:
            for name in ["task4_core_band", "task4_recovery", "task4_v2_results"]:
                p = Path("/kaggle/working") / name / "pretrained_cache/resnet34-b627a593.pth"
                if p.is_file():
                    weights_path = p
                    break
        initial, provenance = v2.resolve_weights(cfg, output, weights_path)
        if (
            provenance["sha256"]
            != "b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f"
        ):
            raise ValueError("Wrong generic ImageNet initializer")
        base.save_json(output / "pretrained_source.json", provenance)
        for name in settings_to_run:
            for fold in folds_to_run:
                folder = output / name / f"fold{fold}"
                if (folder / "completed.json").exists():
                    verify_fold(folder, cfg, name, records[str(fold)])
                    print(f"{name}/fold{fold}: verified complete, reusing", flush=True)
                    continue
                dev = [p for p in train if assignment[p.stem] != fold]
                val = [p for p in train if assignment[p.stem] == fold]
                train_fold(
                    dev, val, labels, cfg, name, fold, folder, initial, device, records[str(fold)]
                )
                status = dict(
                    complete_all_settings=False,
                    engineering_only=engineering,
                    completed_folds={
                        s: [
                            f for f in range(5) if (output / s / f"fold{f}/completed.json").exists()
                        ]
                        for s in NEW_SETTINGS
                    },
                    candidate_csv_generated=False,
                    public_scores_used=False,
                    model_weight_download_required=False,
                )
                base.save_json(output / "hyperparameter_status.json", status)
                paths = package(output)
                if on_package is not None:
                    on_package(output, paths)
        status = analyze(output, cfg, labels, assignment, records, reference_rows, reference_grid)
        status.update(
            engineering_only=engineering,
            environment_differences=differences,
            environment_matches_historical_reference=not differences,
            numerical_repair_summary=summarize(output),
        )
        base.save_json(output / "hyperparameter_status.json", status)
        status = export_selected(output, root, cfg, status, test, device)
        status["numerical_repair_summary"] = summarize(output)
        base.save_json(output / "hyperparameter_status.json", status)
        paths = package(output)
        if on_package is not None:
            on_package(output, paths)
        print(
            (
                "COMPLETE: download hyperparameter_analysis.zip (and candidate CSV if generated)."
                if status["complete_all_settings"]
                else "PARTIAL: remaining settings/folds must finish before parameter selection."
            ),
            flush=True,
        )
    return output
