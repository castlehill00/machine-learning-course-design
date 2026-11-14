"""One five-fold experiment: background/core/converter-band softmax segmentation."""

from __future__ import annotations
import copy
import hashlib
import json
import os
import platform
import shutil
import time
import zipfile
from dataclasses import asdict
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
METHOD = "core_band_3class"
LOSS = dict(
    ce=0.5,
    core_dice=0.25,
    proxy_dice=0.25,
    aux2=0.1,
    aux4=0.2,
    class_order=["background", "core", "band"],
    smooth=1.0,
)


def state_hash(model):
    return abl.digest_json(
        {
            k: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
            for k, v in model.state_dict().items()
        }
    )


class CoreBandUNet(v2.ResNet34UNet):
    def __init__(self, initial=None):
        super().__init__(initial)
        # Hash before replacing heads binds shared initialization to the old L1.
        self.reference_binary_initial_hash = state_hash(self)
        self.head = nn.Conv2d(32, 3, 1)
        self.aux2 = nn.Conv2d(48, 3, 1)
        self.aux4 = nn.Conv2d(64, 3, 1)


def categorical_target(pair):
    c, p = pair[:, :1].float(), pair[:, 1:2].float()
    if torch.any(c > p + 1e-6):
        raise ValueError("Core must be a subset of converter proxy")
    return torch.cat([1 - p, c, p - c], dim=1)


def probabilities(logits):
    p = logits.float().softmax(dim=1)
    return torch.cat([p[:, 1:2], p[:, 1:2] + p[:, 2:3]], dim=1)


def objective(outputs, pair):
    target = categorical_target(pair)
    terms = []
    for logits in outputs:
        truth = F.interpolate(target, size=logits.shape[-2:], mode="area")
        log_p = logits.float().log_softmax(dim=1)
        ce = -(truth * log_p).sum(1).mean()
        pp = probabilities(logits)
        c, p = truth[:, 1:2], truth[:, 1:2] + truth[:, 2:3]

        def dice_loss(pred, gt):
            score = (2 * (pred * gt).sum((1, 2, 3)) + 1) / (
                pred.sum((1, 2, 3)) + gt.sum((1, 2, 3)) + 1
            )
            return 1 - score.mean()

        terms.append((ce, dice_loss(pp[:, :1], c), dice_loss(pp[:, 1:2], p)))
    combined = [(terms[0][j] + 0.1 * terms[1][j] + 0.2 * terms[2][j]) / 1.3 for j in range(3)]
    loss = 0.5 * combined[0] + 0.25 * combined[1] + 0.25 * combined[2]
    return loss, combined


@torch.inference_mode()
def predict(model, image, cfg, device, with_tta=True):
    model.eval()
    if isinstance(image, (str, Path)):
        image = cv2.resize(v2.read_rgb(image), (512, 512), interpolation=cv2.INTER_AREA)
    h, w = image.shape[:2]
    image = cv2.copyMakeBorder(
        image, 0, max(0, cfg.patch - h), 0, max(0, cfg.patch - w), cv2.BORDER_REFLECT_101
    )
    height, width = image.shape[:2]
    coords = [
        (y, x)
        for y in v2.window_starts(height, cfg.patch, cfg.patch // 2)
        for x in v2.window_starts(width, cfg.patch, cfg.patch // 2)
    ]
    weight = np.maximum(np.outer(np.hanning(cfg.patch), np.hanning(cfg.patch)), 0.05).astype(
        np.float32
    )
    accum = np.zeros((2, 2, height, width), np.float32)
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
            with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
                prob = probabilities(model(v2.transform_d4(tensor, rotation, flip))[0])
            prob = v2.inverse_d4(prob, rotation, flip)
            if plain is None:
                plain = prob
                total = prob.clone()
            else:
                total.add_(prob)
        pair = torch.stack([plain, total / len(transforms)]).cpu().numpy()
        for k, (y, x) in enumerate(positions):
            accum[:, :, y : y + cfg.patch, x : x + cfg.patch] += pair[:, k] * weight[None, None]
            weights[y : y + cfg.patch, x : x + cfg.patch] += weight
    result = np.clip(accum / weights[None, None], 0, 1)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite inference")
    if np.any(result[:, 0] > result[:, 1] + 1e-6):
        raise AssertionError("Proxy probability smaller than core")
    result[:, 1] = np.maximum(result[:, 0], result[:, 1])
    return {
        f"{mode}_{head}": result[m, b, :h, :w].copy()
        for m, mode in enumerate(["none", "d4"])
        for b, head in enumerate(["core", "proxy"])
    }


def train_fold(paths, validation, test, labels, cfg, fold, folder, initial, device, reference):
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
    model = CoreBandUNet(initial).to(device)
    train_ids, val_ids = [p.stem for p in paths], [p.stem for p in validation]
    if (
        model.reference_binary_initial_hash != reference["initial_model_hash"]
        or train_ids != reference["train_ids"]
        or val_ids != reference["val_ids"]
    ):
        raise ValueError("Shared initialization/split differs from the paired L1 reference")
    init_hash = state_hash(model)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
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
            or saved["initial_model_hash"] != init_hash
            or saved["train_ids"] != train_ids
            or saved["val_ids"] != val_ids
        ):
            raise ValueError("Checkpoint mismatch")
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        history, first, step = saved["history"], saved["epoch"] + 1, saved["step"]
        base.restore_rng(saved["rng"])
        del saved
        print(f"{METHOD}/fold{fold}: resume {first}/{cfg.epochs}", flush=True)
    for epoch in range(first, cfg.epochs + 1):
        begin = time.perf_counter()
        model.train()
        plan = dataset.set_epoch(epoch)
        values = []
        if plan != reference["plan_hashes"][epoch - 1]:
            raise ValueError("RGB sampling/augmentation plan changed")
        for image, pair in loader:
            v2.set_lrs(optimizer, step, len(loader), cfg)
            image, pair = image.to(device, non_blocking=True), pair.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
                loss, terms = objective(model(image), pair)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite three-class loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                v2.update_ema(ema, model, cfg.ema_decay)
            values.append([float(loss.detach())] + [float(t.detach()) for t in terms])
            step += 1
        monitor_c, monitor_p = None, None
        if epoch % cfg.monitor_every == 0 or epoch == cfg.epochs:
            scores = []
            for path in validation:
                pred = predict(ema, path, cfg, device, False)
                scores.append(
                    [
                        base.dice_score(
                            pred["none_core"] >= 0.5, base.strict_label(labels[path.stem])
                        ),
                        base.dice_score(
                            pred["none_proxy"] >= 0.5, base.converter_label(labels[path.stem])
                        ),
                    ]
                )
            monitor_c, monitor_p = map(float, np.mean(scores, axis=0))
        means = np.mean(values, axis=0)
        row = dict(
            epoch=epoch,
            loss=float(means[0]),
            ce=float(means[1]),
            core_dice_loss=float(means[2]),
            proxy_dice_loss=float(means[3]),
            monitor_core_clean_05=monitor_c,
            monitor_proxy_proxy_05=monitor_p,
            plan_sha256=plan,
            seconds=round(time.perf_counter() - begin, 3),
        )
        history.append(row)
        if shutil.disk_usage(folder).free < 1200 * 1024**2:
            raise OSError("Insufficient disk space for checkpoint")
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
                method=METHOD,
                config=asdict(cfg),
                train_ids=train_ids,
                val_ids=val_ids,
                initial_model_hash=init_hash,
            ),
        )
        abl.write_table(folder / "history.csv", history)
        print(
            f'{METHOD}/fold{fold} {epoch:03d}/{cfg.epochs} loss={row["loss"]:.4f} seconds={row["seconds"]}',
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
            initial_model_hash=init_hash,
        ),
    )
    for name, paths_to_predict in [("validation", validation), ("test", test)]:
        probabilities_ = {}
        for path in paths_to_predict:
            for key, p in predict(ema, path, cfg, device).items():
                probabilities_[f"{key}_{path.stem}"] = p
        np.savez_compressed(folder / f"{name}_probabilities.npz", **probabilities_)
    names = [
        "last.pt",
        "final_ema.pt",
        "history.csv",
        "validation_probabilities.npz",
        "test_probabilities.npz",
    ]
    base.save_json(
        folder / "completed.json",
        dict(
            epoch=cfg.epochs,
            method=METHOD,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=[p.stem for p in test],
            config=asdict(cfg),
            initial_model_hash=init_hash,
            reference_binary_initial_hash=model.reference_binary_initial_hash,
            plan_hashes=[r["plan_sha256"] for r in history],
            sha256={n: base.file_sha(folder / n) for n in names},
        ),
    )
    del model, ema, optimizer, scaler, dataset, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()


def verify_fold(folder, cfg, fold, assignment, reference, test_ids, check_weights=True):
    done = json.loads((folder / "completed.json").read_text())
    if done["method"] != METHOD or done["epoch"] != cfg.epochs or done["config"] != asdict(cfg):
        raise ValueError("Wrong completed experiment")
    for key in ["train_ids", "val_ids", "plan_hashes"]:
        if done[key] != reference[key]:
            raise ValueError(f"Pairing mismatch: {key}")
    if done["reference_binary_initial_hash"] != reference["initial_model_hash"]:
        raise ValueError("Shared initialization mismatch")
    if (
        set(done["val_ids"]) != {i for i, f in assignment.items() if f == fold}
        or done["test_ids"] != test_ids
    ):
        raise ValueError("Wrong OOF/test IDs")
    for name, digest in done["sha256"].items():
        if not check_weights and name.endswith(".pt"):
            continue
        if base.file_sha(folder / name) != digest:
            raise ValueError(f"Artifact changed: {folder/name}")
    for split, ids in [("validation", done["val_ids"]), ("test", test_ids)]:
        with np.load(folder / f"{split}_probabilities.npz", allow_pickle=False) as arrays:
            expected = {
                f"{mode}_{head}_{i}"
                for mode in ["none", "d4"]
                for head in ["core", "proxy"]
                for i in ids
            }
            if set(arrays.files) != expected:
                raise ValueError("Unexpected probability keys")
            for mode in ["none", "d4"]:
                for i in ids:
                    c, p = arrays[f"{mode}_core_{i}"], arrays[f"{mode}_proxy_{i}"]
                    if (
                        c.shape != (512, 512)
                        or p.shape != (512, 512)
                        or not np.isfinite(c).all()
                        or not np.isfinite(p).all()
                    ):
                        raise ValueError("Invalid probability geometry")
                    if c.min() < 0 or p.max() > 1 or np.any(c > p + 1e-6):
                        raise ValueError("Probability hierarchy violated")
    return done


def compare_reference(output, rows, reference_rows):
    old = {(r["experiment"], r["target"], r["id"]): r for r in reference_rows}
    paired = []
    summary = []
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
    for r in rows:
        if not r["experiment"].startswith("proxy_"):
            continue
        key = ("L1_proxy_" + r["experiment"].removeprefix("proxy_"), r["target"], r["id"])
        before = old[key]
        for metric in metrics:
            paired.append(
                dict(
                    experiment=r["experiment"],
                    target=r["target"],
                    id=r["id"],
                    fold=r["fold"],
                    metric=metric,
                    delta=float(r[metric]) - float(before[metric]),
                )
            )
    for e, t, k in sorted({(r["experiment"], r["target"], r["metric"]) for r in paired}):
        rs = [r for r in paired if (r["experiment"], r["target"], r["metric"]) == (e, t, k)]
        ds = [r["delta"] for r in rs]
        fs = [float(np.mean([r["delta"] for r in rs if int(r["fold"]) == f])) for f in range(5)]
        summary.append(
            dict(
                experiment=e,
                target=t,
                metric=k,
                mean_delta=float(np.mean(ds)),
                positive_images=sum(d > 0 for d in ds),
                positive_folds=sum(d > 0 for d in fs),
                **{f"fold{f}_delta": d for f, d in enumerate(fs)},
            )
        )
    abl.write_table(output / "paired_proxy_vs_L1_per_image.csv", paired)
    abl.write_table(output / "paired_proxy_vs_L1_summary.csv", summary)


def aggregate(
    output,
    cfg,
    labels,
    assignment,
    reference_records,
    reference_rows,
    reference_grid,
    test_ids,
    root,
    export=True,
):
    complete = []
    for fold in range(5):
        folder = output / f"fold{fold}"
        if (folder / "completed.json").exists():
            verify_fold(
                folder, cfg, fold, assignment, reference_records[str(fold)], test_ids, False
            )
            complete.append(fold)
    if complete != list(range(5)):
        return dict(
            complete_20_image_oof=False,
            complete_folds=complete,
            summary=[],
            candidate_csv_generated=False,
        )
    truth = abl.targets(labels)
    probs = {f"{mode}_{head}": {} for mode in ["none", "d4"] for head in ["core", "proxy"]}
    test_mean = {
        head: {i: np.zeros((512, 512), np.float64) for i in test_ids} for head in ["core", "proxy"]
    }
    for fold in range(5):
        folder = output / f"fold{fold}"
        done = json.loads((folder / "completed.json").read_text())
        with np.load(folder / "validation_probabilities.npz", allow_pickle=False) as arrays:
            for key in probs:
                for i in done["val_ids"]:
                    probs[key][i] = arrays[f"{key}_{i}"].copy()
        with np.load(folder / "test_probabilities.npz", allow_pickle=False) as arrays:
            for head in test_mean:
                for i in test_ids:
                    test_mean[head][i] += arrays[f"d4_{head}_{i}"].astype(np.float64) / 5
    if any(set(p) != set(assignment) for p in probs.values()):
        raise ValueError("Incomplete OOF")
    rows = []
    grid = []
    calibration = []
    for mode in ["none", "d4"]:
        for head in ["core", "proxy"]:
            values = probs[f"{mode}_{head}"]
            for t in [0.1, 0.4, 0.5]:
                rows += abl.evaluate_masks(
                    f"{head}_{mode}_t{t:g}",
                    {i: p >= t for i, p in values.items()},
                    truth,
                    assignment,
                )
            for target in ["clean", "proxy"]:
                group = []
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
                    group.append(
                        dict(
                            head=head,
                            mode=mode,
                            target=target,
                            threshold=t,
                            mean_dice=float(np.mean(fs)),
                            worst_fold_dice=min(fs),
                            **{f"fold{f}": s for f, s in enumerate(fs)},
                        )
                    )
                grid += group
                best = max(group, key=lambda r: (r["mean_dice"], -abs(r["threshold"] - 0.5)))
                calibration.append(
                    dict(
                        **best,
                        caveat="OOF-tuned development estimate; not independent or nested validation",
                    )
                )
    summary, folds = abl.summarize_rows(rows)
    for name, rs in [
        ("core_band_per_image.csv", rows),
        ("core_band_per_fold.csv", folds),
        ("core_band_summary.csv", summary),
        ("core_band_threshold_grid.csv", grid),
        ("core_band_calibration_development.csv", calibration),
    ]:
        abl.write_table(output / name, rs)
    compare_reference(output, rows, reference_rows)
    selected = next(
        r
        for r in calibration
        if r["head"] == "proxy" and r["mode"] == "d4" and r["target"] == "proxy"
    )
    threshold = selected["threshold"]
    selected_masks = {i: p >= threshold for i, p in probs["d4_proxy"].items()}
    candidate_rows = abl.evaluate_masks(
        "proxy_candidate_selected_threshold", selected_masks, truth, assignment
    )
    candidate_summary, _ = abl.summarize_rows(candidate_rows)
    abl.write_table(output / "candidate_oof_per_image.csv", candidate_rows)
    abl.write_table(output / "candidate_oof_summary.csv", candidate_summary)
    mask_root = output / "oof_binary_masks"
    for tag, masks in [
        ("core_t05", {i: p >= 0.5 for i, p in probs["d4_core"].items()}),
        ("proxy_t05", {i: p >= 0.5 for i, p in probs["d4_proxy"].items()}),
        ("proxy_candidate", selected_masks),
    ]:
        folder = mask_root / tag
        folder.mkdir(parents=True, exist_ok=True)
        for i, m in masks.items():
            Image.fromarray((m * 255).astype(np.uint8)).save(folder / f"{i}.png")
    ref_best = max(
        [r for r in reference_grid if r["mode"] == "d4" and r["target"] == "proxy"],
        key=lambda r: (float(r["mean_dice"]), -abs(float(r["threshold"]) - 0.5)),
    )
    deltas = [float(selected[f"fold{f}"]) - float(ref_best[f"fold{f}"]) for f in range(5)]
    calibrated_comparison = dict(
        new_head="proxy",
        new_threshold=threshold,
        reference_threshold=float(ref_best["threshold"]),
        mean_delta=selected["mean_dice"] - float(ref_best["mean_dice"]),
        positive_folds=sum(d > 0 for d in deltas),
        **{f"fold{f}_delta": d for f, d in enumerate(deltas)},
        caveat="Both thresholds selected on reused OOF; descriptive development comparison",
    )
    exported = None
    if export:
        if any(np.any(test_mean["core"][i] > test_mean["proxy"][i] + 1e-6) for i in test_ids):
            raise AssertionError("Ensemble hierarchy violated")
        exported = v2.export_csv(
            test_mean["proxy"], test_ids, threshold, "submission_core_band_proxy", root, output
        )
        np.savez_compressed(
            output / "test_ensemble_probabilities.npz",
            **{
                f"{h}_{i}": p.astype(np.float32)
                for h, vs in test_mean.items()
                for i, p in vs.items()
            },
        )
    status = dict(
        complete_20_image_oof=True,
        complete_folds=complete,
        summary=summary,
        candidate_oof_summary=candidate_summary,
        selected_proxy_threshold=threshold,
        automatic_threshold_selection=True,
        automatic_final_winner_selection=False,
        threshold_selection_rule="proxy head D4; maximum mean proxy-GT Dice on the predeclared grid; tie nearer 0.5",
        calibrated_proxy_comparison=calibrated_comparison,
        output_head_comparison_warning="Core-head C score and proxy-head P score are from different masks.",
        shared_initialization_and_rgb_plans_verified=True,
        probability_hierarchy_verified=True,
        candidate_csv_generated=exported is not None,
        exported=exported,
        test_inference_complete=export,
        submitted_to_competition=False,
        public_scores_used=False,
        additional_model_or_loss_search=False,
    )
    return status


def package(output):
    files = [p for p in output.rglob("*") if p.is_file() and p.name != "training.lock"]
    paths = {}
    for kind in ["analysis", "oof"]:
        path = output / f"core_band_{kind}.zip"
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
    if (output / "submission_core_band_proxy.csv").exists():
        print(
            "COMPLETE: download CSV + core_band_analysis.zip only. Model checkpoints stay on Kaggle.",
            flush=True,
        )
    else:
        print(
            "PARTIAL checkpoint package only; wait for all five folds. No candidate CSV yet.",
            flush=True,
        )
    return paths


def restore(source, output):
    source = Path(source).resolve()
    if source == output:
        return
    if not source.is_dir() or not (source / "core_band_config.json").exists():
        raise ValueError(
            "RESUME_FROM must be this experiment's full mounted Kaggle Output folder, not a lightweight ZIP"
        )
    if output.is_relative_to(source):
        raise ValueError("Resume target cannot be inside source")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty resume destination")
    if not any(source.glob("fold*/last.pt")):
        raise ValueError("No resumable checkpoints in source")
    shutil.copytree(
        source, output, ignore=shutil.ignore_patterns("*.zip", "*.tmp", "training.lock")
    )


def run(
    data_dir,
    output_dir,
    cfg,
    reference_records,
    reference_rows,
    reference_grid,
    reference_info,
    weights_path=None,
    folds_to_run=(0, 1, 2, 3, 4),
    resume_from=None,
    on_package=None,
    engineering=False,
):
    if not engineering and asdict(cfg) != reference_info["config"]:
        raise ValueError("Keep the paired seed42/final80 L1 recipe")
    if len(set(folds_to_run)) != len(folds_to_run) or not set(folds_to_run) <= set(range(5)):
        raise ValueError("Invalid folds")
    output = Path(output_dir).resolve()
    if resume_from and not (output / "core_band_config.json").exists():
        restore(resume_from, output)
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(
        config=asdict(cfg),
        loss=LOSS,
        source_sha256=base.file_sha(__file__),
        helper_source_sha256={
            Path(p).name: base.file_sha(p) for p in [abl.__file__, v2.__file__, base.__file__]
        },
        reference_info=reference_info,
        reference_records_hash=abl.digest_json(reference_records),
        reference_rows_hash=abl.digest_json(reference_rows),
        reference_grid_hash=abl.digest_json(reference_grid),
        engineering_only=engineering,
    )
    marker = output / "core_band_config.json"
    previous = json.loads(marker.read_text()) if marker.exists() else None
    if previous and any(previous.get(k) != v for k, v in identity.items()):
        raise ValueError("Changed config/source/reference: use a new output directory")
    torch.set_num_threads(cfg.threads)
    cv2.setNumThreads(0)
    with v2.run_lock(output):
        root = base.find_data(data_dir)
        train, test, labels, assignment, ids, audit = v2.audit_v2(root, output, cfg)
        identity["data_fingerprint"] = audit["data_fingerprint"]
        if audit["data_fingerprint"] != reference_info["data_fingerprint"]:
            raise ValueError("Dataset differs from paired reference")
        if previous and previous != identity:
            raise ValueError("Dataset changed")
        base.save_json(marker, identity)
        import PIL, skimage

        current_env = dict(
            python=platform.python_version(),
            torch=str(torch.__version__),
            torchvision=str(v2.torchvision.__version__),
            numpy=np.__version__,
            opencv=cv2.__version__,
            pillow=PIL.__version__,
            skimage=skimage.__version__,
            gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        )
        base.save_json(output / "execution_environment.json", current_env)
        base.save_json(output / "reference_info.json", reference_info)
        abl.write_table(output / "reference_L1_per_image.csv", reference_rows)
        abl.write_table(output / "reference_L1_threshold_grid.csv", reference_grid)
        sources = output / "runtime_source"
        sources.mkdir(exist_ok=True)
        for path in [__file__, abl.__file__, v2.__file__, base.__file__]:
            shutil.copyfile(path, sources / Path(path).name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("Enable GPU for this one five-fold experiment")
        if weights_path is None:
            candidate = Path(
                "/kaggle/working/task4_recovery/pretrained_cache/resnet34-b627a593.pth"
            )
            if candidate.exists():
                weights_path = candidate
        initial, provenance = v2.resolve_weights(cfg, output, weights_path)
        if (
            provenance["sha256"]
            != "b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f"
        ):
            raise ValueError("Wrong generic ImageNet initializer")
        base.save_json(output / "pretrained_source.json", provenance)
        for fold in folds_to_run:
            folder = output / f"fold{fold}"
            if (folder / "completed.json").exists():
                verify_fold(folder, cfg, fold, assignment, reference_records[str(fold)], ids)
                print(f"{METHOD}/fold{fold}: verified complete, reusing", flush=True)
                continue
            dev = [p for p in train if assignment[p.stem] != fold]
            val = [p for p in train if assignment[p.stem] == fold]
            train_fold(
                dev,
                val,
                test,
                labels,
                cfg,
                fold,
                folder,
                initial,
                device,
                reference_records[str(fold)],
            )
            verify_fold(folder, cfg, fold, assignment, reference_records[str(fold)], ids)
            base.save_json(
                output / "core_band_status.json",
                dict(
                    complete_20_image_oof=False,
                    completed_requested_folds=[
                        f for f in range(5) if (output / f"fold{f}/completed.json").exists()
                    ],
                    summary=[],
                    engineering_only=engineering,
                    candidate_csv_generated=False,
                    public_scores_used=False,
                ),
            )
            paths = package(output)
            if on_package is not None:
                on_package(output, paths)
        status = aggregate(
            output,
            cfg,
            labels,
            assignment,
            reference_records,
            reference_rows,
            reference_grid,
            ids,
            root,
        )
        status.update(
            engineering_only=engineering,
            model_weights_local_download_required=False,
            environment_differences={
                k: dict(reference=v, current=current_env.get(k))
                for k, v in reference_info.get("environment", {}).items()
                if current_env.get(k) != v
            },
        )
        base.save_json(output / "core_band_status.json", status)
        paths = package(output)
        if on_package is not None:
            on_package(output, paths)
    return output
