"""Rebuild only native-clean and L1 OOF; fixed fusion diagnostic; explicit backups."""

from __future__ import annotations
import hashlib
import json
import os
import shutil
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path, PurePosixPath

import numpy as np
import torch

import task4_controlled_ablation as abl

base, v2 = abl.base, abl.v2
ARMS = ("native_clean", "L1_proxy")
RULE = dict(
    tta="d4",
    native_center=0.1,
    proxy_center=0.4,
    weight_native=0.5,
    weight_proxy=0.5,
    clip_epsilon=1e-6,
    fused_threshold=0.5,
    interpretation="threshold-centered logits; heuristic alignment, not probability calibration",
)


def safe_archive(z):
    names = z.namelist()
    if len(names) != len(set(names)):
        raise ValueError("Duplicate ZIP paths")
    for item in z.infolist():
        p = PurePosixPath(item.filename)
        if (
            p.is_absolute()
            or ".." in p.parts
            or "\\" in item.filename
            or ":" in item.filename
            or ((item.external_attr >> 16) & 0o170000) == 0o120000
        ):
            raise ValueError("Unsafe ZIP path")
    if z.testzip() is not None:
        raise ValueError("ZIP CRC failed")


def restore(source, output):
    source = Path(source).resolve()
    if source == output:
        return
    if output.is_relative_to(source):
        raise ValueError("Output cannot be inside its resume source")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Resume target must be empty")
    if source.is_dir():
        if not (source / "recovery_config.json").exists():
            raise ValueError("Not a recovery-stage output")
        # OOF bundle is explicitly insufficient to continue training.
        if not any(source.glob("*/fold*/last.pt")):
            raise ValueError("Resume needs full checkpoints, not the OOF-only bundle")
        shutil.copytree(
            source,
            output,
            ignore=shutil.ignore_patterns("*.zip", "*.tmp", "training.lock", "backup_receipt.json"),
        )
    elif source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as z:
            safe_archive(z)
            if "recovery_config.json" not in z.namelist() or not any(
                n.endswith("/last.pt") for n in z.namelist()
            ):
                raise ValueError("Use task4_recovery_full.zip, not an analysis or OOF-only ZIP")
            output.mkdir(parents=True, exist_ok=True)
            for item in z.infolist():
                if item.filename.endswith((".zip", ".tmp", "training.lock", "backup_receipt.json")):
                    continue
                z.extract(item, output)
    else:
        raise ValueError("RESUME_FROM must be a full recovery folder/ZIP")


def verify_fold(folder, arm, fold, cfg, assignment):
    done = json.loads((folder / "completed.json").read_text())
    if done["epoch"] != cfg.epochs or set(done["val_ids"]) != {
        i for i, f in assignment.items() if f == fold
    }:
        raise ValueError(f"Wrong epoch/split in {folder}")
    if set(done["train_ids"]) != set(assignment) - set(done["val_ids"]):
        raise ValueError("Training IDs cross the validation split")
    if arm == "native_clean" and (
        done["recovery_arm"] != arm
        or set(done["train_ids"]) != set(assignment) - set(done["val_ids"])
    ):
        raise ValueError("Native recovery identity mismatch")
    if arm == "L1_proxy" and done["variant"] != "L1_proxy":
        raise ValueError("Expected L1 proxy variant")
    for name, digest in done["sha256"].items():
        if base.file_sha(folder / name) != digest:
            raise ValueError(f"Changed completed artifact: {folder/name}")
    with np.load(folder / "validation_probabilities.npz", allow_pickle=False) as arrays:
        expected = {f"{mode}_{i}" for mode in ["none", "d4"] for i in done["val_ids"]}
        if set(arrays.files) != expected:
            raise ValueError("Unexpected OOF keys")
        for key in arrays.files:
            a = arrays[key]
            if a.shape != (512, 512) or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
                raise ValueError("Invalid OOF probabilities")
    return done


def export_native(model, validation, development, cfg, folder, device):
    values = {}
    for p in validation:
        for mode, prob in v2.predict_pair(model, p, cfg, device).items():
            values[f"{mode}_{p.stem}"] = v2.resize_probability(prob)
    np.savez_compressed(folder / "validation_probabilities.npz", **values)
    names = ["history.csv", "last.pt", "final_ema.pt", "validation_probabilities.npz"]
    base.save_json(
        folder / "completed.json",
        dict(
            recovery_arm="native_clean",
            epoch=cfg.epochs,
            train_ids=[p.stem for p in development],
            val_ids=[p.stem for p in validation],
            checkpoint_selection="predeclared final EMA",
            test_predictions_generated=False,
            sha256={n: base.file_sha(folder / n) for n in names},
        ),
    )


def aligned_fusion(a, b):
    eps = RULE["clip_epsilon"]

    def logit(p):
        p = np.clip(np.asarray(p, np.float64), eps, 1 - eps)
        return np.log(p) - np.log1p(-p)

    z = 0.5 * (logit(a) - logit(0.1)) + 0.5 * (logit(b) - logit(0.4))
    return (1 / (1 + np.exp(-z))).astype(np.float32)


def analyze(output, cfg, labels, assignment, historical=None):
    ready = {
        arm: [f for f in range(5) if (output / arm / f"fold{f}/completed.json").exists()]
        for arm in ARMS
    }
    if any(folds != list(range(5)) for folds in ready.values()):
        return dict(
            complete_20_image_oof=False, complete_folds=ready, summary=[], no_partial_oof_claim=True
        )
    probs = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for fold in range(5):
            folder = output / arm / f"fold{fold}"
            done = verify_fold(folder, arm, fold, cfg, assignment)
            with np.load(folder / "validation_probabilities.npz", allow_pickle=False) as arrays:
                for i in done["val_ids"]:
                    probs[arm][i] = arrays[f"d4_{i}"].copy()
    if any(set(values) != set(assignment) for values in probs.values()):
        raise ValueError("Incomplete OOF")
    truth = abl.targets(labels)
    a, b = probs["native_clean"], probs["L1_proxy"]
    fused = {i: aligned_fusion(a[i], b[i]) for i in assignment}
    methods = {
        "native_d4_t01": {i: p >= 0.1 for i, p in a.items()},
        "proxy_d4_t04": {i: p >= 0.4 for i, p in b.items()},
        "fixed_equal_centered_logit": {i: p >= 0.5 for i, p in fused.items()},
    }
    rows = []
    for method, masks in methods.items():
        rows += abl.evaluate_masks(method, masks, truth, assignment)
    summary, folds = abl.summarize_rows(rows)
    abl.write_table(output / "fusion_per_image.csv", rows)
    abl.write_table(output / "fusion_per_fold.csv", folds)
    abl.write_table(output / "fusion_summary.csv", summary)
    diagnostics = []
    paired = []
    lookup = {(r["experiment"], r["target"], r["id"]): r for r in rows}
    metrics = [
        "dice",
        "precision",
        "recall",
        "cldice",
        "thin_clean_centerline_recall",
        "foreground_fraction",
    ]
    for i in sorted(assignment, key=int):
        ma, mb = methods["native_d4_t01"][i], methods["proxy_d4_t04"][i]
        for target in ["clean", "proxy"]:
            gt = truth[i][target]
            ea, eb = ma != gt, mb != gt
            union = int((ea | eb).sum())
            common = int((ea & eb).sum())
            diagnostics.append(
                dict(
                    id=i,
                    fold=assignment[i],
                    target=target,
                    native_wrong=int(ea.sum()),
                    proxy_wrong=int(eb.sum()),
                    both_wrong=common,
                    native_wrong_proxy_correct=int((ea & ~eb).sum()),
                    proxy_wrong_native_correct=int((eb & ~ea).sum()),
                    error_jaccard=common / union if union else 1.0,
                    mask_disagreement_fraction=float((ma != mb).mean()),
                )
            )
            for reference in ["native_d4_t01", "proxy_d4_t04"]:
                for key in metrics:
                    paired.append(
                        dict(
                            reference=reference,
                            id=i,
                            fold=assignment[i],
                            target=target,
                            metric=key,
                            delta=float(lookup["fixed_equal_centered_logit", target, i][key])
                            - float(lookup[reference, target, i][key]),
                        )
                    )
    abl.write_table(output / "error_complementarity.csv", diagnostics)
    abl.write_table(output / "fusion_paired_per_image.csv", paired)
    paired_summary = []
    for reference in ["native_d4_t01", "proxy_d4_t04"]:
        for target in ["clean", "proxy"]:
            for key in metrics:
                r = [
                    r
                    for r in paired
                    if r["reference"] == reference and r["target"] == target and r["metric"] == key
                ]
                ds = [v["delta"] for v in r]
                fs = [float(np.mean([v["delta"] for v in r if v["fold"] == f])) for f in range(5)]
                paired_summary.append(
                    dict(
                        reference=reference,
                        target=target,
                        metric=key,
                        mean_delta=float(np.mean(ds)),
                        positive_images=sum(v > 0 for v in ds),
                        positive_folds=sum(v > 0 for v in fs),
                        **{f"fold{f}_delta": d for f, d in enumerate(fs)},
                    )
                )
    abl.write_table(output / "fusion_paired_summary.csv", paired_summary)
    np.savez_compressed(output / "fusion_oof_d4_probabilities.npz", **fused)
    if historical is not None:
        comparison = []
        for s in summary:
            if s["experiment"] in historical:
                prior = historical[s["experiment"]][s["target"]]
                comparison.append(
                    dict(
                        method=s["experiment"],
                        target=s["target"],
                        historical_dice=prior,
                        rebuilt_dice=s["dice"],
                        difference=s["dice"] - prior,
                        caveat="rebuild under fixed recipe; not a new independent holdout; exact weights not guaranteed",
                    )
                )
        abl.write_table(output / "historical_vs_rebuilt.csv", comparison)
    return dict(
        complete_20_image_oof=True,
        complete_folds=ready,
        summary=summary,
        fixed_rule=RULE,
        weights_or_threshold_search=False,
        automatic_model_selection=False,
        public_scores_used=False,
        test_predictions_generated=False,
        competition_csv_generated=False,
        caveat="OOF reused for development; aligned logits are a fixed heuristic, not calibrated probabilities or private-score evidence",
    )


def bundle(output, full=True):
    """ZIPs remain temporary until downloaded or saved as version outputs."""
    files = [
        p
        for p in output.rglob("*")
        if p.is_file()
        and p.suffix not in {".zip", ".tmp"}
        and p.name not in {"training.lock", "backup_receipt.json"}
    ]
    result = {}
    for kind in ["analysis", "oof"] + (["full"] if full else []):
        suffixes = {".json", ".csv", ".py", ".md"} | ({".npz"} if kind == "oof" else set())
        selected = files if kind == "full" else [p for p in files if p.suffix in suffixes]
        required = sum(p.stat().st_size for p in selected)
        if shutil.disk_usage(output).free < required + 256 * 1024**2:
            if kind != "full":
                raise OSError(f"Insufficient space for the {kind} backup")
            result[kind] = dict(
                saved=False,
                reason="Insufficient free space; OOF bundle exists, download checkpoints separately",
            )
            print(
                "Full backup skipped: insufficient space. OOF backup is NOT a resumable training backup.",
                flush=True,
            )
            continue
        path = output / f"task4_recovery_{kind}.zip"
        tmp = path.with_suffix(".zip.tmp")
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as z:
            for p in selected:
                z.write(p, p.relative_to(output).as_posix())
        with zipfile.ZipFile(tmp) as z:
            if z.testzip() is not None:
                raise IOError("Backup CRC failed")
        os.replace(tmp, path)
        result[kind] = dict(
            saved=True, path=str(path), bytes=path.stat().st_size, sha256=base.file_sha(path)
        )
    base.save_json(
        output / "backup_receipt.json",
        dict(
            archives=result,
            downloaded_to_user_computer=False,
            warning="ZIP creation is not persistent storage: download both archives before ending session",
        ),
    )
    print(
        "BACKUP READY: download task4_recovery_oof.zip and task4_recovery_full.zip; /kaggle/working is temporary.",
        flush=True,
    )
    return result


def run_recovery(
    data_dir,
    output_dir,
    cfg=None,
    folds_to_run=(0, 1, 2, 3, 4),
    weights_path=None,
    resume_from=None,
    full_backup=True,
    historical=None,
    on_backup=None,
    engineering=False,
):
    cfg = cfg or abl.AblationConfig()
    if not engineering and asdict(cfg) != asdict(abl.AblationConfig()):
        raise ValueError(
            "Recovery fixes the original seed42/80-epoch recipe; use engineering=True only for tests"
        )
    if len(set(folds_to_run)) != len(folds_to_run) or not set(folds_to_run) <= set(range(5)):
        raise ValueError("Invalid folds")
    output = Path(output_dir).resolve()
    if resume_from and not (output / "recovery_config.json").exists():
        restore(resume_from, output)
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(
        config=asdict(cfg),
        source_sha256=base.file_sha(__file__),
        ablation_source_sha256=base.file_sha(abl.__file__),
        v2_source_sha256=base.file_sha(v2.__file__),
        base_source_sha256=base.file_sha(base.__file__),
        arms=ARMS,
        fixed_fusion_rule=RULE,
        engineering_only=engineering,
    )
    # Normalize JSON tuple/list representation for resumed comparisons.
    identity = json.loads(json.dumps(identity))
    marker = output / "recovery_config.json"
    previous = json.loads(marker.read_text()) if marker.exists() else None
    if previous and any(previous.get(k) != v for k, v in identity.items()):
        raise ValueError("Changed source/config; use a new output directory")
    torch.set_num_threads(cfg.threads)
    abl.cv2.setNumThreads(0)
    with v2.run_lock(output):
        root = base.find_data(data_dir)
        train, _, labels, assignment, _, audit = v2.audit_v2(root, output, cfg)
        identity["data_fingerprint"] = audit["data_fingerprint"]
        if previous and previous != identity:
            raise ValueError("Dataset changed")
        base.save_json(marker, identity)
        import PIL, skimage, platform

        base.save_json(
            output / "execution_environment.json",
            dict(
                python=platform.python_version(),
                torch=str(torch.__version__),
                torchvision=str(v2.torchvision.__version__),
                numpy=np.__version__,
                opencv=abl.cv2.__version__,
                pillow=PIL.__version__,
                skimage=skimage.__version__,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            ),
        )
        sources = output / "runtime_source"
        sources.mkdir(exist_ok=True)
        for path in [__file__, abl.__file__, v2.__file__, base.__file__]:
            shutil.copyfile(path, sources / Path(path).name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("Enable GPU for the two-arm recovery")
        initial, provenance = v2.resolve_weights(cfg, output, weights_path)
        if (
            provenance["sha256"]
            != "b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f"
        ):
            raise ValueError("Expected the original official ImageNet initializer")
        base.save_json(output / "pretrained_source.json", provenance)
        native_cfg = v2.ConfigV2(**{k: asdict(cfg)[k] for k in asdict(v2.ConfigV2())})
        for arm in ARMS:
            for fold in folds_to_run:
                folder = output / arm / f"fold{fold}"
                if (folder / "completed.json").exists():
                    verify_fold(folder, arm, fold, cfg, assignment)
                    print(f"{arm}/fold{fold}: verified complete, reusing", flush=True)
                    continue
                dev = [p for p in train if assignment[p.stem] != fold]
                val = [p for p in train if assignment[p.stem] == fold]
                if arm == "native_clean":
                    model = v2.train_fold(
                        dev, val, labels, native_cfg, fold, folder, initial, device
                    )
                    export_native(model, val, dev, native_cfg, folder, device)
                    del model
                    torch.cuda.empty_cache()
                else:
                    abl.train_label_fold(
                        dev, val, labels, cfg, fold, "L1_proxy", folder, initial, device
                    )
                verify_fold(folder, arm, fold, cfg, assignment)
                status = analyze(output, cfg, labels, assignment, historical)
                status.update(
                    engineering_only=engineering,
                    exact_historical_model_hashes_guaranteed=False,
                    historical_submissions_overwritten=False,
                )
                base.save_json(output / "recovery_status.json", status)
                receipt = bundle(output, full_backup)
                if on_backup is not None:
                    on_backup(output, receipt)
        status = analyze(output, cfg, labels, assignment, historical)
        status.update(
            engineering_only=engineering,
            exact_historical_model_hashes_guaranteed=False,
            historical_submissions_overwritten=False,
        )
        base.save_json(output / "recovery_status.json", status)
        receipt = bundle(output, full_backup)
        if on_backup is not None:
            on_backup(output, receipt)
    return output


# ---------- 测试推理：用十个最终 EMA 模型做固定规则融合并导出提交文件 ----------
# runtime 传 None 时使用本模块。
def finish_inference(
    recovery,
    runtime,
    expected,
    data_dir=None,
    output_dir=None,
    implementation_sha=None,
    engineering=False,
    test_limit=None,
):
    rec = runtime if runtime is not None else sys.modules[__name__]
    v2, base, abl = rec.v2, rec.base, rec.abl
    recovery = Path(recovery).resolve()
    config = json.loads((recovery / "recovery_config.json").read_text())
    cfg = abl.AblationConfig(**config["config"])
    if config["engineering_only"] != engineering:
        raise ValueError("Formal/engineering run mismatch")
    if not engineering:
        if asdict(cfg) != asdict(abl.AblationConfig()):
            raise ValueError("Expected the returned final80 recipe")
        if set(expected) != set(rec.ARMS) or any(
            set(v) != set(map(str, range(5))) for v in expected.values()
        ):
            raise ValueError("Need exactly the ten returned model identities")
    root = base.find_data(data_dir)
    cloud_input = Path("/kaggle/input")
    default_output = (
        Path("/kaggle/working/task4_final_inference")
        if recovery.is_relative_to(cloud_input)
        else recovery / "test_inference_fixed_fusion"
    )
    output = Path(output_dir) if output_dir else default_output
    output.mkdir(parents=True, exist_ok=True)
    _, test, _, assignment, ids, audit = v2.audit_v2(root, output / "data_audit", cfg)
    if audit["data_fingerprint"] != config["data_fingerprint"]:
        raise ValueError("Dataset fingerprint changed")
    if test_limit is not None:
        if not engineering:
            raise ValueError("Formal inference must export all 20 images")
        ids = ids[:test_limit]
        test = [p for p in test if p.stem in set(ids)]
    by_id = {p.stem: p for p in test}
    if set(ids) != set(by_id):
        raise ValueError("Test IDs do not match the sample template")
    # Check every weight before starting inference. No optimizer/last.pt loading.
    for arm, folds in expected.items():
        for fold, e in folds.items():
            folder = recovery / arm / f"fold{fold}"
            if not (folder / "final_ema.pt").exists():
                raise FileNotFoundError(
                    f"{folder}/final_ema.pt missing. Use the active recovery session or mount its saved Output; this cell never retrains."
                )
            if base.file_sha(folder / "completed.json") != e["completed_sha256"]:
                raise ValueError(f"Completed record differs from returned OOF: {arm}/fold{fold}")
            done = json.loads((folder / "completed.json").read_text())
            if done["epoch"] != cfg.epochs or done["sha256"]["final_ema.pt"] != e["model_sha256"]:
                raise ValueError("Wrong final model identity")
            if set(done["val_ids"]) != {i for i, f in assignment.items() if f == int(fold)}:
                raise ValueError("Wrong split record")
            if base.file_sha(folder / "final_ema.pt") != e["model_sha256"]:
                raise ValueError(f"Weight hash mismatch: {arm}/fold{fold}")
    identity = dict(
        source_recovery_config=config,
        models=expected,
        rule=rec.RULE,
        test_ids=ids,
        test_image_sha256={i: base.file_sha(by_id[i]) for i in ids},
        ensemble_order="mean probabilities within each five-fold arm, then the frozen aligned-logit fusion",
        implementation_sha256=implementation_sha,
        training_started=False,
        engineering_only=engineering,
    )
    marker = output / "inference_identity.json"
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError("Inference identity changed; use a new output directory")
    base.save_json(marker, identity)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Enable GPU for D4 test inference; no training is needed")
    torch.set_num_threads(cfg.threads)
    abl.cv2.setNumThreads(0)
    means = {arm: {i: np.zeros((512, 512), np.float64) for i in ids} for arm in expected}
    records = []
    for arm, folds in expected.items():
        for fold, e in folds.items():
            cache = output / "cache" / arm / f"fold{fold}"
            cache.mkdir(parents=True, exist_ok=True)
            npz = cache / "test_d4_probabilities.npz"
            meta = cache / "completed.json"
            cache_identity = dict(
                model_sha256=e["model_sha256"],
                test_images=identity["test_image_sha256"],
                config=config["config"],
                arm=arm,
                mode="d4",
                implementation_sha256=implementation_sha,
            )
            if npz.exists() and meta.exists():
                record = json.loads(meta.read_text())
                if (
                    record["identity"] != cache_identity
                    or base.file_sha(npz) != record["probability_sha256"]
                ):
                    raise ValueError("Changed inference cache")
                print(f"{arm}/fold{fold}: verified test cache, reusing", flush=True)
            else:
                path = recovery / arm / f"fold{fold}" / "final_ema.pt"
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if saved["epoch"] != cfg.epochs:
                    raise ValueError("Wrong checkpoint epoch")
                model = v2.ResNet34UNet().to(device)
                model.load_state_dict(saved["model"], strict=True)
                model.eval().requires_grad_(False)
                del saved
                values = {}
                for n, i in enumerate(ids, 1):
                    image = (
                        by_id[i]
                        if arm == "native_clean"
                        else abl.cv2.resize(
                            v2.read_rgb(by_id[i]), (512, 512), interpolation=abl.cv2.INTER_AREA
                        )
                    )
                    probability = v2.predict_pair(model, image, cfg, device)["d4"]
                    if arm == "native_clean":
                        probability = v2.resize_probability(probability)
                    values[i] = probability
                    print(f"TEST ONLY {arm}/fold{fold}: {n}/{len(ids)}", flush=True)
                temporary = npz.with_suffix(".npz.tmp")
                with temporary.open("wb") as f:
                    np.savez_compressed(f, **values)
                os.replace(temporary, npz)
                base.save_json(
                    meta, dict(identity=cache_identity, probability_sha256=base.file_sha(npz))
                )
                del model, values
                torch.cuda.empty_cache()
            with np.load(npz, allow_pickle=False) as arrays:
                if set(arrays.files) != set(ids):
                    raise ValueError("Incomplete test cache")
                for i in ids:
                    p = arrays[i]
                    if (
                        p.shape != (512, 512)
                        or not np.isfinite(p).all()
                        or p.min() < 0
                        or p.max() > 1
                    ):
                        raise ValueError("Invalid test probability")
                    means[arm][i] += p.astype(np.float64) / len(folds)
            records.append(
                dict(
                    arm=arm,
                    fold=int(fold),
                    model_sha256=e["model_sha256"],
                    probability_sha256=base.file_sha(npz),
                )
            )
    fused = {i: rec.aligned_fusion(means["native_clean"][i], means["L1_proxy"][i]) for i in ids}
    for i, p in fused.items():
        count = int((p >= 0.5).sum())
        if not engineering and (count == 0 or count == 512 * 512):
            raise ValueError(f"Empty/full vessel mask: test {i}")
    values = {
        f"{arm}_{i}": p.astype(np.float32) for arm, probs in means.items() for i, p in probs.items()
    }
    values.update({f"fusion_{i}": p for i, p in fused.items()})
    np.savez_compressed(output / "test_ensemble_probabilities.npz", **values)
    exported = v2.export_csv(fused, ids, 0.5, "submission_fixed_equal_logit_fusion", root, output)
    areas = [dict(id=i, foreground_fraction=float((fused[i] >= 0.5).mean())) for i in ids]
    abl.write_table(output / "test_foreground_fraction.csv", areas)
    base.save_json(
        output / "inference_status.json",
        dict(
            complete=True,
            exported=exported,
            models=records,
            rule=rec.RULE,
            test_images=len(ids),
            training_started=False,
            loss_or_threshold_search=False,
            submitted_to_competition=False,
            model_download_to_user_computer_required=False,
            engineering_only=engineering,
            caveat="OOF proxy gain with clean-target tradeoff; no private-score guarantee",
        ),
    )
    small = output / "task4_final_small.zip"
    with zipfile.ZipFile(small, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in output.rglob("*"):
            if (
                p.is_file()
                and "cache" not in p.relative_to(output).parts
                and "data_audit" not in p.relative_to(output).parts
                and p.suffix in {".csv", ".json", ".png"}
            ):
                z.write(p, p.relative_to(output).as_posix())
    with zipfile.ZipFile(small) as z:
        if z.testzip() is not None:
            raise IOError("Small result archive CRC failed")
    print(
        "DONE. Download CSV and task4_final_small.zip only; model weights stay on Kaggle.",
        flush=True,
    )
    print(output / exported["file"], flush=True)
    return output
