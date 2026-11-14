"""Compare published pretrained encoders on fixed Task3 folds, then ensemble coordinates."""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms

from task3_pipeline import (
    DecoderBlock,
    FoveaDataset,
    assign_folds,
    contact_sheet,
    error_metrics,
    export_submission,
    find_data,
    load_manifest,
    predict,
    save_json,
    seed_everything,
    train_model,
)

TASK = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
TASK = TASK.parent if TASK.name == "src" else TASK  # 位于 src/ 时回到任务根目录
ARTIFACT_ROOT = Path(r"outputs")
if Path("/kaggle/working").exists():
    ARTIFACT_ROOT = Path("/kaggle/working/task3_artifacts")
elif not ARTIFACT_ROOT.exists():
    ARTIFACT_ROOT = TASK / "task3_artifacts"
PRETRAINED_ROOT = ARTIFACT_ROOT / "task3_pretrained"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "task3_runs/task3_pretrained_ensemble_seed42"
WEIGHT_URLS = {
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "convnext_tiny": "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
    "retfound_green": "https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/retfoundgreen_statedict.pth",
}

MODEL_SPECS = {
    "resnet34": dict(
        image_size=320,
        batch_size=8,
        lr_backbone=1e-4,
        lr_decoder=1e-3,
        weights=str(TASK.parent / "task2/.torch/hub/checkpoints/resnet34-b627a593.pth"),
        sha256_prefix="b627a593",
        normalization="imagenet",
        source="https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet34.html",
        pretraining="ImageNet-1K, torchvision IMAGENET1K_V1",
    ),
    "convnext_tiny": dict(
        image_size=320,
        batch_size=8,
        lr_backbone=1e-4,
        lr_decoder=1e-3,
        weights=str(Path.home() / ".cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth"),
        sha256_prefix="983f1562",
        normalization="imagenet",
        source="https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html",
        pretraining="ImageNet-1K, torchvision IMAGENET1K_V1",
    ),
    "retfound_green": dict(
        image_size=392,
        batch_size=4,
        lr_backbone=1e-5,
        lr_decoder=1e-3,
        weights=str(PRETRAINED_ROOT / "weights/retfoundgreen_statedict.pth"),
        sha256_prefix="431de5dbc1bebbb32f60e2c0bcf8daa4f8bcbf06f7cb1e1dc97ec589713942e1",
        normalization="half",
        source="https://github.com/justinengelmann/RETFound_Green",
        pretraining="Author retinal foundation checkpoint, ViT-S/14 with registers",
    ),
}


def require_timm():
    try:
        import timm
    except ImportError:
        local = TASK.parent / "task2/.model_deps"
        if local.is_dir():
            sys.path.insert(0, str(local))
        import timm
    return timm


def checked_state(config):
    path = Path(config["weights"])
    if not path.is_file():
        path = PRETRAINED_ROOT / "weights" / WEIGHT_URLS[config["model_name"]].rsplit("/", 1)[-1]
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(
                WEIGHT_URLS[config["model_name"]],
                str(path),
                hash_prefix=config["sha256_prefix"],
                progress=True,
            )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not digest.startswith(config["sha256_prefix"]):
        raise ValueError(f"Pretrained weight hash mismatch: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


class NormalizedDataset(FoveaDataset):
    def __init__(self, data, records, image_size, augment=False, normalization="imagenet"):
        super().__init__(data, records, image_size, augment)
        if normalization == "half":
            self.tensor = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
            )


def make_dataset(config):
    def builder(data, records, image_size, augment=False):
        return NormalizedDataset(data, records, image_size, augment, config["normalization"])

    return builder


class PretrainedHeatmap(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.architecture = config["model_name"]
        state = checked_state(config)
        if self.architecture == "resnet34":
            self.backbone = models.resnet34(weights=None)
            self.backbone.load_state_dict(state, strict=True)
            self.backbone.fc = nn.Identity()
            channels = [64, 128, 256, 512]
        elif self.architecture == "convnext_tiny":
            self.backbone = models.convnext_tiny(weights=None)
            self.backbone.load_state_dict(state, strict=True)
            self.backbone.classifier = nn.Identity()
            channels = [96, 192, 384, 768]
        elif self.architecture == "retfound_green":
            timm = require_timm()
            self.backbone = timm.create_model(
                "vit_small_patch14_reg4_dinov2",
                img_size=config["image_size"],
                num_classes=0,
                pretrained=False,
            )
            self.backbone.load_state_dict(state, strict=True)
            self.token_fusion = nn.Sequential(
                nn.Conv2d(384 * 4, 128, 1, bias=False), nn.GroupNorm(8, 128), nn.ReLU(inplace=True)
            )
            self.token_decoder = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1, bias=False),
                nn.GroupNorm(8, 64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.GroupNorm(8, 32),
                nn.ReLU(inplace=True),
            )
            self.head = nn.Conv2d(32, 1, 1)
            return
        else:
            raise ValueError(self.architecture)
        self.d3 = DecoderBlock(channels[3], channels[2], 128)
        self.d2 = DecoderBlock(128, channels[1], 64)
        self.d1 = DecoderBlock(64, channels[0], 32)
        self.head = nn.Conv2d(32, 1, 1)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, images):
        b = self.backbone
        if self.architecture == "resnet34":
            x = b.maxpool(b.relu(b.bn1(b.conv1(images))))
            c1 = b.layer1(x)
            c2 = b.layer2(c1)
            c3 = b.layer3(c2)
            c4 = b.layer4(c3)
        elif self.architecture == "convnext_tiny":
            features, x = [], images
            for index, block in enumerate(b.features):
                x = block(x)
                if index in [1, 3, 5, 7]:
                    features.append(x)
            c1, c2, c3, c4 = features
        else:
            maps = b.get_intermediate_layers(images, n=[2, 5, 8, 11], reshape=True, norm=True)
            x = self.token_fusion(torch.cat(maps, dim=1))
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = self.token_decoder(x)
            x = F.interpolate(
                x,
                size=(images.shape[-2] // 4, images.shape[-1] // 4),
                mode="bilinear",
                align_corners=False,
            )
            return self.head(x)[:, 0]
        return self.head(self.d1(self.d2(self.d3(c4, c3), c2), c1))[:, 0]


def frame_predictions(records, prediction):
    frame = records[["image_id", "width", "height"]].copy()
    if prediction.shape != (len(frame), 2) or not np.isfinite(prediction).all():
        raise ValueError("Prediction shape/values invalid.")
    frame["nx"], frame["ny"] = prediction[:, 0], prediction[:, 1]
    frame["Fovea_X"] = prediction[:, 0] * frame.width
    frame["Fovea_Y"] = prediction[:, 1] * frame.height
    return frame


def read_predictions(path, records):
    frame = pd.read_csv(path)
    if frame.image_id.duplicated().any() or set(frame.image_id) != set(records.image_id):
        raise ValueError(f"Image IDs do not match: {path}")
    frame = frame.set_index("image_id").loc[records.image_id]
    prediction = frame[["Fovea_X", "Fovea_Y"]].to_numpy() / records[["width", "height"]].to_numpy()
    if not np.isfinite(prediction).all():
        raise ValueError(f"Non-finite predictions: {path}")
    return prediction


def baseline_predictions(valid, test):
    locations = TASK / "run_locations.json"
    baseline_root = (
        Path(json.loads(locations.read_text(encoding="utf-8"))["raw_run_dir"])
        if locations.exists()
        else DEFAULT_OUTPUT.parent / "task3_baseline_seed42"
    )
    frame = pd.read_csv(baseline_root / "validation_predictions.csv")
    if frame.image_id.duplicated().any() or set(frame.image_id) != set(valid.image_id):
        raise ValueError("Baseline validation IDs differ from ensemble validation fold.")
    frame = frame.set_index("image_id").loc[valid.image_id]
    val_pred = frame[["heatmap_x", "heatmap_y"]].to_numpy() / valid[["width", "height"]].to_numpy()
    test_pred = read_predictions(baseline_root / "test_predictions.csv", test)
    return val_pred, test_pred


def compare_and_ensemble(out, valid, test, validation_predictions, test_predictions):
    """Predeclared equal-weight subsets; no continuous weight optimization on 16 samples."""
    truth, sizes = valid[["nx", "ny"]].to_numpy(), valid[["width", "height"]].to_numpy()
    names = sorted(validation_predictions)
    rows = []
    val_candidates, test_candidates, members = {}, {}, {}
    for name in names:
        val_candidates[name] = validation_predictions[name]
        test_candidates[name] = test_predictions[name]
        members[name] = [name]
        rows.append(
            dict(
                method=name,
                kind="single",
                members=name,
                **error_metrics(val_candidates[name], truth, sizes),
            )
        )
    for count in range(2, len(names) + 1):
        for subset in itertools.combinations(names, count):
            name = "equal__" + "__".join(subset)
            prediction = np.stack([validation_predictions[key] for key in subset]).mean(0)
            val_candidates[name] = prediction
            test_candidates[name] = np.stack([test_predictions[key] for key in subset]).mean(0)
            members[name] = list(subset)
            rows.append(
                dict(
                    method=name,
                    kind="equal_ensemble",
                    members="+".join(subset),
                    **error_metrics(prediction, truth, sizes),
                )
            )
    comparison = pd.DataFrame(rows).sort_values("mean_distance_px")
    comparison.to_csv(out / "validation_comparison.csv", index=False)
    singles = comparison[comparison.kind == "single"]
    ensembles = comparison[comparison.kind == "equal_ensemble"]
    best_single = singles.iloc[0]["method"]
    best_ensemble = ensembles.iloc[0]["method"]
    selected = comparison.iloc[0]["method"]
    per_image = valid[["image_id", "width", "height", "x", "y"]].copy()
    errors = []
    for name in names:
        delta = (validation_predictions[name] - truth) * sizes
        errors.append(delta)
        per_image[f"{name}_x"], per_image[f"{name}_y"] = (validation_predictions[name] * sizes).T
        per_image[f"{name}_distance_px"] = np.linalg.norm(delta, axis=1)
    for name, alias in [(best_ensemble, "best_equal_ensemble"), (selected, "selected")]:
        delta = (val_candidates[name] - truth) * sizes
        per_image[f"{alias}_x"], per_image[f"{alias}_y"] = (val_candidates[name] * sizes).T
        per_image[f"{alias}_distance_px"] = np.linalg.norm(delta, axis=1)
    per_image.to_csv(out / "validation_predictions.csv", index=False)
    flattened = np.stack(errors).reshape(len(names), -1)
    pd.DataFrame(np.corrcoef(flattened), index=names, columns=names).to_csv(
        out / "error_vector_correlation.csv"
    )
    pair_rows = []
    for a, b in itertools.combinations(names, 2):
        da = np.linalg.norm((validation_predictions[a] - truth) * sizes, axis=1)
        db = np.linalg.norm((validation_predictions[b] - truth) * sizes, axis=1)
        pair_rows.append(
            dict(
                model_a=a,
                model_b=b,
                a_better_images=int((da < db).sum()),
                b_better_images=int((db < da).sum()),
                mean_prediction_disagreement_px=float(
                    np.linalg.norm(
                        (validation_predictions[a] - validation_predictions[b]) * sizes, axis=1
                    ).mean()
                ),
            )
        )
    pd.DataFrame(pair_rows).to_csv(out / "model_complementarity.csv", index=False)
    info = dict(
        selected=selected,
        best_single=best_single,
        best_equal_ensemble=best_ensemble,
        selected_members=members[selected],
        ensemble_members=members[best_ensemble],
        selected_weights=[1 / len(members[selected])] * len(members[selected]),
        validation_n=len(valid),
        candidate_count=len(comparison),
        selection_metric="mean_distance_px; provisional official metric unavailable",
        limitation="Epochs and subset selected on the same 16-image holdout; selection scores, not unbiased test estimates.",
        weighting="Equal coordinates per member, fixed candidate subsets; no continuous weight fitting.",
        submitted_to_competition=False,
    )
    save_json(out / "selection.json", info)
    return (
        info,
        val_candidates[selected],
        test_candidates[selected],
        test_candidates[best_ensemble],
        comparison,
    )


def run_pretrained(
    model_names=None, epochs=80, output_dir=DEFAULT_OUTPUT, full=True, include_fit=True
):
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment.")
    device = torch.device("cuda")
    data = find_data()
    manifest = load_manifest(data)
    train = assign_folds(manifest[manifest.subset == "train"], 42, 5)
    train.to_csv(out / "split.csv", index=False)
    development, valid = train[train.fold != 0], train[train.fold == 0]
    test = manifest[manifest.subset == "test"]
    model_names = list(model_names or MODEL_SPECS)
    protocol = dict(
        seed=42,
        folds=5,
        fold=0,
        epochs=epochs,
        model_names=model_names,
        validation_metric="mean_distance_px",
        use_tta=False,
        ensemble_candidates="All equal-weight subsets of available pretrained models plus baseline; no fitted weights.",
        dataset_fingerprint=hashlib.sha256(train.to_csv(index=False).encode()).hexdigest(),
    )
    if not include_fit:
        protocol["include_fit"] = False
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Experiment protocol differs; use another output directory.")
    save_json(protocol_path, protocol)
    save_json(out / "model_sources.json", {name: MODEL_SPECS[name] for name in model_names})
    validation_predictions, test_predictions = {}, {}
    if full:
        validation_predictions["resnet18"], test_predictions["resnet18"] = baseline_predictions(
            valid, test
        )
    for name in model_names:
        config = dict(
            MODEL_SPECS[name],
            model_name=name,
            seed=42,
            folds=5,
            fold=0,
            epochs=epochs,
            schedule_epochs=epochs,
            pretrained=True,
            weight_decay=1e-4,
            sigma=1.5,
            dataset_fingerprint=protocol["dataset_fingerprint"],
            architecture="pretrained_encoder_heatmap_v1",
            selection_metric="mean_distance_px",
        )
        config.pop("source")
        config.pop("pretraining")
        builder = make_dataset(config)
        save_json(
            out / "status.json",
            dict(state="running", model=name, stage="validation", updated_unix=time.time()),
        )
        model, best_epoch = train_model(
            data,
            development,
            valid,
            out / name / "fold0",
            config,
            device,
            model_builder=PretrainedHeatmap,
            dataset_builder=builder,
        )
        loader = DataLoader(
            builder(data, valid, config["image_size"]),
            batch_size=config["batch_size"],
            shuffle=False,
        )
        prediction = predict(model, loader, device)
        validation_predictions[name] = prediction
        frame_predictions(valid, prediction).to_csv(
            out / name / "validation_predictions.csv", index=False
        )
        score = error_metrics(
            prediction, valid[["nx", "ny"]].to_numpy(), valid[["width", "height"]].to_numpy()
        )
        save_json(out / name / "validation_metrics.json", dict(best_epoch=best_epoch, **score))
        print(
            f'VALIDATED {name}: {score["mean_distance_px"]:.4f}px @ epoch {best_epoch}', flush=True
        )
        del model, loader
        gc.collect()
        torch.cuda.empty_cache()
        if full:
            save_json(
                out / "status.json",
                dict(state="running", model=name, stage="fulltrain", updated_unix=time.time()),
            )
            full_config = dict(config, epochs=best_epoch, fold="full", selected_from_fold=0)
            model, _ = train_model(
                data,
                train,
                None,
                out / name / "fulltrain",
                full_config,
                device,
                model_builder=PretrainedHeatmap,
                dataset_builder=builder,
            )
            loader = DataLoader(
                builder(data, test, config["image_size"]),
                batch_size=config["batch_size"],
                shuffle=False,
            )
            prediction = predict(model, loader, device)
            test_predictions[name] = prediction
            frame_predictions(test, prediction).to_csv(
                out / name / "test_predictions.csv", index=False
            )
            del model, loader
            gc.collect()
            torch.cuda.empty_cache()
    if full:
        # Optional external zero-shot candidate is included only after its source/coordinate checks finish.
        fit_dir = out / "fit_zeroshot"
        if (
            include_fit
            and (fit_dir / "validation_predictions.csv").is_file()
            and (fit_dir / "test_predictions.csv").is_file()
        ):
            validation_predictions["fit_zeroshot"] = read_predictions(
                fit_dir / "validation_predictions.csv", valid
            )
            test_predictions["fit_zeroshot"] = read_predictions(
                fit_dir / "test_predictions.csv", test
            )
        info, val_pred, test_pred, ensemble_pred, comparison = compare_and_ensemble(
            out, valid, test, validation_predictions, test_predictions
        )
        export_submission(data, test, test_pred, out)
        shutil.copy2(out / "submission.csv", out / "submission_selected.csv")
        export_submission(
            data,
            test,
            ensemble_pred,
            (
                out / "best_ensemble"
                if (out / "best_ensemble").exists()
                else _mkdir(out / "best_ensemble")
            ),
        )
        shutil.copy2(out / "best_ensemble/submission.csv", out / "submission_ensemble.csv")
        contact_sheet(data, valid, out / "validation_predictions.png", val_pred)
        print(comparison.head(12).to_string(index=False), flush=True)
        print("SELECTION", json.dumps(info, ensure_ascii=False), flush=True)
    save_json(out / "status.json", dict(state="completed", updated_unix=time.time()))
    return out


def _mkdir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        run_pretrained(args.models, args.epochs, args.output_dir, not args.validate_only)
    except BaseException as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            args.output_dir / "status.json",
            dict(state="failed", error=str(error), updated_unix=time.time()),
        )
        raise


if __name__ == "__main__":
    cli()
