"""Fresh single-model ResNet34 training on all 80 Task3 images and submission export."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pretrained_ensemble import ARTIFACT_ROOT, MODEL_SPECS, PretrainedHeatmap, make_dataset
from task3_pipeline import (
    assign_folds,
    error_metrics,
    export_submission,
    find_data,
    load_manifest,
    predict,
    save_json,
    train_model,
)

FULLTRAIN_OUTPUT = ARTIFACT_ROOT / "task3_runs/task3_resnet34_fulltrain_seed42"


def run_resnet34_fulltrain(data_dir=None, output_dir=FULLTRAIN_OUTPUT):
    data = find_data(data_dir)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the full training run.")
    device = torch.device("cuda")
    manifest = load_manifest(data)
    train = assign_folds(manifest[manifest.subset == "train"], 42, 5)
    test = manifest[manifest.subset == "test"].copy()
    if len(train) != 80 or len(test) != 20:
        raise ValueError("Expected 80 training images and 20 test images.")
    train.to_csv(out / "train_manifest.csv", index=False)
    test.to_csv(out / "test_manifest.csv", index=False)
    spec = dict(MODEL_SPECS["resnet34"])
    source, pretraining = spec.pop("source"), spec.pop("pretraining")
    config = dict(
        spec,
        model_name="resnet34",
        seed=42,
        folds=5,
        fold="full",
        epochs=43,
        schedule_epochs=80,
        pretrained=True,
        weight_decay=1e-4,
        sigma=1.5,
        dataset_fingerprint=hashlib.sha256(train.to_csv(index=False).encode()).hexdigest(),
        architecture="pretrained_encoder_heatmap_v1",
        selection_metric="mean_distance_px",
        selected_from_fold=0,
    )
    resume_path = out / "checkpoints/last.pt"
    resumed = resume_path.exists()
    start = time.perf_counter()
    save_json(
        out / "run_status.json",
        dict(
            state="training",
            training_images=len(train),
            epochs=43,
            resumed=resumed,
            updated_unix=time.time(),
        ),
    )
    print(
        f"ResNet34 single model | {len(train)} training images | 43 epochs | resume={resumed}",
        flush=True,
    )
    builder = make_dataset(config)
    model, final_epoch = train_model(
        data,
        train,
        None,
        out / "checkpoints",
        config,
        device,
        model_builder=PretrainedHeatmap,
        dataset_builder=builder,
    )
    if final_epoch != 43:
        raise AssertionError("Full training did not complete all 43 epochs.")
    save_json(out / "run_status.json", dict(state="predicting", updated_unix=time.time()))
    loader = DataLoader(
        builder(data, test, config["image_size"]), batch_size=config["batch_size"], shuffle=False
    )
    prediction = predict(model, loader, device, flip_tta=False)
    export_submission(data, test, prediction, out)
    check = pd.read_csv(out / "submission.csv")
    template = pd.read_csv(data / "sample_submission.csv")
    assert len(check) == 40 and check.ImageID.equals(template.ImageID)
    assert np.isfinite(check.value).all()
    for row in check.itertuples():
        image_id, axis = row.ImageID.split("_Fovea_")
        item = test[test.image_id == int(image_id)].iloc[0]
        assert 0 <= row.value < (item.width if axis == "X" else item.height)
    train_loader = DataLoader(
        builder(data, train, config["image_size"]), batch_size=config["batch_size"], shuffle=False
    )
    train_prediction = predict(model, train_loader, device, flip_tta=False)
    fit_metrics = error_metrics(
        train_prediction, train[["nx", "ny"]].to_numpy(), train[["width", "height"]].to_numpy()
    )
    save_json(
        out / "train_fit_metrics.json",
        dict(
            **fit_metrics,
            evaluation="All 80 training images; eval mode, no augmentation/TTA; in-sample, not validation.",
        ),
    )
    provenance = dict(
        model="ResNet34 + heatmap decoder",
        members=1,
        pretrained_source=source,
        pretraining=pretraining,
        training_images=80,
        unique_pixel_groups=int(train.pixel_sha256.nunique()),
        epochs=43,
        cosine_schedule_epochs=80,
        seed=42,
        image_size=320,
        batch_size=8,
        weights_initialization=(
            "ImageNet, freshly initialized run"
            if not resumed
            else "Local fulltrain checkpoint resume"
        ),
        resumed=resumed,
        validation_used_during_this_run=False,
        epoch_selection="Historical fold0: both mean pixel distance and coordinate RMSE minimized at epoch43.",
        historical_validation_images=16,
        historical_validation_mean_distance_px=16.11060954254833,
        historical_validation_coordinate_rmse_px=20.653982153099445,
        flip_tta=False,
        test_images=20,
        submission_rows=40,
        submission_sha256=hashlib.sha256((out / "submission.csv").read_bytes()).hexdigest(),
        submitted_to_competition=False,
        public_score=None,
        seconds=time.perf_counter() - start,
        limitation="Historical holdout scores describe the 64-image training checkpoint; this fulltrain model has no independent test score.",
    )
    save_json(out / "provenance.json", provenance)
    save_json(
        out / "environment.json",
        dict(
            torch=torch.__version__,
            numpy=np.__version__,
            gpu=torch.cuda.get_device_name(),
            device=str(device),
        ),
    )
    save_json(
        out / "run_status.json",
        dict(
            state="completed",
            epochs=43,
            training_images=80,
            submission=str(out / "submission.csv"),
            updated_unix=time.time(),
        ),
    )
    print("Completed:", out / "submission.csv", flush=True)
    return out


class LogTee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        self.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir", type=Path, default=FULLTRAIN_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "train.log").open("a", encoding="utf-8") as log:
        with contextlib.redirect_stdout(LogTee(sys.stdout, log)):
            try:
                run_resnet34_fulltrain(args.data_dir, args.output_dir)
            except BaseException as error:
                save_json(
                    args.output_dir / "run_status.json",
                    dict(state="failed", error=str(error), updated_unix=time.time()),
                )
                raise


if __name__ == "__main__":
    cli()
