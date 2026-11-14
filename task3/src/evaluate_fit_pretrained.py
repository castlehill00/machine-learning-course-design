"""Evaluate the author's supervised EfficientNet-B3 fovea checkpoint without Task3 training.

Reference: https://github.com/berenslab/fundus_image_toolbox (MIT).
Preprocessing follows its validation ResizedCenterCropAndPad transform. Crop/pad
offsets are inverted to return coordinates in the original nonsquare image.
"""

import hashlib
from pathlib import Path
import shutil
import tarfile

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import functional as TF

from pretrained_ensemble import DEFAULT_OUTPUT, PRETRAINED_ROOT, frame_predictions
from task3_pipeline import assign_folds, error_metrics, find_data, load_manifest, save_json


def crop_inverse(points, resized_width, resized_height, crop_size=350):
    top = int(round((resized_height - crop_size) / 2.0))
    left = int(round((resized_width - crop_size) / 2.0))
    return (np.asarray(points) * crop_size + np.array([left, top])) / np.array(
        [resized_width, resized_height]
    )


def validation_preprocess(image):
    """Match the author's ResizedCenterCropAndPad validation transform and record geometry."""
    width, height = image.size
    preview = TF.resize(TF.center_crop(image, height), [350, 350])
    preview = np.asarray(preview)
    black_fraction = (np.mean(preview[:10] == 0) + np.mean(preview[-10:] == 0)) / 2
    pad = black_fraction < 0.5
    if pad:
        resized_height, resized_width = 400, int(400 * width / height)
        image = TF.resize(image, [resized_height, resized_width])
        image = TF.pad(image, [0, 25, 0, 25], fill=0)
        side = resized_height + 50
        image = TF.resize(TF.center_crop(image, side), [350, 350])
        geometry = dict(
            width=resized_width,
            height=resized_height,
            side=side,
            left=(resized_width - side) / 2,
            top_pad=25,
            padded=True,
        )
    else:
        image = TF.resize(TF.center_crop(image, height), [350, 350])
        geometry = dict(
            width=width,
            height=height,
            side=height,
            left=(width - height) / 2,
            top_pad=0,
            padded=False,
        )
    return transforms.ToTensor()(image), geometry


def validation_inverse(points, geometry):
    return (np.asarray(points) * geometry["side"] + [geometry["left"], -geometry["top_pad"]]) / [
        geometry["width"],
        geometry["height"],
    ]


def ensure_fit_weights():
    path = PRETRAINED_ROOT / "weights/fit_efficientnet_b3.pt"
    if path.is_file():
        return path
    archive_path = path.parent / "fit_weights.tar.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.is_file():
        torch.hub.download_url_to_file(
            "https://zenodo.org/records/11174642/files/weights.tar.gz",
            str(archive_path),
            hash_prefix="1c6bb4e7bb82e9c7b89d573cfb3c3d86e5647aa44d547cd257a42e5aab0c27c7",
        )
    with tarfile.open(archive_path) as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith("/multi_efficientnet-b3_best.pt")
        ]
        if len(members) != 1:
            raise ValueError("Expected exactly one author EfficientNet-B3 checkpoint.")
        with archive.extractfile(members[0]) as source, path.open("wb") as target:
            shutil.copyfileobj(source, target)
    return path


def evaluate_fit(output_dir=DEFAULT_OUTPUT):
    torch.set_num_threads(4)
    path = ensure_fit_weights()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == "7f2e94ac17adf9bfbfe2e994970f999ef9bcd900144bf8b6b01f98f03f4f8dc4"
    model = models.efficientnet_b3(weights=None)
    model.classifier = nn.Linear(1536, 4)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    data = find_data()
    manifest = load_manifest(data)
    train = assign_folds(manifest[manifest.subset == "train"])
    valid, test = train[train.fold == 0], manifest[manifest.subset == "test"]
    out = Path(output_dir) / "fit_zeroshot"
    out.mkdir(parents=True, exist_ok=True)
    # Keep the first inference-helper crop experiment; the final pipeline uses the author's validation transform.
    previous = out / "provenance.json"
    if previous.exists() and "ResizedCenterCropAndPad" not in previous.read_text(encoding="utf-8"):
        archive_dir = Path(output_dir) / "fit_inference_crop_archive"
        archive_dir.mkdir(exist_ok=True)
        for file in out.glob("*.json"):
            shutil.copy2(file, archive_dir / file.name)
        for file in out.glob("*.csv"):
            shutil.copy2(file, archive_dir / file.name)
    for width, height in [(523, 350), (353, 350), (350, 350)]:
        point = np.array([0.4, 0.55])
        left, top = int(round((width - 350) / 2)), int(round((height - 350) / 2))
        crop_point = (point * [width, height] - [left, top]) / 350
        assert np.allclose(crop_inverse(crop_point, width, height), point)
    for label, records in [("validation", valid), ("test", test)]:
        images, geometries = [], []
        for row in records.itertuples():
            with Image.open(data / row.subset / row.filename) as im:
                image, geometry = validation_preprocess(im.convert("RGB"))
            images.append(image)
            geometries.append(geometry)
        raw = []
        with torch.inference_mode():
            for start in range(0, len(images), 4):
                raw.extend(model(torch.stack(images[start : start + 4]))[:, :2].numpy())
        prediction = np.stack(
            [validation_inverse(point, geometry) for point, geometry in zip(raw, geometries)]
        )
        frame_predictions(records, prediction).to_csv(out / f"{label}_predictions.csv", index=False)
        if label == "validation":
            score = error_metrics(
                prediction,
                records[["nx", "ny"]].to_numpy(),
                records[["width", "height"]].to_numpy(),
            )
            save_json(out / "validation_metrics.json", score)
            print("FIT zero-shot validation:", score, flush=True)
    save_json(
        out / "provenance.json",
        dict(
            model="Author supervised EfficientNet-B3, fovea and optic disc multitask",
            source="https://github.com/berenslab/fundus_image_toolbox",
            weights_url="https://zenodo.org/records/11174642/files/weights.tar.gz",
            pretrained_datasets=["ADAM", "REFUGE", "IDRID"],
            overlap_status="No GAMMA named in author documentation; image-level overlap not verified.",
            sha256=digest,
            task3_training=False,
            normalization="ToTensor [0,1], no channel standardization",
            preprocessing="Author validation ResizedCenterCropAndPad: PIL center crop, conditional vertical pad, resize350; inverse recorded geometry",
            crop_roundtrip_checked=True,
            device="cpu float32",
        ),
    )


if __name__ == "__main__":
    evaluate_fit()
