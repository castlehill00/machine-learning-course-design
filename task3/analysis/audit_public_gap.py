"""Audit submitted-file disagreement and holdout/fulltrain prediction stability.

No retraining, test labels, competition submissions, or score-based weight fitting.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # 引用 src/ 下的实验代码
from pretrained_ensemble import PretrainedHeatmap, make_dataset
from task3_pipeline import (
    FoveaDataset,
    FoveaHeatmap,
    find_data,
    load_manifest,
    predict,
    seed_everything,
)


def main():
    task = Path(__file__).resolve().parent
    task = task.parent if task.name == "analysis" else task  # 位于 analysis/ 时回到任务根目录
    out = task / "public_score_audit"
    out.mkdir(exist_ok=True)
    base_root = Path(json.loads((task / "run_locations.json").read_text())["raw_run_dir"])
    multi_root = Path(
        json.loads((task / "pretrained_run_locations.json").read_text())["raw_run_dir"]
    )
    data = find_data()
    manifest = load_manifest(data)
    test = manifest[manifest.subset == "test"].copy()
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = test[["image_id", "width", "height"]].copy()
    checks = []
    for name, checkpoint_path in [
        ("resnet18_holdout", base_root / "fold0/best.pt"),
        ("resnet34_holdout", multi_root / "resnet34/fold0/best.pt"),
        ("green_holdout", multi_root / "retfound_green/fold0/best.pt"),
        ("green_full_recheck", multi_root / "retfound_green/fulltrain/best.pt"),
    ]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        if name.startswith("resnet18"):
            model = FoveaHeatmap(pretrained=False)
            dataset = FoveaDataset(data, test, config["image_size"])
        else:
            model = PretrainedHeatmap(config)
            dataset = make_dataset(config)(data, test, config["image_size"])
        model.load_state_dict(checkpoint["model"], strict=True)
        model = model.to(device).eval()
        loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
        prediction = predict(model, loader, device)
        pixels = prediction * test[["width", "height"]].to_numpy()
        result[f"{name}_x"], result[f"{name}_y"] = pixels.T
        checks.append(
            dict(model=name, checkpoint=str(checkpoint_path), epoch=int(checkpoint["epoch"]))
        )
        del model, loader, dataset, checkpoint
        torch.cuda.empty_cache()
    for name, path in [
        ("resnet18_full", base_root / "test_predictions.csv"),
        ("resnet34_full", multi_root / "resnet34/test_predictions.csv"),
        ("green_full", multi_root / "retfound_green/test_predictions.csv"),
    ]:
        saved = pd.read_csv(path).set_index("image_id").loc[test.image_id]
        pixels = saved[["Fovea_X", "Fovea_Y"]].to_numpy()
        result[f"{name}_x"], result[f"{name}_y"] = pixels.T
    full_recheck = result[["green_full_recheck_x", "green_full_recheck_y"]].to_numpy()
    full_cached = result[["green_full_x", "green_full_y"]].to_numpy()
    delta = float(np.abs(full_recheck - full_cached).max())
    if delta > 0.02:
        raise AssertionError(f"Recomputed Green differs from saved predictions: {delta}px")
    summary = []
    for prefix in ["resnet18", "resnet34", "green"]:
        held = result[[f"{prefix}_holdout_x", f"{prefix}_holdout_y"]].to_numpy()
        full = result[[f"{prefix}_full_x", f"{prefix}_full_y"]].to_numpy()
        distance = np.linalg.norm(held - full, axis=1)
        result[f"{prefix}_holdout_to_full_distance_px"] = distance
        worst = int(np.argmax(distance))
        summary.append(
            dict(
                model=prefix,
                mean_shift_px=float(distance.mean()),
                median_shift_px=float(np.median(distance)),
                max_shift_px=float(distance.max()),
                largest_shift_image=int(test.iloc[worst].image_id),
            )
        )
    result.to_csv(out / "holdout_full_test_predictions.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "holdout_full_stability.csv", index=False)
    (out / "checkpoint_checks.json").write_text(
        json.dumps(
            dict(
                checkpoints=checks,
                green_cached_prediction_max_delta_px=delta,
                method="Inference only; no test ground-truth available",
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print("HOLDOUT/FULL STABILITY")
    print(pd.DataFrame(summary).to_string(index=False))
    print("IMAGE 81")
    print(result[result.image_id == 81].to_string(index=False))
    print("Green saved prediction recheck max delta:", delta, "px")


if __name__ == "__main__":
    main()
