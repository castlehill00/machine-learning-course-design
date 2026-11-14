"""Export the selected holdout-trained B2 checkpoint with horizontal-flip TTA."""

import json
import time
import run_pretraining_local as r


def main():
    r.torch.set_num_threads(4)
    r.torch.backends.cudnn.benchmark = False
    r.torch.backends.cudnn.deterministic = True
    if not r.torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")
    folder = r.OUT / "affectnet"
    checkpoint_path = folder / "best.pt"
    ck = r.torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    saved = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    assert ck["config"] == config
    assert ck["epoch"] == saved["best_epoch"] == 9
    assert ck["config"]["classes"] == r.CLASSES
    assert abs(ck["accuracy"] - saved["val_accuracy"]) < 1e-12
    model = r.skeleton().cuda().eval()
    model.load_state_dict(ck["model"], strict=True)

    # Reproduce one original validation batch before exporting test predictions.
    val = r.pd.read_csv(folder / "validation_tta_probabilities.csv").iloc[:64]
    val_frame = val.rename(columns={"true_class": "class"})
    _, check = r.predict(model, r.loader(val_frame, config["image_size"], 64), flip=True)
    max_error = float(abs(check - val[r.CLASSES].to_numpy()).max())
    if max_error > 1e-4:
        raise RuntimeError(f"Validation prediction reproduction failed: {max_error}")
    print("Checkpoint epoch 9 verified; validation reproduction error:", max_error, flush=True)

    template = r.pd.read_csv(r.ROOT / "2026task2/submission.csv", dtype={"ID": str})
    assert list(template.columns) == ["ID", "Emotion"]
    assert len(template) == 7065 and template.ID.is_unique and template.ID.notna().all()
    assert set(template.ID) == {p.name for p in (r.DATA / "test").glob("*.jpg")}
    # Dataset labels are unused by predict; no test labels are read.
    frame = r.pd.DataFrame({"path": ["test/" + s for s in template.ID], "class": "Angry"})
    dl = r.loader(frame, config["image_size"], 64)
    start = time.monotonic()
    print("Predicting 7065 test images with original + horizontal flip...", flush=True)
    base, tta = r.predict(model, dl, flip=True)
    assert tta.shape == (7065, 6) and r.np.isfinite(tta).all()
    assert r.np.allclose(tta.sum(1), 1, atol=1e-6)
    names = [r.CLASSES[i] for i in tta.argmax(1)]
    submission = r.pd.DataFrame({"ID": template.ID, "Emotion": [r.OFFICIAL[n] for n in names]})
    assert submission.ID.tolist() == template.ID.tolist()
    assert submission.Emotion.isin(range(6)).all() and not submission.isna().any().any()
    output = r.OUT / "predictions/affectnet_epoch9"
    output.mkdir(parents=True, exist_ok=True)
    for name, values in [("test_probabilities.csv", base), ("test_tta_probabilities.csv", tta)]:
        probability = r.pd.DataFrame(values, columns=r.CLASSES)
        probability.insert(0, "ID", template.ID)
        probability.to_csv(output / name, index=False)
    r.pd.DataFrame({"ID": template.ID, "EmotionName": names}).to_csv(
        output / "predictions_labels.csv", index=False
    )
    destination = output / "submission_b2_affectnet_epoch9_tta.csv"
    temporary = destination.with_suffix(".csv.tmp")
    submission.to_csv(temporary, index=False)
    temporary.replace(destination)
    reloaded = r.pd.read_csv(destination, dtype={"ID": str})
    assert reloaded.equals(submission)
    r.atomic_json(
        output / "export_summary.json",
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": r.sha(checkpoint_path),
            "epoch": 9,
            "training_scope": "original train split only; NOT full-data training",
            "validation_accuracy": saved["val_accuracy"],
            "validation_tta_accuracy": saved["tta_accuracy"],
            "validation_reproduction_max_error": max_error,
            "tta": "average of original and horizontally flipped softmax probabilities",
            "rows": len(submission),
            "official_mapping": r.OFFICIAL,
            "label_counts": {
                str(k): int(v) for k, v in submission.Emotion.value_counts().sort_index().items()
            },
            "submission_sha256": r.sha(destination),
            "prediction_seconds": time.monotonic() - start,
            "submission": str(destination),
            "leaderboard_score": None,
        },
    )
    print("Verified submission:", destination, flush=True)
    print(submission.head().to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
