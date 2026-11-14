"""Validate B2 + original ResNet18 probability fusion, then export selected TTA fusion."""

import io
import json
import sys
import zipfile
import run_pretraining_local as r
from torchvision import models

OUT = r.ROOT / "task2_ensemble_b2_resnet18"
GRID = [0.0, 0.25, 0.5, 0.75, 1.0]  # Fixed before observing fusion results.


def probabilities(frame, keys, key_column):
    if not frame[key_column].is_unique or set(frame[key_column]) != set(keys):
        raise ValueError(f"Incomplete/duplicate {key_column} alignment.")
    aligned = frame.set_index(key_column).loc[keys]
    values = aligned[r.CLASSES].to_numpy(dtype="float64")
    if not r.np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid probabilities.")
    if not r.np.allclose(values.sum(1), 1, atol=1e-5):
        raise ValueError("Probability rows do not sum to one.")
    return values, aligned


def write_predictions(name, keys, values, key="path"):
    frame = r.pd.DataFrame(values, columns=r.CLASSES)
    frame.insert(0, key, keys)
    frame.to_csv(OUT / name, index=False)


def main():
    OUT.mkdir(exist_ok=True)
    if not r.torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    r.torch.set_num_threads(4)
    r.torch.backends.cudnn.benchmark = False
    r.torch.backends.cudnn.deterministic = True
    b2_folder = r.OUT / "affectnet"
    b2_config = json.loads((b2_folder / "config.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(r.ROOT / "task2_ablation_backup_round2.zip") as z:
        split_bytes = z.read("task2_ablation_seed42/split.csv")
        split = r.pd.read_csv(io.BytesIO(split_bytes))
        baseline_config = json.loads(z.read("task2_ablation_seed42/baseline/config.json"))
        baseline_metrics = json.loads(z.read("task2_ablation_seed42/baseline/metrics.json"))
        baseline_saved = r.pd.read_csv(
            io.BytesIO(z.read("task2_ablation_seed42/baseline/validation_probabilities.csv"))
        )
        checkpoint_bytes = z.read("task2_ablation_seed42/baseline/best.pt")
    split_hash = r.hashlib.sha256(split_bytes).hexdigest()
    assert split_hash == b2_config["split_sha256"] == baseline_config["split_hash"]
    assert baseline_config["classes"] == b2_config["classes"] == r.CLASSES
    assert split.groupby("pixel_sha256").fold.nunique().max() == 1
    val = split[split.fold == 0].reset_index(drop=True)
    assert len(val) == 5656
    for row in val.itertuples(index=False):
        with r.Image.open(r.DATA / row.path) as im:
            gray = im.convert("L")
            fingerprint = r.hashlib.sha256(str(gray.size).encode() + gray.tobytes()).hexdigest()
        assert fingerprint == row.pixel_sha256
    truth = r.np.array([r.CLASSES.index(c) for c in val["class"]])
    keys = val.path.tolist()
    b2_raw, raw_aligned = probabilities(
        r.pd.read_csv(b2_folder / "validation_probabilities.csv"), keys, "path"
    )
    b2_tta, tta_aligned = probabilities(
        r.pd.read_csv(b2_folder / "validation_tta_probabilities.csv"), keys, "path"
    )
    old_raw, old_aligned = probabilities(baseline_saved, keys, "path")
    for aligned in [raw_aligned, tta_aligned, old_aligned]:
        assert aligned.true_class.tolist() == val["class"].tolist()

    ck = r.torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    assert ck["epoch"] == baseline_metrics["best_epoch"] == 20
    assert ck["config"] == baseline_config
    model = models.resnet18(weights=None)
    model.fc = r.nn.Sequential(r.nn.Dropout(0), r.nn.Linear(model.fc.in_features, 6))
    model.load_state_dict(ck["model"], strict=True)
    model.cuda().eval()
    print(
        "Recomputing original holdout ResNet18 validation predictions with flip TTA...", flush=True
    )
    rn_raw, rn_tta = r.predict(model, r.loader(val, 224, 128), flip=True)
    rn_raw = rn_raw.astype("float64")
    rn_tta = rn_tta.astype("float64")
    repro_error = float(abs(rn_raw - old_raw).max())
    repro_prediction_changes = int((rn_raw.argmax(1) != old_raw.argmax(1)).sum())
    if repro_error > 0.02 or repro_prediction_changes > 10:
        raise RuntimeError(
            f"ResNet18 reproduction discrepancy: {repro_error}, {repro_prediction_changes}"
        )
    write_predictions("resnet18_validation_raw.csv", keys, rn_raw)
    write_predictions("resnet18_validation_tta.csv", keys, rn_tta)
    rows = []
    for alpha in GRID:
        raw, _ = r.metrics(alpha * b2_raw + (1 - alpha) * rn_raw, truth)
        tta, _ = r.metrics(alpha * b2_tta + (1 - alpha) * rn_tta, truth)
        rows.append(
            {
                "b2_weight": alpha,
                "resnet18_weight": 1 - alpha,
                "raw_accuracy": raw["accuracy"],
                "tta_accuracy": tta["accuracy"],
                "tta_correct": int(round(tta["accuracy"] * len(truth))),
                "tta_macro_f1": tta["macro_f1"],
                "tta_nll": tta["nll"],
            }
        )
    grid = r.pd.DataFrame(rows)
    grid.to_csv(OUT / "weight_comparison.csv", index=False)
    selected = sorted(rows, key=lambda v: (-v["tta_accuracy"], v["tta_nll"]))[0]
    alpha = selected["b2_weight"]
    combined = alpha * b2_tta + (1 - alpha) * rn_tta
    m, cm = r.metrics(combined, truth)
    b_ok = b2_tta.argmax(1) == truth
    r_ok = rn_tta.argmax(1) == truth
    f_ok = combined.argmax(1) == truth
    pair = {
        "both_correct": int((b_ok & r_ok).sum()),
        "b2_only_correct": int((b_ok & ~r_ok).sum()),
        "resnet18_only_correct": int((~b_ok & r_ok).sum()),
        "both_wrong": int((~b_ok & ~r_ok).sum()),
        "fusion_corrected_b2_errors": int((~b_ok & f_ok).sum()),
        "fusion_broke_b2_correct": int((b_ok & ~f_ok).sum()),
        "net_correct_gain_vs_b2": int(f_ok.sum() - b_ok.sum()),
    }
    r.pd.DataFrame(cm, index=r.CLASSES, columns=r.CLASSES).to_csv(OUT / "confusion_matrix.csv")
    changes = r.pd.DataFrame(
        {
            "path": keys,
            "true_class": val["class"],
            "b2_prediction": [r.CLASSES[i] for i in b2_tta.argmax(1)],
            "resnet18_prediction": [r.CLASSES[i] for i in rn_tta.argmax(1)],
            "fusion_prediction": [r.CLASSES[i] for i in combined.argmax(1)],
            "b2_correct": b_ok,
            "fusion_correct": f_ok,
        }
    )
    changes.to_csv(OUT / "validation_decisions.csv", index=False)
    write_predictions("fusion_validation_tta.csv", keys, combined)
    provenance = {
        "grid": GRID,
        "selection_rule": "max validation TTA accuracy, tie: lower NLL",
        "split_sha256": split_hash,
        "resnet18_checkpoint_sha256": r.hashlib.sha256(checkpoint_bytes).hexdigest(),
        "b2_checkpoint_sha256": r.sha(b2_folder / "best.pt"),
        "resnet18_epoch": 20,
        "b2_epoch": 9,
        "resnet18_reproduction_max_error": repro_error,
        "resnet18_reproduction_prediction_changes": repro_prediction_changes,
        "selected": selected,
        "complementarity": pair,
        "fusion_metrics": m,
        "note": "Weights chosen on this validation set; improvement is not an independent test estimate. No full-data model used.",
        "official_mapping": r.OFFICIAL,
        "submission": None,
    }
    r.atomic_json(OUT / "analysis.json", provenance)
    print(grid.to_string(index=False), flush=True)
    print("Complementarity:", json.dumps(pair), flush=True)
    if alpha in [0.0, 1.0]:
        print(
            "No interior mixture beat the selected single model; no new submission exported.",
            flush=True,
        )
        return

    template = r.pd.read_csv(r.ROOT / "2026task2/submission.csv", dtype={"ID": str})
    assert list(template.columns) == ["ID", "Emotion"] and len(template) == 7065
    assert template.ID.is_unique and template.ID.notna().all()
    assert set(template.ID) == {p.name for p in (r.DATA / "test").glob("*.jpg")}
    b2_test, _ = probabilities(
        r.pd.read_csv(r.OUT / "predictions/affectnet_epoch9/test_tta_probabilities.csv"),
        template.ID.tolist(),
        "ID",
    )
    export = json.loads(
        (r.OUT / "predictions/affectnet_epoch9/export_summary.json").read_text(encoding="utf-8")
    )
    assert export["checkpoint_sha256"] == provenance["b2_checkpoint_sha256"]
    test_frame = r.pd.DataFrame({"path": ["test/" + s for s in template.ID], "class": "Angry"})
    print("Predicting test images with original ResNet18; no retraining...", flush=True)
    _, rn_test = r.predict(model, r.loader(test_frame, 224, 128), flip=True)
    write_predictions("resnet18_test_tta.csv", template.ID.tolist(), rn_test, "ID")
    fused_test = alpha * b2_test + (1 - alpha) * rn_test
    assert r.np.isfinite(fused_test).all() and r.np.allclose(fused_test.sum(1), 1, atol=1e-5)
    write_predictions("fusion_test_tta.csv", template.ID.tolist(), fused_test, "ID")
    names = [r.CLASSES[i] for i in fused_test.argmax(1)]
    submission = r.pd.DataFrame({"ID": template.ID, "Emotion": [r.OFFICIAL[c] for c in names]})
    assert submission.Emotion.isin(range(6)).all() and not submission.isna().any().any()
    destination = OUT / "submission_ensemble_b2_resnet18.csv"
    temp = destination.with_suffix(".csv.tmp")
    submission.to_csv(temp, index=False)
    temp.replace(destination)
    assert r.pd.read_csv(destination).equals(submission)
    provenance.update(
        submission=str(destination),
        submission_sha256=r.sha(destination),
        test_rows=len(submission),
        leaderboard_score=None,
    )
    r.atomic_json(OUT / "analysis.json", provenance)
    print("Verified submission:", destination, flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    main()
