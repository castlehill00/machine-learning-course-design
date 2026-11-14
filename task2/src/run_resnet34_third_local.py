"""Recover a holdout ResNet34, gate third-model fusion, reuse existing full refit."""

import gc
import json
import os
import sys
import traceback
import run_pretraining_local as r
import run_resnet18_fulltrain_local as trainer
import run_multicrop_local as crop

OUT = r.ROOT / "task2_third_resnet34_seed42"
SOURCE = r.ROOT / "task2_multicrop_tta_seed42"
GRID = [0.0, 0.2, 1 / 3]


def read_cache(tag, paths):
    meta = json.loads((SOURCE / (tag + ".json")).read_text())
    assert meta["paths_sha256"] == r.hashlib.sha256("\n".join(paths).encode()).hexdigest()
    assert meta["classes"] == r.CLASSES
    a = r.np.load(SOURCE / (tag + ".npz"))["probabilities"].astype(float)
    assert a.shape == (len(paths), 6, 6)
    crop.check(a.mean(1), len(paths))
    return a.mean(1), meta


def main():
    assert r.torch.cuda.is_available()
    r.torch.set_num_threads(4)
    r.torch.backends.cudnn.benchmark = False
    r.torch.backends.cudnn.deterministic = True
    r.OUT = OUT
    crop.OUT = OUT
    train, val, split_sha = r.load_split()
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    protocol = {
        "third_model": "resnet34",
        "third_weights": GRID,
        "within_original_pair_weights": [0.5, 0.5],
        "views": "original and five 44x44 crops, horizontal flip each, 12 total",
        "holdout_refit_epochs": 14,
        "scheduler_t_max": 25,
        "epoch_reason": "fixed from prior ResNet34 experiment, not selected using this fusion result",
        "minimum_net_correct": 10,
        "minimum_positive_folds": 4,
        "selection": "highest validation accuracy among candidates passing both gates, then lower NLL",
        "split_sha256": split_sha,
        "warning": "Uses existing development set; not an independent test estimate. Full model never used to tune validation fusion.",
    }
    if (OUT / "protocol.json").exists():
        assert json.loads((OUT / "protocol.json").read_text()) == protocol
    r.atomic_json(OUT / "protocol.json", protocol)
    weights = r.WEIGHTS / "resnet34-b627a593.pth"
    assert r.sha(weights).startswith("b627a593")
    full = r.ROOT / "task2_fulltrain_resnet34_seed42/final.pt"
    ck = r.torch.load(full, map_location="cpu", weights_only=True)
    assert ck["config"]["classes"] == r.CLASSES and ck["config"]["epochs"] == 14
    del ck
    config = {
        "pipeline": "resnet34_holdout_fixed14_v1",
        "architecture": "resnet34",
        "seed": 42,
        "classes": r.CLASSES,
        "training_samples": 22619,
        "validation_samples": 5656,
        "epochs": 14,
        "scheduler_t_max": 25,
        "image_size": 224,
        "batch_size": 128,
        "accumulation": 1,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "dropout": 0.0,
        "label_smoothing": 0.0,
        "split_sha256": split_sha,
        "pretrained_sha256": r.sha(weights),
        "selection": "fixed final epoch 14; validation only after training",
    }
    folder = OUT / "holdout_resnet34"
    folder.mkdir(exist_ok=True)
    source = r.torch.load(weights, map_location="cpu", weights_only=True)
    r.atomic_json(
        OUT / "status.json",
        {
            "state": "running",
            "pid": os.getpid(),
            "stage": "training_holdout_resnet34",
            "progress_file": str(folder / "status.json"),
        },
    )
    model, history = trainer.train(train, source, config, folder, architecture="resnet34")
    del model, source
    gc.collect()
    r.torch.cuda.empty_cache()
    b, bmeta = read_cache("validation_b2", val.path.tolist())
    rn, rmeta = read_cache("validation_resnet18", val.path.tolist())
    three = (
        crop.infer("resnet34", folder / "final.pt", val, "validation_resnet34")
        .astype(float)
        .mean(1)
    )
    base = 0.5 * b + 0.5 * rn
    truth = r.np.array([r.CLASSES.index(x) for x in val["class"]])
    folds = r.np.array(
        [
            int(r.hashlib.sha256(("calibration42:" + h).encode()).hexdigest()[:12], 16) % 5
            for h in val.pixel_sha256
        ]
    )
    base_ok = base.argmax(1) == truth
    rows = []
    for weight in GRID:
        values = (1 - weight) * base + weight * three
        metrics, _ = r.metrics(values, truth)
        correct = values.argmax(1) == truth
        gains = [int(correct[folds == i].sum() - base_ok[folds == i].sum()) for i in range(5)]
        rows.append(
            {
                "resnet34_weight": weight,
                "accuracy": metrics["accuracy"],
                "nll": metrics["nll"],
                "macro_f1": metrics["macro_f1"],
                "gain_correct": int(correct.sum() - base_ok.sum()),
                "positive_folds": sum(g > 0 for g in gains),
                "fold_gains": gains,
            }
        )
    r.pd.DataFrame(rows).to_csv(OUT / "validation_comparison.csv", index=False)
    accepted = [v for v in rows if v["gain_correct"] >= 10 and v["positive_folds"] >= 4]
    selected = sorted(accepted, key=lambda v: (-v["accuracy"], v["nll"]))[0] if accepted else None
    report = {
        "protocol": protocol,
        "comparisons": rows,
        "selected": selected,
        "holdout_checkpoint_sha256": r.sha(folder / "final.pt"),
        "training_minutes": sum(h["seconds"] for h in history) / 60,
        "resnet34_alone_validation": r.metrics(three, truth)[0],
        "leaderboard_score": None,
    }
    r.atomic_json(OUT / "analysis.json", report)
    print("VALIDATION RESULT", json.dumps(report), flush=True)
    if selected is None:
        r.atomic_json(
            OUT / "status.json",
            {
                "state": "completed_no_export",
                "pid": os.getpid(),
                "reason": "No third-model mixture passed the fixed validation gate.",
            },
        )
        return
    template = r.pd.read_csv(r.ROOT / "2026task2/submission.csv", dtype={"ID": str})
    assert len(template) == 7065 and template.ID.is_unique
    test = r.pd.DataFrame({"path": ["test/" + s for s in template.ID]})
    bt, _ = read_cache("test_b2_fulltrain", test.path.tolist())
    rt, _ = read_cache("test_resnet18_fulltrain", test.path.tolist())
    ct = crop.infer("resnet34", full, test, "test_resnet34_fulltrain").astype(float).mean(1)
    weight = selected["resnet34_weight"]
    prob = (1 - weight) * (0.5 * bt + 0.5 * rt) + weight * ct
    crop.check(prob, len(test))
    output = r.pd.DataFrame(
        {"ID": template.ID, "Emotion": [r.OFFICIAL[r.CLASSES[i]] for i in prob.argmax(1)]}
    )
    path = OUT / "submission_three_model_crop12.csv"
    output.to_csv(path, index=False)
    assert r.pd.read_csv(path).equals(output)
    probability = r.pd.DataFrame(prob, columns=r.CLASSES)
    probability.insert(0, "ID", template.ID)
    probability.to_csv(OUT / "test_probabilities.csv", index=False)
    report.update(
        submission=str(path),
        submission_sha256=r.sha(path),
        full_resnet34_checkpoint_sha256=r.sha(full),
        test_rows=len(output),
        weights={"b2": (1 - weight) / 2, "resnet18": (1 - weight) / 2, "resnet34": weight},
    )
    r.atomic_json(OUT / "analysis.json", report)
    r.atomic_json(
        OUT / "status.json", {"state": "completed", "pid": os.getpid(), "submission": str(path)}
    )
    print("EXPORT COMPLETE", path, flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    OUT.mkdir(exist_ok=True)
    import msvcrt

    with open(OUT / "run.lock", "a+b") as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise SystemExit("Another third-model run is active.")
        try:
            r.atomic_json(OUT / "status.json", {"state": "starting", "pid": os.getpid()})
            main()
        except BaseException as exc:
            r.atomic_json(
                OUT / "status.json", {"state": "failed", "pid": os.getpid(), "error": str(exc)}
            )
            traceback.print_exc()
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
