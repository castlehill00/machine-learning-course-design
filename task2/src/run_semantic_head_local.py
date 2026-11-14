"""Transfer the semantic expression head, validate, then conditionally refit all data."""

import gc
import json
import os
import sys
import traceback
import run_pretraining_local as r
import run_b2_fulltrain_local as full
import run_multicrop_local as crop

OUT = r.ROOT / "task2_semantic_head_seed42"
PREVIOUS = r.ROOT / "task2_multicrop_tta_seed42"
GRID = [0.0, 0.25, 0.5]  # New B2 weight; ResNet18 stays .5, original B2 is .5-new weight.


def read_cached(tag, paths):
    meta = json.loads((PREVIOUS / (tag + ".json")).read_text())
    assert meta["classes"] == r.CLASSES
    assert meta["paths_sha256"] == r.hashlib.sha256("\n".join(paths).encode()).hexdigest()
    p = r.np.load(PREVIOUS / (tag + ".npz"))["probabilities"].astype(float).mean(1)
    crop.check(p, len(paths))
    return p


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
        "change": "copy pretrained expression classifier rows instead of random head",
        "source_rows": r.SOURCE_ROWS,
        "classes": r.CLASSES,
        "development_epochs": 12,
        "scheduler_t_max": 25,
        "seed": 42,
        "new_b2_weights": GRID,
        "resnet18_weight": 0.5,
        "old_b2_weight": ".5-new_b2_weight",
        "minimum_gain_correct": 20,
        "minimum_positive_folds": 4,
        "checkpoint_selection": "highest raw validation accuracy, then lower NLL",
        "fusion_selection": "highest crop12 validation accuracy among candidates passing both gates, then lower NLL",
        "split_sha256": split_sha,
        "note": "Validation tuning reuses existing development set. Full-data training only if gate passes.",
    }
    if (OUT / "protocol.json").exists():
        assert json.loads((OUT / "protocol.json").read_text()) == protocol
    r.atomic_json(OUT / "protocol.json", protocol)
    states, sources = r.prepare_weights()
    source = states["affectnet"]
    del states
    common = json.loads((r.ROOT / "task2_pretraining_b2_seed42/affectnet/config.json").read_text())
    common.pop("experiment")
    common.update(
        pipeline="b2_semantic_head_v1",
        epochs=12,
        scheduler_t_max=25,
        copy_emotion_head=True,
        scheduler="cosine T_max=25, first 12 epochs",
        head_initialization="copy pretrained classifier rows [0,2,3,4,5,6] in target semantic order",
    )
    assert common["split_sha256"] == split_sha and common["classes"] == r.CLASSES
    result = r.run_experiment("semantic_head", source, train, val, common)
    fresh = (
        crop.infer("b2", OUT / "semantic_head/best.pt", val, "validation_semantic_head")
        .astype(float)
        .mean(1)
    )
    b = read_cached("validation_b2", val.path.tolist())
    rn = read_cached("validation_resnet18", val.path.tolist())
    base = 0.5 * b + 0.5 * rn
    y = r.np.array([r.CLASSES.index(c) for c in val["class"]])
    folds = r.np.array(
        [
            int(r.hashlib.sha256(("calibration42:" + h).encode()).hexdigest()[:12], 16) % 5
            for h in val.pixel_sha256
        ]
    )
    base_ok = base.argmax(1) == y
    rows = []
    for w in GRID:
        p = (0.5 - w) * b + 0.5 * rn + w * fresh
        met, _ = r.metrics(p, y)
        correct = p.argmax(1) == y
        gains = [int(correct[folds == i].sum() - base_ok[folds == i].sum()) for i in range(5)]
        rows.append(
            {
                "new_b2_weight": w,
                **met,
                "gain_correct": int(correct.sum() - base_ok.sum()),
                "positive_folds": sum(g > 0 for g in gains),
                "fold_gains": gains,
            }
        )
    r.pd.DataFrame(rows).to_csv(OUT / "validation_comparison.csv", index=False)
    accepted = [v for v in rows if v["gain_correct"] >= 20 and v["positive_folds"] >= 4]
    selected = sorted(accepted, key=lambda v: (-v["accuracy"], v["nll"]))[0] if accepted else None
    report = {
        "protocol": protocol,
        "model_metrics": result,
        "comparisons": rows,
        "selected": selected,
        "holdout_checkpoint_sha256": r.sha(OUT / "semantic_head/best.pt"),
        "crop12_single_model_metrics": r.metrics(fresh, y)[0],
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
                "reason": "No mixture passed the predeclared validation gates.",
            },
        )
        return
    folder = OUT / "fulltrain"
    folder.mkdir(exist_ok=True)
    frame = r.pd.concat([train, val]).sort_values("path").reset_index(drop=True)
    assert len(frame) == 28275 and frame.path.is_unique
    config = dict(
        common,
        experiment="semantic_head_fulltrain",
        epochs=result["best_epoch"],
        training_samples=28275,
        selection="fixed best epoch inherited from holdout; no full-data validation",
    )
    frame.to_csv(folder / "train_manifest.csv", index=False)
    r.atomic_json(
        OUT / "status.json",
        {
            "state": "running",
            "pid": os.getpid(),
            "stage": "fulltrain",
            "progress_file": str(folder / "status.json"),
        },
    )
    model, history = full.train(frame, source, config, folder)
    del model, source
    gc.collect()
    r.torch.cuda.empty_cache()
    template = r.pd.read_csv(r.ROOT / "2026task2/submission.csv", dtype={"ID": str})
    assert len(template) == 7065 and template.ID.is_unique
    test = r.pd.DataFrame({"path": ["test/" + s for s in template.ID]})
    bp = read_cached("test_b2_fulltrain", test.path.tolist())
    rp = read_cached("test_resnet18_fulltrain", test.path.tolist())
    fp = (
        crop.infer("b2", folder / "final.pt", test, "test_semantic_head_fulltrain")
        .astype(float)
        .mean(1)
    )
    w = selected["new_b2_weight"]
    prob = (0.5 - w) * bp + 0.5 * rp + w * fp
    crop.check(prob, len(template))
    probability = r.pd.DataFrame(prob, columns=r.CLASSES)
    probability.insert(0, "ID", template.ID)
    probability.to_csv(OUT / "test_probabilities.csv", index=False)
    output = r.pd.DataFrame(
        {"ID": template.ID, "Emotion": [r.OFFICIAL[r.CLASSES[i]] for i in prob.argmax(1)]}
    )
    path = OUT / "submission_semantic_head_crop12.csv"
    output.to_csv(path, index=False)
    assert r.pd.read_csv(path).equals(output)
    old = r.pd.read_csv(PREVIOUS / "submission_multicrop_fusion.csv")
    assert old.ID.tolist() == output.ID.tolist()
    report.update(
        submission=str(path),
        submission_sha256=r.sha(path),
        test_rows=len(output),
        full_checkpoint_sha256=r.sha(folder / "final.pt"),
        test_changed_vs_0734=int((old.Emotion != output.Emotion).sum()),
        fulltrain_epochs=len(history),
        weights={"old_b2": 0.5 - w, "resnet18": 0.5, "semantic_b2": w},
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
            raise SystemExit("Another semantic-head experiment is active.")
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
