"""Fixed crop-size ablation using existing models; export at most one candidate."""

import json
import os
import sys
import traceback
import zipfile
import run_pretraining_local as r
import run_multicrop_local as crop

OUT = r.ROOT / "task2_crop_scales_seed42"
SOURCE = r.ROOT / "task2_multicrop_tta_seed42"
MODES = ["crop44", "crop42", "crop46", "multiscale36"]


def previous(tag, frame):
    meta = json.loads((SOURCE / (tag + ".json")).read_text())
    assert meta["classes"] == r.CLASSES and meta["crop"] == 44
    assert meta["paths_sha256"] == r.hashlib.sha256("\n".join(frame.path).encode()).hexdigest()
    p = r.np.load(SOURCE / (tag + ".npz"))["probabilities"].astype(float)
    assert p.shape == (len(frame), 6, 6)
    crop.check(p.mean(1), len(frame))
    return p.mean(1)


def collect(kind, checkpoint, frame, tag):
    values = {44: previous(tag, frame)}
    for size in [42, 46]:
        values[size] = (
            crop.infer(kind, checkpoint, frame, tag + "_crop" + str(size), crop_size=size)
            .astype(float)
            .mean(1)
        )
    return {
        "crop44": values[44],
        "crop42": values[42],
        "crop46": values[46],
        "multiscale36": sum(values.values()) / 3,
    }


def main():
    assert r.torch.cuda.is_available()
    r.torch.set_num_threads(4)
    r.torch.backends.cudnn.benchmark = False
    r.torch.backends.cudnn.deterministic = True
    r.OUT = OUT
    crop.OUT = OUT
    _, val, split_sha = r.load_split()
    val = val.reset_index(drop=True)
    protocol = {
        "candidates": MODES,
        "crop_sizes": [42, 44, 46],
        "b2_weight": 0.5,
        "spatial_views": "full and five positions, horizontal flip each",
        "multiscale": "equal probability average of 12-view predictions at all three crop sizes",
        "acceptance": "net gain >=20 correct and >=4 positive subsets of 5 fixed pixel-group subsets",
        "selection": "maximum accuracy among passing candidates, then lower NLL",
        "split_sha256": split_sha,
        "note": "Sizes and weights fixed before results. Development set has been reused in previous model selection.",
    }
    if (OUT / "protocol.json").exists():
        assert json.loads((OUT / "protocol.json").read_text()) == protocol
    r.atomic_json(OUT / "protocol.json", protocol)
    with zipfile.ZipFile(r.ROOT / "task2_ablation_backup_round2.zip") as z:
        rn_checkpoint = z.read("task2_ablation_seed42/baseline/best.pt")
    b_checkpoint = r.ROOT / "task2_pretraining_b2_seed42/affectnet/best.pt"
    original = json.loads((r.ROOT / "task2_ensemble_b2_resnet18/analysis.json").read_text())
    assert r.hashlib.sha256(rn_checkpoint).hexdigest() == original["resnet18_checkpoint_sha256"]
    assert r.sha(b_checkpoint) == original["b2_checkpoint_sha256"]
    b = collect("b2", b_checkpoint, val, "validation_b2")
    rn = collect("resnet18", rn_checkpoint, val, "validation_resnet18")
    truth = r.np.array([r.CLASSES.index(c) for c in val["class"]])
    subsets = r.np.array(
        [
            int(r.hashlib.sha256(("calibration42:" + h).encode()).hexdigest()[:12], 16) % 5
            for h in val.pixel_sha256
        ]
    )
    base = 0.5 * b["crop44"] + 0.5 * rn["crop44"]
    base_ok = base.argmax(1) == truth
    rows = []
    predictions = {}
    for mode in MODES:
        p = 0.5 * b[mode] + 0.5 * rn[mode]
        predictions[mode] = p
        metrics, _ = r.metrics(p, truth)
        ok = p.argmax(1) == truth
        gains = [int(ok[subsets == i].sum() - base_ok[subsets == i].sum()) for i in range(5)]
        rows.append(
            {
                "mode": mode,
                **metrics,
                "gain_correct": int(ok.sum() - base_ok.sum()),
                "positive_subsets": sum(g > 0 for g in gains),
                "subset_gains": gains,
            }
        )
    r.pd.DataFrame(rows).to_csv(OUT / "validation_comparison.csv", index=False)
    r.np.savez_compressed(OUT / "validation_fusion_probabilities.npz", **predictions)
    passed = [v for v in rows if v["gain_correct"] >= 20 and v["positive_subsets"] >= 4]
    selected = sorted(passed, key=lambda v: (-v["accuracy"], v["nll"]))[0] if passed else None
    report = {
        "protocol": protocol,
        "comparisons": rows,
        "selected": selected,
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
                "reason": "No crop-scale candidate passed the fixed validation gates.",
            },
        )
        return
    template = r.pd.read_csv(r.ROOT / "2026task2/submission.csv", dtype={"ID": str})
    assert len(template) == 7065 and template.ID.is_unique
    assert set(template.ID) == {p.name for p in (r.DATA / "test").glob("*.jpg")}
    frame = r.pd.DataFrame({"path": ["test/" + s for s in template.ID]})
    mode = selected["mode"]

    def test_prediction(kind, ck, tag):
        if mode == "multiscale36":
            return collect(kind, ck, frame, tag)[mode]
        size = int(mode[4:])
        return (
            crop.infer(kind, ck, frame, tag + "_crop" + str(size), crop_size=size)
            .astype(float)
            .mean(1)
        )

    bp = test_prediction("b2", r.ROOT / "task2_fulltrain_b2_seed42/final.pt", "test_b2_fulltrain")
    rp = test_prediction(
        "resnet18", r.ROOT / "task2_fulltrain_resnet18_seed42/final.pt", "test_resnet18_fulltrain"
    )
    prob = 0.5 * bp + 0.5 * rp
    crop.check(prob, len(frame))
    df = r.pd.DataFrame(prob, columns=r.CLASSES)
    df.insert(0, "ID", template.ID)
    df.to_csv(OUT / "test_probabilities.csv", index=False)
    submission = r.pd.DataFrame(
        {"ID": template.ID, "Emotion": [r.OFFICIAL[r.CLASSES[i]] for i in prob.argmax(1)]}
    )
    path = OUT / "submission_crop_scales.csv"
    submission.to_csv(path, index=False)
    assert r.pd.read_csv(path).equals(submission)
    old = r.pd.read_csv(SOURCE / "submission_multicrop_fusion.csv")
    assert old.ID.tolist() == submission.ID.tolist()
    report.update(
        submission=str(path),
        submission_sha256=r.sha(path),
        test_rows=len(frame),
        test_changed_vs_0734=int((old.Emotion != submission.Emotion).sum()),
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
            raise SystemExit("Another crop-scale experiment is active.")
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
