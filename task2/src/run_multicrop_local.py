"""Fixed three-way crop-TTA ablation on holdout models, then full-model export."""

import io
import json
import os
import sys
import time
import traceback
import zipfile
import run_pretraining_local as r

OUT = r.ROOT / "task2_multicrop_tta_seed42"
MODES = ["flip2", "center4", "crop12"]
MIN_GAIN = 10  # Predeclared minimum extra correct validation predictions for export.


def check(values, n):
    assert values.shape == (n, 6) and r.np.isfinite(values).all()
    assert (values >= 0).all() and r.np.allclose(values.sum(-1), 1, atol=1e-5)


class Views(r.Dataset):
    def __init__(self, frame, size, crop_size=44):
        self.frame = frame.reset_index(drop=True)
        self.transform = r.eval_transform(size)
        if not 1 <= crop_size <= 48:
            raise ValueError("Crop must fit the original image.")
        self.crop = r.transforms.FiveCrop(crop_size)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        with r.Image.open(r.DATA / self.frame.iloc[index]["path"]) as im:
            im = im.convert("RGB")
            if im.size != (48, 48):
                raise ValueError("This experiment requires original 48x48 images.")
            views = (im, *self.crop(im))  # full, TL, TR, BL, BR, center
            return r.torch.stack([self.transform(v) for v in views])


def get_model(kind, checkpoint):
    if kind == "b2":
        model = r.skeleton()
    elif kind in ("resnet18", "resnet34"):
        model = getattr(r.torchvision.models, kind)(weights=None)
        model.fc = r.nn.Sequential(r.nn.Dropout(0), r.nn.Linear(512, 6))
    else:
        raise ValueError("Unsupported architecture.")
    ck = r.torch.load(
        io.BytesIO(checkpoint) if isinstance(checkpoint, bytes) else checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    assert ck["config"]["classes"] == r.CLASSES
    model.load_state_dict(ck["model"], strict=True)
    return model.cuda().eval()


@r.torch.inference_mode()
def infer(kind, checkpoint, frame, tag, crop_size=44):
    signature = {
        "checkpoint_sha256": (
            r.hashlib.sha256(checkpoint).hexdigest()
            if isinstance(checkpoint, bytes)
            else r.sha(checkpoint)
        ),
        "paths_sha256": r.hashlib.sha256("\n".join(frame.path).encode()).hexdigest(),
        "crop": crop_size,
        "size": 260 if kind == "b2" else 224,
        "views": ["full", "top_left", "top_right", "bottom_left", "bottom_right", "center"],
        "flip_each_view": True,
        "classes": r.CLASSES,
    }
    cache = OUT / (tag + ".npz")
    meta = OUT / (tag + ".json")
    if cache.exists() and meta.exists() and json.loads(meta.read_text()) == signature:
        saved = r.np.load(cache)["probabilities"]
        assert saved.shape == (len(frame), 6, 6)
        for j in range(6):
            check(saved[:, j, :], len(frame))
        print("Reusing completed predictions", tag, flush=True)
        return saved
    model = get_model(kind, checkpoint)
    batch = 64
    loader = r.DataLoader(
        Views(frame, signature["size"], crop_size),
        batch_size=batch,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    output = []
    start = time.monotonic()
    for step, x in enumerate(loader, 1):
        probs = []
        for j in range(6):
            view = x[:, j].cuda(non_blocking=True)
            with r.torch.autocast("cuda"):
                logits = model(view)
                flipped = model(view.flip(-1))
            probs.append(
                ((logits.float().softmax(1) + flipped.float().softmax(1)) / 2).cpu().numpy()
            )
        output.append(r.np.stack(probs, axis=1))
        if step == 1 or step % 10 == 0 or step == len(loader):
            elapsed = time.monotonic() - start
            print(f"{tag}: batch {step}/{len(loader)} elapsed={elapsed:.0f}s", flush=True)
            r.atomic_json(
                OUT / "status.json",
                {
                    "state": "running",
                    "pid": os.getpid(),
                    "stage": tag,
                    "batch": step,
                    "batches": len(loader),
                    "seconds": elapsed,
                },
            )
    result = r.np.concatenate(output)
    for j in range(6):
        check(result[:, j, :], len(frame))
    with cache.with_suffix(".npz.tmp").open("wb") as f:
        r.np.savez_compressed(f, probabilities=result)
    cache.with_suffix(".npz.tmp").replace(cache)
    r.atomic_json(meta, signature)
    del model, loader
    r.torch.cuda.empty_cache()
    return result


def combine(views, mode):
    if mode == "flip2":
        return views[:, 0, :].astype("float64")
    if mode == "center4":
        return views[:, [0, 5], :].astype("float64").mean(1)
    if mode == "crop12":
        return views.astype("float64").mean(1)
    raise ValueError(mode)


def paired_summary(before, after, truth):
    a = before.argmax(1) == truth
    b = after.argmax(1) == truth
    wins, losses = int((b & ~a).sum()), int((a & ~b).sum())
    # Pixel-identical images are resampled together for descriptive uncertainty.
    split = r.pd.read_csv(OUT / "split.csv")
    groups = split[split.fold == 0].pixel_sha256.to_numpy()
    _, codes = r.np.unique(groups, return_inverse=True)
    sizes = r.np.bincount(codes)
    delta = r.np.bincount(codes, weights=b.astype(int) - a.astype(int))
    rng = r.np.random.default_rng(42)
    boot = []
    for _ in range(2000):
        sampled = rng.integers(0, len(sizes), len(sizes))
        boot.append(100 * delta[sampled].sum() / sizes[sampled].sum())
    return {
        "corrected": wins,
        "new_errors": losses,
        "net_correct": wins - losses,
        "change_pp": 100 * (wins - losses) / len(truth),
        "descriptive_group_bootstrap_95_interval_pp": r.np.quantile(boot, [0.025, 0.975]).tolist(),
        "interval_note": "Conditional on these fixed models; not adjusted for development-set selection.",
    }


def main():
    assert r.torch.cuda.is_available()
    r.torch.set_num_threads(4)
    r.torch.backends.cudnn.benchmark = False
    r.torch.backends.cudnn.deterministic = True
    r.OUT = OUT
    _, val, split_sha = r.load_split()
    val = val.reset_index(drop=True)
    truth = r.np.array([r.CLASSES.index(s) for s in val["class"]])
    protocol = {
        "modes": MODES,
        "crop_pixels": 44,
        "original_pixels": 48,
        "b2_weight": 0.5,
        "selection": "highest validation fusion accuracy, then lower NLL, then fewer views",
        "minimum_net_correct_gain_to_export": MIN_GAIN,
        "split_sha256": split_sha,
        "note": "Predeclared crop search. Reuses development split; no full-data model in validation.",
    }
    if (OUT / "protocol.json").exists():
        assert json.loads((OUT / "protocol.json").read_text()) == protocol
    r.atomic_json(OUT / "protocol.json", protocol)
    with zipfile.ZipFile(r.ROOT / "task2_ablation_backup_round2.zip") as z:
        rn_checkpoint = z.read("task2_ablation_seed42/baseline/best.pt")
    b_checkpoint = r.ROOT / "task2_pretraining_b2_seed42/affectnet/best.pt"
    old_analysis = json.loads((r.ROOT / "task2_ensemble_b2_resnet18/analysis.json").read_text())
    assert r.hashlib.sha256(rn_checkpoint).hexdigest() == old_analysis["resnet18_checkpoint_sha256"]
    assert r.sha(b_checkpoint) == old_analysis["b2_checkpoint_sha256"]
    b = infer("b2", b_checkpoint, val, "validation_b2")
    rn = infer("resnet18", rn_checkpoint, val, "validation_resnet18")
    # Reproduce saved flip-only predictions before accepting the new experiment.
    for pred, old_path in [
        (
            b[:, 0, :],
            r.ROOT / "task2_pretraining_b2_seed42/affectnet/validation_tta_probabilities.csv",
        ),
        (rn[:, 0, :], r.ROOT / "task2_ensemble_b2_resnet18/resnet18_validation_tta.csv"),
    ]:
        old = r.pd.read_csv(old_path).set_index("path").loc[val.path, r.CLASSES].to_numpy()
        if abs(pred - old).max() > 0.02 or (pred.argmax(1) != old.argmax(1)).sum() > 10:
            raise RuntimeError("Baseline reproduction mismatch.")
    results = []
    fusion = {}
    for mode in MODES:
        bp, rp = combine(b, mode), combine(rn, mode)
        fusion[mode] = 0.5 * bp + 0.5 * rp
        bm, _ = r.metrics(bp, truth)
        rm, _ = r.metrics(rp, truth)
        fm, _ = r.metrics(fusion[mode], truth)
        results.append(
            {
                "mode": mode,
                "b2_accuracy": bm["accuracy"],
                "resnet18_accuracy": rm["accuracy"],
                "fusion_accuracy": fm["accuracy"],
                "fusion_nll": fm["nll"],
                "fusion_macro_f1": fm["macro_f1"],
                "correct": int((fusion[mode].argmax(1) == truth).sum()),
            }
        )
    r.pd.DataFrame(results).to_csv(OUT / "validation_comparison.csv", index=False)
    selected = sorted(
        results, key=lambda m: (-m["fusion_accuracy"], m["fusion_nll"], MODES.index(m["mode"]))
    )[0]
    paired = paired_summary(fusion["flip2"], fusion[selected["mode"]], truth)
    accepted = selected["mode"] != "flip2" and paired["net_correct"] >= MIN_GAIN
    report = {
        "protocol": protocol,
        "results": results,
        "selected": selected,
        "paired_comparison": paired,
        "accepted_for_fulltrain_export": accepted,
        "leaderboard_score": None,
    }
    r.atomic_json(OUT / "analysis.json", report)
    print("VALIDATION RESULT", json.dumps(report), flush=True)
    if not accepted:
        r.atomic_json(
            OUT / "status.json",
            {
                "state": "completed_no_export",
                "pid": os.getpid(),
                "reason": "No candidate passed the predefined gain threshold.",
            },
        )
        return
    template = r.pd.read_csv(r.ROOT / "2026task2/submission.csv", dtype={"ID": str})
    assert (
        template.columns.tolist() == ["ID", "Emotion"]
        and len(template) == 7065
        and template.ID.is_unique
    )
    assert set(template.ID) == {p.name for p in (r.DATA / "test").glob("*.jpg")}
    test = r.pd.DataFrame({"path": ["test/" + s for s in template.ID]})
    bfull = r.ROOT / "task2_fulltrain_b2_seed42/final.pt"
    rfull = r.ROOT / "task2_fulltrain_resnet18_seed42/final.pt"
    bprob = combine(infer("b2", bfull, test, "test_b2_fulltrain"), selected["mode"])
    rnprob = combine(infer("resnet18", rfull, test, "test_resnet18_fulltrain"), selected["mode"])
    prob = 0.5 * bprob + 0.5 * rnprob
    check(prob, len(test))
    df = r.pd.DataFrame(prob, columns=r.CLASSES)
    df.insert(0, "ID", template.ID)
    df.to_csv(OUT / "fusion_test_probabilities.csv", index=False)
    submission = r.pd.DataFrame(
        {"ID": template.ID, "Emotion": [r.OFFICIAL[r.CLASSES[i]] for i in prob.argmax(1)]}
    )
    path = OUT / "submission_multicrop_fusion.csv"
    submission.to_csv(path, index=False)
    assert r.pd.read_csv(path).equals(submission)
    old = r.pd.read_csv(
        r.ROOT / "task2_fulltrain_resnet18_seed42/submission_b2_resnet18_both_fulltrain_equal.csv"
    )
    assert old.ID.tolist() == submission.ID.tolist()
    report.update(
        submission=str(path),
        submission_sha256=r.sha(path),
        test_rows=len(test),
        official_mapping=r.OFFICIAL,
        test_changed_vs_0726=int((old.Emotion != submission.Emotion).sum()),
        full_checkpoints={"b2": r.sha(bfull), "resnet18": r.sha(rfull)},
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
            raise SystemExit("Another multicrop process is running.")
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
