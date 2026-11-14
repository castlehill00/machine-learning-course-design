"""Nine-epoch full-data B2 refit and fixed 50:50 fusion with the original ResNet18."""

import json
import os
import sys
import time
import traceback
import run_pretraining_local as r

OUT = r.ROOT / "task2_fulltrain_b2_seed42"


def train(frame, source, config, folder, stop_after=None):
    folder.mkdir(exist_ok=True)
    cfg_file = folder / "config.json"
    if cfg_file.exists() and json.loads(cfg_file.read_text(encoding="utf-8")) != config:
        raise ValueError("Configuration mismatch; refusing to mix runs.")
    r.atomic_json(cfg_file, config)
    r.seed_all(42)
    model = r.build_model(source, copy_emotion_head=config.get("copy_emotion_head", False)).cuda()
    head_hash = r.state_hash(model.classifier.state_dict())
    optimizer = r.torch.optim.AdamW(
        [
            {
                "params": [
                    p for n, p in model.named_parameters() if not n.startswith("classifier.")
                ],
                "lr": config["backbone_lr"],
            },
            {"params": model.classifier.parameters(), "lr": config["head_lr"]},
        ],
        weight_decay=config["weight_decay"],
    )
    scheduler = r.torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["scheduler_t_max"])
    scaler = r.torch.amp.GradScaler("cuda", init_scale=4096)
    history = []
    if (folder / "last.pt").exists():
        ck = r.torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
        if ck["config"] != config or ck["head_hash"] != head_hash:
            raise ValueError("Checkpoint mismatch.")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        history = ck["history"]
        del ck
        print("Resumed after epoch", len(history), flush=True)
    dl = r.loader(frame, config["image_size"], config["batch_size"], train=True)
    for epoch in range(len(history) + 1, config["epochs"] + 1):
        r.seed_all(42 + epoch)
        dl.generator.manual_seed(42 + epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        start = time.monotonic()
        total, correct, loss_sum, updates = 0, 0, 0.0, 0
        for step, (x, y) in enumerate(dl, 1):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            window_start = (
                ((step - 1) // config["accumulation"])
                * config["accumulation"]
                * config["batch_size"]
            )
            count = min(config["batch_size"] * config["accumulation"], len(frame) - window_start)
            with r.torch.autocast("cuda"):
                logits = model(x)
                loss = r.nn.functional.cross_entropy(logits, y)
            if not r.torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss.")
            scaler.scale(loss * (len(y) / count)).backward()
            if step % config["accumulation"] == 0 or step == len(dl):
                scaler.unscale_(optimizer)
                r.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                updates += int(scaler.get_scale() >= scale)
                optimizer.zero_grad(set_to_none=True)
            total += len(y)
            correct += (logits.argmax(1) == y).sum().item()
            loss_sum += loss.item() * len(y)
            if step == 1 or step % 40 == 0 or step == len(dl):
                print(
                    f'Epoch {epoch}/{config["epochs"]} batch {step}/{len(dl)} elapsed={time.monotonic()-start:.0f}s',
                    flush=True,
                )
                r.atomic_json(
                    folder / "status.json",
                    {
                        "state": "running",
                        "pid": os.getpid(),
                        "epoch": epoch,
                        "batch": step,
                        "batches": len(dl),
                        "completed_epochs": len(history),
                    },
                )
        assert total == len(frame) and updates > 0
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / total,
                "train_accuracy": correct / total,
                "backbone_lr": optimizer.param_groups[0]["lr"],
                "head_lr": optimizer.param_groups[1]["lr"],
                "images_seen": total,
                "optimizer_updates": updates,
                "seconds": time.monotonic() - start,
            }
        )
        scheduler.step()
        r.atomic_save(
            folder / "last.pt",
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "history": history,
                "config": config,
                "head_hash": head_hash,
            },
        )
        r.pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        print(
            f"Epoch {epoch} saved: train_accuracy={correct/total:.4f}; no validation set",
            flush=True,
        )
        if stop_after is not None and epoch >= stop_after:
            return None
    r.atomic_save(
        folder / "final.pt",
        {"model": model.state_dict(), "config": config, "epoch": config["epochs"]},
    )
    return model, history


def export(model, history, config):
    r.atomic_json(
        OUT / "status.json", {"state": "predicting", "pid": os.getpid(), "completed_epochs": 9}
    )
    template = r.pd.read_csv(r.ROOT / "2026task2/submission.csv", dtype={"ID": str})
    assert list(template.columns) == ["ID", "Emotion"] and len(template) == 7065
    assert template.ID.is_unique and template.ID.notna().all()
    assert set(template.ID) == {p.name for p in (r.DATA / "test").glob("*.jpg")}
    test = r.pd.DataFrame({"path": ["test/" + s for s in template.ID], "class": "Angry"})
    print("Predicting all test images with flip TTA...", flush=True)
    _, prob = r.predict(model, r.loader(test, 260, 64), flip=True)
    old_path = r.ROOT / "task2_ensemble_b2_resnet18/resnet18_test_tta.csv"
    old = r.pd.read_csv(old_path)
    assert old.ID.is_unique and set(old.ID) == set(template.ID)
    rn = old.set_index("ID").loc[template.ID, r.CLASSES].to_numpy()
    submissions = {}
    for name, values in [
        ("b2_fulltrain", prob),
        ("b2_fulltrain_resnet18_equal", 0.5 * prob + 0.5 * rn),
    ]:
        assert values.shape == (7065, 6) and r.np.isfinite(values).all()
        assert (values >= 0).all() and r.np.allclose(values.sum(1), 1, atol=1e-5)
        probability = r.pd.DataFrame(values, columns=r.CLASSES)
        probability.insert(0, "ID", template.ID)
        probability.to_csv(OUT / (name + "_probabilities.csv"), index=False)
        labels = [r.OFFICIAL[r.CLASSES[i]] for i in values.argmax(1)]
        submission = r.pd.DataFrame({"ID": template.ID, "Emotion": labels})
        path = OUT / ("submission_" + name + ".csv")
        tmp = path.with_suffix(".csv.tmp")
        submission.to_csv(tmp, index=False)
        tmp.replace(path)
        assert r.pd.read_csv(path).equals(submission)
        submissions[name] = {"path": str(path), "sha256": r.sha(path)}
    r.atomic_json(
        OUT / "summary.json",
        {
            "config": config,
            "epochs": len(history),
            "training_minutes": sum(h["seconds"] for h in history) / 60,
            "final_checkpoint_sha256": r.sha(OUT / "final.pt"),
            "resnet18_probabilities_sha256": r.sha(old_path),
            "test_rows": 7065,
            "official_mapping": r.OFFICIAL,
            "fusion_b2_weight": 0.5,
            "note": "Fixed fusion weight inherited from prior holdout experiment, not retuned. No full-data validation score.",
            "submissions": submissions,
            "leaderboard_scores": None,
        },
    )
    r.atomic_json(
        OUT / "status.json", {"state": "completed", "pid": os.getpid(), "submissions": submissions}
    )
    print("TRAINING AND EXPORT COMPLETE", json.dumps(submissions), flush=True)


def main():
    if not r.torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    r.torch.set_num_threads(4)
    r.torch.backends.cudnn.benchmark = False
    r.torch.backends.cudnn.deterministic = True
    original = r.OUT
    old_config = json.loads((original / "affectnet/config.json").read_text(encoding="utf-8"))
    r.OUT = OUT
    a, b, split_hash = r.load_split()
    frame = r.pd.concat([a, b]).sort_values("path").reset_index(drop=True)
    assert len(frame) == 28275 and frame.path.is_unique
    states, sources = r.prepare_weights()
    config = dict(
        old_config,
        pipeline="b2_fulltrain_v1",
        experiment="affectnet_fulltrain",
        epochs=9,
        scheduler_t_max=25,
        scheduler="cosine T_max=25, first 9 epochs",
        selection="fixed final epoch 9 from prior holdout experiment; no validation",
        training_samples=28275,
        split_sha256=split_hash,
        sources=sources,
        initialization="original AffectNet pretrained backbone plus same random six-class head; not epoch9 fine-tuned model",
    )
    # The full training frame contains both old partitions; fold is retained only for provenance.
    frame.to_csv(OUT / "train_manifest.csv", index=False)
    model, history = train(frame, states["affectnet"], config, OUT)
    export(model, history, config)


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
            raise SystemExit("Another full B2 process is running.")
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
