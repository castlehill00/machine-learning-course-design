"""Fixed-split EfficientNet-B2 pretraining comparison; restart to resume.

Run from any directory: python run_pretraining_local.py
No test-set prediction or full-data training is performed during model selection.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parent
ROOT = ROOT.parent if ROOT.name == "src" else ROOT  # 位于 src/ 时回到任务根目录
sys.path.insert(0, str(ROOT / ".model_deps"))
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
import timm

DATA = ROOT / "2026task2/fer_data/fer_data"
OUT = ROOT / "task2_pretraining_b2_seed42"
WEIGHTS = ROOT / ".torch/hub/checkpoints"
CLASSES = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
SOURCE_ROWS = [0, 2, 3, 4, 5, 6]
OFFICIAL = {"Angry": 0, "Fear": 1, "Happy": 2, "Sad": 3, "Surprise": 4, "Neutral": 5}
EMOTION_SHA = "e687fd9e7bee45486c7325f11222740f841c3bebac531badb903a50e82b9ac0f"
ARCH = "tf_efficientnet_b2.ns_jft_in1k"
EXPERIMENTS = ["affectnet", "generic_ns"]


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def atomic_save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def state_hash(state):
    return hashlib.sha256(
        b"".join(v.detach().cpu().numpy().tobytes() for v in state.values())
    ).hexdigest()


def status(state, **fields):
    atomic_json(
        OUT / "status.json",
        dict(state=state, pid=os.getpid(), updated_at=time.strftime("%Y-%m-%d %H:%M:%S"), **fields),
    )


def skeleton(classes=6):
    # The author's serialized EmotiEffNet uses MAX pooling, unlike timm's default.
    return timm.create_model(
        ARCH,
        pretrained=False,
        num_classes=classes,
        global_pool="max",
        drop_rate=0.0,
        drop_path_rate=0.0,
    )


def prepare_weights():
    original = WEIGHTS / "enet_b2_7.pt"
    if sha(original) != EMOTION_SHA:
        raise ValueError("Author checkpoint SHA256 differs from the inspected download.")
    converted = WEIGHTS / "enet_b2_7_state.pt"
    if not converted.exists():
        # Only deserialize the inspected author checkpoint; subsequent loads use weights_only.
        author = torch.load(original, map_location="cpu", weights_only=False).eval()
        reconstructed = skeleton(7).eval()
        reconstructed.load_state_dict(author.state_dict(), strict=True)
        with Image.open(next((DATA / "train/Happy").glob("*.jpg"))) as im:
            x = eval_transform(260)(im.convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            expected, actual = author(x), reconstructed(x)
        if not torch.allclose(expected, actual, atol=1e-6, rtol=1e-5):
            raise RuntimeError("Reconstructed model does not match author inference.")
        atomic_save(
            converted,
            {
                "model": author.state_dict(),
                "source_sha256": EMOTION_SHA,
                "max_reconstruction_error": (expected - actual).abs().max().item(),
            },
        )
        print("Author reconstruction verified:", (expected - actual).abs().max().item(), flush=True)
    generic = WEIGHTS / "tf_efficientnet_b2_ns-00306e48.pth"
    if not sha(generic).startswith("00306e48"):
        raise ValueError("Generic checkpoint hash mismatch.")
    source = torch.load(converted, map_location="cpu", weights_only=True)
    if source["source_sha256"] != EMOTION_SHA:
        raise ValueError("Converted source mismatch.")
    return {
        "affectnet": source["model"],
        "generic_ns": torch.load(generic, map_location="cpu", weights_only=True),
    }, {
        "affectnet": {
            "url": "https://raw.githubusercontent.com/sb-ai-lab/EmotiEffLib/main/models/affectnet_emotions/enet_b2_7.pt",
            "sha256": EMOTION_SHA,
            "pretraining": "VGGFace2 and AffectNet; author metadata names NS backbone",
        },
        "generic_ns": {
            "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2_ns-00306e48.pth",
            "sha256": sha(generic),
            "pretraining": "ImageNet-1k plus unlabeled JFT-300M, Noisy Student",
        },
    }


def build_model(source, copy_emotion_head=False):
    model = skeleton()
    backbone = {k: v for k, v in source.items() if not k.startswith("classifier.")}
    missing, unexpected = model.load_state_dict(backbone, strict=False)
    if set(missing) != {"classifier.weight", "classifier.bias"} or unexpected:
        raise ValueError(f"Backbone mismatch: {missing}, {unexpected}")
    with torch.random.fork_rng():
        torch.manual_seed(42)
        head = nn.Linear(model.num_features, 6)
    if copy_emotion_head:
        head.load_state_dict(
            {
                "weight": source["classifier.weight"][SOURCE_ROWS],
                "bias": source["classifier.bias"][SOURCE_ROWS],
            }
        )
    model.classifier = head
    return model


def eval_transform(size):
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def train_transform(size):
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(
                10,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            eval_transform(size),
        ]
    )


class Faces(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.labels = np.array([CLASSES.index(c) for c in self.frame["class"]])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        with Image.open(DATA / self.frame.iloc[index]["path"]) as im:
            image = im.convert("RGB")
        return self.transform(image), int(self.labels[index])


def loader(frame, size, batch, train=False):
    return DataLoader(
        Faces(frame, train_transform(size) if train else eval_transform(size)),
        batch_size=batch,
        shuffle=train,
        num_workers=0,
        pin_memory=True,
        generator=torch.Generator().manual_seed(42),
        drop_last=False,
    )


def load_split():
    with zipfile.ZipFile(ROOT / "task2_ablation_backup_round2.zip") as z:
        raw = z.read("task2_ablation_seed42/split.csv")
    frame = pd.read_csv(io.BytesIO(raw))
    if len(frame) != 28275 or not frame.path.is_unique or frame.isna().any().any():
        raise ValueError("Invalid saved split.")
    if set(frame["class"]) != set(CLASSES) or not frame.fold.isin(range(5)).all():
        raise ValueError("Invalid split classes/folds.")
    if frame.groupby("pixel_sha256").fold.nunique().max() != 1:
        raise ValueError("Duplicate image group crosses folds.")
    for row in frame.itertuples(index=False):
        path = DATA / row.path
        if path.parent.name != row[1]:
            raise ValueError("Class/path mismatch.")
        with Image.open(path) as im:
            gray = im.convert("L")
            fingerprint = hashlib.sha256(str(gray.size).encode() + gray.tobytes()).hexdigest()
        if fingerprint != row.pixel_sha256:
            raise ValueError(f"Image differs from original split: {row.path}")
    if set(frame.path) != {
        p.relative_to(DATA).as_posix() for p in (DATA / "train").glob("*/*.jpg")
    }:
        raise ValueError("Saved split does not cover the current training data.")
    destination = OUT / "split.csv"
    if destination.exists() and destination.read_bytes() != raw:
        raise ValueError("Output split changed.")
    destination.write_bytes(raw)
    train, val = frame[frame.fold != 0], frame[frame.fold == 0]
    assert len(train) == 22619 and len(val) == 5656
    print(
        f"Original split and all pixel hashes verified: train={len(train)}, val={len(val)}",
        flush=True,
    )
    return train, val, hashlib.sha256(raw).hexdigest()


@torch.inference_mode()
def predict(model, dl, flip=False):
    model.eval()
    base, augmented = [], []
    for x, _ in dl:
        x = x.cuda(non_blocking=True)
        with torch.autocast("cuda"):
            logits = model(x)
            other = model(x.flip(-1)) if flip else None
        prob = logits.float().softmax(1)
        base.append(prob.cpu().numpy())
        augmented.append(((prob + other.float().softmax(1)) / 2 if flip else prob).cpu().numpy())
    base, augmented = np.concatenate(base), np.concatenate(augmented)
    if not np.isfinite(base).all() or not np.isfinite(augmented).all():
        raise RuntimeError("Nonfinite predictions.")
    return base, augmented


def metrics(prob, truth):
    cm = np.zeros((6, 6), dtype=np.int64)
    np.add.at(cm, (truth, prob.argmax(1)), 1)
    tp = cm.diagonal()
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "nll": float(-np.log(prob[np.arange(len(truth)), truth].clip(1e-12, 1)).mean()),
        "macro_f1": float((2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)).mean()),
        "recall": dict(zip(CLASSES, (tp / np.maximum(cm.sum(1), 1)).tolist())),
    }, cm


def save_predictions(folder, frame, base, tta):
    for name, prob in [
        ("validation_probabilities.csv", base),
        ("validation_tta_probabilities.csv", tta),
    ]:
        df = pd.DataFrame(prob, columns=CLASSES)
        df.insert(0, "true_class", frame["class"].tolist())
        df.insert(0, "path", frame.path.tolist())
        df.to_csv(folder / name, index=False)


def zero_shot(source, val, config):
    folder = OUT / "affectnet_direct_transfer"
    folder.mkdir(exist_ok=True)
    cfg = {
        "source": config["sources"]["affectnet"],
        "split_sha256": config["split_sha256"],
        "image_size": 260,
        "source_rows": SOURCE_ROWS,
        "classes": CLASSES,
        "note": "Untrained diagnostic: remove Disgust logits and renormalize six classes.",
    }
    if (folder / "config.json").exists() and json.loads(
        (folder / "config.json").read_text()
    ) != cfg:
        raise ValueError("Direct-transfer config mismatch.")
    if (folder / "metrics.json").exists():
        return
    atomic_json(folder / "config.json", cfg)
    status("running", experiment="affectnet_direct_transfer")
    model = build_model(source, copy_emotion_head=True).cuda()
    dl = loader(val, 260, config["batch_size"])
    base, tta = predict(model, dl, flip=True)
    m, cm = metrics(base, dl.dataset.labels)
    mt, _ = metrics(tta, dl.dataset.labels)
    save_predictions(folder, val, base, tta)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(folder / "confusion_matrix.csv")
    atomic_json(folder / "metrics.json", {"raw": m, "tta": mt, "training_epochs": 0})
    print(
        f'Direct transfer (no fine-tuning): raw={m["accuracy"]:.4f}, TTA={mt["accuracy"]:.4f}',
        flush=True,
    )
    del model, dl
    torch.cuda.empty_cache()


def run_experiment(name, source, train, val, common, stop_after=None):
    folder = OUT / name
    folder.mkdir(exist_ok=True)
    config = dict(common, experiment=name)
    config_file = folder / "config.json"
    if config_file.exists() and json.loads(config_file.read_text()) != config:
        raise ValueError(f"{name}: configuration changed; choose another output directory.")
    atomic_json(config_file, config)
    if (folder / "metrics.json").exists() and (folder / "best.pt").exists():
        print(name, "already complete; reusing", flush=True)
        return json.loads((folder / "metrics.json").read_text())
    seed_all(42)
    model = build_model(source, copy_emotion_head=common.get("copy_emotion_head", False)).cuda()
    head_hash = state_hash(model.classifier.state_dict())
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p for n, p in model.named_parameters() if not n.startswith("classifier.")
                ],
                "lr": common["backbone_lr"],
            },
            {"params": model.classifier.parameters(), "lr": common["head_lr"]},
        ],
        weight_decay=common["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, common.get("scheduler_t_max", common["epochs"])
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=4096)
    history, best_acc, best_nll, best_epoch, elapsed = [], -1.0, float("inf"), 0, 0.0
    last_path = folder / "last.pt"
    if last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=True)
        if ck["config"] != config or ck["head_hash"] != head_hash:
            raise ValueError("Resume checkpoint mismatch.")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        history, best_acc, best_nll, best_epoch, elapsed = (
            ck[k] for k in ["history", "best_acc", "best_nll", "best_epoch", "seconds"]
        )
        del ck
        print(f"{name}: resumed after epoch {len(history)}", flush=True)
    dl = loader(train, common["image_size"], common["batch_size"], train=True)
    vd = loader(val, common["image_size"], common["batch_size"])
    truth = vd.dataset.labels
    for epoch in range(len(history) + 1, common["epochs"] + 1):
        start = time.monotonic()
        seed_all(42 + epoch)
        dl.generator.manual_seed(42 + epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, correct, loss_sum, updates = 0, 0, 0.0, 0
        for step, (x, y) in enumerate(dl, 1):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            # Exact sample weighting for the final, possibly incomplete accumulation window.
            window_start = (
                ((step - 1) // common["accumulation"])
                * common["accumulation"]
                * common["batch_size"]
            )
            window_samples = min(
                common["batch_size"] * common["accumulation"], len(train) - window_start
            )
            with torch.autocast("cuda"):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss.")
            scaler.scale(loss * (len(y) / window_samples)).backward()
            if step % common["accumulation"] == 0 or step == len(dl):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                updates += int(scaler.get_scale() >= scale)
                optimizer.zero_grad(set_to_none=True)
            loss_sum += loss.item() * len(y)
            total += len(y)
            correct += (logits.argmax(1) == y).sum().item()
            if step == 1 or step % 40 == 0 or step == len(dl):
                print(
                    f'{name} epoch {epoch:02d}/{common["epochs"]} batch {step}/{len(dl)} '
                    f"elapsed={time.monotonic()-start:.0f}s",
                    flush=True,
                )
                status(
                    "running",
                    experiment=name,
                    epoch=epoch,
                    epochs=common["epochs"],
                    batch=step,
                    batches=len(dl),
                    completed_epochs=len(history),
                    best_accuracy=best_acc,
                )
        if total != len(train) or updates == 0:
            raise RuntimeError("Incomplete epoch or no optimizer updates.")
        base, _ = predict(model, vd)
        vm, _ = metrics(base, truth)
        row = dict(
            epoch=epoch,
            train_loss=loss_sum / total,
            train_accuracy=correct / total,
            val_accuracy=vm["accuracy"],
            val_nll=vm["nll"],
            val_macro_f1=vm["macro_f1"],
            backbone_lr=optimizer.param_groups[0]["lr"],
            head_lr=optimizer.param_groups[1]["lr"],
            images_seen=total,
            optimizer_updates=updates,
            seconds=time.monotonic() - start,
        )
        history.append(row)
        elapsed += row["seconds"]
        if vm["accuracy"] > best_acc or (vm["accuracy"] == best_acc and vm["nll"] < best_nll):
            best_acc, best_nll, best_epoch = vm["accuracy"], vm["nll"], epoch
            atomic_save(
                folder / "best.pt",
                {
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "accuracy": best_acc,
                    "nll": best_nll,
                },
            )
        scheduler.step()
        atomic_save(
            last_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config,
                "head_hash": head_hash,
                "history": history,
                "best_acc": best_acc,
                "best_nll": best_nll,
                "best_epoch": best_epoch,
                "seconds": elapsed,
            },
        )
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        print(
            f'{name} epoch {epoch:02d}: train={correct/total:.4f}, val={vm["accuracy"]:.4f}, '
            f'NLL={vm["nll"]:.4f}, best={best_acc:.4f} @ {best_epoch}',
            flush=True,
        )
        if stop_after is not None and epoch >= stop_after:
            return None
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    base, tta = predict(model, vd, flip=True)
    raw, cm = metrics(base, truth)
    flip_metrics, _ = metrics(tta, truth)
    clean_loader = loader(train, common["image_size"], common["batch_size"])
    train_prob, _ = predict(model, clean_loader)
    clean_accuracy = metrics(train_prob, clean_loader.dataset.labels)[0]["accuracy"]
    result = dict(
        experiment=name,
        best_epoch=checkpoint["epoch"],
        val_accuracy=raw["accuracy"],
        tta_accuracy=flip_metrics["accuracy"],
        macro_f1=raw["macro_f1"],
        val_nll=raw["nll"],
        train_eval_accuracy=clean_accuracy,
        gap_pp=100 * (clean_accuracy - raw["accuracy"]),
        training_minutes=elapsed / 60,
        initial_head_hash=head_hash,
        parameters=sum(p.numel() for p in model.parameters()),
        **{f"recall_{c}": v for c, v in raw["recall"].items()},
    )
    save_predictions(folder, val, base, tta)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(folder / "confusion_matrix.csv")
    atomic_json(folder / "metrics.json", result)
    print("RESULT", json.dumps(result), flush=True)
    del model, optimizer, scheduler, scaler, checkpoint, dl, vd, clean_loader
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing accidental CPU training.")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    status("starting")
    states, sources = prepare_weights()
    train, val, split_sha = load_split()
    common = {
        "pipeline": "b2_pretraining_v1",
        "architecture": ARCH,
        "pooling": "max",
        "sources": sources,
        "split_sha256": split_sha,
        "seed": 42,
        "classes": CLASSES,
        "epochs": 25,
        "image_size": 260,
        "batch_size": 64,
        "accumulation": 2,
        "effective_batch_size": 128,
        "backbone_lr": 1e-4,
        "head_lr": 3e-4,
        "weight_decay": 1e-4,
        "dropout": 0.0,
        "label_smoothing": 0.0,
        "workers": 0,
        "augmentation": "horizontal flip; affine 10deg, translate .05, scale .95-1.05; brightness/contrast .15",
        "normalization": "ImageNet",
        "resize": "bilinear 260x260, no crop",
        "selection": "highest raw validation accuracy, then lowest NLL",
        "head_initialization": "identical random nn.Linear(1408,6), seed42, no transferred classifier",
        "scheduler": "cosine T_max=25",
        "amp_initial_scale": 4096,
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "timm": timm.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    cfg_path = OUT / "config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text(encoding="utf-8")) != common:
        raise ValueError("Run configuration changed.")
    atomic_json(cfg_path, common)
    heads = [state_hash(build_model(states[name]).classifier.state_dict()) for name in EXPERIMENTS]
    if len(set(heads)) != 1:
        raise RuntimeError("Paired classifier initialization differs.")
    print("Matched architecture, classifier initialization, split and budget verified.", flush=True)
    if args.check_only:
        status("checked")
        return
    zero_shot(states["affectnet"], val, common)
    results = []
    for name in EXPERIMENTS:
        results.append(run_experiment(name, states[name], train, val, common))
        pd.DataFrame(results).to_csv(OUT / "comparison.csv", index=False)
    winner = sorted(results, key=lambda r: (-r["val_accuracy"], r["val_nll"]))[0]
    atomic_json(
        OUT / "selection.json",
        {
            "selected": winner,
            "historical_resnet34_val_accuracy": 0.6779,
            "historical_resnet34_tta_accuracy": 0.6908,
            "next_step": "Review held-out results before full-data retraining. No test prediction performed.",
        },
    )
    status("completed", results=results, selected=winner["experiment"])
    print(
        "ALL EXPERIMENTS COMPLETE. Held-out comparison saved:", OUT / "comparison.csv", flush=True
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    OUT.mkdir(exist_ok=True)
    import msvcrt

    lock = open(OUT / "run.lock", "a+b")
    lock.seek(0)
    if not lock.read(1):
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit("An experiment process is already running.")
    try:
        main()
    except BaseException as exc:
        status("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        traceback.print_exc()
        raise
    finally:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()
