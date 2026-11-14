"""Time-bounded ZIAN fine-tuning: fixed fold-0 holdout, then full80 refit."""

from pathlib import Path
import csv
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
ROOT = ROOT.parent if ROOT.name == "src" else ROOT  # 位于 src/ 时回到任务根目录
BASE = ROOT / "zian_transfer_20261027"
SOURCE = BASE / "source"
RUNTIME = BASE / "runtime"
shutil.copytree(SOURCE, RUNTIME, dirs_exist_ok=True)
path = RUNTIME / "models/fovea_net.py"
original = path.read_text(encoding="utf-8")
old = "get_max_preds(heatmap_ds_pred.cpu().numpy())"
new = "get_max_preds(heatmap_ds_pred.detach().cpu().numpy())"
assert original.count(old) == 1
patched = original.replace(old, new)
path.write_text(patched, encoding="utf-8")
(BASE / "runtime_patch.json").write_text(
    json.dumps(
        dict(
            file="models/fovea_net.py",
            old=old,
            new=new,
            reason="Allow training with predicted integer-argmax crops; inference unchanged.",
            original_sha256=hashlib.sha256(original.encode()).hexdigest(),
            patched_sha256=hashlib.sha256(patched.encode()).hexdigest(),
        ),
        indent=2,
    )
)
sys.path[:0] = [str(ROOT / ".zian_deps"), str(RUNTIME)]
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from config.common import _C
from models.fovea_net import get_fovea_net
from core.inference import get_final_preds

torch.set_num_threads(2)
cv2.setNumThreads(2)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DATA = ROOT / "2026task3/detection"
OUT = BASE / "finetune"
OUT.mkdir(exist_ok=True)
cfg = _C.clone()
cfg.merge_from_file(str(SOURCE / "experiments/refuge.yaml"))
cfg.MODEL.SELF_ATTEN = True
cfg.MODEL.TRIP_ROI = True
cfg.MODEL.CO_ATTEN = True
cfg.MODEL.HRNET_TYPE = 0
cfg.TEST.DEBUG = False
cfg.DEBUG.DEBUG = False
cfg.freeze()
MEAN = torch.tensor([0.158, 0.224, 0.350], device="cuda").reshape(1, 3, 1, 1)
STD = torch.tensor([0.143, 0.175, 0.255], device="cuda").reshape(1, 3, 1, 1)
YY, XX = torch.meshgrid(
    torch.arange(256, device="cuda"), torch.arange(256, device="cuda"), indexing="ij"
)


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_model():
    torch.manual_seed(42)
    m = get_fovea_net(cfg, is_train=False)
    state = torch.load(
        BASE / "model_best_L1038_TL907_hrnet_SATA.pth", map_location="cpu", weights_only=True
    )
    m.load_state_dict(state, strict=True)
    del state
    return m.cuda()


def sample(row, aug=False):
    image = cv2.imread(str(DATA / "train" / row["filename"]))
    image = cv2.resize(image, (1064, 1064), interpolation=cv2.INTER_LINEAR)[20:1044, 20:1044]
    x = float(row["x"]) * 1064 / int(row["width"]) - 20
    y = float(row["y"]) * 1064 / int(row["height"]) - 20
    if aug and random.random() < 0.5:
        image = image[:, ::-1]
        x = 1023 - x
    tensor = (
        torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0).cuda().float()
        / 255
    )
    if aug:
        tensor = (tensor * random.uniform(0.9, 1.1)).clamp(0, 1)
    return (tensor - MEAN) / STD, torch.tensor([x, y], device="cuda")


def target(xy):
    # Match the author's rounded 5x5 sigma=2 Gaussian target (zero outside +/-2).
    cx, cy = torch.floor(xy + 0.5)
    dx, dy = XX - cx, YY - cy
    return (torch.exp(-(dx * dx + dy * dy) / 8) * (dx.abs() <= 2) * (dy.abs() <= 2))[None, None]


@torch.inference_mode()
def evaluate(model, rows):
    model.eval()
    outputs = []
    for row in rows:
        x, _ = sample(row)
        coarse, fine, offsets, meta = model(x, {})
        _, _, coord, _, _ = get_final_preds(
            cfg, coarse.cpu().numpy(), fine.cpu().numpy(), offsets, meta
        )
        px = (float(coord[0, 0]) + 20) * int(row["width"]) / 1064
        py = (float(coord[0, 1]) + 20) * int(row["height"]) / 1064
        outputs.append(
            dict(
                image_id=int(row["image_id"]),
                pred_x=px,
                pred_y=py,
                x=float(row["x"]),
                y=float(row["y"]),
                squared_error=(px - float(row["x"])) ** 2 + (py - float(row["y"])) ** 2,
            )
        )
        model.debug_iteration = 0
    return dict(
        rmse_coordinate_px=math.sqrt(sum(r["squared_error"] for r in outputs) / (2 * len(outputs))),
        predictions=outputs,
    )


def train(model, rows, epochs, held=None):
    backbone = [p for name, p in model.named_parameters() if name.startswith("hrnet")]
    heads = [p for name, p in model.named_parameters() if not name.startswith("hrnet")]
    optimizer = torch.optim.AdamW(
        [dict(params=backbone, lr=1e-5), dict(params=heads, lr=5e-5)],
        weight_decay=1e-4,
        foreach=False,
    )
    best = evaluate(model, held) if held else None
    if held:
        best["epoch"] = 0
        save("holdout_baseline.json", best)
    history = []
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        # Batch=1: retain pretrained running statistics for stable transfer.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        order = list(rows)
        random.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0
        skipped_roi = 0
        for i, row in enumerate(order):
            image, xy = sample(row, True)
            model.debug_iteration = 0
            coarse, fine, sub, meta = model(image, {})
            tc = target(xy / 4)
            center = meta["roi_center"].to("cuda")[0]
            local = xy - center + 128
            loss = 0.5 * F.mse_loss(coarse, tc)
            if bool(((local >= 0) & (local < 256)).all()):
                tf = target(local)
                loss = (
                    loss + 0.5 * F.mse_loss(fine, tf) + 0.125 * sum(F.mse_loss(s, tf) for s in sub)
                )
            else:
                skipped_roi += 1
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            (loss / 4).backward()
            loss_total += float(loss.detach())
            if (i + 1) % 4 == 0 or i + 1 == len(order):
                norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise ValueError("Nonfinite gradient")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if i == 0:
                print(
                    f"epoch={epoch} first-step loss={float(loss.detach()):.7f} gpuMB={torch.cuda.max_memory_allocated()/2**20:.0f}",
                    flush=True,
                )
            del coarse, fine, sub, meta, image, loss
        item = dict(
            epoch=epoch,
            loss=loss_total / len(order),
            skipped_roi=skipped_roi,
            seconds=time.time() - start,
        )
        if held:
            metric = evaluate(model, held)
            item["rmse_coordinate_px"] = metric["rmse_coordinate_px"]
            if item["rmse_coordinate_px"] < best["rmse_coordinate_px"]:
                best = dict(metric, epoch=epoch)
                torch.save(model.state_dict(), OUT / "holdout_best.pth")
                save("holdout_best.json", best)
        history.append(item)
        save("holdout_history.json" if held else "full_history.json", history)
        print(json.dumps(item), flush=True)
        if held and epoch >= 6 and epoch - best["epoch"] >= 4:
            break
    return best


def main():
    random.seed(42)
    rows = list(csv.DictReader((ROOT / "flair_transfer_local_20261027/split.csv").open()))
    rows = [r for r in rows if r["subset"] == "train"]
    held = [r for r in rows if int(r["fold"]) == 0]
    fit = [r for r in rows if int(r["fold"]) != 0]
    assert len(rows) == 80 and len(held) == 16
    assert not {r["pixel_sha256"] for r in held} & {r["pixel_sha256"] for r in fit}
    save(
        "protocol.json",
        dict(
            started_local=time.strftime("%Y-%m-%d %H:%M:%S"),
            max_holdout_epochs=12,
            holdout_ids=[int(r["image_id"]) for r in held],
            fit_ids=[int(r["image_id"]) for r in fit],
            split="Existing grouped fold0; only one holdout, not full OOF",
            seed=42,
            accumulation=4,
            backbone_lr=1e-5,
            head_lr=5e-5,
            optimizer="AdamW",
            batch_norm="frozen running statistics",
            augmentation="horizontal flip 0.5; brightness gain 0.9-1.1",
            crop_centers="Predicted coarse argmax during both training and inference; no GT crops",
            full_refit="Selected epoch * 64/80 rounded up, minimum1; only if holdout RMSE improves >0.5px",
            public_score_used=False,
        ),
    )
    model = load_model()
    best = train(model, fit, 12, held)
    save("selection.json", best)
    baseline = json.loads((OUT / "holdout_baseline.json").read_text())
    del model
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    if best["epoch"] == 0 or baseline["rmse_coordinate_px"] - best["rmse_coordinate_px"] < 0.5:
        save("status.json", dict(state="completed_no_improvement", best=best["rmse_coordinate_px"]))
        return
    epochs = max(1, math.ceil(best["epoch"] * len(fit) / len(rows)))
    random.seed(42)
    model = load_model()
    train(model, rows, epochs)
    checkpoint = OUT / "full80.pth"
    torch.save(model.state_dict(), checkpoint)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(ROOT / "run_zian_transfer.py"),
            "--runtime-source",
            "--weights",
            str(checkpoint),
            "--output-dir",
            str(OUT / "full80_predictions"),
        ],
        check=True,
        cwd=str(ROOT.parent),
    )
    save(
        "status.json",
        dict(
            state="completed",
            full_epochs=epochs,
            selected_holdout_rmse=best["rmse_coordinate_px"],
            baseline_holdout_rmse=baseline["rmse_coordinate_px"],
        ),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        save("status.json", dict(state="failed", error=repr(exc)))
        raise
