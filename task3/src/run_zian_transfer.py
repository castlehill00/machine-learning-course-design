"""Direct REFUGE ZIAN transfer; author inference, no Task3 fitting or GT crops."""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import sys
import time

ROOT = Path(__file__).resolve().parent
ROOT = ROOT.parent if ROOT.name == "src" else ROOT  # 位于 src/ 时回到任务根目录
BASE = ROOT / "zian_transfer_20261027"
OUT = BASE
sys.path[:0] = [
    str(ROOT / ".zian_deps"),
    str(BASE / ("runtime" if "--runtime-source" in sys.argv else "source")),
]
import cv2
import numpy as np
import torch
from config.common import _C
from models.fovea_net import get_fovea_net
from core.inference import get_final_preds


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--runtime-source", action="store_true")
    args = parser.parse_args()
    if args.output_dir:
        OUT = args.output_dir.resolve()
        OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    cfg = _C.clone()
    cfg.merge_from_file(str(BASE / "source/experiments/refuge.yaml"))
    cfg.MODEL.SELF_ATTEN = True
    cfg.MODEL.TRIP_ROI = True
    cfg.MODEL.CO_ATTEN = True
    cfg.MODEL.HRNET_TYPE = 0
    cfg.DEBUG.DEBUG = False
    cfg.TEST.DEBUG = False
    cfg.freeze()
    print("Building author ZIAN, REFUGE, SELF/TRIP/CO attention enabled.", flush=True)
    model = get_fovea_net(cfg, is_train=False)
    weight = args.weights or BASE / "model_best_L1038_TL907_hrnet_SATA.pth"
    state = torch.load(weight, map_location="cpu", weights_only=True)
    print("Checkpoint type:", type(state).__name__, "first keys:", list(state)[:5], flush=True)
    if "state_dict" in state:
        state = state["state_dict"]
    if all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    del state
    model.eval().cuda()
    iw, ih = cfg.MODEL.IMAGE_SIZE
    cw, ch = cfg.MODEL.CROP_SIZE
    left, top = (iw - cw) // 2, (ih - ch) // 2
    mean = torch.tensor([0.158, 0.224, 0.350]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.143, 0.175, 0.255]).view(1, 3, 1, 1).cuda()
    data = ROOT / "2026task3/detection"
    labels = {
        int(r["data"]): (float(r["Fovea_X"]), float(r["Fovea_Y"]))
        for r in csv.DictReader((data / "fovea_localization_train_GT.csv").open())
    }
    protocol = dict(
        model=(
            "Task3 fine-tuned ZIAN"
            if args.weights
            else "Author REFUGE ZIAN direct inference; no Task3 fitting"
        ),
        checkpoint_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
        strict_load=True,
        image_size=[iw, ih],
        crop_size=[cw, ch],
        color="OpenCV BGR as author",
        mean=mean.cpu().flatten().tolist(),
        std=std.cpu().flatten().tolist(),
        decoder="Author integer heatmap argmax",
        coordinate_inverse="(crop_prediction + crop_offset) * original_size / resize_size",
        tta=False,
        fp32=True,
        torch=torch.__version__,
        config=cfg.dump(),
        limitation="Train diagnostics are not cross-validation; external training overlap not audited.",
    )
    save_json(OUT / "inference_protocol.json", protocol)
    start = time.time()
    rows = []
    files = [
        (split, p) for split in ["train", "test"] for p in sorted((data / split).glob("*.jpg"))
    ]
    if args.limit:
        files = files[: args.limit]
    for idx, (split, path) in enumerate(files):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(path)
        h, w = bgr.shape[:2]
        resized = cv2.resize(bgr, (iw, ih), interpolation=cv2.INTER_LINEAR)
        cropped = np.ascontiguousarray(resized[top : top + ch, left : left + cw].transpose(2, 0, 1))
        tensor = torch.from_numpy(cropped).unsqueeze(0).cuda().float().div_(255)
        tensor = (tensor - mean) / std
        with torch.inference_mode():
            coarse, fine, offsets, meta = model(tensor, {})
            lr, _, final, _, _ = get_final_preds(
                cfg, coarse.cpu().numpy(), fine.cpu().numpy(), offsets, meta
            )
        image_id = int(path.stem)
        row = dict(
            split=split,
            image_id=image_id,
            width=w,
            height=h,
            Fovea_X=float((final[0, 0] + left) * w / iw),
            Fovea_Y=float((final[0, 1] + top) * h / ih),
            coarse_X=float((lr[0, 0] + left) * w / iw),
            coarse_Y=float((lr[0, 1] + top) * h / ih),
        )
        for axis, size in [("X", w), ("Y", h)]:
            if not math.isfinite(row["Fovea_" + axis]) or not 0 <= row["Fovea_" + axis] < size:
                raise ValueError(f"Invalid inverse coordinate: {row}")
            # Check forward/inverse geometry in original pixels, including anisotropic resize.
            restored = (
                row["Fovea_" + axis] * [iw, ih][axis == "Y"] / size - [left, top][axis == "Y"]
            )
            assert abs(restored - final[0, int(axis == "Y")]) < 1e-3
        if split == "train":
            x, y = labels[image_id]
            row.update(
                target_X=x, target_Y=y, error_px=math.hypot(row["Fovea_X"] - x, row["Fovea_Y"] - y)
            )
        rows.append(row)
        save_json(OUT / "predictions_progress.json", rows)
        print(
            f'{idx+1}/{len(files)} {split}/{path.name}: ({row["Fovea_X"]:.2f},{row["Fovea_Y"]:.2f}) '
            f'error={row.get("error_px", "test")} elapsed={time.time()-start:.1f}s',
            flush=True,
        )
        del coarse, fine, offsets, meta, tensor
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    train = [r for r in rows if r["split"] == "train"]
    summary = dict(
        n_train=len(train), n_test=len(rows) - len(train), elapsed_seconds=time.time() - start
    )
    for mode in ["fine", "coarse"]:
        sq = [
            sum((r[prefix + a] - r["target_" + a]) ** 2 for a in ["X", "Y"])
            for r in train
            for prefix in ["Fovea_" if mode == "fine" else "coarse_"]
        ]
        summary[mode] = dict(
            rmse_coordinate_px=math.sqrt(sum(sq) / (2 * len(sq))),
            mean_distance_px=sum(math.sqrt(v) for v in sq) / len(sq),
            max_distance_px=math.sqrt(max(sq)),
        )
    test = {r["image_id"]: r for r in rows if r["split"] == "test"}
    if len(test) == 20:
        template = list(csv.DictReader((data / "sample_submission.csv").open()))
        output_rows = []
        for item in template:
            ident, _, axis = item["ImageID"].split("_")
            output_rows.append(
                dict(ImageID=item["ImageID"], value=f'{test[int(ident)]["Fovea_"+axis]:.6f}')
            )
        assert len(output_rows) == 40 and len({r["ImageID"] for r in output_rows}) == 40
        target = OUT / "submission_zian_refuge_direct.csv"
        with target.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ImageID", "value"])
            writer.writeheader()
            writer.writerows(output_rows)
        summary["submission"] = str(target)
    save_json(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
