"""Compare label constructions only; none of these Dice values are model scores."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
ROOT = ROOT.parent if ROOT.name == "analysis" else ROOT  # 位于 analysis/ 时回到任务根目录


def main():
    labels = ROOT / "2026task-4/segmentation/train/label"
    rows = []
    for path in sorted(labels.glob("*.jpg"), key=lambda p: int(p.stem)):
        raw = np.asarray(Image.open(path).convert("L"))
        clean = (
            cv2.resize((raw > 127).astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)
            > 0
        )
        proxy = (
            np.asarray(Image.open(path).convert("L").resize((512, 512), Image.Resampling.BILINEAR))
            > 0
        )

        def dice(a, b):
            return float(2 * (a & b).sum() / (a.sum() + b.sum()))

        row = dict(
            id=path.stem,
            clean_vs_proxy_dice=dice(clean, proxy),
            clean_fraction=float(clean.mean()),
            proxy_fraction=float(proxy.mean()),
        )
        for radius in [1, 2]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
            dilated = cv2.dilate(clean.astype(np.uint8), kernel) > 0
            row[f"clean_dilate_radius{radius}_vs_proxy_dice"] = dice(dilated, proxy)
        added = proxy & ~clean
        distance = cv2.distanceTransform((~clean).astype(np.uint8), cv2.DIST_L2, 5)
        for width in [1, 2, 3]:
            row[f"proxy_extra_fraction_within_{width}px"] = float(
                (distance[added] <= width + 1e-6).mean()
            )
        row["proxy_missing_clean_pixels"] = int((clean & ~proxy).sum())
        rows.append(row)
    assert len(rows) == 20
    with (ROOT / "label_boundary_diagnostic_20261101.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "id"}
    result["interpretation"] = (
        "label construction geometry only; NOT model predictions or validation scores"
    )
    (ROOT / "label_boundary_diagnostic_20261101.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
