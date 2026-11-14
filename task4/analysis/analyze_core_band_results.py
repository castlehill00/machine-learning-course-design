"""Audit the returned five-fold core/band experiment without loading weights.

Returned source is hashed, never imported. Binary metrics are independently
recomputed from lossless PNGs and original labels. Historical probabilities
are used only for the already frozen L1/fusion comparisons.
"""

import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
ROOT = ROOT.parent if ROOT.name == "analysis" else ROOT  # 位于 analysis/ 时回到任务根目录
sys.path.insert(0, str(ROOT / ".ablation_test_deps"))

import csv
import hashlib
import json
import math
import random
import zipfile

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

OUT = ROOT / "core_band_results_20261107_cc538477"
REC = ROOT / "recovery_results_20261103_5596e091"
DATA = ROOT / "2026task-4/segmentation"
METRICS = (
    "dice",
    "precision",
    "recall",
    "centerline_precision",
    "centerline_recall",
    "cldice",
    "foreground_fraction",
    "thin_clean_centerline_recall",
)
GRID = (0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
PINS = {
    "task4_core_band.py": "a45a8137b7de215f43e69a987993c097a366e669b881fb5989e65cd9b9007eeb",
    "task4_controlled_ablation.py": "8353368810c8988d5e2d4a349de8bec049e50dc838e1f9397691496d835f0485",
    "task4_kaggle_v2.py": "42dbe8eb0347828b2b3ebb5cac49a82856232da741a6126972fa5800f65b0016",
    "task4_pipeline.py": "bf4fef344a0965b8a1ca23fc734ee2938b7316e2e65b769f947750c8a842a135",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def close(a, b, tol=1e-10):
    assert math.isclose(float(a), float(b), rel_tol=0, abs_tol=tol), (a, b)


def write(name, rows):
    with (OUT / name).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def check_frozen_rows(actual, expected):
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert a.keys() == e.keys()
        for k, v in e.items():
            if isinstance(v, (int, float)):
                close(a[k], v)
            else:
                assert a[k] == v, (k, a[k], v)


def binary(path):
    with Image.open(path) as im:
        assert im.size == (512, 512) and im.mode == "L"
        a = np.asarray(im)
    assert set(np.unique(a)) == {0, 255}
    return a > 0


def metrics(mask, truth):
    pred_sk = skeletonize(mask)
    c, p, sc, sp, thin = truth
    n = int(mask.sum())
    assert n > 0
    result = {}
    for name, gt, sk in [("clean", c, sc), ("proxy", p, sp)]:
        tp = int((mask & gt).sum())
        cp = float((pred_sk & gt).sum() / pred_sk.sum())
        cr = float((sk & mask).sum() / sk.sum())
        result[name] = dict(
            dice=float((2 * tp + 1e-7) / (n + int(gt.sum()) + 1e-7)),
            precision=tp / n,
            recall=tp / int(gt.sum()),
            centerline_precision=cp,
            centerline_recall=cr,
            cldice=2 * cp * cr / (cp + cr) if cp + cr else 0.0,
            foreground_fraction=float(mask.mean()),
            thin_clean_centerline_recall=float((mask & thin).sum() / thin.sum()),
        )
    return result


def encode(mask):
    a = mask.ravel(order="C").astype(np.int8)
    q = np.flatnonzero(np.diff(np.r_[0, a, 0]))
    return " ".join(map(str, np.column_stack((q[::2] + 1, q[1::2] - q[::2])).ravel()))


def decode(value):
    words = value.split()
    assert words and len(words) % 2 == 0 and all(w.isdigit() for w in words)
    runs = np.array([int(w) for w in words], np.int64).reshape(-1, 2)
    starts, lengths = runs.T
    assert np.all(starts >= 1) and np.all(lengths > 0)
    assert np.all(starts + lengths - 1 <= 512 * 512)
    assert np.all(starts[1:] >= starts[:-1] + lengths[:-1])
    flat = np.zeros(512 * 512, bool)
    for s, n in runs:
        flat[s - 1 : s - 1 + n] = True
    return flat.reshape(512, 512)


def rebuild_plans(train_ids, clean, fold, cfg):
    positive = [np.argwhere(clean[i]) for i in train_ids]
    hashes = []
    for epoch in range(1, 81):
        plans = []
        for index in range(cfg["patches_per_epoch"]):
            rng = random.Random((cfg["seed"] + 1009 * fold) * 10**10 + epoch * 10**6 + index)
            i = rng.randrange(len(train_ids))
            patch = cfg["patch"]
            crop = rng.randint(int(patch * 0.9), min(int(patch * 1.1), 512))
            if rng.random() < 0.3 and len(positive[i]):
                y, x = positive[i][rng.randrange(len(positive[i]))]
                top = max(0, min(int(y) - crop // 2, 512 - crop))
                left = max(0, min(int(x) - crop // 2, 512 - crop))
            else:
                top, left = rng.randint(0, 512 - crop), rng.randint(0, 512 - crop)
            plans.append(
                (
                    i,
                    top,
                    left,
                    crop,
                    rng.randrange(4),
                    rng.random() < 0.5,
                    rng.uniform(0.85, 1.15),
                    rng.uniform(-0.04, 0.04),
                    rng.uniform(0.85, 1.15),
                    [rng.uniform(0.95, 1.05) for _ in range(3)],
                )
            )
        hashes.append(digest(plans))
    return hashes


def check_aggregates(rows, folds, summaries):
    lookup = {(r["experiment"], r["target"], r["id"]): r for r in rows}
    assert len(lookup) == len(rows)
    fl = {(r["experiment"], r["target"], int(r["fold"])): r for r in folds}
    for s in summaries:
        e, t = s["experiment"], s["target"]
        group = [lookup[e, t, str(i)] for i in range(1, 21)]
        assert int(s["n_images"]) == 20 and str(s["complete_20_image_oof"]) == "True"
        for k in METRICS:
            close(s[k], np.mean([float(r[k]) for r in group]))
        fs = []
        for f in range(5):
            subset = [r for r in group if int(r["fold"]) == f]
            assert len(subset) == 4
            fs.append(float(np.mean([float(r["dice"]) for r in subset])))
            if folds:
                fr = fl[e, t, f]
                assert int(fr["n_images"]) == 4
                for k in METRICS:
                    close(fr[k], np.mean([float(r[k]) for r in subset]))
        close(s["worst_fold_dice"], min(fs))
        close(s["fold_dice_std"], np.std(fs))
    return lookup, fl


def main():
    checks = {}
    archive = OUT / "received_core_band_analysis.zip"
    assert sha(archive) == "cc538477c4bc45fcedb584c712d1f7daeb72dfe7e039c470d4d154e3e982b9b9"
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist()))
        assert sum(n.startswith("oof_binary_masks/") for n in z.namelist()) == 60
        assert sum(n.startswith("masks_512_white_vessels/") for n in z.namelist()) == 20
        for i in z.infolist():
            p = PurePosixPath(i.filename)
            assert not p.is_absolute() and ".." not in p.parts
            assert "\\" not in i.filename and ":" not in i.filename
            assert ((i.external_attr >> 16) & 0o170000) != 0o120000
            assert p.suffix not in (".pt", ".npz")
            assert (OUT / i.filename).read_bytes() == z.read(i)
    checks["original_zip_crc_safe_paths_and_all_returned_files_unchanged"] = True
    config = load(OUT / "core_band_config.json")
    cfg = config["config"]
    status = load(OUT / "core_band_status.json")
    frozen = load(ROOT / "core_band_reference_frozen.json")
    records = {str(f): load(REC / f"L1_proxy/fold{f}/completed.json") for f in range(5)}
    for name, pin in PINS.items():
        assert sha(ROOT / name) == sha(OUT / "runtime_source" / name) == pin
    assert config["source_sha256"] == PINS["task4_core_band.py"]
    assert config["helper_source_sha256"] == {
        n: h for n, h in PINS.items() if n != "task4_core_band.py"
    }
    assert config["loss"] == dict(
        ce=0.5,
        core_dice=0.25,
        proxy_dice=0.25,
        aux2=0.1,
        aux4=0.2,
        class_order=["background", "core", "band"],
        smooth=1.0,
    )
    assert cfg == load(REC / "recovery_config.json")["config"]
    assert not config["engineering_only"] and not status["engineering_only"]
    assert config["reference_records_hash"] == frozen["identity"]["records_hash"] == digest(records)
    for key in ("rows", "grid"):
        assert config[f"reference_{key}_hash"] == digest(frozen[key])
    check_frozen_rows(table(OUT / "reference_L1_per_image.csv"), frozen["rows"])
    check_frozen_rows(table(OUT / "reference_L1_threshold_grid.csv"), frozen["grid"])
    info = load(OUT / "reference_info.json")
    assert info == config["reference_info"]
    env = load(OUT / "execution_environment.json")
    assert env == info["environment"] == load(REC / "execution_environment.json")
    assert status["environment_differences"] == {}
    assert cfg["seed"] == 42 and cfg["folds"] == 5 and cfg["epochs"] == 80
    assert status["complete_20_image_oof"] and status["complete_folds"] == list(range(5))
    for key in (
        "shared_initialization_and_rgb_plans_verified",
        "probability_hierarchy_verified",
        "candidate_csv_generated",
        "test_inference_complete",
        "automatic_threshold_selection",
    ):
        assert status[key] is True
    for key in (
        "submitted_to_competition",
        "public_scores_used",
        "additional_model_or_loss_search",
        "model_weights_local_download_required",
        "automatic_final_winner_selection",
    ):
        assert status[key] is False
    checks[
        "four_pinned_sources_fixed_loss_recipe_frozen_reference_and_recorded_environment_match"
    ] = True

    manifest = table(OUT / "manifest.csv")
    assert manifest == table(REC / "manifest.csv") and len(manifest) == 40
    assignment = {r["id"]: int(r["fold"]) for r in manifest if r["split"] == "train"}
    assert {f: sorted([int(i) for i in assignment if assignment[i] == f]) for f in range(5)} == {
        0: [2, 6, 9, 10],
        1: [3, 14, 17, 20],
        2: [1, 4, 5, 11],
        3: [15, 16, 18, 19],
        4: [7, 8, 12, 13],
    }
    truth, clean, stable = {}, {}, []
    for r in manifest:
        ip = DATA / r["split"] / "image" / f'{r["id"]}.jpg'
        assert sha(ip) == r["image_sha256"]
        with Image.open(ip) as im:
            assert im.size == (int(r["width"]), int(r["height"]))
        if r["split"] == "train":
            lp = DATA / "train/label" / f'{r["id"]}.jpg'
            assert sha(lp) == r["label_sha256"]
            with Image.open(lp) as im:
                im = im.convert("L")
                c = cv2.resize(
                    (np.asarray(im) > 127).astype(np.uint8),
                    (512, 512),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                p = np.asarray(im.resize((512, 512), Image.Resampling.BILINEAR)) > 0
            assert not (c & ~p).any()
            sc, sp = skeletonize(c), skeletonize(p)
            thin = sc & (cv2.distanceTransform(c.astype(np.uint8), cv2.DIST_L2, 5) <= 1.5)
            truth[r["id"]] = c, p, sc, sp, thin
            clean[r["id"]] = c
            close(c.mean(), r["vessel_fraction"])
            close(p.mean(), r["converter_fraction"])
        row = {
            k: r[k]
            for k in ("split", "id", "width", "height", "image_sha256", "label_sha256", "fold")
        }
        if row["split"] == "train":
            row["fold"] = int(row["fold"])
        stable.append(row)
    assert digest(stable) == config["data_fingerprint"] == frozen["identity"]["data_fingerprint"]
    assert load(OUT / "audit_summary.json")["converter"]["source_sha256"] == sha(
        DATA / "segmentation_to_csv.py"
    )
    checks["all40_original_images_20_labels_frozen_folds_targets_and_fingerprint_verified"] = True

    histories, seconds = {}, 0.0
    for f in range(5):
        done = load(OUT / f"fold{f}/completed.json")
        ref = records[str(f)]
        assert (
            done["method"] == "core_band_3class" and done["epoch"] == 80 and done["config"] == cfg
        )
        for k in ("train_ids", "val_ids", "plan_hashes"):
            assert done[k] == ref[k]
        assert done["reference_binary_initial_hash"] == ref["initial_model_hash"]
        assert done["initial_model_hash"] != done["reference_binary_initial_hash"]
        assert done["test_ids"] == [str(i) for i in range(1, 21)]
        assert set(done["val_ids"]) == {i for i in assignment if assignment[i] == f}
        assert set(done["train_ids"]) == set(assignment) - set(done["val_ids"])
        assert sha(OUT / f"fold{f}/history.csv") == done["sha256"]["history.csv"]
        assert all(
            len(h) == 64 and set(h) <= set("0123456789abcdef") for h in done["sha256"].values()
        )
        rows = table(OUT / f"fold{f}/history.csv")
        assert [int(r["epoch"]) for r in rows] == list(range(1, 81))
        assert [r["plan_sha256"] for r in rows] == done["plan_hashes"]
        assert rebuild_plans(done["train_ids"], clean, f, cfg) == done["plan_hashes"]
        for r in rows:
            for k in ("loss", "ce", "core_dice_loss", "proxy_dice_loss", "seconds"):
                assert math.isfinite(float(r[k])) and float(r[k]) >= 0
            close(
                r["loss"],
                0.5 * float(r["ce"])
                + 0.25 * float(r["core_dice_loss"])
                + 0.25 * float(r["proxy_dice_loss"]),
                1e-6,
            )
            monitored = int(r["epoch"]) % 10 == 0
            for k in ("monitor_core_clean_05", "monitor_proxy_proxy_05"):
                assert bool(r[k]) == monitored
                if monitored:
                    assert 0 <= float(r[k]) <= 1
            seconds += float(r["seconds"])
        histories[f] = rows
    checks[
        "five_final80_folds_all400_history_rows_loss_arithmetic_and_rebuilt_RGB_plans_verified"
    ] = True
    print(
        "Verified sources, dataset, all 400 epochs and independently rebuilt sampling plans.",
        flush=True,
    )

    rows = table(OUT / "core_band_per_image.csv")
    folds = table(OUT / "core_band_per_fold.csv")
    summaries = table(OUT / "core_band_summary.csv")
    assert (len(rows), len(folds), len(summaries)) == (480, 120, 24)
    expected = {
        f"{h}_{m}_t{t:g}"
        for h in ("core", "proxy")
        for m in ("none", "d4")
        for t in (0.1, 0.4, 0.5)
    }
    assert {r["experiment"] for r in rows} == expected
    lookup, fl = check_aggregates(rows, folds, summaries)
    for r in rows:
        assert int(r["fold"]) == assignment[r["id"]]
        assert all(math.isfinite(float(r[k])) and 0 <= float(r[k]) <= 1 for k in METRICS)
        gt = truth[r["id"]][0 if r["target"] == "clean" else 1]
        n = round(float(r["foreground_fraction"]) * 512 * 512)
        tp = round(float(r["recall"]) * int(gt.sum()))
        close(r["foreground_fraction"], n / (512 * 512))
        close(r["precision"], tp / n)
        close(r["recall"], tp / int(gt.sum()))
        close(r["dice"], (2 * tp + 1e-7) / (n + int(gt.sum()) + 1e-7))
        a, b = float(r["centerline_precision"]), float(r["centerline_recall"])
        close(r["cldice"], 2 * a * b / (a + b) if a + b else 0.0)
        for k in ("foreground_fraction", "thin_clean_centerline_recall"):
            close(
                lookup[r["experiment"], "clean", r["id"]][k],
                lookup[r["experiment"], "proxy", r["id"]][k],
            )
    for s in status["summary"]:
        other = next(
            r for r in summaries if (r["experiment"], r["target"]) == (s["experiment"], s["target"])
        )
        for k in METRICS:
            close(s[k], other[k])
    for f in range(5):
        close(histories[f][-1]["monitor_core_clean_05"], fl["core_none_t0.5", "clean", f]["dice"])
        close(histories[f][-1]["monitor_proxy_proxy_05"], fl["proxy_none_t0.5", "proxy", f]["dice"])
    checks["480_image_rows_120_fold_rows_24_summaries_and_final_monitors_verified"] = True

    grid = table(OUT / "core_band_threshold_grid.csv")
    calibrated = table(OUT / "core_band_calibration_development.csv")
    assert len(grid) == 104 and len(calibrated) == 8
    for h in ("core", "proxy"):
        for m in ("none", "d4"):
            for target in ("clean", "proxy"):
                group = [r for r in grid if (r["head"], r["mode"], r["target"]) == (h, m, target)]
                assert tuple(float(r["threshold"]) for r in group) == GRID
                for r in group:
                    fs = [float(r[f"fold{f}"]) for f in range(5)]
                    close(r["mean_dice"], np.mean(fs))
                    close(r["worst_fold_dice"], min(fs))
                    t = float(r["threshold"])
                    if t in (0.1, 0.4, 0.5):
                        for f in range(5):
                            close(fs[f], fl[f"{h}_{m}_t{t:g}", target, f]["dice"])
                best = max(
                    group, key=lambda r: (float(r["mean_dice"]), -abs(float(r["threshold"]) - 0.5))
                )
                chosen = next(
                    r for r in calibrated if (r["head"], r["mode"], r["target"]) == (h, m, target)
                )
                assert all(chosen[k] == best[k] for k in best)
    selected = next(
        r for r in calibrated if (r["head"], r["mode"], r["target"]) == ("proxy", "d4", "proxy")
    )
    threshold = float(selected["threshold"])
    assert threshold == status["selected_proxy_threshold"] == 0.4
    checks["104_predeclared_grid_rows_8_selections_and_candidate_threshold_rule_verified"] = True

    reference_rows = table(OUT / "reference_L1_per_image.csv")
    old = {(r["experiment"], r["target"], r["id"]): r for r in reference_rows}
    paired = table(OUT / "paired_proxy_vs_L1_per_image.csv")
    ps = table(OUT / "paired_proxy_vs_L1_summary.csv")
    assert len(paired) == 1920 and len(ps) == 96
    paired_keys = set()
    for r in paired:
        key = r["experiment"], r["target"], r["id"], r["metric"]
        assert key not in paired_keys
        paired_keys.add(key)
        assert int(r["fold"]) == assignment[r["id"]]
        base_e = "L1_proxy_" + r["experiment"].removeprefix("proxy_")
        close(
            r["delta"],
            float(lookup[r["experiment"], r["target"], r["id"]][r["metric"]])
            - float(old[base_e, r["target"], r["id"]][r["metric"]]),
        )
    for s in ps:
        g = [
            r
            for r in paired
            if (r["experiment"], r["target"], r["metric"])
            == (s["experiment"], s["target"], s["metric"])
        ]
        ds = [float(r["delta"]) for r in g]
        assert len(ds) == 20
        close(s["mean_delta"], np.mean(ds))
        assert int(s["positive_images"]) == sum(d > 0 for d in ds)
        fs = [np.mean([float(r["delta"]) for r in g if int(r["fold"]) == f]) for f in range(5)]
        assert int(s["positive_folds"]) == sum(d > 0 for d in fs)
        for f in range(5):
            close(s[f"fold{f}_delta"], fs[f])
    checks["1920_paired_image_deltas_and96_paired_summaries_verified"] = True

    cr = table(OUT / "candidate_oof_per_image.csv")
    cs = table(OUT / "candidate_oof_summary.csv")
    assert len(cr) == 40 and len(cs) == 2
    candidate, _ = check_aggregates(cr, [], cs)
    recomputed, masks = [], {}
    for tag, experiment in [
        ("core_t05", "core_d4_t0.5"),
        ("proxy_t05", "proxy_d4_t0.5"),
        ("proxy_candidate", "proxy_d4_t0.4"),
    ]:
        masks[tag] = {}
        for i in assignment:
            mask = binary(OUT / f"oof_binary_masks/{tag}/{i}.png")
            masks[tag][i] = mask
            for target, values in metrics(mask, truth[i]).items():
                for k in METRICS:
                    close(values[k], lookup[experiment, target, i][k])
                    if tag == "proxy_candidate":
                        close(
                            values[k], candidate["proxy_candidate_selected_threshold", target, i][k]
                        )
                recomputed.append(dict(mask=tag, id=i, fold=assignment[i], target=target, **values))
    assert len(recomputed) == 120
    for i in assignment:
        assert not (masks["core_t05"][i] & ~masks["proxy_t05"][i]).any()
        assert not (masks["proxy_t05"][i] & ~masks["proxy_candidate"][i]).any()
    write("local_recomputed_binary_metrics.csv", recomputed)
    checks["all60_lossless_OOF_masks_120_pixel_and_skeleton_rows_independently_recomputed"] = True
    checks["fixed05_binary_core_subset_proxy_and_threshold_monotonicity_verified"] = True

    # Replay historical masks from existing OOF probabilities; no new selection.
    comparison_rows, comparison_summary, errors = [], [], []
    refs = {"L1_proxy_t04": {}, "fixed_fusion": {}}
    for f in range(5):
        p = REC / f"L1_proxy/fold{f}/validation_probabilities.npz"
        assert sha(p) == records[str(f)]["sha256"]["validation_probabilities.npz"]
        with np.load(p, allow_pickle=False) as z:
            for i in records[str(f)]["val_ids"]:
                refs["L1_proxy_t04"][i] = z[f"d4_{i}"] >= 0.4
                for mode in ("none", "d4"):
                    for t in (0.1, 0.4, 0.5):
                        e = f"L1_proxy_{mode}_t{t:g}"
                        for target, values in metrics(z[f"{mode}_{i}"] >= t, truth[i]).items():
                            for k in METRICS:
                                close(values[k], old[e, target, i][k])
    fusion_lookup = {
        (r["experiment"], r["target"], r["id"]): r for r in table(REC / "fusion_per_image.csv")
    }
    with np.load(REC / "fusion_oof_d4_probabilities.npz", allow_pickle=False) as z:
        assert set(z.files) == set(assignment)
        for i in z.files:
            refs["fixed_fusion"][i] = z[i] >= 0.5
            for target, values in metrics(refs["fixed_fusion"][i], truth[i]).items():
                for k in METRICS:
                    close(values[k], fusion_lookup["fixed_equal_centered_logit", target, i][k])
    for name, baseline in refs.items():
        for i in assignment:
            new = masks["proxy_candidate"][i]
            old_m = baseline[i]
            before, after = metrics(old_m, truth[i]), metrics(new, truth[i])
            for target in ("clean", "proxy"):
                for k in METRICS:
                    comparison_rows.append(
                        dict(
                            reference=name,
                            id=i,
                            fold=assignment[i],
                            target=target,
                            metric=k,
                            before=before[target][k],
                            after=after[target][k],
                            delta=after[target][k] - before[target][k],
                        )
                    )
            c, p = truth[i][:2]
            errors.append(
                dict(
                    reference=name,
                    id=i,
                    fold=assignment[i],
                    reference_fn_core=int((c & ~old_m).sum()),
                    new_fn_core=int((c & ~new).sum()),
                    reference_fn_band=int((p & ~c & ~old_m).sum()),
                    new_fn_band=int((p & ~c & ~new).sum()),
                    reference_fp=int((old_m & ~p).sum()),
                    new_fp=int((new & ~p).sum()),
                    corrected_proxy_errors=int(((old_m != p) & (new == p)).sum()),
                    introduced_proxy_errors=int(((old_m == p) & (new != p)).sum()),
                )
            )
        for target in ("clean", "proxy"):
            for k in METRICS:
                g = [
                    r
                    for r in comparison_rows
                    if (r["reference"], r["target"], r["metric"]) == (name, target, k)
                ]
                ds = [r["delta"] for r in g]
                fs = [float(np.mean([r["delta"] for r in g if r["fold"] == f])) for f in range(5)]
                comparison_summary.append(
                    dict(
                        reference=name,
                        target=target,
                        metric=k,
                        before=float(np.mean([r["before"] for r in g])),
                        after=float(np.mean([r["after"] for r in g])),
                        mean_delta=float(np.mean(ds)),
                        positive_images=sum(d > 0 for d in ds),
                        positive_folds=sum(d > 0 for d in fs),
                        **{f"fold{f}_delta": d for f, d in enumerate(fs)},
                    )
                )
    write("local_candidate_comparisons_per_image.csv", comparison_rows)
    write("local_candidate_comparisons_summary.csv", comparison_summary)
    write("local_candidate_pixel_errors.csv", errors)
    checks["frozen_L1_240_rows_and_fusion40_rows_recomputed_from_historical_probabilities"] = True
    checks["candidate_vs_L1_and_previous_fusion_comparisons_and_error_counts_recomputed"] = True

    with (OUT / "submission_core_band_proxy.csv").open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)
        assert reader.fieldnames == ["Id", "Predicted"]
    ids = [r["Id"] for r in table(DATA / "sample_submission.csv")]
    assert [r["Id"] for r in csv_rows] == ids == [str(i) for i in range(1, 21)]
    csv_sha = sha(OUT / "submission_core_band_proxy.csv")
    assert (
        csv_sha
        == sha(OUT / "received_submission_core_band_proxy.csv")
        == status["exported"]["sha256"]
    )
    assert csv_sha == "e4d7f6b92c8e4a748ad2fcf8fd247bf10ccef7ab801ba0f4957e2e7066fbc562"
    assert status["exported"]["threshold"] == threshold and status["exported"]["rows"] == 20
    areas, differences = [], []
    previous = {
        r["Id"]: decode(r["Predicted"])
        for r in table(ROOT / "submission_fixed_equal_logit_fusion.csv")
    }
    assert (
        sha(ROOT / "submission_fixed_equal_logit_fusion.csv")
        == "51abf18c801dab4b49e26ce51b51b1063ef67a6075a1b2790f5bb12b0fcdd075"
    )
    for r in csv_rows:
        i = r["Id"]
        mask = decode(r["Predicted"])
        assert encode(mask) == r["Predicted"]
        mp = OUT / f"masks_512_white_vessels/submission_core_band_proxy/{i}.png"
        assert np.array_equal(mask, binary(mp))
        with Image.open(mp) as im:
            official = np.asarray(im.convert("L").resize((512, 512), Image.Resampling.BILINEAR)) > 0
        assert encode(official) == r["Predicted"]
        assert 0 < mask.sum() < mask.size
        areas.append(float(mask.mean()))
        differences.append(
            dict(
                id=i,
                new_fraction=float(mask.mean()),
                previous_fraction=float(previous[i].mean()),
                changed_pixels=int((mask != previous[i]).sum()),
            )
        )
    write("local_test_prediction_differences.csv", differences)
    checks["20_template_IDs_RLEs_PNGs_and_official_converter_pixel_semantics_verified"] = True
    checks["previous_084276_submission_bytes_unchanged"] = True

    primary = next(
        r
        for r in ps
        if (r["experiment"], r["target"], r["metric"]) == ("proxy_d4_t0.5", "proxy", "dice")
    )
    best_ref = max(
        [r for r in frozen["grid"] if r["mode"] == "d4" and r["target"] == "proxy"],
        key=lambda r: (r["mean_dice"], -abs(r["threshold"] - 0.5)),
    )
    calibrated_comparison = status["calibrated_proxy_comparison"]
    assert calibrated_comparison["new_threshold"] == threshold
    close(calibrated_comparison["reference_threshold"], best_ref["threshold"])
    for f in range(5):
        close(
            calibrated_comparison[f"fold{f}_delta"],
            float(selected[f"fold{f}"]) - best_ref[f"fold{f}"],
        )
    close(calibrated_comparison["mean_delta"], float(selected["mean_dice"]) - best_ref["mean_dice"])
    assert calibrated_comparison["positive_folds"] == 0
    checks["calibrated_comparison_matches_frozen_L1_reference_and_predeclared_rule"] = True
    error_totals = {}
    for name in refs:
        g = [r for r in errors if r["reference"] == name]
        error_totals[name] = {
            k: sum(r[k] for r in g) for k in g[0] if k not in ("reference", "id", "fold")
        }
    audit = dict(
        date="2026-11-07",
        checks=checks,
        analysis_archive_sha256=sha(archive),
        csv_sha256=csv_sha,
        formal_training_complete=True,
        new_training_arms=1,
        folds=5,
        epochs_per_fold=80,
        logged_training_and_monitor_minutes=seconds / 60,
        selected_proxy_threshold=threshold,
        primary_fixed05_comparison=primary,
        candidate_comparisons=comparison_summary,
        pixel_error_totals=error_totals,
        foreground_mean=float(np.mean(areas)),
        foreground_min=min(areas),
        foreground_max=max(areas),
        decision="Do not promote or submit this candidate; retain the existing fixed fusion.",
        threshold_selection_is_reused_OOF_development_tuning=True,
        public_scores_used_to_select_new_method=False,
        additional_training_required=False,
        weight_download_required=False,
        candidate_submitted_by_agent=False,
        limitations=[
            "No new weights or floating probabilities in the light package: local checks do not replay neural inference.",
            "At new thresholds beyond saved masks, numerical tables are checked for consistency but not independently rebuilt from float arrays.",
            "Shared initialization and completed test inference are supported by pinned implementation and returned provenance, not direct weight inspection.",
            "Repeatedly reused 20-image OOF is development evidence, not an independent or nested evaluation; private leaderboard remains unknown.",
            "Observed results apply to this fixed three-class loss/recipe, not all possible core/band methods.",
        ],
    )
    (OUT / "local_analysis_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in audit.items() if k != "candidate_comparisons"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
