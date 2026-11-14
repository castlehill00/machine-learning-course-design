"""Independent pixel-level replay of returned recovery probabilities; no weights."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROOT = ROOT.parent if ROOT.name == "analysis" else ROOT  # 位于 analysis/ 时回到任务根目录
sys.path.insert(0, str(ROOT / ".ablation_test_deps"))
import csv
import hashlib
import json
import math
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # 引用 src/ 下的实验代码
import task4_recovery as rec

OUT = ROOT / "recovery_results_20261103_5596e091"
base, v2, abl = rec.base, rec.v2, rec.abl
METRICS = [
    "dice",
    "precision",
    "recall",
    "centerline_precision",
    "centerline_recall",
    "cldice",
    "foreground_fraction",
    "thin_clean_centerline_recall",
]


def read(name):
    with (OUT / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-10), (a, b)


def main():
    config = json.loads((OUT / "recovery_config.json").read_text())
    cfg = abl.AblationConfig(**config["config"])
    assert config["source_sha256"] == base.file_sha(rec.__file__)
    for name, key in [
        ("task4_pipeline.py", "base_source_sha256"),
        ("task4_kaggle_v2.py", "v2_source_sha256"),
        ("task4_controlled_ablation.py", "ablation_source_sha256"),
        ("task4_recovery.py", "source_sha256"),
    ]:
        assert (
            base.file_sha(ROOT / name)
            == base.file_sha(OUT / "runtime_source" / name)
            == config[key]
        )
    assert config["fixed_fusion_rule"] == rec.RULE and not config["engineering_only"]
    status = json.loads((OUT / "recovery_status.json").read_text())
    assert status["complete_20_image_oof"] and all(
        status["complete_folds"][a] == list(range(5)) for a in rec.ARMS
    )
    for key in [
        "weights_or_threshold_search",
        "automatic_model_selection",
        "public_scores_used",
        "test_predictions_generated",
        "competition_csv_generated",
    ]:
        assert status[key] is False
    data = ROOT / "2026task-4/segmentation"
    train, _, labels, assignment, _, audit = v2.audit_v2(data, OUT / "local_data_audit", cfg)
    assert audit["data_fingerprint"] == config["data_fingerprint"]
    probs = {a: {} for a in rec.ARMS}
    seconds = 0.0
    expected = {a: {} for a in rec.ARMS}
    for arm in rec.ARMS:
        for fold in range(5):
            folder = OUT / arm / f"fold{fold}"
            done = json.loads((folder / "completed.json").read_text())
            assert done["epoch"] == 80 and set(done["val_ids"]) == {
                i for i, f in assignment.items() if f == fold
            }
            assert set(done["train_ids"]) == set(assignment) - set(done["val_ids"])
            for name in ["history.csv", "validation_probabilities.npz"]:
                assert base.file_sha(folder / name) == done["sha256"][name]
            h = read(f"{arm}/fold{fold}/history.csv")
            assert [int(r["epoch"]) for r in h] == list(range(1, 81))
            seconds += sum(float(r["seconds"]) for r in h)
            expected[arm][str(fold)] = dict(
                record=done, completed_json_sha256=base.file_sha(folder / "completed.json")
            )
            with np.load(folder / "validation_probabilities.npz", allow_pickle=False) as arrays:
                assert set(arrays.files) == {
                    f"{m}_{i}" for m in ["none", "d4"] for i in done["val_ids"]
                }
                for i in done["val_ids"]:
                    for mode in ["none", "d4"]:
                        p = arrays[f"{mode}_{i}"]
                        assert (
                            p.shape == (512, 512)
                            and np.isfinite(p).all()
                            and 0 <= p.min() <= p.max() <= 1
                        )
                    probs[arm][i] = arrays[f"d4_{i}"].copy()
            gt = [base.strict_label(labels[i]) for i in done["val_ids"]]
            with np.load(folder / "validation_probabilities.npz") as arrays:
                clean = float(
                    np.mean(
                        [
                            base.dice_score(arrays[f"none_{i}"] >= 0.5, g)
                            for i, g in zip(done["val_ids"], gt)
                        ]
                    )
                )
                last = h[-1]
                close(
                    clean, last["monitor_dice_05" if arm == "native_clean" else "monitor_clean_05"]
                )
    truth = abl.targets(labels)
    fused = {
        i: rec.aligned_fusion(probs["native_clean"][i], probs["L1_proxy"][i]) for i in assignment
    }
    with np.load(OUT / "fusion_oof_d4_probabilities.npz", allow_pickle=False) as arrays:
        assert set(arrays.files) == set(assignment)
        drift = max(float(np.max(np.abs(fused[i] - arrays[i]))) for i in assignment)
        disagreements = sum(
            int(((fused[i] >= 0.5) != (arrays[i] >= 0.5)).sum()) for i in assignment
        )
        assert drift <= 1e-7 and disagreements == 0
    masks = {
        "native_d4_t01": {i: p >= 0.1 for i, p in probs["native_clean"].items()},
        "proxy_d4_t04": {i: p >= 0.4 for i, p in probs["L1_proxy"].items()},
        "fixed_equal_centered_logit": {i: p >= 0.5 for i, p in fused.items()},
    }
    rows = []
    for name, p in masks.items():
        rows += abl.evaluate_masks(name, p, truth, assignment)
    saved = {(r["experiment"], r["target"], r["id"]): r for r in read("fusion_per_image.csv")}
    assert len(saved) == len(rows) == 120
    for r in rows:
        original = saved[r["experiment"], r["target"], r["id"]]
        for k in METRICS:
            close(r[k], original[k])
    summary, folds = abl.summarize_rows(rows)
    for name, recomputed, keys in [
        ("fusion_summary.csv", summary, ["experiment", "target"]),
        ("fusion_per_fold.csv", folds, ["experiment", "target", "fold"]),
    ]:
        lookup = {tuple(str(r[k]) for k in keys): r for r in read(name)}
        for r in recomputed:
            original = lookup[tuple(str(r[k]) for k in keys)]
            for k in METRICS:
                close(r[k], original[k])
            if "worst_fold_dice" in r:
                close(r["worst_fold_dice"], original["worst_fold_dice"])
    overlap = {(r["id"], r["target"]): r for r in read("error_complementarity.csv")}
    for i in assignment:
        for target in ["clean", "proxy"]:
            ea = masks["native_d4_t01"][i] != truth[i][target]
            eb = masks["proxy_d4_t04"][i] != truth[i][target]
            r = overlap[i, target]
            for k, m in [
                ("native_wrong", ea),
                ("proxy_wrong", eb),
                ("both_wrong", ea & eb),
                ("native_wrong_proxy_correct", ea & ~eb),
                ("proxy_wrong_native_correct", eb & ~ea),
            ]:
                assert int(r[k]) == int(m.sum())
    metrics_lookup = {(r["experiment"], r["target"], r["id"]): r for r in rows}
    for r in read("fusion_paired_per_image.csv"):
        close(
            r["delta"],
            metrics_lookup["fixed_equal_centered_logit", r["target"], r["id"]][r["metric"]]
            - metrics_lookup[r["reference"], r["target"], r["id"]][r["metric"]],
        )
    for r in read("fusion_paired_summary.csv"):
        ds = [
            metrics_lookup["fixed_equal_centered_logit", r["target"], i][r["metric"]]
            - metrics_lookup[r["reference"], r["target"], i][r["metric"]]
            for i in assignment
        ]
        close(r["mean_delta"], np.mean(ds))
        assert int(r["positive_images"]) == sum(d > 0 for d in ds)
        fs = [
            float(np.mean([d for i, d in zip(assignment, ds) if assignment[i] == f]))
            for f in range(5)
        ]
        assert int(r["positive_folds"]) == sum(d > 0 for d in fs)
        for f, v in enumerate(fs):
            close(r[f"fold{f}_delta"], v)
    abl.write_table(OUT / "local_recomputed_per_image.csv", rows)
    result = dict(
        all_four_returned_sources_match=True,
        all_ten_folds_80_epochs_complete=True,
        original_data_and_oof_hashes_verified=True,
        all120_pixel_and_skeleton_rows_independently_recomputed=True,
        all30_fold_rows_and6_summaries_verified=True,
        error_overlap_and_paired_deltas_verified=True,
        fusion_probability_max_drift=drift,
        fusion_mask_disagreements=disagreements,
        training_monitor_minutes=seconds / 60,
        local_model_weights_required=False,
        extra_training_started=False,
        private_score=None,
        conclusion="Fixed fusion improves proxy Dice in all five folds; clean Dice decreases in all five. One test-inference candidate, no weight search.",
    )
    base.save_json(OUT / "analysis_audit.json", result)
    base.save_json(OUT / "expected_inference_checkpoints.json", expected)
    (ROOT / "恢复与固定融合结果_20261103.md").write_text(
        """# 任务4恢复与固定融合结果

两组模型各完成5折、每折80轮训练。根据OOF概率在本地重新计算了像素级与骨架级指标、逐折结果和错误重叠情况，融合公式重算后在0.5阈值下的二值预测逐像素一致。

| 固定方法 | 干净Dice | 代理Dice | 最差代理折Dice |
|---|---:|---:|---:|
| 原尺寸D4阈值0.1（本次重建） | 0.743276 | 0.826899 | 0.808757 |
| L1 D4阈值0.4（本次重建） | 0.730368 | 0.826044 | 0.800791 |
| 预先固定的等权中心化logit融合 | 0.739780 | 0.829778 | 0.808980 |

融合与原尺寸模型相比：代理Dice平均+0.002879，16/20张图、5/5折提高；干净Dice−0.003496，五折均降低；干净clDice+0.003383，五折均提高，但细中心线召回−0.002280。与L1相比，融合的代理Dice+0.003734，17/20张图、5/5折提高。最差代理折相对原尺寸模型只提高0.000223。

融合在代理目标上有小幅提升，但干净目标略有下降，两个目标之间存在取舍。融合规则为事先固定的唯一规则，不再搜索权重或阈值；据此生成一份融合测试候选，同时保留原提交。

测试推理在Kaggle上用现有权重完成：每个系列的五折模型等权平均概率，再按同样的中心化等权logit规则融合，阈值固定为0.5，生成一份提交文件。

模型权重保留在Kaggle的输出中，本地只保存OOF结果和最终提交文件。
""",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
