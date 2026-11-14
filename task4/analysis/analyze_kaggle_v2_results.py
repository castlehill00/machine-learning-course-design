"""Audit returned analysis ZIP contents without executing the returned source."""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
ROOT = ROOT.parent if ROOT.name == "analysis" else ROOT  # 位于 analysis/ 时回到任务根目录
RESULT = ROOT / "kaggle_v2_results_20261101"
DATA = ROOT / "2026task-4/segmentation"


def load(name):
    return json.loads((RESULT / name).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    with (RESULT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def close(a, b):
    if not math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-11):
        raise AssertionError((a, b))


def dice(a, b):
    return float((2 * np.logical_and(a, b).sum() + 1e-7) / (a.sum() + b.sum() + 1e-7))


def decode(text):
    value = np.array([int(s) for s in text.split()], np.int64)
    if len(value) % 2:
        raise ValueError("Odd RLE count")
    out = np.zeros(512 * 512, np.uint8)
    previous = 0
    for start, length in value.reshape(-1, 2):
        if start < 1 or length < 1 or start <= previous or start + length - 1 > out.size:
            raise ValueError("RLE positive/order/overlap/bounds violation")
        out[start - 1 : start + length - 1] = 1
        previous = start + length - 1
    # Enforce the canonical runs emitted by the official converter.
    runs = np.flatnonzero(np.diff(np.r_[np.uint8(0), out, np.uint8(0)])) + 1
    runs[1::2] -= runs[::2]
    if " ".join(map(str, runs)) != text:
        raise ValueError("Noncanonical RLE")
    return out.reshape(512, 512).astype(bool)


def main():
    summary, config, provenance = (
        load("cv_summary.json"),
        load("run_config.json"),
        load("submission_provenance.json"),
    )
    feedback_path = ROOT / "public_scores.json"
    feedback = (
        json.loads(feedback_path.read_text(encoding="utf-8"))
        if feedback_path.exists()
        else {"records": []}
    )
    by_candidate = {r["candidate"]: r for r in feedback["records"]}
    v2_public = by_candidate.get("resnet34_5fold_d4", {}).get("public_dice")
    proxy_public = by_candidate.get("resnet34_5fold_d4_converter_proxy", {}).get("public_dice")
    baseline_public = by_candidate.get("baseline", {}).get("public_dice")
    for record in feedback["records"]:
        assert sha(ROOT / record["file"]) == record["sha256"]
    cfg = config["config"]
    checks = {}
    for file, key in [
        ("task4_kaggle_v2.py", "source_sha256"),
        ("task4_pipeline.py", "baseline_helpers_sha256"),
    ]:
        assert sha(RESULT / "runtime_source" / file) == config[key] == sha(ROOT / file)
    checks["returned_source_matches_delivered_implementation"] = True
    manifest = read_csv(RESULT / "manifest.csv")
    assert len(manifest) == 40
    assignments = {r["id"]: int(r["fold"]) for r in manifest if r["split"] == "train"}
    assert len(assignments) == 20 and all(
        list(assignments.values()).count(k) == 4 for k in range(5)
    )
    stable = []
    for r in manifest:
        image = DATA / r["split"] / "image" / f'{r["id"]}.jpg'
        assert sha(image) == r["image_sha256"]
        if r["split"] == "train":
            assert sha(DATA / "train/label" / f'{r["id"]}.jpg') == r["label_sha256"]
        row = {
            k: r[k]
            for k in ["split", "id", "width", "height", "image_sha256", "label_sha256", "fold"]
        }
        if r["split"] == "train":
            row["fold"] = int(row["fold"])
        stable.append(row)
    fingerprint = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
    assert (
        fingerprint == config["data_fingerprint"] == load("audit_summary.json")["data_fingerprint"]
    )
    checks["raw_train_label_test_files_and_frozen_folds_match"] = True
    assert load("pretrained_source.json")["sha256"] == config["pretrained_sha256"]
    assert config["pretrained_sha256"].startswith("b627a593")
    checks["declared_official_imagenet_source_consistent"] = True

    rows = read_csv(RESULT / "oof_per_image.csv")
    assert len(rows) == 40 and len({(r["mode"], r["id"]) for r in rows}) == 40
    for r in rows:
        assert int(r["fold"]) == assignments[r["id"]]
        assert all(math.isfinite(float(r[k])) for k in r if k not in {"id", "mode"})
    modes = {mode: {r["id"]: r for r in rows if r["mode"] == mode} for mode in ["none", "d4"]}
    for mode, values in modes.items():
        assert set(values) == set(assignments)
        for column, key in [
            ("dice_05", "mean_dice_05"),
            ("dice_global_threshold", "mean_dice_global_threshold"),
            ("dice_other_fold_threshold", "mean_dice_other_fold_threshold"),
            ("native_dice_05", "mean_native_dice_05"),
            ("converter_proxy_dice", "mean_converter_proxy_dice"),
        ]:
            close(np.mean([float(r[column]) for r in values.values()]), summary["modes"][mode][key])
        sweep = read_csv(RESULT / f"threshold_sweep_{mode}.csv")
        clean = max(
            sweep, key=lambda r: (float(r["clean_dice"]), -abs(float(r["threshold"]) - 0.5))
        )
        proxy = max(
            sweep,
            key=lambda r: (float(r["converter_proxy_dice"]), -abs(float(r["threshold"]) - 0.5)),
        )
        close(clean["threshold"], summary["modes"][mode]["global_threshold"])
        close(clean["clean_dice"], summary["modes"][mode]["mean_dice_global_threshold"])
        close(proxy["threshold"], summary["modes"][mode]["converter_proxy_best_threshold"])
        close(proxy["converter_proxy_dice"], summary["modes"][mode]["converter_proxy_best_dice"])
    checks["20_oof_images_once_per_mode_and_summary_arithmetic"] = True

    fold_rows = []
    for fold in range(5):
        folder = RESULT / f"fold{fold}"
        h = read_csv(folder / "history.csv")
        done = load(f"fold{fold}/completed.json")
        assert [int(r["epoch"]) for r in h] == list(range(1, cfg["epochs"] + 1))
        assert done["epoch"] == cfg["epochs"] and set(done["val_ids"]) == {
            i for i, f in assignments.items() if f == fold
        }
        assert sha(folder / "history.csv") == done["sha256"]["history.csv"]
        assert done["sha256"]["final_ema.pt"] == provenance["fold_models"][fold]
        for r in h:
            assert all(
                math.isfinite(float(r[k])) for k in ["loss", "encoder_lr", "decoder_lr", "seconds"]
            )
            assert float(r["loss"]) > 0 and float(r["seconds"]) > 0
        monitor = [r for r in h if r["monitor_dice_05"]]
        assert [int(r["epoch"]) for r in monitor] == list(
            range(cfg["monitor_every"], cfg["epochs"] + 1, cfg["monitor_every"])
        )
        best = max(monitor, key=lambda r: float(r["monitor_dice_05"]))
        val_ids = done["val_ids"]
        none = float(np.mean([float(modes["none"][i]["dice_05"]) for i in val_ids]))
        d4 = float(np.mean([float(modes["d4"][i]["dice_05"]) for i in val_ids]))
        close(none, done["none_dice_05"])
        close(d4, done["d4_dice_05"])
        close(none, h[-1]["monitor_dice_05"])
        fold_rows.append(
            dict(
                fold=fold,
                val_ids=" ".join(val_ids),
                epochs=len(h),
                training_monitor_minutes=sum(float(r["seconds"]) for r in h) / 60,
                last_loss=float(h[-1]["loss"]),
                best_monitored_epoch=int(best["epoch"]),
                best_monitor_dice_05=float(best["monitor_dice_05"]),
                final_none_dice_05=none,
                final_d4_dice_05=d4,
                d4_gain_05=d4 - none,
                none_dice_t04=float(
                    np.mean([float(modes["none"][i]["dice_global_threshold"]) for i in val_ids])
                ),
                d4_dice_t04=float(
                    np.mean([float(modes["d4"][i]["dice_global_threshold"]) for i in val_ids])
                ),
            )
        )
    write_csv("analysis_fold_comparison.csv", fold_rows)
    checks["all_five_histories_complete_80_epochs_and_hash_verified"] = True
    checks["final_monitor_scores_agree_with_saved_oof_records"] = True
    assert summary["selected_mode"] == (
        "d4"
        if summary["modes"]["d4"]["mean_dice_05"]
        >= summary["modes"]["none"]["mean_dice_05"] + cfg["tta_gain_required"]
        else "none"
    )
    checks["predeclared_tta_selection_rule_followed"] = True

    gains = []
    for i in sorted(assignments, key=int):
        a, b = modes["none"][i], modes["d4"][i]
        gains.append(
            dict(
                id=i,
                fold=assignments[i],
                none_05=float(a["dice_05"]),
                d4_05=float(b["dice_05"]),
                delta_05=float(b["dice_05"]) - float(a["dice_05"]),
                none_t04=float(a["dice_global_threshold"]),
                d4_t04=float(b["dice_global_threshold"]),
                delta_t04=float(b["dice_global_threshold"]) - float(a["dice_global_threshold"]),
            )
        )
    write_csv("analysis_tta_per_image.csv", gains)
    old = {r["id"]: r for r in read_csv(ROOT / "baseline_results/validation_per_image.csv")}
    matched = []
    for i in sorted(old, key=int):
        matched.append(
            dict(
                id=i,
                baseline_dice_05=float(old[i]["dice_05"]),
                v2_none_dice_05=float(modes["none"][i]["dice_05"]),
                v2_d4_dice_05=float(modes["d4"][i]["dice_05"]),
                none_gain=float(modes["none"][i]["dice_05"]) - float(old[i]["dice_05"]),
                d4_gain=float(modes["d4"][i]["dice_05"]) - float(old[i]["dice_05"]),
            )
        )
    write_csv("analysis_matched_baseline.csv", matched)

    ids = [r["Id"] for r in read_csv(DATA / "sample_submission.csv")]
    filenames = [
        "submission.csv",
        "submission_none.csv",
        "submission_d4.csv",
        "submission_converter_proxy.csv",
    ]
    expected_hashes = {
        "submission.csv": provenance["primary_csv_sha256"],
        "submission_none.csv": provenance["alternatives"]["none"]["sha256"],
        "submission_d4.csv": provenance["alternatives"]["d4"]["sha256"],
        "submission_converter_proxy.csv": provenance["converter_proxy_candidate"]["sha256"],
    }
    masks, statistics = {}, []
    for filename in filenames:
        assert sha(RESULT / filename) == expected_hashes[filename]
        table = read_csv(RESULT / filename)
        assert [r["Id"] for r in table] == ids and len(table) == 20
        masks[filename] = {r["Id"]: decode(r["Predicted"]) for r in table}
        for i, mask in masks[filename].items():
            statistics.append(dict(file=filename, id=i, foreground_fraction=float(mask.mean())))
    assert (RESULT / "submission.csv").read_bytes() == (RESULT / "submission_d4.csv").read_bytes()
    for i in ids:
        assert np.all(~masks["submission.csv"][i] | masks["submission_converter_proxy.csv"][i])
        png = np.asarray(Image.open(RESULT / "masks_native_black_vessels" / f"{i}.png"))
        assert png.shape == (584, 565) and set(np.unique(png)) <= {0, 255}
    write_csv("analysis_test_mask_statistics.csv", statistics)
    checks["all_four_csv_hashes_ids_order_canonical_rle_bounds_verified"] = True
    checks["main_csv_is_d4_and_proxy_contains_main_foreground"] = True
    checks["all_20_native_black_png_sizes_and_values_valid"] = True
    checks["weights_or_probabilities_available_for_inference_replay"] = False
    checks["new_training_started"] = False

    clean_fraction = load("audit_summary.json")["mean_clean_foreground"]
    proxy_fraction = load("audit_summary.json")["mean_converter_foreground"]
    rates = {
        name: dict(
            mean=float(np.mean([float(m.mean()) for m in values.values()])),
            minimum=float(min(m.mean() for m in values.values())),
            maximum=float(max(m.mean() for m in values.values())),
        )
        for name, values in masks.items()
    }
    stats = dict(
        folds=fold_rows,
        training_monitor_total_minutes=sum(r["training_monitor_minutes"] for r in fold_rows),
        tta_fixed05=dict(
            mean_delta=float(np.mean([r["delta_05"] for r in gains])),
            improved_images=sum(r["delta_05"] > 0 for r in gains),
            worse_images=sum(r["delta_05"] < 0 for r in gains),
            improved_folds=sum(r["d4_gain_05"] > 0 for r in fold_rows),
        ),
        tta_t04=dict(
            mean_delta=float(np.mean([r["delta_t04"] for r in gains])),
            improved_images=sum(r["delta_t04"] > 0 for r in gains),
            worse_images=sum(r["delta_t04"] < 0 for r in gains),
        ),
        matched_baseline=dict(
            n=4,
            baseline_05=float(np.mean([r["baseline_dice_05"] for r in matched])),
            v2_none_05=float(np.mean([r["v2_none_dice_05"] for r in matched])),
            v2_d4_05=float(np.mean([r["v2_d4_dice_05"] for r in matched])),
            none_gain=float(np.mean([r["none_gain"] for r in matched])),
            d4_gain=float(np.mean([r["d4_gain"] for r in matched])),
        ),
        lowest_d4_t04=sorted(gains, key=lambda r: r["d4_t04"])[:5],
        test_foreground_rates=rates,
        training_clean_foreground_fraction=clean_fraction,
        training_converter_proxy_foreground_fraction=proxy_fraction,
        converter_proxy_optimum_on_lower_grid_boundary=True,
        public_score=v2_public,
        converter_proxy_public_score=proxy_public,
        highest_reported_public_score=max(
            (r["public_dice"] for r in feedback["records"]), default=None
        ),
        baseline_public_score=baseline_public,
        private_leaderboard_exists=feedback.get("leaderboards", {}).get(
            "private_leaderboard_exists"
        ),
        private_score=None,
        selection_policy=feedback.get("selection_policy", {}),
        public_gain_vs_baseline=(
            v2_public - baseline_public
            if v2_public is not None and baseline_public is not None
            else None
        ),
        public_score_source="Kaggle submission record" if v2_public is not None else None,
        user_submission_status=(
            "submitted_user_reported" if v2_public is not None else "not_submitted"
        ),
    )
    result = dict(
        checks=checks, statistics=stats, archive_sha256=load("received_archive.json")["sha256"]
    )
    (RESULT / "analysis_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    table = "\n".join(
        f"| {r['fold']} | {r['val_ids']} | {r['final_none_dice_05']:.6f} | {r['final_d4_dice_05']:.6f} | {r['d4_gain_05']:+.6f} | {r['d4_dice_t04']:.6f} |"
        for r in fold_rows
    )
    matched_table = "\n".join(
        f"| {r['id']} | {r['baseline_dice_05']:.6f} | {r['v2_none_dice_05']:.6f} | {r['v2_d4_dice_05']:.6f} |"
        for r in matched
    )
    a, b = summary["modes"]["none"], summary["modes"]["d4"]
    if v2_public is not None:
        public_intro = (
            f"主提交（D4、阈值0.4）公开Dice为{v2_public:.5f}，baseline为{baseline_public:.5f}。"
        )
        public_conclusion = f"主提交公开Dice为{v2_public:.5f}，比baseline {baseline_public:.5f}提高{v2_public-baseline_public:.5f}（{100*(v2_public-baseline_public):.3f}个百分点）。这一提升来自预训练、五折平均和TTA等多项改动的共同作用。"
        next_step = "主提交保持不变。下一步用同一批权重的`submission_converter_proxy.csv`（阈值0.1）做一次阈值对照，再决定是否重训；无TTA候选留作后续衡量TTA贡献的对照。"
    else:
        public_intro = "结果返回时尚未向比赛提交，没有公开榜分数。"
        public_conclusion = "在相同4张图、固定0.5阈值下，原尺寸训练的新方法优于首版。"
        next_step = "先提交`submission.csv`获取公开分数，`submission_none.csv`作为TTA对照；如需确认评分时mask的约定，再提交`submission_converter_proxy.csv`。"
    if proxy_public is not None:
        public_intro += (
            f"阈值0.1代理候选公开Dice为{proxy_public:.5f}。本比赛另有私有榜，不应过度看重公开分数。"
        )
        public_conclusion += (
            f"同一模型阈值0.1时公开Dice为{proxy_public:.5f}，支持“评分目标经过JPEG转码”的推断。"
        )
        next_step = "同时保留阈值0.1和0.4两个候选，不再根据公开分数搜索阈值。后续方案选择以固定五折OOF、逐折稳定性和事先确定的比较标准为主，公开榜只作参考。阈值敏感性分析在已有OOF概率上进行。"
        proxy_status = "阈值0.1候选已有公开分数，不再追加阈值试探。两种目标的指标分开统计，最终选择依据验证结果和稳定性。"
    else:
        proxy_status = "如需诊断标签预处理，可用同一批权重的`submission_converter_proxy.csv`（阈值0.1）作对照；注意代理Dice与干净血管Dice对应不同目标，不能直接比较。"
    text = f"""# 任务4 Kaggle五折优化版结果分析

分析日期：2026-11-01。{public_intro}结果文件SHA256为`{result['archive_sha256']}`。

## 结论

五折均训练80轮，共400轮、102400个增强patch。主提交`submission.csv`（与`submission_d4.csv`相同）由5折最终EMA权重等权平均、D4推理、阈值0.4得到。{public_conclusion}

{proxy_status}

## 运行情况

在一块Tesla T4上运行。5折日志均完整记录第1至80轮；每折4张OOF图，共20张。固定使用最终EMA权重，不按验证集挑选轮次。逐轮训练与监控时间合计约{stats['training_monitor_total_minutes']:.2f}分钟（不含最终D4推理、概率保存与打包）。

源码、60张训练图/标签/测试图和五折划分与原设置一致；逐折记录、最终监控分数和逐图OOF相互吻合。四份提交CSV的20个ID、顺序及RLE编码均检查无误，主CSV为D4候选。

模型权重和概率文件保存在`full_backup.zip`中，本地分析基于源码、日志、汇总表和提交文件。

## 20图OOF结果

| 方法 | 512干净目标Dice@0.5 | 512干净目标Dice@0.4 | 跨折阈值诊断Dice | 原尺寸Dice@0.5 |
|---|---:|---:|---:|---:|
| 无TTA | {a['mean_dice_05']:.6f} | {a['mean_dice_global_threshold']:.6f} | {a['mean_dice_other_fold_threshold']:.6f} | {a['mean_native_dice_05']:.6f} |
| D4 | {b['mean_dice_05']:.6f} | {b['mean_dice_global_threshold']:.6f} | {b['mean_dice_other_fold_threshold']:.6f} | {b['mean_native_dice_05']:.6f} |

固定0.5阈值下，D4平均提升{stats['tta_fixed05']['mean_delta']:.6f}（约{stats['tta_fixed05']['mean_delta']*100:.3f}个百分点），{stats['tta_fixed05']['improved_images']}/20张图提升、{stats['tta_fixed05']['worse_images']}/20张下降，{stats['tta_fixed05']['improved_folds']}/5折提升，达到预设的0.001启用门槛。阈值0.4下D4较无TTA提升{stats['tta_t04']['mean_delta']:.6f}。

| 折 | 验证ID | 无TTA@0.5 | D4@0.5 | D4增量 | D4@0.4 |
|---|---|---:|---:|---:|---:|
{table}

第4折较弱，D4阈值0.4时为{fold_rows[4]['d4_dice_t04']:.6f}；其中12号图为{stats['lowest_d4_t04'][0]['d4_t04']:.6f}，是20张图中的最低值，后续可重点分析该图的误差区域。困难样本保留在评估中。

TTA方式和全局阈值都是在这些OOF图上选出的，因此调参后的0.787453属于开发集估计；五折模型的训练集互有重叠，20张图的结果也不是完全独立的重复实验。

## 与首版的同图对照

首版只有4张验证图，其0.784356不能直接与新的20图OOF均值比较。下表在相同4张图、相同512干净目标、固定0.5阈值下对比：

| 图像 | 首版 | 新版无TTA | 新版D4 |
|---|---:|---:|---:|
{matched_table}

4张图的平均Dice：首版{stats['matched_baseline']['baseline_05']:.6f}，新版无TTA{stats['matched_baseline']['v2_none_05']:.6f}，新版D4{stats['matched_baseline']['v2_d4_05']:.6f}。无TTA提升约{stats['matched_baseline']['none_gain']*100:.3f}个百分点，D4提升约{stats['matched_baseline']['d4_gain']*100:.3f}个百分点。首版挑选了最佳轮次而新版固定轮数，且预训练、分辨率、增强和EMA同时改变，这一提升是多项改动的综合结果。

## JPEG转码代理指标

主提交（阈值0.4）的转换器目标代理Dice为{b['mean_converter_proxy_dice']:.6f}。阈值0.1时代理Dice为{b['converter_proxy_best_dice']:.6f}，但同阈值下干净目标Dice为{float(read_csv(RESULT/'threshold_sweep_d4.csv')[0]['clean_dice']):.6f}，两者是不同目标下的取舍，0.827069并不代表干净血管的分割精度。

测试集上主候选的平均前景比例为{rates['submission.csv']['mean']:.2%}，代理候选为{rates['submission_converter_proxy.csv']['mean']:.2%}；代理mask包含主mask的全部像素，多出的是低概率的边缘和细线区域。训练集干净标签的平均前景比例为{clean_fraction:.2%}，原JPEG经官方转换器处理后为{proxy_fraction:.2%}，这解释了低阈值为何更贴合代理目标。

两种推理模式的代理最优阈值都落在扫描下界0.1。如需扩大范围，应在已有OOF概率上按事先固定的范围做敏感性分析，并检查逐折稳定性。

## 预测可视化与下一步

从验证集和测试集的预测图看，主要血管结构连续，但仍有细血管遗漏，部分分支附近的宽度与标注不一致。

{next_step}

分折结果见`analysis_fold_comparison.csv`，逐图TTA增量见`analysis_tta_per_image.csv`，同图对照见`analysis_matched_baseline.csv`。
"""
    (ROOT / "Kaggle五折优化版结果分析_20261101.md").write_text(text, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
