"""Recompute failure/selection diagnostics from cached predictions; no model imports."""

import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

ROOT = Path(__file__).resolve().parent
ROOT = ROOT.parent if ROOT.name == "analysis" else ROOT  # 位于 analysis/ 时回到任务根目录
RESULT = ROOT / "kaggle_refine_results_20261025"
OUTPUT = ROOT / "refinement_diagnosis_20261027"


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_if_available(curves, coarse_rmse, guarded_rmse, raw_rmse, blend_rmse, best_shared):
    try:
        import matplotlib
    except ModuleNotFoundError:
        return False
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})
    axis = axes[0]
    axis.plot(
        [r["epoch"] for r in curves],
        [r["pooled_same_epoch_rmse_px"] for r in curves],
        color="#285d90",
        lw=1.8,
        label="One common epoch across folds",
    )
    axis.axhline(
        coarse_rmse, color="#777777", ls="--", lw=1.5, label=f"Coarse selected: {coarse_rmse:.2f}"
    )
    axis.axhline(
        guarded_rmse,
        color="#b45536",
        ls=":",
        lw=1.8,
        label=f"Per-fold epoch selection: {guarded_rmse:.2f}",
    )
    axis.scatter(
        best_shared["epoch"],
        best_shared["pooled_same_epoch_rmse_px"],
        color="#285d90",
        s=36,
        zorder=3,
    )
    axis.set(
        xlabel="Refinement epoch",
        ylabel="Pooled coordinate RMSE (px)",
        title="Strong sensitivity to checkpoint selection",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(fontsize=8, loc="upper right")
    axis = axes[1]
    vals = [coarse_rmse, raw_rmse, guarded_rmse, blend_rmse]
    axis.bar(range(4), vals, color=["#868b8e", "#c66a51", "#4b89a6", "#497a5a"], width=0.65)
    axis.set_xticks(range(4), ["Coarse", "Raw fine", "Fine +\nfallback", "Equal\nblend"])
    for i, value in enumerate(vals):
        axis.text(i, value + 0.5, f"{value:.2f}", ha="center", fontsize=9)
    axis.set(
        ylabel="Coordinate RMSE (px)",
        title="Same selected fine checkpoints",
        ylim=(0, max(vals) * 1.17),
    )
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    fig.suptitle("Development diagnostics: not an independent test estimate", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTPUT / "refinement_diagnosis.png", dpi=170)
    plt.close(fig)
    return True


def main():
    OUTPUT.mkdir(exist_ok=True)
    oof = read(RESULT / "oof_refine384.csv")
    coarse = read(RESULT / "oof_coarse512.csv")
    comparisons = {r["arm"]: r for r in read(RESULT / "comparison.csv")}
    assert len(oof) == 80 and len({r["image_id"] for r in oof}) == 80
    raw_sse = []
    guard_sse = []
    details = []
    for r in oof:
        raw = math.hypot(
            float(r["raw_refined_x"]) - float(r["gt_x"]),
            float(r["raw_refined_y"]) - float(r["gt_y"]),
        )
        guarded = math.hypot(
            float(r["Fovea_X"]) - float(r["gt_x"]), float(r["Fovea_Y"]) - float(r["gt_y"])
        )
        raw_sse.append(raw * raw)
        guard_sse.append(guarded * guarded)
        details.append(
            dict(
                image_id=int(r["image_id"]),
                fold=int(r["fold"]),
                raw_distance_px=raw,
                guarded_distance_px=guarded,
                coarse_distance_px=math.hypot(
                    float(r["coarse_x"]) - float(r["gt_x"]), float(r["coarse_y"]) - float(r["gt_y"])
                ),
                fallback=int(r["fallback"]),
                shift_px=float(r["shift_px"]),
            )
        )
    raw_rmse = math.sqrt(sum(raw_sse) / 160)
    guarded_rmse = math.sqrt(sum(guard_sse) / 160)
    coarse_rmse = float(comparisons["coarse512"]["rmse_coordinate_px"])
    blend_rmse = float(comparisons["coarse_refine_equal"]["rmse_coordinate_px"])
    assert math.isclose(
        guarded_rmse, float(comparisons["refine384"]["rmse_coordinate_px"]), abs_tol=1e-9
    )
    history = [read(RESULT / f"fold_{fold}/history.csv") for fold in range(5)]
    curves = []
    for e in range(1, 61):
        row = dict(epoch=e)
        values = []
        for fold, h in enumerate(history):
            assert len(h) == 60 and int(h[e - 1]["epoch"]) == e
            value = float(h[e - 1]["rmse_coordinate_px"])
            values.append(value)
            row[f"fold_{fold}_rmse_px"] = value
        row["pooled_same_epoch_rmse_px"] = math.sqrt(statistics.mean(v * v for v in values))
        curves.append(row)
    best_shared = min(curves, key=lambda r: r["pooled_same_epoch_rmse_px"])
    best_per_fold = [min(h, key=lambda r: float(r["rmse_coordinate_px"])) for h in history]
    selected_rmse = math.sqrt(
        statistics.mean(float(r["rmse_coordinate_px"]) ** 2 for r in best_per_fold)
    )
    assert math.isclose(selected_rmse, guarded_rmse, abs_tol=1e-9)
    jitter_rms = math.sqrt(2 * 160**2 / 3)
    coarse_rms = math.sqrt(
        statistics.mean(float(r["dx"]) ** 2 + float(r["dy"]) ** 2 for r in coarse)
    )
    tests = []
    for fold in range(5):
        for r in read(RESULT / f"fold_{fold}/test_refine.csv"):
            tests.append(
                dict(
                    image_id=int(r["image_id"]),
                    fold=fold,
                    coarse_x=float(r["coarse_x"]),
                    coarse_y=float(r["coarse_y"]),
                    refined_x=float(r["Fovea_X"]),
                    refined_y=float(r["Fovea_Y"]),
                    raw_refined_x=float(r["raw_refined_x"]),
                    raw_refined_y=float(r["raw_refined_y"]),
                    shift_px=float(r["shift_px"]),
                    fallback=int(r["fallback"]),
                )
            )
    test96 = [r for r in tests if r["image_id"] == 96]

    def spread(label):
        x = statistics.mean(r[f"{label}_x"] for r in test96)
        y = statistics.mean(r[f"{label}_y"] for r in test96)
        return math.sqrt(
            statistics.mean((r[f"{label}_x"] - x) ** 2 + (r[f"{label}_y"] - y) ** 2 for r in test96)
        )

    reference = ROOT / "kaggle_cv_results_20261025/submission_resnet34_512_5fold.csv"
    control = RESULT / "submission_coarse512_5fold.csv"
    assert control.read_bytes() == reference.read_bytes()
    exported = ROOT / "submission_resnet34_512_control.csv"
    shutil.copy2(control, exported)
    template = read(ROOT / "2026task3/detection/sample_submission.csv")
    candidate = read(exported)
    assert len(candidate) == 40 and [r["ImageID"] for r in candidate] == [
        r["ImageID"] for r in template
    ]
    public_path = ROOT / "public_score_audit/public_scores.json"
    public_records = (
        json.loads(public_path.read_text(encoding="utf-8")) if public_path.exists() else {}
    )
    control_result = public_records.get("resnet34_512_coarse_control_5fold")
    refined_result = public_records.get("resnet34_coarse_refine_equal_5fold")
    if (
        control_result
        and control_result["sha256"] != hashlib.sha256(exported.read_bytes()).hexdigest()
    ):
        raise ValueError("Control public score belongs to another file.")
    control_score = control_result["score"] if control_result else None
    refined_score = refined_result["score"] if refined_result else None
    paired_comparison = (
        dict(
            coarse_score=control_score,
            coarse_refine_equal_score=refined_score,
            increase=round(refined_score - control_score, 5),
            relative_increase=(refined_score / control_score - 1),
            interpretation="This fixed refinement combination worsens the same-source coarse model on the reported public subset; mechanism/private performance unproven.",
        )
        if control_score is not None and refined_score is not None
        else None
    )
    write(
        OUTPUT / "raw_vs_guarded_oof.csv",
        sorted(details, key=lambda r: r["raw_distance_px"], reverse=True),
    )
    write(OUTPUT / "epoch_sensitivity.csv", curves)
    write(OUTPUT / "test_model_shifts.csv", tests)
    statistics_result = dict(
        raw_refiner_selected_checkpoints_rmse_px=raw_rmse,
        guarded_refiner_selected_checkpoints_rmse_px=guarded_rmse,
        guarded_blend_rmse_px=blend_rmse,
        coarse_selected_checkpoints_rmse_px=coarse_rmse,
        validation_fallback_count=sum(r["fallback"] for r in details),
        raw_sse_share_in_fallback_cases=sum(
            raw_sse[i] for i, r in enumerate(details) if r["fallback"]
        )
        / sum(raw_sse),
        sse_reduction_due_to_fallback=1 - sum(guard_sse) / sum(raw_sse),
        selected_epochs=[int(r["epoch"]) for r in best_per_fold],
        best_shared_epoch_in_retrospect=best_shared,
        last_epoch_pooled_rmse_px=curves[-1]["pooled_same_epoch_rmse_px"],
        jitter_center_rms_before_affine_px=jitter_rms,
        coarse_oof_center_rms_px=coarse_rms,
        jitter_to_coarse_rms_ratio=jitter_rms / coarse_rms,
        test96_coarse_spread_px=spread("coarse"),
        test96_refiner_spread_px=spread("refined"),
        control_submission=str(exported),
        control_sha256=hashlib.sha256(exported.read_bytes()).hexdigest(),
        control_public_score=control_score,
        same_source_public_comparison=paired_comparison,
        official_metric_confirmed=False,
        actual_training=False,
        interpretation="All metrics are descriptive development diagnostics. Same-epoch selection is also post hoc, not an unbiased estimate of selection bias. Test disagreement is not ground-truth error.",
    )
    (OUTPUT / "diagnosis.json").write_text(
        json.dumps(statistics_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    statistics_result["plot_generated"] = plot_if_available(
        curves, coarse_rmse, guarded_rmse, raw_rmse, blend_rmse, best_shared
    )
    (OUTPUT / "diagnosis.json").write_text(
        json.dumps(statistics_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if paired_comparison:
        public_heading = "## 5. 同源公开对照：精修组合相对粗定位退步"
        public_context = (
            f"2026-10-27提交的同源512粗定位对照公开分数为{control_score:.5f}；"
            f"相同粗预测加入精修并等权后为{refined_score:.5f}，"
            f'高{paired_comparison["increase"]:.5f}（约{paired_comparison["relative_increase"]:.2%}）。'
            "说明在这批公开样本上，加入精修后的组合比其自身的粗定位起点更差，这一比较排除了与320全量模型的输入差异。"
            "该结果只针对本次精修组合，具体是哪些图片退步、私有榜上如何仍不得而知。"
        )
        public_control = (
            f'对照文件为`submission_resnet34_512_control.csv`，SHA256为 `{statistics_result["control_sha256"]}`，'
            "与原512五折粗预测提交字节一致。分数取自Kaggle提交记录。"
            "原320全量模型（12.31624）仍为公开榜最佳。"
        )
        first_step = "以14.16538的粗定位结果作为下一轮精修消融的起点。"
    else:
        public_heading = "## 5. 公开榜比较中的混杂因素"
        public_context = (
            "12.31624是320输入、全部80张训练图、43轮单模型；16.45402则是512粗定位五折"
            "（每模型约64图）加局部精修及等权平均。上轮12.37299又包含320/512两个分辨率的集成。"
            "由于没有相同512粗定位单独提交的公开分数，4.13778的差距不能全部归因于精修。"
        )
        public_control = (
            f"已导出相同原512粗预测对照`submission_resnet34_512_control.csv`，SHA256为 "
            f'`{statistics_result["control_sha256"]}`，与原512五折粗预测完全相同。'
            "提交后与16.45402比较，即可判断精修组合相对自身起点的变化。"
        )
        first_step = "提交同源512粗定位对照，单独衡量精修的贡献（无需重新训练）。"
    report = f"""# 精修方案公开分数反转：原因排查

日期：2026-10-27。分析基于已保存的80张OOF预测、5折逐轮记录、20张测试图的逐折预测和公开分数，未重新训练，也未改动候选坐标。

## 评估口径说明

“精修在五折中均有改善”这一结果的前提是：每折从60轮中按该折16张验证图挑选误差最低的权重，并经过固定回退后再统计。它反映的是整个流程（含轮次选择与回退）的表现，而不是精修网络本身的稳定改善，也不是完全独立的五折泛化估计。

## 1. 原始精修输出存在严重失效，被回退规则掩盖

| 同一批选中权重/预测 | 坐标RMSE（像素） |
|---|---:|
| 原512粗定位 | {coarse_rmse:.4f} |
| 精修原始输出，不应用回退 | {raw_rmse:.4f} |
| 精修应用固定回退 | {guarded_rmse:.4f} |
| 粗定位与已回退精修等权 | {blend_rmse:.4f} |

0024和0049两张验证图的原始精修误差分别约299.91和399.18像素（精确值见raw_vs_guarded_oof.csv），占原始精修平方误差的{statistics_result['raw_sse_share_in_fallback_cases']:.2%}。固定回退替换这两点后，总平方误差减少{statistics_result['sse_reduction_due_to_fallback']:.2%}。

回退是事先设定的保护规则，问题在于网络仍会出现少量大错。现有阈值只检查精修点与粗定位点的距离是否超过256像素，并不判断精修本身是否可信。测试图0096在第3折移动约90.91像素，未触发回退，但与其他折明显不一致：其精修折间RMS分歧为{spread('refined'):.2f}像素，粗定位为{spread('coarse'):.2f}像素。由于没有测试标签，无法确定0096是否预测错误。

## 2. 精修收益依赖于逐折挑选训练轮次

各折最佳轮次为{statistics_result['selected_epochs']}。若各折统一使用同一个epoch，在全部60轮中最好的是第{best_shared['epoch']}轮，整体RMSE为{best_shared['pooled_same_epoch_rmse_px']:.4f}，仍高于原粗定位的{coarse_rmse:.4f}；第60轮为{curves[-1]['pooled_same_epoch_rmse_px']:.4f}。

这说明精修的改善并不对应一个稳定的训练阶段，而是依赖在各折小验证集上反复挑选。第一阶段的轮次也参考了同一外层验证折，第二阶段又从60轮中挑选，最后再选择方案，因此这一评估偏乐观（统一epoch本身也是事后选出的，不宜直接作为新的最优设置）。

这与[Cawley与Talbot关于模型选择过拟合的研究](https://jmlr.org/papers/v11/cawley10a.html)一致。[Varma与Simon](https://linus.nci.nih.gov/techreport/Varma-Simon-CrossValid.pdf)指出，评估调参后的方法应把选择过程放在内层交叉验证中，外层只用于最终评估。

## 3. 训练与推理时裁剪中心分布不一致

训练时以标注点为中心、每轴均匀扰动±160像素生成裁剪中心，二维RMS位移为{jitter_rms:.2f}像素；而原粗定位OOF误差的二维RMS为{coarse_rms:.2f}像素，前者约为后者的{jitter_rms/coarse_rms:.2f}倍。此外训练还有旋转和平移增强，因此训练与测试时裁剪中心的偏移分布并不相同。

随机扰动本身是为了提高鲁棒性，但大范围的均匀扰动可能使模型倾向于在宽范围内重新搜索，而测试时主要需要小幅修正。下一轮可在外层训练折内部生成真实的粗预测，按其误差分布构造大部分训练裁剪，再混入少量大偏移。注意不能直接使用其他外层模型的OOF作为内层训练的裁剪中心，因为这些模型可能已见过当前外层验证图片。

## 4. 结构上缺少全局上下文

当前精修网络只输入单个768像素的原图ROI，没有使用整幅图像、粗定位热图或粗层特征。ZIAN论文指出中央凹等标志点缺乏明显的局部特征，其方法融合了多尺度ROI和粗层上下文；论文的消融实验显示简单裁剪的贡献较小，特征融合更关键。本实现相当于其中的简单裁剪基线。[ZIAN方法与消融](https://arxiv.org/html/2210.02445v2)

因此，加入大尺度ROI或粗定位特征，或把网络改为保守的局部残差修正，是下一步值得验证的方向。

{public_heading}

{public_context}

{public_control}

## 已排除的问题

以下环节均已检查无误：原图与裁剪坐标的往返换算、边缘补黑、X/Y与宽高对应、验证裁剪不读取标注、重复图分在同一折、五折测试坐标平均以及提交文件格式。80张OOF的标注点都落在768像素裁剪范围内，因此数据格式错误或目标未被裁入都不是主要原因。

本次分析未重新加载模型权重查看热图。报告中的RMSE均为开发集指标。20张测试图中公开榜只占50%，样本很少，且没有公开子集名单和标注，无法逐图归因。

## 下一步优先顺序

1. {first_step}
2. 固定外层分组与独立评估规则，把轮次/阈值选择放入训练折内部；同时记录原始精修与回退后两套指标。
3. 做单因素消融：先改变训练裁剪中心分布，再试带粗定位信息的残差/多尺度精修，不同时更换骨干、数据、轮次和融合权重。

本次排查未进行新的训练或提交；原12.31624模型及其提交文件保留。

"""
    # Use measured, rather than hand-rounded, hard-case distances in the report.
    hard = {r["image_id"]: r["raw_distance_px"] for r in details if r["fallback"]}
    report = report.replace("约299.91和399.18像素", f"约{hard[24]:.2f}和{hard[49]:.2f}像素")
    if statistics_result["plot_generated"]:
        report += (
            "\n![训练轮次与回退诊断](refinement_diagnosis_20261027/refinement_diagnosis.png)\n"
        )
    (ROOT / "精修反转原因排查_20261027.md").write_text(report, encoding="utf-8")
    assert "torch" not in __import__("sys").modules
    print(json.dumps(statistics_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
