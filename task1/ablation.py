"""Task 1 消融实验：在 ConvNeXt-Tiny 上做 5 折交叉验证，每次只改一个设置，其余与最终方案相同。

python ablation.py           依次运行还没跑过的实验（可以随时中断，重新运行会跳过已完成的）
python ablation.py 7         只运行列表里的前 7 个实验
python ablation.py summary   汇总已完成的实验，打印结果表

每个实验调用 train.py（RUN_NAME 模式），结果在 logs/<实验名>.json / .pt / .log。
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
LOGS = HERE / "logs"
COMMON = {"MODELS": ["convnext_tiny"]}  # 所有消融都只用 ConvNeXt-Tiny
EXPERIMENTS = {  # 实验名: (相对最终方案的改动, 说明)
    "base": ({}, "最终方案"),
    "no_pretrain": ({"PRETRAINED": False}, "不用 ImageNet 预训练"),
    "no_rotate": ({"ROTATE": False}, "去掉 90° 旋转增强"),
    "no_label_smoothing": ({"LABEL_SMOOTHING": 0.0}, "去掉标签平滑"),
    "drop_path_1x": ({"DROP_PATH_SCALE": 1.0}, "随机深度用默认强度"),
    "img_224": ({"IMG_SIZE": 224}, "输入 224×224"),
    "seed_1": ({"SEED": 1}, "最终方案，换随机种子"),
    "epochs_5": ({"EPOCHS": 5}, "训练 5 轮"),
    "epochs_15": ({"EPOCHS": 15}, "训练 15 轮"),
    "lr_3e-5": ({"LR": 3e-5}, "学习率 3e-5"),
    "lr_3e-4": ({"LR": 3e-4}, "学习率 3e-4"),
    "seed_2": ({"SEED": 2}, "最终方案，换随机种子"),
}
VIEW_SETS = {
    "1 视角（原图）": [0],
    "3 视角（原图+水平+垂直翻转）": [0, 4, 6],
    "8 视角（4 旋转×翻转）": list(range(8)),
}


def run_pending(limit=None):
    LOGS.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for name, (changes, note) in list(EXPERIMENTS.items())[:limit]:
        if (LOGS / f"{name}.json").exists():
            print(f"跳过已完成的 {name}")
            continue
        args = [f"RUN_NAME={name}"] + [f"{k}={v!r}" for k, v in {**COMMON, **changes}.items()]
        print(f"=== {name}（{note}）: {' '.join(args[1:])}", flush=True)
        for attempt in (1, 2, 3):  # 偶发崩溃（如内存紧张）时自动重试
            with open(LOGS / f"{name}.log", "w", encoding="utf-8") as log:
                code = subprocess.call(
                    [sys.executable, "-u", str(HERE / "train.py"), *args],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=HERE,
                )
            if code == 0:
                break
            print(f"    第 {attempt} 次运行失败，退出码 {code}", flush=True)
        else:
            sys.exit(f"{name} 失败，见 logs/{name}.log")
        print(
            f"    完成：OOF 准确率 {json.loads((LOGS / f'{name}.json').read_text(encoding='utf-8'))['oof_acc']:.4f}",
            flush=True,
        )


def mcnemar_p(a_correct, b_correct):
    """精确 McNemar 检验（两组预测在同一批样本上的配对比较），返回双侧 p 值。"""
    b = int((a_correct & ~b_correct).sum())
    c = int((~a_correct & b_correct).sum())
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def summary():
    done = [n for n in EXPERIMENTS if (LOGS / f"{n}.json").exists()]
    if "base" not in done:
        sys.exit("还没有 base 的结果")
    raw = {n: torch.load(LOGS / f"{n}.pt") for n in done}
    labels = raw["base"]["labels"]
    correct = {n: raw[n]["oof_views"]["convnext_tiny"].mean(1).argmax(1) == labels for n in done}

    print("| 实验 | 改动 | OOF 准确率 | 与最终方案差 | McNemar p | 两种禾草互认 | 耗时(分) |")
    print("|---|---|---|---|---|---|---|")
    base_acc = correct["base"].float().mean().item()
    for n in done:
        info = json.loads((LOGS / f"{n}.json").read_text(encoding="utf-8"))
        acc = correct[n].float().mean().item()
        conf = info["confusion"]
        grass = conf[0][2] + conf[2][0]  # Black-grass ↔ Loose Silky-bent
        minutes = info["models"]["convnext_tiny"]["minutes"]
        diff = "—" if n == "base" else f"{(acc - base_acc) * 100:+.1f}"
        p = "—" if n == "base" else f"{mcnemar_p(correct['base'], correct[n]):.3f}"
        print(f"| {n} | {EXPERIMENTS[n][1]} | {acc:.3f} | {diff} | {p} | {grass} | {minutes:.0f} |")

    seeds = [correct[n].float().mean().item() for n in ("base", "seed_1", "seed_2") if n in correct]
    if len(seeds) > 1:
        mean = sum(seeds) / len(seeds)
        sd = (sum((s - mean) ** 2 for s in seeds) / (len(seeds) - 1)) ** 0.5
        print(
            f"\n同一方案换随机种子：{', '.join(f'{s:.3f}' for s in seeds)}；均值 {mean:.3f}，标准差 {sd * 100:.2f} 个百分点"
        )

    print("\n测试时增强（base 的同一批模型，只改预测时的视角数）：")
    views = raw["base"]["oof_views"]["convnext_tiny"]
    for label, idx in VIEW_SETS.items():
        acc = (views[:, idx].mean(1).argmax(1) == labels).float().mean().item()
        print(f"  {label}: {acc:.3f}")


if __name__ == "__main__":
    if sys.argv[1:] == ["summary"]:
        summary()
    else:
        run_pending(int(sys.argv[1]) if sys.argv[1:] else None)
