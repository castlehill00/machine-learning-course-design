"""汇总第一章用到的所有数字，写到 facts.json（图和表都只从这里取数）。

python facts.py   （在装了 torch 的环境里运行，读 task1/logs 下的消融结果）
"""

import json, math, statistics, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import datasets, models

sys.stdout.reconfigure(encoding="utf-8")
TASK1 = Path(__file__).resolve().parents[1]
LOGS = TASK1 / "logs"
OUT = Path(__file__).resolve().parent / "facts.json"
data_dir = next(p.parent for p in TASK1.rglob("train") if (p.parent / "test").is_dir())
train = datasets.ImageFolder(data_dir / "train")
classes = train.classes
labels = torch.tensor(train.targets)
facts = {"classes": classes, "data_dir": str(data_dir)}

# ---------- 数据：数量与图片尺寸 ----------
sizes = [Image.open(p).size for p, _ in train.samples]
test_files = sorted((data_dir / "test").iterdir())
test_sizes = [Image.open(p).size for p in test_files]
per_class = {}
for k, c in enumerate(classes):
    w = [s[0] for s, y in zip(sizes, train.targets) if y == k]
    h = [s[1] for s, y in zip(sizes, train.targets) if y == k]
    per_class[c] = dict(
        n=len(w),
        w_median=statistics.median(w),
        w_min=min(w),
        w_max=max(w),
        square=sum(a == b for a, b in zip(w, h)),
    )
facts["data"] = dict(
    n_train=len(sizes),
    n_test=len(test_files),
    per_class=per_class,
    w_min=min(s[0] for s in sizes),
    w_max=max(s[0] for s in sizes),
    square_train=sum(a == b for a, b in sizes),
    test_w_median=statistics.median(s[0] for s in test_sizes),
    test_w_min=min(s[0] for s in test_sizes),
    test_w_max=max(s[0] for s in test_sizes),
)
# 只凭图片宽度区分两种禾草：在 200 张图上取最好的阈值
bg, lsb = classes.index("Black-grass"), classes.index("Loose Silky-bent")
pairs = [(s[0], y) for s, y in zip(sizes, train.targets) if y in (bg, lsb)]
best = max((sum((w > t) == (y == bg) for w, y in pairs), t) for t in sorted({w for w, _ in pairs}))
facts["data"]["width_rule"] = dict(acc=best[0] / len(pairs), threshold=best[1], n=len(pairs))

# ---------- 候选骨干网络 ----------
facts["backbones"] = {}
for name in ["convnext_tiny", "efficientnet_v2_s", "swin_t"]:
    weights = models.get_model_weights(name).DEFAULT
    m = models.get_model(name, weights=None)
    head = [mod for mod in m.modules() if isinstance(mod, torch.nn.Linear)][-1]
    params = (
        sum(p.numel() for p in m.parameters())
        - head.weight.numel()
        - head.bias.numel()
        + head.in_features * 5
        + 5
    )
    sd = max(
        (
            mod.p
            for mod in m.modules()
            if isinstance(mod, models.convnext.StochasticDepth.__mro__[0])
            or type(mod).__name__ == "StochasticDepth"
        ),
        default=None,
    )
    tf = weights.transforms()
    facts["backbones"][name] = dict(
        params_m=round(params / 1e6, 2),
        imagenet_top1=weights.meta["_metrics"]["ImageNet-1K"]["acc@1"],
        drop_path_default=sd,
        pretrain_crop=tf.crop_size[0],
        weights=str(weights).split(".")[-1],
    )

# ---------- 早期三模型 5 折（09-29，不去背景）与去背景（10-08）：来自当时的运行日志 ----------
facts["three_model"] = {
    "folds": {
        "convnext_tiny": [0.92, 0.94, 0.93, 0.95, 0.91],
        "efficientnet_v2_s": [0.92, 0.88, 0.91, 0.96, 0.92],
        "swin_t": [0.94, 0.94, 0.92, 0.97, 0.94],
    },
    "oof": {
        "convnext_tiny": 0.930,
        "efficientnet_v2_s": 0.918,
        "swin_t": 0.942,
        "ensemble": 0.938,
        "convnext+effnet": 0.928,
    },
    "confusion": [
        [86, 4, 10, 0, 0],
        [3, 94, 0, 0, 3],
        [9, 1, 90, 0, 0],
        [0, 0, 0, 100, 0],
        [0, 1, 0, 0, 99],
    ],
    "mask_oof": {
        "convnext_tiny": 0.942,
        "efficientnet_v2_s": 0.924,
        "swin_t": 0.922,
        "ensemble": 0.934,
    },
    "mask_confusion": [
        [84, 4, 12, 0, 0],
        [4, 95, 0, 0, 1],
        [12, 0, 88, 0, 0],
        [0, 0, 0, 100, 0],
        [0, 0, 0, 0, 100],
    ],
    "holdout_100": {
        "convnext_tiny": 0.91,
        "efficientnet_v2_s": 0.87,
        "swin_t": 0.94,
        "ensemble": 0.90,
    },
}
facts["leaderboard"] = {
    "best": 0.9682,
    "best_correct": round(0.9682 * 378),
    "mask": 0.9444,
    "mask_correct": round(0.9444 * 378),
    "mask_vs_cv_disagree": 18,
}

# ---------- 消融（ConvNeXt-Tiny，5 折，同一划分） ----------
names = [
    "base",
    "seed_1",
    "no_pretrain",
    "img_224",
    "epochs_5",
    "epochs_15",
    "lr_3e-5",
    "lr_3e-4",
    "no_rotate",
    "no_label_smoothing",
    "drop_path_1x",
]
raw = {n: torch.load(LOGS / f"{n}.pt", weights_only=False) for n in names}
info = {n: json.loads((LOGS / f"{n}.json").read_text(encoding="utf-8")) for n in names}
assert all(raw[n]["folds"] == raw["base"]["folds"] for n in names)
assert torch.equal(raw["base"]["labels"], labels)
views = {n: raw[n]["oof_views"]["convnext_tiny"] for n in names}
ok = {n: views[n].mean(1).argmax(1) == labels for n in names}


def mcnemar(a, b):
    x, y = int((a & ~b).sum()), int((~a & b).sum())
    k = x + y
    p = 1.0 if k == 0 else min(1.0, 2 * sum(math.comb(k, i) for i in range(min(x, y) + 1)) / 2**k)
    return x, y, p


VIEW_SETS = {"1": [0], "2": [0, 4], "3": [0, 4, 6], "4": [0, 1, 2, 3], "8": list(range(8))}
runs = {}
for n in names:
    c = info[n]["confusion"]
    m = info[n]["models"]["convnext_tiny"]
    xb, yb, pb = mcnemar(ok["base"], ok[n])
    xs, ys, ps = mcnemar(ok["seed_1"], ok[n])
    runs[n] = dict(
        acc=ok[n].float().mean().item(),
        correct=int(ok[n].sum()),
        vs_base=[xb, yb, pb],
        vs_seed1=[xs, ys, ps],
        grass=c[bg][lsb] + c[lsb][bg],
        recall=[c[i][i] for i in range(5)],
        confusion=c,
        folds=m["fold_acc"],
        minutes=m["minutes"],
        loss=torch.tensor(m["fold_losses"]).mean(0).tolist(),
        config=info[n]["config"],
        tta={
            k: int((views[n][:, v].mean(1).argmax(1) == labels).sum()) for k, v in VIEW_SETS.items()
        },
    )
facts["runs"] = runs
repeats = [0.930, runs["base"]["acc"], runs["seed_1"]["acc"]]
facts["repeats"] = dict(values=repeats, mean=statistics.mean(repeats), sd=statistics.stdev(repeats))
facts["ls_floor"] = -(0.92 * math.log(0.92) + 4 * 0.02 * math.log(0.02))

# ---------- 困难样本：9 个 384 分辨率的预训练配置里的判错次数 ----------
pre384 = [n for n in names if n not in ("no_pretrain", "img_224")]
wrong = sum((~ok[n]).int() for n in pre384)
facts["hard"] = dict(configs=pre384, hist=torch.bincount(wrong, minlength=len(pre384) + 1).tolist())
items = []
for i in torch.nonzero(wrong == len(pre384)).flatten().tolist():
    preds = Counter(classes[int(views[n][i].mean(0).argmax())] for n in pre384)
    p_true = statistics.mean(float(views[n][i].mean(0)[labels[i]]) for n in pre384)
    path = Path(train.samples[i][0])
    items.append(
        dict(
            index=i,
            file=f"{path.parent.name}/{path.name}",
            true=classes[labels[i]],
            preds=dict(preds),
            p_true=round(p_true, 3),
            size=list(sizes[i]),
        )
    )
facts["hard"]["always_wrong"] = items
all_w = [s[0] for s in sizes]
facts["hard"]["w_median_all"] = statistics.median(all_w)
facts["hard"]["w_median_always_wrong"] = statistics.median(it["size"][0] for it in items)
facts["hard"]["w_median_always_right"] = statistics.median(
    all_w[i] for i in range(len(all_w)) if wrong[i] == 0
)


# ---------- 背景线索：画面里的白色标签/标尺，以及取景尺度 ----------
def white_frac(path):
    """低饱和、高亮度像素（白色标签、标尺、花盆边缘）占画面的比例。"""
    im = Image.open(path).convert("RGB")
    im.thumbnail((256, 256))
    h, s, v = [np.asarray(c, dtype=int) for c in im.convert("HSV").split()]
    return float(((s < 35) & (v > 170)).mean())


err = sum((~ok[n]).float() for n in pre384) / len(pre384)  # 每张图在 9 个配置里的平均错误率
ruler = torch.tensor([white_frac(p) > 0.05 for p, _ in train.samples])
width = torch.tensor([s[0] for s in sizes])
t = facts["data"]["width_rule"]["threshold"]
facts["ruler_share"] = {c: ruler[labels == k].float().mean().item() for k, c in enumerate(classes)}
facts["context"] = {}
for k in (bg, lsb):
    m = labels == k
    groups = {
        "ruler": m & ruler,
        "no_ruler": m & ~ruler,
        "small": m & (width <= t),
        "large": m & (width > t),
    }
    facts["context"][classes[k]] = {
        g: dict(n=int(sel.sum()), err=err[sel].mean().item()) for g, sel in groups.items()
    }
for it in items:
    it["ruler"] = bool(ruler[it["index"]])

OUT.write_text(json.dumps(facts, ensure_ascii=False, indent=1), encoding="utf-8")
print("已写出", OUT)
print(
    json.dumps(
        {k: facts[k] for k in ("data", "backbones", "repeats", "ls_floor")},
        ensure_ascii=False,
        indent=1,
    )
)
print("TTA:", {n: r["tta"] for n, r in runs.items()})
print(
    "hard hist",
    facts["hard"]["hist"],
    "| 宽度中位数 全部/总错/总对:",
    facts["hard"]["w_median_all"],
    facts["hard"]["w_median_always_wrong"],
    facts["hard"]["w_median_always_right"],
)
for it in items:
    print(it)
