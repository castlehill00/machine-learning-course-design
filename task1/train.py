"""Task 1 植物分类：多个预训练模型微调后集成预测，生成提交文件。

python train.py

- FULL_TRAIN = False：5 折交叉验证。每折用 80% 训练、20% 验证，500 张图都当过一次验证集，
  得到较可靠的准确率；所有折模型对测试集的概率取平均，写出 submission.csv。
- FULL_TRAIN = True：每个模型直接用全部 500 张训练一次，没有验证分数，快约 5 倍，
  写出 submission_full.csv。
- 下面的常量都可以在命令行覆盖，方便做消融实验，例如：
  python train.py RUN_NAME=no_rotate MODELS=["convnext_tiny"] ROTATE=False
Kaggle：添加比赛数据、开启 GPU 和 Internet（下载预训练权重），直接运行。
"""

import ast
import functools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.ops import StochasticDepth

MODELS = ["convnext_tiny", "efficientnet_v2_s", "swin_t"]  # 结构各异，集成效果更好
FOLDS = 5
EPOCHS = 10
BATCH_SIZE = 16  # 8GB 显存可用
LR = 1e-4
DROP_PATH_SCALE = 1.5  # 把模型自带的随机深度概率放大 1.5 倍，加强正则化
LABEL_SMOOTHING = 0.1
IMG_SIZE = 384  # 分辨率高一些，更容易区分细叶禾草
ROTATE = True  # 训练时随机旋转 90° 的倍数
PRETRAINED = True  # False：随机初始化，只用于消融实验
FULL_TRAIN = False  # 见文件开头说明
REMOVE_BG = False  # 去背景：只保留绿色植物，砂砾、标尺、条形码涂黑（5 折验证未见提升，默认关闭）
SEED = 42  # 训练过程的随机种子；5 折划分固定，不随它变
RUN_NAME = ""  # 非空时只做交叉验证，结果存到 logs/<RUN_NAME>.json，不预测测试集（消融实验用）

# 命令行覆盖上面的常量，例如 EPOCHS=15 或 MODELS=["swin_t"]
for arg in sys.argv[1:]:
    key, _, value = arg.partition("=")
    if key.isupper() and key in globals():
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass  # 不是 Python 字面量，就当作字符串
        globals()[key] = value

random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 路径：自动找到同时含 train/ 和 test/ 的数据目录 ----------
on_kaggle = Path("/kaggle/input").exists()
root = Path("/kaggle/input") if on_kaggle else Path(__file__).parent
out_dir = Path("/kaggle/working") if on_kaggle else Path(__file__).parent
data_dir = next(p.parent for p in root.rglob("train") if (p.parent / "test").is_dir())
print("数据目录:", data_dir, "| 设备:", device)


# ---------- 数据 ----------
def remove_background(img):
    """按 HSV 颜色阈值保留绿色的植物，其余像素涂黑。"""
    if not REMOVE_BG:
        return img
    h, s, v = [np.asarray(c, dtype=int) for c in img.convert("HSV").split()]
    green = (
        (h > 40) & (h < 110) & (s > 50) & (v > 30)
    )  # PIL 的色相范围是 0~255，40~110 约为黄绿到青绿
    mask = Image.fromarray(green.astype(np.uint8) * 255).filter(
        ImageFilter.MaxFilter(5)
    )  # 稍微膨胀，保住叶片边缘
    return Image.composite(img, Image.new("RGB", img.size), mask)


@functools.cache
def load_image(path):
    """读图并去背景；结果缓存在内存里，每张训练图只处理一次（大图去背景很慢）。"""
    return remove_background(Image.open(path).convert("RGB"))


# 俯拍的植物旋转 90° 的倍数或翻转后仍是同一种植物，所以这些增强不会引入错误信息
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
rotate = transforms.RandomChoice([transforms.RandomRotation((a, a)) for a in (0, 90, 180, 270)])
train_tf = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
        rotate if ROTATE else nn.Identity(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        normalize,
    ]
)
eval_tf = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ]
)

# 同一批图片建两份 ImageFolder：训练用增强，验证不用
train_set = datasets.ImageFolder(data_dir / "train", train_tf, loader=load_image)
val_set = datasets.ImageFolder(data_dir / "train", eval_tf, loader=load_image)
classes = train_set.classes
labels = torch.tensor(train_set.targets)

perm = torch.randperm(len(train_set), generator=torch.Generator().manual_seed(42)).tolist()
folds = [perm[i::FOLDS] for i in range(FOLDS)]  # 随机分成 5 份，所有实验用同一种划分

if not RUN_NAME:  # 消融实验不需要测试集
    test_ids = sorted(p.name for p in (data_dir / "test").iterdir())
    test_data = [
        (eval_tf(remove_background(Image.open(data_dir / "test" / f).convert("RGB"))), 0)
        for f in test_ids
    ]
    test_loader = DataLoader(test_data, BATCH_SIZE)
print(f"类别: {classes}\n训练图 {len(train_set)}（{'全量训练' if FULL_TRAIN else f'{FOLDS} 折'}）")


def build_model(name):
    """加载预训练模型，把最后一个全连接层换成 5 类输出。"""
    model = models.get_model(name, weights="DEFAULT" if PRETRAINED else None)
    for m in model.modules():
        if isinstance(m, StochasticDepth):
            m.p *= DROP_PATH_SCALE
    head_name, head = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)][-1]
    model.set_submodule(head_name, nn.Linear(head.in_features, len(classes)))
    return model.to(device)


@torch.no_grad()
def predict_views(model, loader):
    """返回每张图在 8 种视角（4 个旋转角度 × 是否水平翻转）下的类别概率，形状 [图片数, 8, 类别数]。
    对视角取平均就是测试时增强（TTA）。"""
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(device)
        views = [x.rot90(k, (2, 3)) for k in range(4)]
        views += [v.flip(3) for v in views]
        out.append(torch.stack([model(v).softmax(1) for v in views], 1).cpu())
    return torch.cat(out)


def accuracy(probs, idx):
    return (probs[idx].argmax(1) == labels[idx]).float().mean().item()


def train(name, train_idx):
    """训练一个模型，返回模型和每轮的训练 loss。"""
    model = build_model(name)
    loader = DataLoader(Subset(train_set, train_idx), BATCH_SIZE, shuffle=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    losses = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        scheduler.step()
        losses.append(total_loss / len(train_idx))
        print(f"  epoch {epoch:2d} | loss {losses[-1]:.4f}")
    return model, losses


# ---------- 交叉验证：每张训练图都由没见过它的模型预测一次（OOF） ----------
all_idx = list(range(len(train_set)))
oof_all, test_probs = 0, 0
results = {}  # 消融实验要保存的结果
for name in MODELS:
    start = time.time()
    if FULL_TRAIN:
        print(f"[{name}] 全量训练")
        model, _ = train(name, all_idx)
        test_probs += predict_views(model, test_loader).mean(1)
        del model
        torch.cuda.empty_cache()
        continue

    oof_views = torch.zeros(len(train_set), 8, len(classes))
    fold_acc, fold_losses = [], []
    for k, val_idx in enumerate(folds):
        print(f"[{name}] 第 {k + 1}/{FOLDS} 折")
        train_idx = [i for j, fold in enumerate(folds) if j != k for i in fold]
        model, losses = train(name, train_idx)
        oof_views[val_idx] = predict_views(model, DataLoader(Subset(val_set, val_idx), BATCH_SIZE))
        if not RUN_NAME:
            test_probs += predict_views(model, test_loader).mean(1)
        fold_acc.append(accuracy(oof_views.mean(1), val_idx))
        fold_losses.append(losses)
        print(f"[{name}] 第 {k + 1} 折验证准确率 {fold_acc[-1]:.4f}")
        del model
        torch.cuda.empty_cache()

    oof = oof_views.mean(1)
    oof_all += oof
    print(
        f"[{name}] 5 折总准确率 {accuracy(oof, all_idx):.4f}"
        f" | 目前集成 {accuracy(oof_all, all_idx):.4f}"
    )
    results[name] = dict(
        fold_acc=fold_acc,
        fold_losses=fold_losses,
        minutes=(time.time() - start) / 60,
        oof_views=oof_views,
    )

# 混淆矩阵：行是真实类别，列是预测类别
if not FULL_TRAIN:
    confusion = torch.zeros(len(classes), len(classes), dtype=torch.long)
    for t, p in zip(labels, oof_all.argmax(1)):
        confusion[t, p] += 1
    print(pd.DataFrame(confusion.numpy(), index=classes, columns=classes).to_string())

if RUN_NAME:
    # ---------- 消融实验：保存设置、逐折准确率、混淆矩阵；逐视角的 OOF 概率另存 .pt ----------
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    config = {
        k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, list))
    }
    summary = dict(
        config=config,
        classes=classes,
        confusion=confusion.tolist(),
        oof_acc=accuracy(oof_all, all_idx),
        models={n: {k: v for k, v in r.items() if k != "oof_views"} for n, r in results.items()},
    )
    (log_dir / f"{RUN_NAME}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    torch.save(
        dict(labels=labels, folds=folds, oof_views={n: r["oof_views"] for n, r in results.items()}),
        log_dir / f"{RUN_NAME}.pt",
    )
    print("已保存", log_dir / f"{RUN_NAME}.json")
else:
    # ---------- 写出提交文件 ----------
    preds = test_probs.argmax(1)
    submission = pd.DataFrame({"ID": test_ids, "Category": [classes[i] for i in preds]})
    out_path = out_dir / ("submission_full.csv" if FULL_TRAIN else "submission.csv")
    submission.to_csv(out_path, index=False)
    print("已保存", out_path)
    print(submission["Category"].value_counts())
