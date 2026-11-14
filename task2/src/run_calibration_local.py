"""Cross-fit low-parameter calibration on cached holdout crop-TTA probabilities."""

from pathlib import Path
import hashlib
import json
import math
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ROOT = ROOT.parent if ROOT.name == "src" else ROOT  # 位于 src/ 时回到任务根目录
SOURCE = ROOT / "task2_multicrop_tta_seed42"
OUT = ROOT / "task2_calibration_seed42"
CLASSES = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
OFFICIAL = np.array([0, 1, 2, 5, 3, 4])
METHODS = ["baseline", "weight_only", "temperature_equal", "temperature_weight"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def load_cached(tag, paths):
    meta = json.loads((SOURCE / (tag + ".json")).read_text())
    assert meta["classes"] == CLASSES and meta["crop"] == 44
    assert meta["paths_sha256"] == hashlib.sha256("\n".join(paths).encode()).hexdigest()
    a = np.load(SOURCE / (tag + ".npz"))["probabilities"].astype(float)
    assert a.shape == (len(paths), 6, 6)
    assert np.isfinite(a).all() and (a >= 0).all() and np.allclose(a.sum(2), 1, atol=1e-5)
    return a.mean(1), meta


def scale(p, temperature):
    if temperature == 1.0:
        return p.copy()
    log = np.log(p.clip(1e-12, 1)) / temperature
    exp = np.exp(log - log.max(1, keepdims=True))
    return exp / exp.sum(1, keepdims=True)


def nll(p, y):
    return float(-np.log(p[np.arange(len(y)), y].clip(1e-12, 1)).mean())


def bounded_minimize(func, low, high):
    a, b = low, high
    g = (math.sqrt(5) - 1) / 2
    c = b - g * (b - a)
    d = a + g * (b - a)
    fc = func(c)
    fd = func(d)
    for _ in range(55):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = func(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = func(d)
    return min([low, high, (a + b) / 2], key=func)


def fit(method, b, r, y):
    tb = tr = 1.0
    w = 0.5
    if "temperature" in method:
        tb = math.exp(
            bounded_minimize(
                lambda logt: nll(scale(b, math.exp(logt)), y), math.log(0.5), math.log(4.0)
            )
        )
        tr = math.exp(
            bounded_minimize(
                lambda logt: nll(scale(r, math.exp(logt)), y), math.log(0.5), math.log(4.0)
            )
        )
    if method in ["weight_only", "temperature_weight"]:
        bp, rp = scale(b, tb), scale(r, tr)
        w = bounded_minimize(lambda alpha: nll(alpha * bp + (1 - alpha) * rp, y), 0.25, 0.75)
    params = {"b2_temperature": tb, "resnet18_temperature": tr, "b2_weight": w}
    if "bias" in method:
        p = w * scale(b, tb) + (1 - w) * scale(r, tr)
        params["class_bias"] = fit_bias(p, y).tolist()
    return params


def fit_bias(p, y):
    logp = np.log(p.clip(1e-12, 1))
    bias = np.zeros(6)
    target = np.bincount(y, minlength=6) / len(y)

    def objective(v):
        z = logp + v
        e = np.exp(z - z.max(1, keepdims=True))
        q = e / e.sum(1, keepdims=True)
        return nll(q, y) + 0.05 / 2 * (v * v).sum(), q

    for _ in range(40):
        old, q = objective(bias)
        grad = q.mean(0) - target + 0.05 * bias
        hessian = np.diag(q.mean(0)) - q.T @ q / len(y) + 0.05 * np.eye(6)
        direction = np.linalg.solve(hessian, grad)
        step = 1.0
        while step > 1e-8:
            candidate = np.clip(bias - step * direction, -0.35, 0.35)
            if objective(candidate)[0] <= old + 1e-12:
                break
            step *= 0.5
        if np.max(abs(candidate - bias)) < 1e-9:
            break
        bias = candidate
    assert np.isfinite(bias).all() and (abs(bias) <= 0.35).all()
    return bias


def apply(b, r, parameters):
    p = parameters["b2_weight"] * scale(b, parameters["b2_temperature"]) + (
        1 - parameters["b2_weight"]
    ) * scale(r, parameters["resnet18_temperature"])
    if "class_bias" in parameters:
        log = np.log(p.clip(1e-12, 1)) + np.array(parameters["class_bias"])
        exp = np.exp(log - log.max(1, keepdims=True))
        p = exp / exp.sum(1, keepdims=True)
    return p


def metrics(p, y):
    cm = np.zeros((6, 6), dtype=int)
    np.add.at(cm, (y, p.argmax(1)), 1)
    return {
        "accuracy": float(np.trace(cm) / len(y)),
        "correct": int(np.trace(cm)),
        "nll": nll(p, y),
        "macro_f1": float((2 * np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1), 1)).mean()),
    }


def main():
    OUT.mkdir(exist_ok=True)
    protocol = {
        "candidates": METHODS,
        "calibration": "softmax(log(mean crop probabilities)/T), per model",
        "temperature_range": [0.5, 4.0],
        "b2_weight_range": [0.25, 0.75],
        "fitting_objective": "NLL; temperatures fit separately; optional fusion weight fit after temperatures",
        "cross_fit": "5 deterministic folds by pixel_sha256 group hash, no group crosses folds",
        "selection": "highest out-of-fold accuracy, then lower NLL, then simpler method",
        "acceptance": "at least 10 extra correct overall and positive gain in at least 4 of 5 folds",
        "seed": 42,
        "warning": "Architecture and TTA selected previously using same development set; not an independent test estimate.",
    }
    if any("bias" in m for m in METHODS):
        protocol.update(
            class_bias_range=[-0.35, 0.35],
            bias_l2=0.05,
            bias_fitting="penalized NLL on calibration folds, after optional per-model temperatures",
        )
    write_json(OUT / "protocol.json", protocol)
    split = pd.read_csv(SOURCE / "split.csv")
    val = split[split.fold == 0].reset_index(drop=True)
    paths = val.path.tolist()
    y = np.array([CLASSES.index(c) for c in val["class"]])
    assert len(val) == 5656
    b, bmeta = load_cached("validation_b2", paths)
    r, rmeta = load_cached("validation_resnet18", paths)
    original = json.loads((ROOT / "task2_ensemble_b2_resnet18/analysis.json").read_text())
    assert bmeta["checkpoint_sha256"] == original["b2_checkpoint_sha256"]
    assert rmeta["checkpoint_sha256"] == original["resnet18_checkpoint_sha256"]
    folds = np.array(
        [
            int(hashlib.sha256(("calibration42:" + h).encode()).hexdigest()[:12], 16) % 5
            for h in val.pixel_sha256
        ]
    )
    assert (
        pd.DataFrame({"group": val.pixel_sha256, "fold": folds})
        .groupby("group")
        .fold.nunique()
        .max()
        == 1
    )
    val.assign(calibration_fold=folds).to_csv(OUT / "validation_folds.csv", index=False)
    predictions = {m: np.zeros_like(b) for m in METHODS}
    parameters = []
    fold_rows = []
    for fold in range(5):
        train, test = folds != fold, folds == fold
        assert len(set(y[train])) == len(set(y[test])) == 6
        for method in METHODS:
            params = fit(method, b[train], r[train], y[train])
            pred = apply(b[test], r[test], params)
            predictions[method][test] = pred
            parameters.append({"fold": fold, "method": method, **params})
            fold_rows.append(
                {
                    "fold": fold,
                    "method": method,
                    "samples": int(test.sum()),
                    **metrics(pred, y[test]),
                }
            )
    baseline = metrics(predictions["baseline"], y)
    comparisons = []
    for method in METHODS:
        row = {"method": method, **metrics(predictions[method], y)}
        row["gain_correct"] = row["correct"] - baseline["correct"]
        row["gain_pp"] = 100 * (row["accuracy"] - baseline["accuracy"])
        gains = [
            next(v["correct"] for v in fold_rows if v["fold"] == fold and v["method"] == method)
            - next(
                v["correct"] for v in fold_rows if v["fold"] == fold and v["method"] == "baseline"
            )
            for fold in range(5)
        ]
        row["positive_folds"] = sum(v > 0 for v in gains)
        comparisons.append(row)
    ranked = sorted(
        comparisons, key=lambda m: (-m["accuracy"], m["nll"], METHODS.index(m["method"]))
    )
    accepted = [x for x in ranked if x["gain_correct"] >= 10 and x["positive_folds"] >= 4]
    selected = accepted[0] if accepted else None
    pd.DataFrame(comparisons).to_csv(OUT / "oof_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "fold_comparison.csv", index=False)
    pd.DataFrame(parameters).to_csv(OUT / "fitted_parameters.csv", index=False)
    np.savez_compressed(OUT / "validation_oof_probabilities.npz", **predictions)
    report = {
        "protocol": protocol,
        "baseline": baseline,
        "comparisons": comparisons,
        "selected": selected,
        "holdout_sources": {"b2": bmeta, "resnet18": rmeta},
        "leaderboard_score": None,
    }
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)
    if selected is not None:
        method = selected["method"]
        params = fit(method, b, r, y)
        report["fitted_all_validation_parameters"] = params
        template = pd.read_csv(ROOT / "2026task2/submission.csv", dtype={"ID": str})
        test_paths = ["test/" + s for s in template.ID]
        bt, btmeta = load_cached("test_b2_fulltrain", test_paths)
        rt, rtmeta = load_cached("test_resnet18_fulltrain", test_paths)
        prob = apply(bt, rt, params)
        assert np.isfinite(prob).all() and np.allclose(prob.sum(1), 1, atol=1e-8)
        current = pd.read_csv(SOURCE / "submission_multicrop_fusion.csv")
        assert current.ID.tolist() == template.ID.tolist()
        submission = pd.DataFrame({"ID": template.ID, "Emotion": OFFICIAL[prob.argmax(1)]})
        path = OUT / "submission_calibrated_fusion.csv"
        submission.to_csv(path, index=False)
        assert pd.read_csv(path).equals(submission)
        probability = pd.DataFrame(prob, columns=CLASSES)
        probability.insert(0, "ID", template.ID)
        probability.to_csv(OUT / "test_probabilities.csv", index=False)
        report.update(
            submission=str(path),
            submission_sha256=sha(path),
            test_rows=len(template),
            changed_vs_0734=int((submission.Emotion != current.Emotion).sum()),
            full_model_sources={"b2": btmeta, "resnet18": rtmeta},
        )
        print("EXPORTED", path, params, flush=True)
    else:
        print(
            "No candidate passed the predeclared cross-fit gate. No submission exported.",
            flush=True,
        )
    write_json(OUT / "analysis.json", report)


if __name__ == "__main__":
    if "--bias" in sys.argv:
        OUT = ROOT / "task2_bias_calibration_seed42"
        METHODS = ["baseline", "bias_only", "temperature_bias"]
    main()
