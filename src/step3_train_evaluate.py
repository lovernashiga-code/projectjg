"""Step 3: train Softmax classifiers on frozen features and evaluate every partial-face variant.

  python step3_train_evaluate.py                     # LFW, the 70/30 split with seed 42 used in the slides
  python step3_train_evaluate.py --dataset fei
  python step3_train_evaluate.py --seeds 0 1 2 3 4   # also repeat on 5 other random splits (mean +/- std)
  python step3_train_evaluate.py --dataset custom --fit-all   # demo model trained on every image

Methods
  M0  Eigenfaces (PCA) + SVM            trained on full faces        (classical baseline)
  M1  VGG-Face + Softmax                trained on full faces
  M2  VGG-Face + Softmax                trained on full + partial faces (transfer learning on partial data)
  M3  FaceNet(VGGFace2) + Softmax       trained on full faces
  M4  FaceNet(VGGFace2) + Softmax       trained on full + partial faces

Outputs in <results>/
  metrics.json        method x variant: accuracy, macro precision/recall/F1, ROC-AUC, top-5, coverage@tau
  targets.json        targets T1-T3 from slide 5 checked against the measured accuracies
  predictions.csv     every test image x method x variant: true name, prediction, confidence
  top_confusions.csv  most frequent (true -> predicted) mistakes per method x variant
  probs.npz, split.npz          used by step 4
  metrics_seeds.json  with --seeds: mean and std over the repeated splits
and models/<dataset>_<M1..M4>.joblib, the trained classifiers used by predict.py, webcam.py and app.py.
"""
import argparse
import csv
import json
from collections import Counter

import cv2
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, top_k_accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

from common import MODELS, OCCLUDED, SEED, TAU, VARIANTS, apply_variant, out_dir

METHODS = {
    "M0": "Eigenfaces + SVM (Full)",
    "M1": "VGG-Face + Softmax (Full)",
    "M2": "VGG-Face + Softmax (Full+Partial)",
    "M3": "FaceNet + Softmax (Full)",
    "M4": "FaceNet + Softmax (Full+Partial)",
}
# deep methods: backbone and which variants of the training images they learn from
DEEP = {"M1": ("vggface", ["full"]), "M2": ("vggface", list(VARIANTS)),
        "M3": ("facenet", ["full"]), "M4": ("facenet", list(VARIANTS))}


def metrics(y, prob, classes):
    pred = classes[prob.argmax(1)]
    accepted = prob.max(1) >= TAU
    try:
        auc = roc_auc_score(y, prob[:, 1] if len(classes) == 2 else prob,
                            multi_class="ovr", average="macro", labels=classes)
    except ValueError:      # a class missing from a tiny test set
        auc = float("nan")
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, average="macro", zero_division=0),
        "recall": recall_score(y, pred, average="macro", zero_division=0),
        "f1": f1_score(y, pred, average="macro", zero_division=0),
        "auc": auc,
        "top5": top_k_accuracy_score(y, prob, k=min(5, len(classes)), labels=classes),
        "coverage@tau": accepted.mean(),
        "accuracy@tau": (pred[accepted] == y[accepted]).mean() if accepted.any() else float("nan"),
    }


def softmax_clf():
    return LogisticRegression(C=10, max_iter=3000)


def eigenfaces_clf(n_train):
    return make_pipeline(PCA(min(150, n_train - 1), whiten=True, random_state=SEED),
                         SVC(C=10, gamma="scale", break_ties=True))


def svm_scores(clf, X):
    """Softmax over the SVM's one-vs-rest decision values: the argmax is exactly clf.predict(X).
    (Platt-scaled predict_proba is fitted by internal cross-validation and can disagree with predict.)"""
    z = clf.decision_function(X)
    z = np.stack([-z, z], 1) if z.ndim == 1 else z
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def gray(faces, v):
    return np.stack([cv2.resize(cv2.cvtColor(apply_variant(f, v), cv2.COLOR_RGB2GRAY), (64, 64))
                     for f in faces]).reshape(len(faces), -1) / 255.0


def fit_deep(m, feats, y, idx):
    bb, used = DEEP[m]
    F = feats[bb]
    return softmax_clf().fit(np.concatenate([F[v][idx] for v in used]), np.tile(y[idx], len(used)))


def run_split(faces, y, feats, seed):
    """Train every method on one stratified 70/30 split and test it on every variant."""
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed)
    results, probs, clfs = {}, {}, {}

    def evaluate(m, clf, test_x, scores=lambda c, X: c.predict_proba(X)):
        for v in VARIANTS:
            p = scores(clf, test_x(v))
            results[(m, v)] = metrics(y[idx_te], p, clf.classes_)
            probs[f"{m}|{v}"] = p

    clf = eigenfaces_clf(len(idx_tr)).fit(gray(faces[idx_tr], "full"), y[idx_tr])
    evaluate("M0", clf, lambda v: gray(faces[idx_te], v), svm_scores)
    print(f"[seed {seed}] M0 done", flush=True)
    for m, (bb, _) in DEEP.items():
        if bb not in feats:
            continue
        clfs[m] = fit_deep(m, feats, y, idx_tr)
        evaluate(m, clfs[m], lambda v: feats[bb][v][idx_te])
        print(f"[seed {seed}] {m} done", {v: round(results[(m, v)]["accuracy"], 3) for v in VARIANTS}, flush=True)
    return idx_tr, idx_te, results, probs, clfs


def check_targets(results):
    """T1-T3 (slide 5), measured on VGG-Face; T3 compares training (ข) M2 against (ก) M1 (slide 11)."""
    if ("M1", "full") not in results:
        return {}
    acc = lambda m, v: float(results[(m, v)]["accuracy"])          # noqa: E731
    worst = min(acc("M1", v) for v in OCCLUDED)
    hardest = [v for v in OCCLUDED if np.isclose(acc("M1", v), worst)]   # ties are all reported
    t2 = {v: acc("M2", v) for v in ("mask", "sunglasses")}
    t3 = {v: acc("M2", v) - acc("M1", v) for v in hardest}
    return {
        "T1": {"desc": "Accuracy on full faces, M2", "target": 0.95,
               "value": acc("M2", "full"), "pass": acc("M2", "full") >= 0.95},
        "T2": {"desc": "Accuracy with mask / sunglasses, M2", "target": 0.85,
               "value": t2, "pass": min(t2.values()) >= 0.85},
        "T3": {"desc": "Acc(M2) - Acc(M1) on the variant where M1 is worst", "target": 0.10,
               "M1_worst_variant": hardest, "value": t3, "pass": min(t3.values()) >= 0.10},
    }


def write_predictions(path, probs, y, idx_te, names, files):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "variant", "index", "file", "true", "predicted", "confidence", "correct", "accepted"])
        for key, p in probs.items():
            m, v = key.split("|")
            for i, row in zip(idx_te, p):
                k = int(row.argmax())
                w.writerow([m, v, int(i), files[i], names[y[i]], names[k], f"{row[k]:.4f}",
                            int(k == y[i]), int(row[k] >= TAU)])


def write_confusions(path, probs, y_te, names, top=5):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "variant", "true", "predicted", "count"])
        for key, p in probs.items():
            pred = p.argmax(1)
            pairs = Counter((t, q) for t, q in zip(y_te, pred) if t != q)
            for (t, q), c in pairs.most_common(top):
                w.writerow([*key.split("|"), names[t], names[q], c])


def save_models(dataset, clfs, names, fit_on):
    MODELS.mkdir(exist_ok=True)
    for m, clf in clfs.items():
        bb, used = DEEP[m]
        path = MODELS / f"{dataset}_{m}.joblib"
        joblib.dump({"method": m, "method_name": METHODS[m], "dataset": dataset, "backbone": bb,
                     "trained_on_variants": used, "fit_on": fit_on, "classifier": clf,
                     "names": [str(n) for n in names], "tau": TAU}, path)
        print("saved", path)


def print_table(results, key="accuracy"):
    methods = sorted({m for m, _ in results})
    print(f"\n{key} (%)".ljust(14) + "".join(f"{m:>8}" for m in methods))
    for v in VARIANTS:
        print(v.ljust(14) + "".join(f"{100 * results[(m, v)][key]:8.1f}" for m in methods))


def main(dataset, seeds, fit_all):
    out = out_dir(dataset)
    d = np.load(out / "faces.npz")
    faces, y, names = d["faces"], d["labels"], [str(n) for n in d["names"]]
    files = d["files"] if "files" in d else np.full(len(y), "")
    feats = {}
    for bb in ("vggface", "facenet"):
        if (out / f"features_{bb}.npz").exists():
            F = np.load(out / f"features_{bb}.npz")
            feats[bb] = {v: normalize(F[v]) for v in VARIANTS}
        else:
            print(f"features_{bb}.npz not found: skipping its methods")

    idx_tr, idx_te, results, probs, clfs = run_split(faces, y, feats, SEED)
    np.savez(out / "split.npz", train=idx_tr, test=idx_te)
    np.savez_compressed(out / "probs.npz", y_test=y[idx_te], idx_test=idx_te, **probs)
    rows = [{"method": m, "method_name": METHODS[m], "variant": v, **{k: round(float(x), 4) for k, x in r.items()}}
            for (m, v), r in results.items()]
    (out / "metrics.json").write_text(json.dumps(rows, indent=2))
    write_predictions(out / "predictions.csv", probs, y, idx_te, names, files)
    write_confusions(out / "top_confusions.csv", probs, y[idx_te], names)
    targets = check_targets(results)
    (out / "targets.json").write_text(json.dumps(targets, indent=2, ensure_ascii=False))
    print_table(results)
    for t, r in targets.items():
        print(f"{t} {'PASS' if r['pass'] else 'FAIL'}: {r['desc']} = {r['value']} (target {r['target']})")

    if fit_all:   # demo model: same recipe, every image of every person
        clfs = {m: fit_deep(m, feats, y, np.arange(len(y))) for m in clfs}
    save_models(dataset, clfs, names, "all images" if fit_all else f"70% train split (seed {SEED})")

    if seeds:
        per = {}
        for s in seeds:
            r = results if s == SEED else run_split(faces, y, feats, s)[2]
            for k, mtr in r.items():
                per.setdefault(k, []).append(mtr)
        summary = [{"method": m, "method_name": METHODS[m], "variant": v, "seeds": seeds,
                    **{f"{k}_{f}": round(float(fn([r[k] for r in lst])), 4)
                       for k in lst[0] for f, fn in (("mean", np.nanmean), ("std", np.nanstd))}}
                   for (m, v), lst in per.items()]
        (out / "metrics_seeds.json").write_text(json.dumps(summary, indent=2))
        print_table({k: {"accuracy": np.mean([r["accuracy"] for r in lst])} for k, lst in per.items()})
        print(f"(mean over seeds {seeds})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="lfw")
    ap.add_argument("--seeds", type=int, nargs="*", default=[], help="extra random splits for mean +/- std")
    ap.add_argument("--fit-all", action="store_true",
                    help="save classifiers refit on every image (for the webcam demo) instead of the train split")
    a = ap.parse_args()
    main(a.dataset, a.seeds, a.fit_all)
