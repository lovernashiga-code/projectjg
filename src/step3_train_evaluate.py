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
  M1+aug ... M4+aug  the same, plus the augmented copies of the training images (slide 6), when step 2
                     was run with --augment K. The test images and their copies are never trained on.

Split: 70/30 stratified by image, so every person is in both sets but no photo is (its occluded
copies stay on the same side). People recorded in two or more capture sessions (custom webcam data)
instead keep one whole session as their test set, because consecutive webcam frames are near-duplicates.

Outputs in <results>/
  metrics.json        method x variant: accuracy, macro precision/recall/F1, ROC-AUC, top-5, coverage@tau
  open_set.json       with strangers (unknown.npz): over tau, how many known test faces get the right name,
                      a wrong name or "Unknown", and how many strangers wrongly get a name
  targets.json        targets T1-T3 from slide 5 checked against the measured accuracies
  predictions.csv     every test image x method x variant: true name, prediction, confidence
  top_confusions.csv  most frequent (true -> predicted) mistakes per method x variant
  probs.npz, split.npz          used by step 4 (split.npz also says how the split was made)
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
AUG = "+aug"     # suffix: also trained on the augmented copies of the training images
METHODS.update({m + AUG: METHODS[m] + " + augmentation" for m in DEEP})


def backbone(m):
    return DEEP[m.removesuffix(AUG)][0]


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


def fit_deep(m, feats, aug, y, idx):
    """Softmax on the training images idx; an AUG method also gets every augmented copy of those images."""
    bb, used = DEEP[m.removesuffix(AUG)]
    X = [feats[bb][v][idx] for v in used]
    if m.endswith(AUG):
        X += [copy[idx] for v in used for copy in aug[bb][v]]
    return softmax_clf().fit(np.concatenate(X), np.tile(y[idx], len(X)))


def make_split(y, sessions, names, seed):
    """70/30 stratified split of the images. A person with photos from two or more capture sessions keeps
    one whole session (chosen at random) as test data instead, so near-identical webcam frames never end
    up on both sides. Returns train indices, test indices and a one-line description."""
    rng = np.random.default_rng(seed)
    test, by_session, single = np.zeros(len(y), bool), np.zeros(len(y), bool), []
    for c in np.unique(y):
        own = np.where(y == c)[0]
        ss = sorted(set(sessions[own]))
        if "" in ss:            # LFW / FEI: no sessions
            continue
        if len(ss) < 2:
            single.append(names[c])
            continue
        by_session[own] = True
        test[own[sessions[own] == ss[rng.integers(len(ss))]]] = True
    if single:
        print(f"only one capture session for {', '.join(single)}: their 70/30 split mixes near-duplicate frames "
              "and overestimates accuracy; record a second session with capture_faces.py")
    if not by_session.any():   # the original split, unchanged
        idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed)
        return idx_tr, idx_te, "70/30 stratified split of the images"
    rest = np.where(~by_session)[0]
    idx_tr, idx_te = np.where(by_session & ~test)[0], np.where(test)[0]
    if len(rest):
        r_tr, r_te = train_test_split(rest, test_size=0.3, stratify=y[rest], random_state=seed)
        idx_tr, idx_te = np.concatenate([r_tr, idx_tr]), np.concatenate([r_te, idx_te])
    n = len(np.unique(y[by_session]))
    return idx_tr, idx_te, (f"one held-out capture session per person for {n} people"
                            + (", 70/30 stratified split of the images for the rest" if len(rest) else ""))


def run_split(faces, y, sessions, names, feats, aug, seed):
    """Train every method on one split and test it on every variant."""
    idx_tr, idx_te, how = make_split(y, sessions, names, seed)
    results, probs, clfs = {}, {}, {}

    def evaluate(m, clf, test_x, scores=lambda c, X: c.predict_proba(X)):
        for v in VARIANTS:
            p = scores(clf, test_x(v))
            results[(m, v)] = metrics(y[idx_te], p, clf.classes_)
            probs[f"{m}|{v}"] = p

    clf = eigenfaces_clf(len(idx_tr)).fit(gray(faces[idx_tr], "full"), y[idx_tr])
    evaluate("M0", clf, lambda v: gray(faces[idx_te], v), svm_scores)
    print(f"[seed {seed}] M0 done", flush=True)
    for m in [m for base in DEEP for m in (base, base + AUG)]:
        bb = backbone(m)
        if bb not in feats or (m.endswith(AUG) and bb not in aug):
            continue
        clfs[m] = fit_deep(m, feats, aug, y, idx_tr)
        evaluate(m, clfs[m], lambda v: feats[bb][v][idx_te])
        print(f"[seed {seed}] {m} done", {v: round(results[(m, v)]["accuracy"], 3) for v in VARIANTS}, flush=True)
    return idx_tr, idx_te, results, probs, clfs, how


TAUS = np.round(np.arange(0, 1, 0.05), 2)


def open_set(clf, f_known, y_known, f_strangers):
    """Over tau: shares of known test faces given the right name / a wrong name / "Unknown",
    and the share of strangers (never in training) wrongly given a name."""
    pk, pu = clf.predict_proba(f_known), clf.predict_proba(f_strangers)
    ok = clf.classes_[pk.argmax(1)] == y_known
    ck, cu = pk.max(1), pu.max(1)
    return [{"tau": float(t),
             "known_correct": float((ok & (ck >= t)).mean()),
             "known_wrong": float((~ok & (ck >= t)).mean()),
             "known_unknown": float((ck < t).mean()),
             "stranger_accepted": float((cu >= t).mean())} for t in TAUS]


def tau_for(rows, max_strangers=0.05):
    """Smallest tau on the grid at which at most this share of strangers gets a name."""
    return next((r for r in rows if r["stranger_accepted"] <= max_strangers), None)


def run_open_set(out, clfs, feats, y, idx_te):
    """open_set.json for every deep method x variant; prints the full / mask summary."""
    stranger = {}
    for bb in {backbone(m) for m in clfs}:
        if (out / f"unknown_features_{bb}.npz").exists():
            F = np.load(out / f"unknown_features_{bb}.npz")
            stranger[bb] = {v: normalize(F[v]) for v in VARIANTS}
    if not stranger:
        print("no unknown_features_*.npz: skipping the stranger test (see step 1 and 2)")
        return {}
    rows = {(m, v): open_set(clf, feats[backbone(m)][v][idx_te], y[idx_te], stranger[backbone(m)][v])
            for m, clf in clfs.items() if backbone(m) in stranger for v in VARIANTS}
    n = len(next(iter(stranger.values()))["full"])
    (out / "open_set.json").write_text(json.dumps(
        {"n_strangers": n, "n_known_test": int(len(idx_te)), "tau_default": TAU,
         "rows": [{"method": m, "variant": v, **r} for (m, v), rs in rows.items() for r in rs]}, indent=1))
    print(f"\nstrangers: {n} photos of people never seen in training. 'Accepted' = given a name, not Unknown")
    at = int(np.argmin(np.abs(TAUS - TAU)))
    for (m, v), rs in rows.items():
        if v in ("full", "mask"):
            best = tau_for(rs)
            pick = (f"tau {best['tau']:.2f}: strangers {100 * best['stranger_accepted']:5.1f}%, known right "
                    f"{100 * best['known_correct']:5.1f}%" if best else "no tau keeps strangers <= 5%")
            print(f"{m:7} {v:5}  tau {TAU}: strangers accepted {100 * rs[at]['stranger_accepted']:5.1f}%, "
                  f"known right {100 * rs[at]['known_correct']:5.1f}%  |  {pick}")
    return rows


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


def save_models(dataset, clfs, names, fit_on, open_rows):
    MODELS.mkdir(exist_ok=True)
    for m, clf in clfs.items():
        bb, used = DEEP[m.removesuffix(AUG)]
        path = MODELS / f"{dataset}_{m}.joblib"
        best = tau_for(open_rows.get((m, "full"), []))
        joblib.dump({"method": m, "method_name": METHODS[m], "dataset": dataset, "backbone": bb,
                     "trained_on_variants": used, "fit_on": fit_on, "classifier": clf,
                     "names": [str(n) for n in names], "tau": TAU,
                     "tau_strangers_5pct": best["tau"] if best else None}, path)
        print("saved", path)


def print_table(results, key="accuracy"):
    methods = sorted({m for m, _ in results})
    print(f"\n{key} (%)".ljust(14) + "".join(f"{m:>8}" for m in methods))
    for v in VARIANTS:
        print(v.ljust(14) + "".join(f"{100 * results[(m, v)][key]:8.1f}" for m in methods))


def load_augmented(out, feats, n):
    """{backbone: {variant: (K, N, D)}} from features_aug_*.npz (step 2 --augment K), if current."""
    aug = {}
    for bb in feats:
        path = out / f"features_aug_{bb}.npz"
        if not path.exists():
            continue
        if path.stat().st_mtime < (out / "faces.npz").stat().st_mtime:
            print(f"{path.name} is older than faces.npz: ignored (rerun step 2 with --augment)")
            continue
        A = np.load(path)
        if A["full"].shape[1] != n:
            print(f"{path.name} does not match faces.npz: ignored (rerun step 2 with --augment)")
            continue
        aug[bb] = {v: normalize(A[v].reshape(-1, A[v].shape[-1])).reshape(A[v].shape) for v in VARIANTS}
        print(f"{path.name}: {A['full'].shape[0]} augmented copies per image -> methods with {AUG}")
    return aug


def main(dataset, seeds, fit_all):
    out = out_dir(dataset)
    d = np.load(out / "faces.npz")
    faces, y, names = d["faces"], d["labels"], [str(n) for n in d["names"]]
    files = d["files"] if "files" in d else np.full(len(y), "")
    sessions = d["sessions"] if "sessions" in d else np.full(len(y), "")
    feats = {}
    for bb in ("vggface", "facenet"):
        if (out / f"features_{bb}.npz").exists():
            F = np.load(out / f"features_{bb}.npz")
            feats[bb] = {v: normalize(F[v]) for v in VARIANTS}
        else:
            print(f"features_{bb}.npz not found: skipping its methods")
    aug = load_augmented(out, feats, len(y))

    idx_tr, idx_te, results, probs, clfs, how = run_split(faces, y, sessions, names, feats, aug, SEED)
    print("split:", how)
    np.savez(out / "split.npz", train=idx_tr, test=idx_te, how=how)
    np.savez_compressed(out / "probs.npz", y_test=y[idx_te], idx_test=idx_te, **probs)
    rows = [{"method": m, "method_name": METHODS[m], "variant": v, **{k: round(float(x), 6) for k, x in r.items()},
             **({"augmented_copies": len(aug[backbone(m)]["full"])} if m.endswith(AUG) else {})}
            for (m, v), r in results.items()]
    (out / "metrics.json").write_text(json.dumps(rows, indent=2))
    write_predictions(out / "predictions.csv", probs, y, idx_te, names, files)
    write_confusions(out / "top_confusions.csv", probs, y[idx_te], names)
    targets = check_targets(results)
    (out / "targets.json").write_text(json.dumps(targets, indent=2, ensure_ascii=False))
    print_table(results)
    for t, r in targets.items():
        print(f"{t} {'PASS' if r['pass'] else 'FAIL'}: {r['desc']} = {r['value']} (target {r['target']})")

    open_rows = run_open_set(out, clfs, feats, y, idx_te)

    if fit_all:   # demo model: same recipe, every image of every person
        clfs = {m: fit_deep(m, feats, aug, y, np.arange(len(y))) for m in clfs}
    save_models(dataset, clfs, names, "all images" if fit_all else f"train split (seed {SEED}): {how}", open_rows)

    if seeds:
        per = {}
        for s in seeds:
            r = results if s == SEED else run_split(faces, y, sessions, names, feats, aug, s)[2]
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
