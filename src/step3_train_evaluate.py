"""Step 3: train Softmax classifiers on frozen features and evaluate every partial-face variant.

Methods
  M0  Eigenfaces (PCA) + SVM            trained on full faces        (classical baseline)
  M1  VGG-Face + Softmax                trained on full faces
  M2  VGG-Face + Softmax                trained on full + partial faces (transfer learning on partial data)
  M3  FaceNet(VGGFace2) + Softmax       trained on full faces
  M4  FaceNet(VGGFace2) + Softmax       trained on full + partial faces
"""
import json

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, top_k_accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

from common import RESULTS, SEED, VARIANTS, apply_variant

METHODS = {
    "M0": "Eigenfaces + SVM (Full)",
    "M1": "VGG-Face + Softmax (Full)",
    "M2": "VGG-Face + Softmax (Full+Partial)",
    "M3": "FaceNet + Softmax (Full)",
    "M4": "FaceNet + Softmax (Full+Partial)",
}


def metrics(y, prob, classes):
    pred = classes[prob.argmax(1)]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, average="macro", zero_division=0),
        "recall": recall_score(y, pred, average="macro", zero_division=0),
        "f1": f1_score(y, pred, average="macro", zero_division=0),
        "auc": roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=classes),
        "top5": top_k_accuracy_score(y, prob, k=5, labels=classes),
    }


def softmax_clf():
    return LogisticRegression(C=10, max_iter=3000)


def gray(faces, v):
    return np.stack([cv2.resize(cv2.cvtColor(apply_variant(f, v), cv2.COLOR_RGB2GRAY), (64, 64))
                     for f in faces]).reshape(len(faces), -1) / 255.0


def main():
    d = np.load(RESULTS / "faces.npz")
    faces, y = d["faces"], d["labels"]
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=SEED)
    np.savez(RESULTS / "split.npz", train=idx_tr, test=idx_te)
    variants = list(VARIANTS)
    results, probs = {}, {}

    # M0: Eigenfaces + SVM
    clf = make_pipeline(PCA(150, whiten=True, random_state=SEED),
                        SVC(C=10, gamma="scale", probability=True, random_state=SEED))
    clf.fit(gray(faces[idx_tr], "full"), y[idx_tr])
    for v in variants:
        p = clf.predict_proba(gray(faces[idx_te], v))
        results[("M0", v)] = metrics(y[idx_te], p, clf.classes_)
        probs[f"M0|{v}"] = p
    print("M0 done", flush=True)

    for bb, (m_full, m_part) in {"vggface": ("M1", "M2"), "facenet": ("M3", "M4")}.items():
        F = np.load(RESULTS / f"features_{bb}.npz")
        F = {v: normalize(F[v]) for v in variants}
        train_sets = {
            m_full: (F["full"][idx_tr], y[idx_tr]),
            m_part: (np.concatenate([F[v][idx_tr] for v in variants]), np.tile(y[idx_tr], len(variants))),
        }
        for m, (Xtr, ytr) in train_sets.items():
            clf = softmax_clf().fit(Xtr, ytr)
            for v in variants:
                p = clf.predict_proba(F[v][idx_te])
                results[(m, v)] = metrics(y[idx_te], p, clf.classes_)
                probs[f"{m}|{v}"] = p
            print(m, "done", {v: round(results[(m, v)]["accuracy"], 3) for v in variants}, flush=True)

    np.savez_compressed(RESULTS / "probs.npz", y_test=y[idx_te], idx_test=idx_te, **probs)
    out = [{"method": m, "method_name": METHODS[m], "variant": v, **{k: round(float(x), 4) for k, x in r.items()}}
           for (m, v), r in results.items()]
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2))
    print("saved", len(out), "rows")


if __name__ == "__main__":
    main()
