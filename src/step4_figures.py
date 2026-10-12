"""Step 4: figures and tables for the presentation (results/figures/*.png, results/summary.json)."""
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

from common import FIG, RESULTS, VARIANT_TH, VARIANTS, apply_variant

matplotlib.rcParams.update({
    "font.family": ["Leelawadee UI", "Tahoma", "sans-serif"],
    "font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8f98", "axes.labelcolor": "#33373d", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb", "axes.grid": False,
})
INK, MUTED = "#1f2430", "#6b6f76"
COL = {"M0": "#a3a39e", "M1": "#9cc2ee", "M2": "#2a78d6", "M3": "#f4b08f", "M4": "#d95926"}
LABEL = {"M0": "Eigenfaces + SVM (เต็ม)", "M1": "VGG-Face (เทรนภาพเต็ม)",
         "M2": "VGG-Face (เทรนเต็ม+บางส่วน)", "M3": "FaceNet (เทรนภาพเต็ม)",
         "M4": "FaceNet (เทรนเต็ม+บางส่วน)"}
OCC = [v for v in VARIANTS if v != "full"]


def short(name):
    parts = name.split("_")
    return f"{parts[0]} {parts[-1]}" if len(parts) > 1 else name


def save(fig, name):
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    d = np.load(RESULTS / "faces.npz")
    faces, labels, names = d["faces"], d["labels"], d["names"]
    P = np.load(RESULTS / "probs.npz")
    y, idx_te = P["y_test"], P["idx_test"]
    M = json.loads((RESULTS / "metrics.json").read_text())
    acc = {(r["method"], r["variant"]): r["accuracy"] for r in M}
    rng = np.random.default_rng(0)

    # 1. sample images per class
    show = np.argsort(-np.bincount(labels))[:4].tolist() + rng.choice(len(names), 4, replace=False).tolist()
    fig, ax = plt.subplots(4, 8, figsize=(14, 7.6))
    for c, k in enumerate(dict.fromkeys(show)):
        imgs = rng.choice(np.where(labels == k)[0], 4, replace=False)
        for r, i in enumerate(imgs):
            ax[r, c].imshow(faces[i]); ax[r, c].axis("off")
        ax[0, c].set_title(short(str(names[k])).replace("_", " "), fontsize=11, color=INK)
    fig.subplots_adjust(wspace=0.1, hspace=0.04)
    save(fig, "samples_per_class.png")

    # 2. partial-face variants on one face
    i = np.where(labels == show[0])[0][3]
    fig, ax = plt.subplots(1, len(VARIANTS), figsize=(15, 2.9))
    for a, v in zip(ax, VARIANTS):
        a.imshow(apply_variant(faces[i], v)); a.axis("off")
        a.set_title(VARIANT_TH[v], fontsize=14, color=INK)
    save(fig, "variants.png")

    # 4. images per class
    cnt = np.bincount(labels)
    order = np.argsort(-cnt)
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.bar(range(len(cnt)), cnt[order], color="#2a78d6", width=0.8)
    ax.set_xlim(-1, len(cnt)); ax.set_ylabel("จำนวนภาพ")
    ax.set_xlabel(f"บุคคล (คลาส) ทั้งหมด {len(cnt)} คน เรียงจากมากไปน้อย"); ax.set_xticks([])
    ax.yaxis.grid(True, color="#e6e6e3"); ax.set_axisbelow(True)
    save(fig, "class_distribution.png")

    # 5. accuracy by variant, grouped bars
    methods = ["M0", "M1", "M2", "M3", "M4"]
    fig, ax = plt.subplots(figsize=(15, 5.6))
    w = 0.16
    for j, m in enumerate(methods):
        xs = np.arange(len(VARIANTS)) + (j - 2) * w
        vals = [acc[(m, v)] * 100 for v in VARIANTS]
        ax.bar(xs, vals, w * 0.88, color=COL[m], label=LABEL[m])
        if m in ("M2", "M4"):
            for x, val in zip(xs, vals):
                ax.text(x, val + 1, f"{val:.0f}", ha="center", fontsize=10.5, color=INK)
    ax.set_xticks(range(len(VARIANTS)), [VARIANT_TH[v] for v in VARIANTS], fontsize=13)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 108)
    ax.yaxis.grid(True, color="#e6e6e3"); ax.set_axisbelow(True)
    ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=12)
    save(fig, "accuracy_by_variant.png")

    # 6. confusion matrices on the hardest variant (lower half): full 62x62, and top-12 classes M1 vs M2
    best = "M2"
    for m in ["M1", "M2"]:
        cm = confusion_matrix(y, P[f"{m}|lower_half"].argmax(1), labels=range(len(names)), normalize="true")
        fig, ax = plt.subplots(figsize=(7.2, 6.4))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xlabel("คลาสที่ทำนาย (62 คน)"); ax.set_ylabel("คลาสจริง (62 คน)")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="สัดส่วน (normalize ต่อแถว)")
        save(fig, f"confusion_full_{m}_lower_half.png")

    top = np.argsort(-np.bincount(y))[:12]
    tick = [short(str(names[k])).replace("_", " ") for k in top]
    m_ = np.isin(y, top)
    for m in ["M1", "M2"]:
        pred_all = P[f"{m}|lower_half"][m_].argmax(1)
        # predictions outside the 12 shown classes are pooled into an "other" column
        cm = confusion_matrix(y[m_], np.where(np.isin(pred_all, top), pred_all, -1), labels=list(top) + [-1])[:12]
        fig, ax = plt.subplots(figsize=(9.6, 7.6))
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=15)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                if cm[r, c]:
                    ax.text(c, r, cm[r, c], ha="center", va="center", fontsize=12,
                            color="#fcfcfb" if cm[r, c] > 8 else INK)
        ax.set_xticks(range(13), tick + ["อื่น ๆ"], rotation=45, ha="right", fontsize=12)
        ax.set_yticks(range(12), tick, fontsize=12)
        ax.set_xlabel("คลาสที่ทำนาย"); ax.set_ylabel("คลาสจริง")
        save(fig, f"confusion_top12_{m}_lower_half.png")

    # 7. ROC curves (micro-average) per variant, M2
    Y = label_binarize(y, classes=range(len(names)))
    roc = {}
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    pal = ["#1f2430", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
    for v, c in zip(VARIANTS, pal):
        fpr, tpr, _ = roc_curve(Y.ravel(), P[f"{best}|{v}"].ravel())
        roc[v] = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{VARIANT_TH[v]} (AUC {roc[v]:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#b5b5b0", lw=1)
    ax.set_xscale("log"); ax.set_xlim(1e-4, 1); ax.set_ylim(0, 1.01)
    ax.set_xlabel("False Positive Rate (log scale)"); ax.set_ylabel("True Positive Rate")
    ax.legend(frameon=False, fontsize=11, loc="lower right")
    save(fig, f"roc_{best}.png")

    # 8. M1 vs M2 ROC on mask (effect of training with partial faces)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for m in ["M0", "M1", "M2", "M3", "M4"]:
        fpr, tpr, _ = roc_curve(Y.ravel(), P[f"{m}|mask"].ravel())
        ax.plot(fpr, tpr, color=COL[m], lw=2, label=f"{LABEL[m]} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#b5b5b0", lw=1)
    ax.set_xscale("log"); ax.set_xlim(1e-4, 1); ax.set_ylim(0, 1.01)
    ax.set_xlabel("False Positive Rate (log scale)"); ax.set_ylabel("True Positive Rate")
    ax.legend(frameon=False, fontsize=11, loc="lower right")
    save(fig, "roc_mask_methods.png")

    # 9. correct / wrong examples (M2 on occluded variants)
    picks = []
    for want_ok in (True, False):
        for v in ["mask", "sunglasses", "upper_half", "lower_half", "eyes_only", "nose_mouth"]:
            p = P[f"{best}|{v}"]
            ok = p.argmax(1) == y
            cand = np.where(ok == want_ok)[0]
            if len(cand):
                t = rng.choice(cand)
                picks.append((v, t, p[t].argmax(), p[t].max(), want_ok))
    fig, ax = plt.subplots(2, 6, figsize=(17, 7))
    for a, (v, t, pr, conf, ok) in zip(ax.ravel(), picks):
        a.imshow(apply_variant(faces[idx_te[t]], v)); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(True); s.set_color("#1baf7a" if ok else "#e34948"); s.set_linewidth(4)
        a.set_title(f"{'ถูก' if ok else 'ผิด'} · {VARIANT_TH[v]}", fontsize=12, color=INK)
        a.set_xlabel(f"จริง: {short(str(names[y[t]]))}\nทาย: {short(str(names[pr]))} ({conf:.0%})",
                     fontsize=11, color=INK)
    fig.subplots_adjust(hspace=0.5, wspace=0.32)
    save(fig, "examples_correct_wrong.png")

    # 10. confidence threshold: accuracy of accepted vs coverage (all occluded tests pooled)
    pool = np.concatenate([P[f"{best}|{v}"] for v in OCC])
    yy = np.tile(y, len(OCC))
    conf, ok = pool.max(1), pool.argmax(1) == yy
    taus = np.linspace(0, 0.95, 20)
    thr = [{"tau": round(float(t), 2), "coverage": float((conf >= t).mean()),
            "accuracy": float(ok[conf >= t].mean()) if (conf >= t).any() else None} for t in taus]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.plot(taus, [r["coverage"] * 100 for r in thr], color="#eb6834", lw=2, marker="o", ms=5, label="Coverage (% ที่ระบบตอบ)")
    ax.plot(taus, [r["accuracy"] * 100 for r in thr], color="#2a78d6", lw=2, marker="o", ms=5, label="Accuracy ของภาพที่ตอบ")
    ax.set_xlabel("เกณฑ์ค่าความมั่นใจ τ"); ax.set_ylabel("%"); ax.set_ylim(0, 102)
    ax.yaxis.grid(True, color="#e6e6e3"); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left")
    save(fig, f"threshold_{best}.png")

    summary = {"roc_auc_micro_M2": roc, "threshold_M2": thr}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("figures saved")


if __name__ == "__main__":
    main()
