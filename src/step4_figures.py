"""Step 4: figures and tables for the presentation.

  python step4_figures.py                 # LFW -> results/figures/*.png, results/summary.json, results/RESULTS.md
  python step4_figures.py --dataset fei   # also pose_fei.png: accuracy at each of the 14 FEI poses
"""
import argparse
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

from common import OCCLUDED, TAU, VARIANT_TH, VARIANTS, apply_variant, augment, out_dir

matplotlib.rcParams.update({
    "font.family": ["Leelawadee UI", "Tahoma", "Noto Sans Thai", "Sarabun", "Garuda", "sans-serif"],
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
OCC = OCCLUDED


def short(name):
    parts = name.split("_")
    return f"{parts[0]} {parts[-1]}" if len(parts) > 1 else name


FIG = None   # set in main(): <results>/figures


def save(fig, name):
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(dataset):
    global FIG
    out = out_dir(dataset)
    FIG = out / "figures"
    d = np.load(out / "faces.npz")
    faces, labels, names = d["faces"], d["labels"], d["names"]
    P = np.load(out / "probs.npz")
    y, idx_te = P["y_test"], P["idx_test"]
    M = json.loads((out / "metrics.json").read_text())
    acc = {(r["method"], r["variant"]): r["accuracy"] for r in M}
    methods = [m for m in ["M0", "M1", "M2", "M3", "M4"] if (m, "full") in acc]
    rng = np.random.default_rng(0)

    # 1. sample images per class
    n_show = min(4, len(names))
    show = np.argsort(-np.bincount(labels))[:n_show].tolist() + rng.choice(len(names), n_show, replace=False).tolist()
    show = list(dict.fromkeys(show))
    fig, ax = plt.subplots(4, len(show), figsize=(1.75 * len(show), 7.6), squeeze=False)
    for a in ax.ravel():
        a.axis("off")
    for c, k in enumerate(show):
        own = np.where(labels == k)[0]
        imgs = rng.choice(own, min(4, len(own)), replace=False)
        for r, i in enumerate(imgs):
            ax[r, c].imshow(faces[i])
        ax[0, c].set_title(short(str(names[k])).replace("_", " "), fontsize=11, color=INK)
    fig.subplots_adjust(wspace=0.1, hspace=0.04)
    save(fig, "samples_per_class.png")

    # 2. partial-face variants on one face
    own = np.where(labels == show[0])[0]
    i = own[min(3, len(own) - 1)]
    fig, ax = plt.subplots(1, len(VARIANTS), figsize=(15, 2.9))
    for a, v in zip(ax, VARIANTS):
        a.imshow(apply_variant(faces[i], v)); a.axis("off")
        a.set_title(VARIANT_TH[v], fontsize=14, color=INK)
    save(fig, "variants.png")

    # 3. data augmentation (slide 6): random training copies of the same face, plain and with a mask
    rng_aug = np.random.default_rng(1)          # own stream: the other figures keep their samples
    copies = [faces[i]] + [augment(faces[i], rng_aug) for _ in range(5)]
    fig, ax = plt.subplots(2, len(copies), figsize=(2.2 * len(copies), 4.9))
    for c, img in enumerate(copies):
        for r, v in enumerate(["full", "mask"]):
            ax[r, c].imshow(apply_variant(img, v)); ax[r, c].axis("off")
        ax[0, c].set_title("ภาพเดิม" if c == 0 else f"สุ่มครั้งที่ {c}", fontsize=12, color=INK)
    fig.subplots_adjust(wspace=0.06, hspace=0.06)
    save(fig, "augmentation.png")

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
    fig, ax = plt.subplots(figsize=(15, 5.6))
    w = 0.8 / len(methods)
    for j, m in enumerate(methods):
        xs = np.arange(len(VARIANTS)) + (j - (len(methods) - 1) / 2) * w
        vals = [acc[(m, v)] * 100 for v in VARIANTS]
        ax.bar(xs, vals, w * 0.88, color=COL[m], label=LABEL[m])
        if m in ("M2", "M4"):
            for x, val in zip(xs, vals):
                ax.text(x, val + 1, f"{val:.0f}", ha="center", fontsize=10.5, color=INK)
    ax.set_xticks(range(len(VARIANTS)), [VARIANT_TH[v] for v in VARIANTS], fontsize=13)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 108)
    ax.yaxis.grid(True, color="#e6e6e3"); ax.set_axisbelow(True)
    ax.legend(ncol=len(methods), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=12)
    save(fig, "accuracy_by_variant.png")

    # 5b. the same methods with and without augmented training copies (step 2 --augment)
    pairs = [m for m in methods if (m + "+aug", "full") in acc]
    if pairs:
        augmentation_figure(acc, pairs)

    # 6. confusion matrices on the hardest variant (lower half): full 62x62, and top-12 classes M1 vs M2
    best = "M2" if "M2" in methods else methods[-1]
    for m in [x for x in ["M1", "M2"] if x in methods]:
        cm = confusion_matrix(y, P[f"{m}|lower_half"].argmax(1), labels=range(len(names)), normalize="true")
        fig, ax = plt.subplots(figsize=(7.2, 6.4))
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xlabel(f"คลาสที่ทำนาย ({len(names)} คน)"); ax.set_ylabel(f"คลาสจริง ({len(names)} คน)")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="สัดส่วน (normalize ต่อแถว)")
        save(fig, f"confusion_full_{m}_lower_half.png")

    top = np.argsort(-np.bincount(y))[:12]
    tick = [short(str(names[k])).replace("_", " ") for k in top]
    m_ = np.isin(y, top)
    for m in [x for x in ["M1", "M2"] if x in methods]:
        pred_all = P[f"{m}|lower_half"][m_].argmax(1)
        # predictions outside the 12 shown classes are pooled into an "other" column
        cm = confusion_matrix(y[m_], np.where(np.isin(pred_all, top), pred_all, -1), labels=list(top) + [-1])[:len(top)]
        fig, ax = plt.subplots(figsize=(9.6, 7.6))
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=15)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                if cm[r, c]:
                    ax.text(c, r, cm[r, c], ha="center", va="center", fontsize=12,
                            color="#fcfcfb" if cm[r, c] > 8 else INK)
        ax.set_xticks(range(len(top) + 1), tick + ["อื่น ๆ"], rotation=45, ha="right", fontsize=12)
        ax.set_yticks(range(len(top)), tick, fontsize=12)
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
    for m in methods:
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
    for a in ax.ravel()[len(picks):]:
        a.axis("off")
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

    summary = {f"roc_auc_micro_{best}": roc, f"threshold_{best}": thr}

    # 11. FEI: accuracy at each of the 14 poses (image numbers) and how often MTCNN finds the face
    if "pose_idx" in d and (d["pose_idx"] >= 0).any():
        summary["pose"] = pose_figure(out, d["pose_idx"], d["yaw"], y, idx_te, P, methods)

    # 12. strangers: who gets a name at each tau (open-set test)
    if (out / "open_set.json").exists():
        open_set_figure(out, best)

    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    write_report(out, dataset, M, methods, names, labels)
    print("figures saved to", FIG)


def pose_figure(out, pose_idx, yaw, y, idx_te, P, methods):
    stats = json.loads((out / "dataset_stats.json").read_text()).get("per_image_number", {})
    nums = sorted(int(k) for k in stats) or sorted(set(pose_idx[idx_te].tolist()))
    pi = pose_idx[idx_te]
    lines = [(m, v) for m, v in [("M1", "full"), ("M2", "full"), ("M2", "mask"), ("M2", "sunglasses")] if m in methods]
    table = {}
    for n in nums:
        sel = pi == n
        row = {"test_images": int(sel.sum()),
               "detected": stats.get(str(n), {}).get("detected"), "images": stats.get(str(n), {}).get("images"),
               "median_yaw": float(np.nanmedian(yaw[pose_idx == n])) if (pose_idx == n).any() else None}
        for m, v in lines:
            row[f"{m}|{v}"] = float((P[f"{m}|{v}"][sel].argmax(1) == y[sel]).mean()) if sel.any() else None
        table[n] = row

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    pal = {"M1|full": "#9cc2ee", "M2|full": "#2a78d6", "M2|mask": "#e87ba4", "M2|sunglasses": "#4a3aa7"}
    w = 0.8 / max(1, len(lines))
    for j, (m, v) in enumerate(lines):
        xs = np.arange(len(nums)) + (j - (len(lines) - 1) / 2) * w
        vals = [100 * table[n][f"{m}|{v}"] if table[n][f"{m}|{v}"] is not None else np.nan for n in nums]
        a1.bar(xs, vals, w * 0.9, color=pal[f"{m}|{v}"], label=f"{LABEL[m]} · ทดสอบ{VARIANT_TH[v]}")
    a1.set_ylabel("Accuracy (%)"); a1.set_ylim(0, 105)
    a1.yaxis.grid(True, color="#e6e6e3"); a1.set_axisbelow(True)
    a1.legend(ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=11)
    det = [100 * table[n]["detected"] / table[n]["images"] if table[n]["images"] else 0 for n in nums]
    a2.bar(range(len(nums)), det, 0.6, color="#a3a39e")
    for x, n in enumerate(nums):
        if table[n]["median_yaw"] is not None:
            a2.text(x, 4, f"yaw {table[n]['median_yaw']:+.2f}", ha="center", fontsize=9, color=INK, rotation=90)
    a2.set_ylabel("MTCNN พบหน้า (%)"); a2.set_ylim(0, 105)
    a2.set_xticks(range(len(nums)), [f"{n:02d}" for n in nums])
    a2.set_xlabel("ภาพที่ (FEI image number, แต่ละเลขคือมุมหน้าเดียวกันของทุกคน)")
    save(fig, "pose_fei.png")
    return table


def fmt(x):
    return "–" if x is None or x != x else f"{x:.3f}"     # x != x: NaN


def augmentation_figure(acc, pairs):
    bars = [x for m in pairs for x in (m, m + "+aug")]
    fig, ax = plt.subplots(figsize=(15, 5.6))
    w = 0.8 / len(bars)
    for j, m in enumerate(bars):
        base, is_aug = m.removesuffix("+aug"), m.endswith("+aug")
        xs = np.arange(len(VARIANTS)) + (j - (len(bars) - 1) / 2) * w
        ax.bar(xs, [100 * acc[(m, v)] for v in VARIANTS], w * 0.9, color=COL[base], hatch="///" if is_aug else None,
               edgecolor="#fcfcfb", linewidth=0.5, label=LABEL[base] + (" + augmentation" if is_aug else ""))
    ax.set_xticks(range(len(VARIANTS)), [VARIANT_TH[v] for v in VARIANTS], fontsize=13)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 108)
    ax.yaxis.grid(True, color="#e6e6e3"); ax.set_axisbelow(True)
    ax.legend(ncol=min(4, len(bars)), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=11)
    save(fig, "augmentation_effect.png")


def open_set_figure(out, best):
    opn = json.loads((out / "open_set.json").read_text())
    rows = [r for r in opn["rows"] if r["method"] == best]
    shown = [v for v in ("full", "mask", "sunglasses") if any(r["variant"] == v for r in rows)]
    if not shown:
        return
    fig, axes = plt.subplots(1, len(shown), figsize=(5.4 * len(shown), 4.9), sharey=True, squeeze=False)
    for a, v in zip(axes[0], shown):
        rs = [r for r in rows if r["variant"] == v]
        t = [r["tau"] for r in rs]
        a.plot(t, [100 * r["known_correct"] for r in rs], color="#2a78d6", lw=2.2, label="คนที่รู้จัก: ตอบชื่อถูก")
        a.plot(t, [100 * r["known_wrong"] for r in rs], color="#e34948", lw=2, label="คนที่รู้จัก: ตอบเป็นคนอื่น")
        a.plot(t, [100 * r["stranger_accepted"] for r in rs], color="#eb6834", lw=2.2, ls="--",
               label=f"ภาพคนแปลกหน้า {opn['n_strangers']} ภาพ: ถูกตอบเป็นชื่อคนที่รู้จัก")
        a.axvline(TAU, color="#b5b5b0", lw=1, ls=":")
        a.set_title(f"ทดสอบ{VARIANT_TH[v]}", fontsize=13, color=INK)
        a.set_xlabel("เกณฑ์ความมั่นใจ τ (ต่ำกว่านี้ตอบ \"ไม่รู้จัก\")")
        a.set_ylim(0, 102)
        a.yaxis.grid(True, color="#e6e6e3"); a.set_axisbelow(True)
    axes[0, 0].set_ylabel("% ของภาพ")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, ncol=3, fontsize=11,
               loc="upper center", bbox_to_anchor=(0.5, -0.02))
    save(fig, f"open_set_{best}.png")


def write_report(out, dataset, M, methods, names, labels):
    """results/RESULTS.md: the numbers behind the slides, regenerated from metrics.json and targets.json."""
    acc = {(r["method"], r["variant"]): r for r in M}
    split = np.load(out / "split.npz") if (out / "split.npz").exists() else {}
    how = str(split["how"]) if "how" in split else "70/30 stratified split of the images"
    lines = [f"# Results: {dataset}", "",
             f"{len(names)} people, {len(labels)} aligned faces. Test set: {how} (seed 42).", "",
             "## Accuracy (%) by test variant", "",
             "| Variant | " + " | ".join(f"{m} {METHOD_SHORT[m]}" for m in methods) + " |",
             "|---|" + "---:|" * len(methods)]
    for v in VARIANTS:
        row = " | ".join(f"{100 * acc[(m, v)]['accuracy']:.1f}" for m in methods)
        lines.append(f"| {v} ({VARIANT_TH[v]}) | {row} |")
    best = "M2" if "M2" in methods else methods[-1]
    lines += ["", f"## {best}: all metrics", "",
              "| Variant | Accuracy | Precision | Recall | F1 | ROC-AUC | Top-5 | Coverage@τ | Accuracy@τ |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for v in VARIANTS:
        r = acc[(best, v)]
        lines.append(f"| {v} | " + " | ".join(fmt(r.get(k)) for k in
                     ["accuracy", "precision", "recall", "f1", "auc", "top5", "coverage@tau", "accuracy@tau"]) + " |")
    tpath = out / "targets.json"
    targets = json.loads(tpath.read_text()) if tpath.exists() else {}
    if targets:
        lines += ["", "## Targets (slide 5)", "", "| | Target | Measured | |", "|---|---|---|---|"]
        for t, r in targets.items():
            val = r["value"]
            val = ", ".join(f"{k} {x:+.3f}" if t == "T3" else f"{k} {x:.3f}" for k, x in val.items()) \
                if isinstance(val, dict) else f"{val:.3f}"
            lines.append(f"| {t} {r['desc']} | {'+' if t == 'T3' else '≥ '}{r['target']} | {val} | "
                         f"{'PASS' if r['pass'] else 'FAIL'} |")
    aug = [m for m in methods if (m + "+aug", "full") in acc]
    if aug:
        k = acc[(aug[0] + "+aug", "full")].get("augmented_copies", "K")
        cols = [x for m in aug for x in (m, m + "+aug")]
        lines += ["", "## Data augmentation (slide 6)", "",
                  f"Accuracy (%) without and with {k} augmented copies of every training image "
                  "(rotation ±15°, random crop, flip, brightness/contrast, noise; test images never augmented).", "",
                  "| Variant | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
        for v in VARIANTS:
            lines.append(f"| {v} | " + " | ".join(f"{100 * acc[(m, v)]['accuracy']:.1f}" for m in cols) + " |")
    op = out / "open_set.json"
    if op.exists():
        opn = json.loads(op.read_text())
        R = {(r["method"], r["variant"], r["tau"]): r for r in opn["rows"] if r["method"] == best}
        lines += ["", f"## Strangers ({best}, open-set test)", "",
                  f"{opn['n_strangers']} photos of people who are not in the training set, tested like the "
                  f"{opn['n_known_test']} known test faces. *Accepted* = given a name instead of \"Unknown\".", "",
                  f"| Variant | τ = {TAU}: known, right name | known, wrong name | strangers accepted "
                  "| τ for ≤ 5 % strangers accepted | known, right name at that τ |",
                  "|---|---:|---:|---:|---:|---:|"]
        for v in VARIANTS:
            rs = [R[k] for k in sorted(R) if k[1] == v]
            if not rs:
                continue
            at = min(rs, key=lambda r: abs(r["tau"] - TAU))
            ok = next((r for r in rs if r["stranger_accepted"] <= 0.05), None)
            lines.append(f"| {v} | {100 * at['known_correct']:.1f} % | {100 * at['known_wrong']:.1f} % | "
                         f"{100 * at['stranger_accepted']:.1f} % | " +
                         (f"{ok['tau']:.2f} | {100 * ok['known_correct']:.1f} % |" if ok else "– | – |"))
    sp = out / "metrics_seeds.json"
    if sp.exists():
        S = {(r["method"], r["variant"]): r for r in json.loads(sp.read_text())}
        seeds = next(iter(S.values()))["seeds"]
        lines += ["", f"## Accuracy (%) mean ± std over splits with seeds {seeds}", "",
                  "| Variant | " + " | ".join(methods) + " |", "|---|" + "---:|" * len(methods)]
        for v in VARIANTS:
            row = " | ".join(f"{100 * S[(m, v)]['accuracy_mean']:.1f} ± {100 * S[(m, v)]['accuracy_std']:.1f}"
                             for m in methods)
            lines.append(f"| {v} | {row} |")
    figs = sorted(p.name for p in FIG.glob("*.png"))
    lines += ["", "## Figures", ""] + [f"- [{f}](figures/{f})" for f in figs]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


METHOD_SHORT = {"M0": "Eigenfaces+SVM", "M1": "VGG-Face full", "M2": "VGG-Face full+partial",
                "M3": "FaceNet full", "M4": "FaceNet full+partial"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="lfw")
    main(ap.parse_args().dataset)
