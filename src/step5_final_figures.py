"""Step 5: extra figures for the final presentation (run after step 3 and step 4).

  python step5_final_figures.py                # LFW -> results/figures/final/*.png + final_numbers.json
  python step5_final_figures.py --dataset fei

What it adds to step 4 (the feedback from the progress presentation):
  confusion_grid.png              the confusion matrix of EVERY experiment: 5 methods x 7 test variants
  confusion/<method>_<variant>.png   each of those matrices on its own (with the names on the axes)
  confusion_row_<method>.png      the 7 matrices of one method side by side (one slide per method)
  cm_top12_<variant>_M1_vs_M2.png    M1 vs M2 on the 12 people M1 confuses most, with counts
  variant_rules.png               one real aligned face with the box of each variant V0-V6 drawn on it
  one_person_variants.png         one person, several real photos, each shown as V0-V6
  dataset_variants.png            what the 7 test sets look like: 8 people x 7 variants
  example_*.png + example.json    one real test photo through every step with the real numbers
                                  (raw photo, MTCNN box + 5 landmarks, aligned face, variant, fc7 vector,
                                  top-5 of M1 and M2), replacing the made-up numbers of slide 9
  real_predictions_<variant>.png  real test photos: what M1 and M2 answered, green = right, red = wrong
  final_numbers.json              split sizes per variant and training set sizes per method
"""
import argparse
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
from facenet_pytorch import MTCNN
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix

import step4_figures  # noqa: F401  (same fonts and colours as step 4)
from common import ROOT, TEMPLATE, VARIANT_TH, VARIANTS, apply_variant, out_dir

INK, MUTED, BLUE, RED, GREEN = "#1f2430", "#6b6f76", "#2a78d6", "#d03b3b", "#2f9e44"
METHOD_TH = {"M0": "M0 Eigenfaces+SVM", "M1": "M1 VGG-Face (เต็ม)", "M2": "M2 VGG-Face (เต็ม+บางส่วน)",
             "M3": "M3 FaceNet (เต็ม)", "M4": "M4 FaceNet (เต็ม+บางส่วน)"}
VCODE = {v: f"V{i}" for i, v in enumerate(VARIANTS)}
FIG = None


def short(name):
    parts = str(name).split("_")
    return f"{parts[0]} {parts[-1]}" if len(parts) > 1 else str(name)


def save(fig, name):
    fig.savefig(FIG / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


def cm_of(P, y, m, v, n):
    return confusion_matrix(y, P[f"{m}|{v}"].argmax(1), labels=range(n))


def norm_rows(cm):
    return cm / np.maximum(cm.sum(1, keepdims=True), 1)


def confusion_figures(P, y, names, methods, acc):
    n = len(names)
    # every experiment on one page
    fig, ax = plt.subplots(len(methods), len(VARIANTS), figsize=(2.3 * len(VARIANTS), 2.45 * len(methods)))
    for r, m in enumerate(methods):
        for c, v in enumerate(VARIANTS):
            a = ax[r, c]
            a.imshow(norm_rows(cm_of(P, y, m, v, n)), cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(True); s.set_color("#c9ccd1")
            a.set_title(f"{100 * acc[(m, v)]:.1f}%", fontsize=12, color=INK, pad=3)
            if r == 0:
                a.text(0.5, 1.28, f"{VCODE[v]} {VARIANT_TH[v]}", transform=a.transAxes, ha="center",
                       fontsize=12.5, color=INK, weight="bold")
            if c == 0:
                a.set_ylabel(METHOD_TH[m], fontsize=11.5, color=INK, rotation=0, ha="right", va="center",
                             labelpad=8)
    fig.subplots_adjust(wspace=0.08, hspace=0.32)
    save(fig, "confusion_grid.png")

    # every matrix on its own, names on the axes
    (FIG / "confusion").mkdir(exist_ok=True)
    labels = [short(x) for x in names]
    for m in methods:
        for v in VARIANTS:
            cm = cm_of(P, y, m, v, n)
            fig, a = plt.subplots(figsize=(11, 10))
            a.imshow(norm_rows(cm), cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
            a.set_xticks(range(n), labels, rotation=90, fontsize=6.5)
            a.set_yticks(range(n), labels, fontsize=6.5)
            a.set_xlabel("ทายว่าเป็น (Predicted)"); a.set_ylabel("คนจริง (True)")
            a.set_title(f"{METHOD_TH[m]} · ทดสอบ {VCODE[v]} {VARIANT_TH[v]} · Accuracy {100 * acc[(m, v)]:.1f}%",
                        fontsize=13, color=INK)
            fig.savefig(FIG / "confusion" / f"{m}_{v}.png", dpi=130, bbox_inches="tight")
            plt.close(fig)

    # one method per figure: its 7 matrices in a row
    for m in methods:
        fig, ax = plt.subplots(1, len(VARIANTS), figsize=(2.6 * len(VARIANTS), 3.3))
        for a, v in zip(ax, VARIANTS):
            a.imshow(norm_rows(cm_of(P, y, m, v, n)), cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
            a.set_xticks([]); a.set_yticks([])
            a.set_title(f"{VCODE[v]} {VARIANT_TH[v]}\n{100 * acc[(m, v)]:.1f}%", fontsize=12, color=INK)
        fig.subplots_adjust(wspace=0.08)
        save(fig, f"confusion_row_{m}.png")


def top12_figures(P, y, names, acc, variants=("mask", "sunglasses", "lower_half", "eyes_only", "nose_mouth",
                                               "upper_half"), k=12):
    """M1 vs M2 on the k people of M1's most frequent (true -> predicted) mistakes, both ends of each pair,
    plus a column "others" = mistakes that went to someone outside these k."""
    n = len(names)
    out = {}
    for v in variants:
        cm1 = cm_of(P, y, "M1", v, n).copy()
        off = cm1.copy(); np.fill_diagonal(off, 0)
        keep = []
        for flat in np.argsort(-off, axis=None, kind="stable"):
            for c in np.unravel_index(flat, off.shape):
                if c not in keep and len(keep) < k:
                    keep.append(int(c))
            if len(keep) >= k:
                break
        keep = np.array(keep)
        fig, ax = plt.subplots(1, 2, figsize=(16.5, 7.4))
        for a, m in zip(ax, ["M1", "M2"]):
            cm = cm_of(P, y, m, v, n)
            sub = cm[np.ix_(keep, keep)]
            others = cm[keep].sum(1) - sub.sum(1)
            full = np.concatenate([sub, others[:, None]], 1)
            rows = cm[keep].sum(1, keepdims=True)
            shade = full / np.maximum(rows, 1)
            a.imshow(shade, cmap="Blues", vmin=0, vmax=1)
            for i in range(len(keep)):
                for j in range(len(keep) + 1):
                    if full[i, j]:
                        dark = shade[i, j] > 0.55
                        a.text(j, i, full[i, j], ha="center", va="center", fontsize=10.5,
                               color="white" if dark else (INK if i == j else RED), weight="bold" if i != j else None)
            a.axvline(len(keep) - 0.5, color=MUTED, lw=1)
            lab = [short(names[c]) for c in keep]
            a.set_xticks(range(len(keep) + 1), lab + ["อื่น ๆ"], rotation=60, ha="right", fontsize=10)
            a.set_yticks(range(len(keep)), lab, fontsize=10)
            a.set_xlabel("ทายว่าเป็น"); a.set_ylabel("คนจริง")
            right = int(np.trace(cm)); total = int(cm.sum())
            a.set_title(METHOD_TH[m] + "\n" + f"ทั้งชุด: ถูก {right}/{total} ภาพ = {100 * acc[(m, v)]:.1f}%",
                        fontsize=13, color=INK)
        fig.suptitle(f"ทดสอบ {VCODE[v]} {VARIANT_TH[v]} · คนในคู่ที่ M1 สับสนบ่อยที่สุด (ตัวเลข = จำนวนภาพ, "
                     "แดง = ทายผิด)", fontsize=13.5, color=INK, y=1.0)
        fig.subplots_adjust(wspace=0.42)
        save(fig, f"cm_top12_{v}_M1_vs_M2.png")
        out[v] = [str(names[c]) for c in keep]
    return out


def variant_rules(face):
    fig, ax = plt.subplots(1, len(VARIANTS), figsize=(2.3 * len(VARIANTS), 3.2))
    for a, v in zip(ax, VARIANTS):
        a.imshow(apply_variant(face, v)); a.set_xticks([]); a.set_yticks([])
        mode, box = VARIANTS[v]
        if box is not None:
            x0, y0, x1, y1 = box
            a.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=2,
                                  ec=GREEN if mode == "keep" else RED, ls="--"))
            rule = f"{'เหลือ' if mode == 'keep' else 'ปิด'} x {x0}–{x1}\ny {y0}–{y1}"
        else:
            rule = "ไม่บดบัง\n(ภาพอ้างอิง)"
        a.set_title(f"{VCODE[v]} {VARIANT_TH[v]}", fontsize=12, color=INK)
        a.set_xlabel(rule, fontsize=10.5, color=MUTED)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "variant_rules.png")


def one_person(faces, labels, names, k, n_photos=4):
    own = np.where(labels == k)[0][:n_photos]
    fig, ax = plt.subplots(len(own), len(VARIANTS), figsize=(2.05 * len(VARIANTS), 2.1 * len(own)))
    for r, i in enumerate(own):
        for c, v in enumerate(VARIANTS):
            ax[r, c].imshow(apply_variant(faces[i], v)); ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(f"{VCODE[v]} {VARIANT_TH[v]}", fontsize=11.5, color=INK)
    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    save(fig, "one_person_variants.png")
    return own.tolist()


def dataset_variants(faces, labels, names, rng, n_people=8):
    people = rng.choice(len(names), n_people, replace=False)
    fig, ax = plt.subplots(len(VARIANTS), n_people, figsize=(1.75 * n_people, 1.8 * len(VARIANTS)))
    for c, k in enumerate(people):
        i = rng.choice(np.where(labels == k)[0])
        for r, v in enumerate(VARIANTS):
            ax[r, c].imshow(apply_variant(faces[i], v)); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            for s in ax[r, c].spines.values():
                s.set_visible(False)
            if c == 0:
                ax[r, c].set_ylabel(f"{VCODE[v]}\n{VARIANT_TH[v]}", fontsize=11, color=INK, rotation=0,
                                    ha="right", va="center", labelpad=6)
        ax[0, c].set_title(short(names[k]), fontsize=10, color=INK)
    fig.subplots_adjust(wspace=0.04, hspace=0.05)
    save(fig, "dataset_variants.png")


def worked_example(d, P, y, idx_te, names, feats, ex_variant="mask"):
    """A test photo that M2 gets right under the mask while M1 gets it wrong (or the first one M2 gets right)."""
    faces, files = d["faces"], d["files"]
    p1, p2 = P[f"M1|{ex_variant}"], P[f"M2|{ex_variant}"]
    rows = np.arange(len(y))
    good = rows[(p2.argmax(1) == y) & (p1.argmax(1) != y) & (p2.max(1) >= 0.5)]   # M2 right and accepted at tau
    if not len(good):
        good = rows[p2.argmax(1) == y]
    j = int(good[np.argmax(p2[good, y[good]])]) if len(good) else 0
    i = int(idx_te[j])
    raw = np.array(ImageOps.exif_transpose(Image.open(ROOT / str(files[i]))).convert("RGB"))
    mt = MTCNN(keep_all=True, device="cpu")
    boxes, probs, lms = mt.detect(raw, landmarks=True)
    boxes, probs, lms = (np.asarray(a, np.float32) for a in (boxes, probs, lms))
    centres = (boxes[:, :2] + boxes[:, 2:]) / 2
    k = int(np.argmin(np.linalg.norm(centres - np.array(raw.shape[1::-1]) / 2, axis=1)))
    box, pts, det = boxes[k], lms[k], float(probs[k])
    M, _ = cv2.estimateAffinePartial2D(pts.astype(np.float32), TEMPLATE, method=cv2.LMEDS)
    angle = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    eye_tilt = float(np.degrees(np.arctan2(pts[1, 1] - pts[0, 1], pts[1, 0] - pts[0, 0])))

    def img(name, im, marks=None, rect=None, size=3.2):
        fig, a = plt.subplots(figsize=(size, size))
        a.imshow(im); a.axis("off")
        if rect is not None:
            a.add_patch(Rectangle(rect[:2], rect[2] - rect[0], rect[3] - rect[1], fill=False, ec=BLUE, lw=2.5))
        if marks is not None:
            a.scatter(marks[:, 0], marks[:, 1], s=40, c="#ffb000", edgecolors="k", linewidths=0.8, zorder=3)
        save(fig, name)

    img("example_raw.png", raw)
    img("example_mtcnn.png", raw, pts, box)
    img("example_aligned.png", faces[i], TEMPLATE)
    img("example_variant.png", apply_variant(faces[i], ex_variant))
    fig, ax = plt.subplots(1, len(VARIANTS), figsize=(15, 2.6))
    for a, v in zip(ax, VARIANTS):
        a.imshow(apply_variant(faces[i], v)); a.axis("off"); a.set_title(f"{VCODE[v]}", fontsize=12)
    save(fig, "example_all_variants.png")

    f = feats[ex_variant][i]
    fig, a = plt.subplots(figsize=(5.2, 2.6))
    a.bar(range(48), f[:48], color=BLUE, width=0.75)
    a.set_xlabel(f"48 ค่าแรกจาก {len(f):,} มิติ (fc7, ReLU)", fontsize=11); a.set_yticks([])
    save(fig, "example_features.png")

    top = {}
    for m, p in (("M1", p1), ("M2", p2)):
        order = np.argsort(-p[j])[:5]
        top[m] = [[str(names[c]), float(p[j, c])] for c in order]
        fig, a = plt.subplots(figsize=(3.3, 2.5))
        vals = [p[j, c] for c in order][::-1]
        labs = [short(names[c]) for c in order][::-1]
        cols = [GREEN if c == y[j] else ("#9aa0a6" if r else RED) for r, c in enumerate(order)][::-1]
        a.barh(range(5), vals, color=cols)
        a.set_yticks(range(5), labs, fontsize=14); a.set_xlim(0, 1.15); a.set_xticks([0, 0.5, 1])
        a.axvline(0.5, color=MUTED, ls="--", lw=1); a.text(0.52, -0.75, "τ = 0.5", fontsize=12, color=MUTED)
        for r, val in enumerate(vals):
            a.text(val + 0.01, r, f"{val:.2f}", va="center", fontsize=13)
        a.set_title(METHOD_TH[m], fontsize=14, color=INK)
        save(fig, f"example_top5_{m}.png")
    info = {"file": str(files[i]), "true": str(names[y[j]]), "raw_shape": list(raw.shape),
            "mtcnn_box": [round(float(b)) for b in box], "mtcnn_prob": round(det, 4),
            "landmarks": [[round(float(a)), round(float(b))] for a, b in pts],
            "template": TEMPLATE.round(1).tolist(), "eye_tilt_deg": round(eye_tilt, 1),
            "align_rotation_deg": round(angle, 1), "align_scale": round(scale, 3),
            "variant": ex_variant, "feature_dim": int(len(f)),
            "feature_first8": [round(float(x), 3) for x in f[:8]], "top5": top}
    (FIG / "example.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return info


def real_predictions(d, P, y, idx_te, names, v, rng, n=8):
    faces = d["faces"]
    p1, p2 = P[f"M1|{v}"], P[f"M2|{v}"]
    flips = np.where((p2.argmax(1) == y) & (p1.argmax(1) != y))[0]
    both = np.where((p2.argmax(1) == y) & (p1.argmax(1) == y))[0]
    miss = np.where(p2.argmax(1) != y)[0]
    pick = np.concatenate([rng.choice(flips, min(4, len(flips)), replace=False),
                           rng.choice(both, min(2, len(both)), replace=False),
                           rng.choice(miss, min(2, len(miss)), replace=False)])[:n].astype(int)
    fig, ax = plt.subplots(1, len(pick), figsize=(2.6 * len(pick), 4.0))
    for a, j in zip(np.atleast_1d(ax), pick):
        a.imshow(apply_variant(faces[idx_te[j]], v)); a.set_xticks([]); a.set_yticks([])
        lines = [f"จริง: {short(names[y[j]])}"]
        for m, p in (("M1", p1), ("M2", p2)):
            k = p[j].argmax()
            lines.append(f"{m}: {short(names[k]).split(' ')[-1]} {p[j, k]:.2f} {'✓' if k == y[j] else '✗'}")
        a.set_xlabel("\n".join(lines), fontsize=11, color=INK, linespacing=1.4)
        ok2 = p2[j].argmax() == y[j]
        for s in a.spines.values():
            s.set_visible(True); s.set_color(GREEN if ok2 else RED); s.set_linewidth(3)
    fig.subplots_adjust(wspace=0.12)
    save(fig, f"real_predictions_{v}.png")


def members_confusion(P, y, names, sessions, labels):
    """Custom data (capture_faces.py): confusion matrix of the group members only, on their held-out session.
    Rows = members, columns = members + "LFW คนอื่น" (any other class). One figure per method, 7 variants.
    Returns the member names, or [] when the dataset has no capture sessions (LFW, FEI)."""
    members = sorted({int(c) for c, ss in zip(labels, sessions) if ss})
    if not members:
        return []
    rows = np.isin(y, members)
    if not rows.any():
        return []
    col = {c: i for i, c in enumerate(members)}
    keys = sorted({k.split("|")[0] for k in P.files if "|" in k})
    for m in [k for k in keys if k != "M0"]:
        fig, ax = plt.subplots(1, len(VARIANTS), figsize=(3.0 * len(VARIANTS), 3.6))
        for a, v in zip(ax, VARIANTS):
            pred = P[f"{m}|{v}"].argmax(1)[rows]
            cm = np.zeros((len(members), len(members) + 1), int)
            for t, q in zip(y[rows], pred):
                cm[col[int(t)], col.get(int(q), len(members))] += 1
            shade = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            a.imshow(shade, cmap="Blues", vmin=0, vmax=1)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    if cm[i, j]:
                        a.text(j, i, cm[i, j], ha="center", va="center", fontsize=10,
                               color="white" if shade[i, j] > 0.55 else (INK if i == j else RED))
            lab = [names[c] for c in members]
            a.set_xticks(range(len(members) + 1), lab + ["LFW คนอื่น"], rotation=60, ha="right", fontsize=9)
            a.set_yticks(range(len(members)), lab if v == "full" else [""] * len(members), fontsize=10)
            acc_m = np.trace(cm[:, :len(members)]) / max(cm.sum(), 1)
            a.set_title(f"{VCODE[v]} {VARIANT_TH[v]}" + "\n" + f"{100 * acc_m:.1f}%", fontsize=11.5, color=INK)
        fig.suptitle(f"{METHOD_TH.get(m.removesuffix('+aug'), m)}{' + aug' if m.endswith('+aug') else ''} · "
                     "สมาชิกกลุ่ม (ชุด Test = session ที่โมเดลไม่เคยเห็น)", fontsize=13, color=INK, y=1.04)
        fig.subplots_adjust(wspace=0.15)
        save(fig, f"members_confusion_{m}.png")
    return [names[c] for c in members]


def main(dataset):
    global FIG
    out = out_dir(dataset)
    FIG = out / "figures" / "final"
    FIG.mkdir(parents=True, exist_ok=True)
    d = np.load(out / "faces.npz")
    faces, labels, names = d["faces"], d["labels"], [str(n) for n in d["names"]]
    P = np.load(out / "probs.npz")
    y, idx_te = P["y_test"], P["idx_test"]
    S = np.load(out / "split.npz")
    M = json.loads((out / "metrics.json").read_text())
    acc = {(r["method"], r["variant"]): r["accuracy"] for r in M}
    methods = [m for m in ["M0", "M1", "M2", "M3", "M4"] if (m, "full") in acc]
    rng = np.random.default_rng(7)

    confusion_figures(P, y, names, methods, acc)
    top12 = top12_figures(P, y, names, acc)
    biggest = int(np.argmax(np.bincount(labels)))
    variant_rules(faces[np.where(labels == biggest)[0][0]])
    person_imgs = one_person(faces, labels, names, biggest)
    dataset_variants(faces, labels, names, rng)
    F = np.load(out / "features_vggface.npz")
    example = worked_example(d, P, y, idx_te, names, {v: F[v] for v in VARIANTS})
    for v in ("mask", "sunglasses", "lower_half", "eyes_only"):
        real_predictions(d, P, y, idx_te, names, v, np.random.default_rng(3))

    sessions = d["sessions"] if "sessions" in d.files else np.full(len(labels), "")
    members = members_confusion(P, y, names, sessions, labels)
    n_tr, n_te = len(S["train"]), len(S["test"])
    numbers = {
        "dataset": dataset, "people": len(names), "images": int(len(labels)), "train_images": n_tr,
        "test_images": n_te, "split": str(S["how"]),
        "per_variant": {VCODE[v] + " " + v: {"train": n_tr, "test": n_te, "rule": VARIANTS[v][0],
                                             "box": VARIANTS[v][1]} for v in VARIANTS},
        "train_set_size": {"M0": n_tr, "M1": n_tr, "M3": n_tr, "M2": n_tr * len(VARIANTS),
                           "M4": n_tr * len(VARIANTS)},
        "one_person": {"name": names[biggest], "images": person_imgs},
        "top12_people": top12, "example": example, "members": members,
        "accuracy": {m: {v: round(100 * acc[(m, v)], 1) for v in VARIANTS} for m in methods},
    }
    (out / "final_numbers.json").write_text(json.dumps(numbers, indent=2, ensure_ascii=False))
    print("figures in", FIG)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="lfw")
    main(ap.parse_args().dataset)
