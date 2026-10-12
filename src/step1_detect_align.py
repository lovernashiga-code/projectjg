"""Step 1: pick identities, detect faces with MTCNN, align to a 5-point template.

  python step1_detect_align.py                      # LFW (default, slide 7)
  python step1_detect_align.py --dataset fei        # FEI Face Database: 200 people x 14 poses
  python step1_detect_align.py --dataset custom     # data/custom/<person>/<session>/*.jpg (capture_faces.py)
  python step1_detect_align.py --dataset lfw+custom # LFW people plus the group members
  python step1_detect_align.py --strangers-only     # only (re)build the stranger set, keep faces.npz

Output in <results>/:
  faces.npz      aligned 224x224 RGB faces, labels, names, source files, capture session (custom data),
                 the FEI image number (1-14, -1 otherwise) and a yaw estimate from the landmarks
  unknown.npz    strangers for the open-set test: up to 500 LFW people who are not classes here
                 (one photo each) plus every photo in data/unknown/; step 3 measures how many of them
                 get a name instead of "Unknown"
  dataset_stats.json
"""
import argparse
import json
from collections import Counter

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image, ImageOps

from common import (DATA, DATASETS, IMG, IMG_EXT, N_STRANGERS, ROOT, SEED, align, find_lfw,
                    list_fei, out_dir, yaw_score)


def rel(f):
    return str(f.relative_to(ROOT) if f.is_relative_to(ROOT) else f)


def pick_lfw(rng, cfg):
    people = {}
    for d in sorted(p for p in find_lfw().iterdir() if p.is_dir()):
        files = sorted(d.glob("*.jpg"))
        if len(files) >= cfg["min_imgs"]:
            if cfg["max_imgs"] and len(files) > cfg["max_imgs"]:
                files = sorted(rng.choice(files, cfg["max_imgs"], replace=False))
            people[d.name] = [(f, -1, "") for f in files]
    return people


def pick_folders(root, cfg):
    """One sub-folder per person, optionally split into capture sessions:
    <root>/<person>/<session>/*.jpg (capture_faces.py). Loose photos in <root>/<person>/
    form a session called "default"."""
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found: put photos in {root}/<person>/*.jpg "
                                "or record them with capture_faces.py")
    people = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(f for f in d.rglob("*") if f.suffix.lower() in IMG_EXT)
        if len(files) >= cfg["min_imgs"]:
            people[d.name] = [(f, -1, f.relative_to(d).parts[0] if f.parent != d else "default")
                              for f in files]
    return people


def pick_people(dataset):
    rng = np.random.default_rng(SEED)
    people = {}
    for part in dataset.split("+"):
        cfg = DATASETS[part]
        if part == "lfw":
            got = pick_lfw(rng, cfg)
        elif part == "fei":
            got = {p: [(f, idx, "") for f, idx in v] for p, v in list_fei().items() if len(v) >= cfg["min_imgs"]}
        else:
            got = pick_folders(DATA / part, cfg)
        people.update({p: [(*t, cfg["fallback_crop"]) for t in v] for p, v in got.items()})
    return people


def pick_strangers(known):
    """[(identity, file)]: up to N_STRANGERS LFW people who are not in `known` (one random photo each)
    plus every photo under data/unknown/ (e.g. friends outside the group recorded with capture_faces.py)."""
    rng = np.random.default_rng(SEED + 1)        # own stream: the known-people selection stays unchanged
    out = []
    try:
        others = [d for d in sorted(find_lfw().iterdir()) if d.is_dir() and d.name not in known]
    except FileNotFoundError:
        others = []
    for i in rng.permutation(len(others))[:N_STRANGERS]:
        files = sorted(others[i].glob("*.jpg"))
        if files:
            out.append((f"LFW/{others[i].name}", files[rng.integers(len(files))]))
    extra = DATA / "unknown"
    if extra.is_dir():
        out += [(f"unknown/{f.parent.relative_to(extra)}", f)
                for f in sorted(extra.rglob("*")) if f.suffix.lower() in IMG_EXT]
    return out


def load_face(mtcnn, f, can_fallback):
    """(aligned face, yaw, detected) for the face nearest the image centre; None if there is no face."""
    img = np.array(ImageOps.exif_transpose(Image.open(f)).convert("RGB"))
    boxes, _, lms = mtcnn.detect(img, landmarks=True)
    if boxes is None:
        if not can_fallback:
            return None
        # LFW images are centred on the face: fall back to a centre crop
        return cv2.resize(img[45:205, 45:205], (IMG, IMG)), np.nan, False
    # several faces may appear; take the one nearest the image centre
    boxes, lms = np.asarray(boxes, np.float32), np.asarray(lms, np.float32)
    centres = (boxes[:, :2] + boxes[:, 2:]) / 2
    k = np.argmin(np.linalg.norm(centres - np.array(img.shape[1::-1], np.float32) / 2, axis=1))
    return align(img, lms[k]), yaw_score(lms[k]), True


def build_known(mtcnn, dataset, out):
    people = pick_people(dataset)
    names = list(people)
    faces, labels, files, pose_idx, yaws, sessions = [], [], [], [], [], []
    fallback, skipped = 0, []
    for li, name in enumerate(names):
        for f, idx, session, can_fallback in people[name]:
            got = load_face(mtcnn, f, can_fallback)
            if got is None:
                skipped.append(rel(f))
                continue
            faces.append(got[0])
            yaws.append(got[1])
            fallback += not got[2]
            labels.append(li)
            files.append(rel(f))
            pose_idx.append(idx)
            sessions.append(session)
        print(f"{li + 1}/{len(names)} {name}: {len(people[name])}", flush=True)

    # people whose every image was skipped would leave an empty class: drop and relabel
    used = sorted(set(labels))
    remap = {old: new for new, old in enumerate(used)}
    names = [names[i] for i in used]
    labels = np.array([remap[i] for i in labels])
    np.savez_compressed(out / "faces.npz", faces=np.stack(faces).astype(np.uint8), labels=labels,
                        names=np.array(names), files=np.array(files), pose_idx=np.array(pose_idx),
                        yaw=np.array(yaws, np.float32), sessions=np.array(sessions))
    stats = {
        "dataset": dataset,
        "n_images": int(len(labels)),
        "n_classes": len(names),
        "mtcnn_detected": int(len(labels) - fallback),
        "mtcnn_fallback": int(fallback),
        "skipped_no_face": len(skipped),
        "per_class": {names[k]: v for k, v in sorted(Counter(labels.tolist()).items())},
    }
    if any(sessions):
        stats["sessions_per_class"] = {names[k]: sorted({s for s, l in zip(sessions, labels) if l == k})
                                       for k in range(len(names))}
    if any(i >= 0 for i in pose_idx):
        # FEI: how often MTCNN finds the face at each of the 14 poses
        total = Counter(idx for v in people.values() for _, idx, _, _ in v)
        kept = Counter(pose_idx)
        stats["per_image_number"] = {int(i): {"images": total[i], "detected": kept.get(i, 0)}
                                     for i in sorted(total)}
    stats["skipped_files"] = skipped
    (out / "dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in stats.items() if k not in ("per_class", "skipped_files", "sessions_per_class")}))
    return names


def build_strangers(mtcnn, known, out):
    picks = pick_strangers(set(known))
    if not picks:
        print("no strangers: put LFW in data/ or photos in data/unknown/ to measure the open-set error")
        return
    faces, names, files = [], [], []
    for i, (who, f) in enumerate(picks, 1):
        got = load_face(mtcnn, f, False)
        if got is not None:
            faces.append(got[0])
            names.append(who)
            files.append(rel(f))
        if i % 100 == 0:
            print(f"strangers {i}/{len(picks)}", flush=True)
    np.savez_compressed(out / "unknown.npz", faces=np.stack(faces).astype(np.uint8),
                        names=np.array(names), files=np.array(files))
    print(f"unknown.npz: {len(faces)} strangers ({len(picks) - len(faces)} without a detected face)")


def main(dataset, known=True, strangers=True):
    torch.set_num_threads(8)
    mtcnn = MTCNN(keep_all=True, device="cpu")
    out = out_dir(dataset)
    if known:
        names = build_known(mtcnn, dataset, out)
    else:
        names = [str(n) for n in np.load(out / "faces.npz")["names"]]
    if strangers:
        build_strangers(mtcnn, names, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="lfw", help="lfw, fei, custom, or a combination such as lfw+custom")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--strangers-only", action="store_true", help="keep faces.npz, rebuild unknown.npz")
    g.add_argument("--no-strangers", action="store_true", help="skip unknown.npz")
    a = ap.parse_args()
    main(a.dataset, known=not a.strangers_only, strangers=not a.no_strangers)
