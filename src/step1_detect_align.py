"""Step 1: pick identities, detect faces with MTCNN, align to a 5-point template.

  python step1_detect_align.py                      # LFW (default, slide 7)
  python step1_detect_align.py --dataset fei        # FEI Face Database: 200 people x 14 poses
  python step1_detect_align.py --dataset custom     # data/custom/<person>/*.jpg (capture_faces.py)
  python step1_detect_align.py --dataset lfw+custom # LFW people plus the group members

Output: <results>/faces.npz with aligned 224x224 RGB faces, labels, names, source files,
the FEI image number (1-14, -1 for other datasets) and a yaw estimate from the landmarks,
plus <results>/dataset_stats.json.
"""
import argparse
import json
from collections import Counter

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image, ImageOps

from common import (DATA, DATASETS, IMG, IMG_EXT, ROOT, SEED, align, find_lfw, list_fei,
                    out_dir, yaw_score)


def pick_lfw(rng, cfg):
    people = {}
    for d in sorted(p for p in find_lfw().iterdir() if p.is_dir()):
        files = sorted(d.glob("*.jpg"))
        if len(files) >= cfg["min_imgs"]:
            if cfg["max_imgs"] and len(files) > cfg["max_imgs"]:
                files = sorted(rng.choice(files, cfg["max_imgs"], replace=False))
            people[d.name] = [(f, -1) for f in files]
    return people


def pick_folders(root, cfg):
    """One sub-folder per person (custom dataset)."""
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found: put photos in {root}/<person>/*.jpg "
                                "or record them with capture_faces.py")
    people = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
        if len(files) >= cfg["min_imgs"]:
            people[d.name] = [(f, -1) for f in files]
    return people


def pick_people(dataset):
    rng = np.random.default_rng(SEED)
    people = {}
    for part in dataset.split("+"):
        cfg = DATASETS[part]
        if part == "lfw":
            got = pick_lfw(rng, cfg)
        elif part == "fei":
            got = {p: v for p, v in list_fei().items() if len(v) >= cfg["min_imgs"]}
        else:
            got = pick_folders(DATA / part, cfg)
        people.update({p: [(f, idx, cfg["fallback_crop"]) for f, idx in v] for p, v in got.items()})
    return people


def main(dataset):
    torch.set_num_threads(8)
    mtcnn = MTCNN(keep_all=True, device="cpu")
    people = pick_people(dataset)
    names = list(people)
    faces, labels, files, pose_idx, yaws = [], [], [], [], []
    fallback, skipped = 0, []
    for li, name in enumerate(names):
        for f, idx, can_fallback in people[name]:
            img = np.array(ImageOps.exif_transpose(Image.open(f)).convert("RGB"))
            boxes, probs, lms = mtcnn.detect(img, landmarks=True)
            if boxes is None:
                if not can_fallback:
                    skipped.append(str(f.relative_to(ROOT) if f.is_relative_to(ROOT) else f))
                    continue
                # LFW images are centred on the face: fall back to a centre crop
                fallback += 1
                c = img[45:205, 45:205]
                faces.append(cv2.resize(c, (IMG, IMG)))
                yaws.append(np.nan)
            else:
                # several faces may appear; take the one nearest the image centre
                boxes, lms = np.asarray(boxes, np.float32), np.asarray(lms, np.float32)
                centres = (boxes[:, :2] + boxes[:, 2:]) / 2
                centre = np.array(img.shape[1::-1], np.float32) / 2
                k = np.argmin(np.linalg.norm(centres - centre, axis=1))
                faces.append(align(img, lms[k]))
                yaws.append(yaw_score(lms[k]))
            labels.append(li)
            files.append(str(f.relative_to(ROOT) if f.is_relative_to(ROOT) else f))
            pose_idx.append(idx)
        print(f"{li + 1}/{len(names)} {name}: {len(people[name])}", flush=True)

    # people whose every image was skipped would leave an empty class: drop and relabel
    used = sorted(set(labels))
    remap = {old: new for new, old in enumerate(used)}
    names = [names[i] for i in used]
    labels = np.array([remap[i] for i in labels])
    faces = np.stack(faces).astype(np.uint8)
    out = out_dir(dataset)
    np.savez_compressed(out / "faces.npz", faces=faces, labels=labels, names=np.array(names),
                        files=np.array(files), pose_idx=np.array(pose_idx), yaw=np.array(yaws, np.float32))
    stats = {
        "dataset": dataset,
        "n_images": int(len(labels)),
        "n_classes": len(names),
        "mtcnn_detected": int(len(labels) - fallback),
        "mtcnn_fallback": int(fallback),
        "skipped_no_face": len(skipped),
        "per_class": {names[k]: v for k, v in sorted(Counter(labels.tolist()).items())},
    }
    if any(i >= 0 for i in pose_idx):
        # FEI: how often MTCNN finds the face at each of the 14 poses
        total = Counter(idx for v in people.values() for _, idx, _ in v)
        kept = Counter(pose_idx)
        stats["per_image_number"] = {int(i): {"images": total[i], "detected": kept.get(i, 0)}
                                     for i in sorted(total)}
    stats["skipped_files"] = skipped
    (out / "dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in stats.items() if k not in ("per_class", "skipped_files")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="lfw", help="lfw, fei, custom, or a combination such as lfw+custom")
    main(ap.parse_args().dataset)
