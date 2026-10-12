"""Step 1: pick LFW identities, detect faces with MTCNN, align to a 5-point template.

Output: results/faces.npz with aligned 224x224 RGB faces, labels and names.
"""
import json
from collections import Counter

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image

from common import DATA, IMG, MAX_IMGS, MIN_IMGS, RESULTS, SEED, TEMPLATE

LFW_DIR = DATA / "lfw_home" / "lfw"


def pick_people():
    rng = np.random.default_rng(SEED)
    people = {}
    for d in sorted(LFW_DIR.iterdir()):
        files = sorted(d.glob("*.jpg"))
        if len(files) >= MIN_IMGS:
            if len(files) > MAX_IMGS:
                files = sorted(rng.choice(files, MAX_IMGS, replace=False))
            people[d.name] = files
    return people


def align(img, pts):
    M, _ = cv2.estimateAffinePartial2D(pts.astype(np.float32), TEMPLATE, method=cv2.LMEDS)
    return cv2.warpAffine(img, M, (IMG, IMG), borderMode=cv2.BORDER_REFLECT)


def main():
    torch.set_num_threads(8)
    mtcnn = MTCNN(keep_all=True, device="cpu")
    people = pick_people()
    names = list(people)
    faces, labels, fallback = [], [], 0
    for li, name in enumerate(names):
        for f in people[name]:
            img = np.array(Image.open(f).convert("RGB"))
            boxes, probs, lms = mtcnn.detect(img, landmarks=True)
            if boxes is None:
                # LFW images are centred on the face: fall back to a centre crop
                fallback += 1
                c = img[45:205, 45:205]
                faces.append(cv2.resize(c, (IMG, IMG)))
            else:
                # several faces may appear; take the one nearest the image centre
                boxes, lms = np.asarray(boxes, np.float32), np.asarray(lms, np.float32)
                centres = (boxes[:, :2] + boxes[:, 2:]) / 2
                k = np.argmin(np.linalg.norm(centres - np.array([125, 125]), axis=1))
                faces.append(align(img, lms[k]))
            labels.append(li)
        print(f"{li + 1}/{len(names)} {name}: {len(people[name])}", flush=True)

    faces = np.stack(faces).astype(np.uint8)
    labels = np.array(labels)
    np.savez_compressed(RESULTS / "faces.npz", faces=faces, labels=labels, names=np.array(names))
    stats = {
        "n_images": int(len(labels)),
        "n_classes": len(names),
        "mtcnn_detected": int(len(labels) - fallback),
        "mtcnn_fallback": int(fallback),
        "per_class": {names[k]: v for k, v in sorted(Counter(labels.tolist()).items())},
    }
    (RESULTS / "dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in stats.items() if k != "per_class"}))


if __name__ == "__main__":
    main()
