"""Step 2: build every partial-face variant and extract deep features.

  python step2_extract_features.py                    # LFW, both backbones
  python step2_extract_features.py --dataset fei vggface
  python step2_extract_features.py --augment 2        # also 2 augmented copies of every face (slide 6)
  python step2_extract_features.py --force            # re-extract even if the features are up to date

Backbones (both frozen, used as feature extractors = transfer learning), see backbones.py:
  - vggface : VGG-Face fc7, 4096-d
  - facenet : FaceNet (InceptionResnetV1, VGGFace2), 512-d
Output in <results>/, one array per variant:
  features_<backbone>.npz          the aligned faces in faces.npz
  unknown_features_<backbone>.npz  the strangers in unknown.npz
  features_aug_<backbone>.npz      with --augment K: (K, N, D) features of K random copies of every face
                                   (common.augment, then the occlusion). Copies are made for every face,
                                   but step 3 trains only on the copies of its training images.
A file newer than its faces file (and with the same K) is kept as it is. A GPU is used when available;
--augment multiplies the extraction time by K.
"""
import argparse
import time

import numpy as np
import torch

from backbones import BACKBONES, DEVICE, extract, load_backbone
from common import SEED, VARIANTS, apply_variant, augment, out_dir


def stale(src, dst, force, copies=0):
    if force or not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        return True
    if copies:
        with np.load(dst) as f:
            return f["full"].shape[0] != copies
    return False


def variant_features(model, prep, faces, name):
    feats = {}
    for v in VARIANTS:
        t = time.time()
        feats[v] = extract(model, prep, np.stack([apply_variant(f, v) for f in faces]))
        print(f"{name} {v}: {feats[v].shape} in {time.time() - t:.0f}s", flush=True)
    return feats


def augmented_features(model, prep, faces, copies, name):
    rng = np.random.default_rng(SEED)
    per_copy = [variant_features(model, prep, np.stack([augment(f, rng) for f in faces]),
                                 f"{name} copy {k + 1}/{copies}") for k in range(copies)]
    return {v: np.stack([c[v] for c in per_copy]) for v in VARIANTS}


def main(dataset, backbones, force, copies):
    torch.set_num_threads(8)
    out = out_dir(dataset)
    if not (out / "faces.npz").exists():
        raise SystemExit(f"{out / 'faces.npz'} not found: run step1_detect_align.py --dataset {dataset} first")
    print("device:", DEVICE)
    for bb in backbones:
        jobs = [(out / "faces.npz", out / f"features_{bb}.npz", 0),
                (out / "unknown.npz", out / f"unknown_features_{bb}.npz", 0)]
        if copies:
            jobs.append((out / "faces.npz", out / f"features_aug_{bb}.npz", copies))
        todo = []
        for src, dst, k in jobs:
            if not src.exists():
                continue
            if stale(src, dst, force, k):
                todo.append((src, dst, k))
            else:
                print(f"{dst.name} is up to date, skipping (--force to redo)")
        if not todo:
            continue
        model, prep = load_backbone(bb)
        for src, dst, k in todo:
            faces = np.load(src)["faces"]
            feats = (augmented_features(model, prep, faces, k, dst.name) if k
                     else variant_features(model, prep, faces, dst.name))
            np.savez_compressed(dst, **feats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="lfw")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--augment", type=int, default=0, metavar="K",
                    help="augmented copies per face (0 = none; 2 for LFW, 5 for a small custom set)")
    ap.add_argument("backbones", nargs="*", help=f"any of {', '.join(BACKBONES)} (default: both)")
    a = ap.parse_args()
    main(a.dataset, a.backbones or ["facenet", "vggface"], a.force, a.augment)
