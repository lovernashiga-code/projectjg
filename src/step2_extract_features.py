"""Step 2: build every partial-face variant and extract deep features.

  python step2_extract_features.py                    # LFW, both backbones
  python step2_extract_features.py --dataset fei vggface
  python step2_extract_features.py --force            # re-extract even if the features are up to date

Backbones (both frozen, used as feature extractors = transfer learning), see backbones.py:
  - vggface : VGG-Face fc7, 4096-d
  - facenet : FaceNet (InceptionResnetV1, VGGFace2), 512-d
Output: <results>/features_<backbone>.npz (and unknown_features_<backbone>.npz for the strangers
in unknown.npz) with one array per variant. A file newer than its faces file is kept as it is.
"""
import argparse
import time

import numpy as np
import torch

from backbones import BACKBONES, extract, load_backbone
from common import VARIANTS, apply_variant, out_dir

JOBS = [("faces.npz", "features_{}.npz"), ("unknown.npz", "unknown_features_{}.npz")]


def main(dataset, backbones, force):
    torch.set_num_threads(8)
    out = out_dir(dataset)
    if not (out / "faces.npz").exists():
        raise SystemExit(f"{out / 'faces.npz'} not found: run step1_detect_align.py --dataset {dataset} first")
    for bb in backbones:
        todo = []
        for src, dst in JOBS:
            src, dst = out / src, out / dst.format(bb)
            if not src.exists():
                continue
            if not force and dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
                print(f"{dst.name} is up to date, skipping (--force to redo)")
                continue
            todo.append((src, dst))
        if not todo:
            continue
        model, prep = load_backbone(bb)
        for src, dst in todo:
            faces = np.load(src)["faces"]
            feats = {}
            for v in VARIANTS:
                t = time.time()
                imgs = np.stack([apply_variant(f, v) for f in faces])
                feats[v] = extract(model, prep, imgs)
                print(f"{dst.name} {v}: {feats[v].shape} in {time.time() - t:.0f}s", flush=True)
            np.savez_compressed(dst, **feats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="lfw")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("backbones", nargs="*", help=f"any of {', '.join(BACKBONES)} (default: both)")
    a = ap.parse_args()
    main(a.dataset, a.backbones or ["facenet", "vggface"], a.force)
