"""Step 2: build every partial-face variant and extract deep features.

  python step2_extract_features.py                    # LFW, both backbones
  python step2_extract_features.py --dataset fei vggface

Backbones (both frozen, used as feature extractors = transfer learning), see backbones.py:
  - vggface : VGG-Face fc7, 4096-d
  - facenet : FaceNet (InceptionResnetV1, VGGFace2), 512-d
Output: <results>/features_<backbone>.npz with one array per variant.
"""
import argparse
import time

import numpy as np
import torch

from backbones import BACKBONES, extract, load_backbone
from common import VARIANTS, apply_variant, out_dir


def main(dataset, backbones):
    torch.set_num_threads(8)
    out = out_dir(dataset)
    faces = np.load(out / "faces.npz")["faces"]
    for bb in backbones:
        model, prep = load_backbone(bb)
        feats = {}
        for v in VARIANTS:
            t = time.time()
            imgs = np.stack([apply_variant(f, v) for f in faces])
            feats[v] = extract(model, prep, imgs)
            print(f"{bb} {v}: {feats[v].shape} in {time.time() - t:.0f}s", flush=True)
        np.savez_compressed(out / f"features_{bb}.npz", **feats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="lfw")
    ap.add_argument("backbones", nargs="*", help=f"any of {', '.join(BACKBONES)} (default: both)")
    a = ap.parse_args()
    main(a.dataset, a.backbones or ["facenet", "vggface"])
