"""Step 2: build every partial-face variant and extract deep features.

Backbones (both frozen, used as feature extractors = transfer learning):
  - vggface : VGG-Face (VGG-16 trained on 2.6M faces / 2,622 people), fc7 4096-d
  - facenet : InceptionResnetV1 trained on VGGFace2 (3.3M faces / 9,131 people), 512-d
Output: results/features_<backbone>.npz with one array per variant.
"""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

from common import RESULTS, VARIANTS, WEIGHTS, apply_variant
from vggface import load_vggface


def build(name):
    if name == "vggface":
        model = load_vggface(WEIGHTS / "rcmalli_vggface_tf_vgg16.h5")
        prep = lambda x: x                                     # noqa: E731
    else:
        model = InceptionResnetV1(pretrained="vggface2").eval()
        prep = lambda x: (F.interpolate(x, size=160, mode="bilinear", align_corners=False) - 127.5) / 128  # noqa: E731
    return model, prep


@torch.no_grad()
def extract(model, prep, faces, bs=32):
    out = []
    for i in range(0, len(faces), bs):
        x = torch.from_numpy(faces[i:i + bs]).permute(0, 3, 1, 2).float()
        out.append(model(prep(x)).numpy())
    return np.concatenate(out).astype(np.float32)


def main(backbones):
    torch.set_num_threads(8)
    faces = np.load(RESULTS / "faces.npz")["faces"]
    for bb in backbones:
        model, prep = build(bb)
        feats = {}
        for v in VARIANTS:
            t = time.time()
            imgs = np.stack([apply_variant(f, v) for f in faces])
            feats[v] = extract(model, prep, imgs)
            print(f"{bb} {v}: {feats[v].shape} in {time.time() - t:.0f}s", flush=True)
        np.savez_compressed(RESULTS / f"features_{bb}.npz", **feats)


if __name__ == "__main__":
    main(sys.argv[1:] or ["facenet", "vggface"])
