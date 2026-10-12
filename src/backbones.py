"""Frozen feature extractors (transfer learning: the pretrained weights are never updated).

  vggface : VGG-Face (VGG-16 trained on 2.6M faces / 2,622 people), fc7 4096-d
  facenet : InceptionResnetV1 trained on VGGFace2 (3.3M faces / 9,131 people), 512-d
"""
import numpy as np
import torch
import torch.nn.functional as F

from common import WEIGHTS

BACKBONES = ("vggface", "facenet")


def load_backbone(name):
    """Return (model, prep): prep maps a uint8-range RGB batch (N, 3, 224, 224) to the model input."""
    if name == "vggface":
        from vggface import load_vggface
        path = WEIGHTS / "rcmalli_vggface_tf_vgg16.h5"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run download_data.py --weights (see README)")
        model = load_vggface(path)
        prep = lambda x: x                                     # noqa: E731
    elif name == "facenet":
        from facenet_pytorch import InceptionResnetV1
        model = InceptionResnetV1(pretrained="vggface2").eval()
        prep = lambda x: (F.interpolate(x, size=160, mode="bilinear", align_corners=False) - 127.5) / 128  # noqa: E731
    else:
        raise ValueError(f"unknown backbone {name!r}, expected one of {BACKBONES}")
    return model, prep


@torch.no_grad()
def extract(model, prep, faces, bs=32):
    """faces: (N, 224, 224, 3) uint8 aligned faces -> (N, D) float32 features."""
    out = []
    for i in range(0, len(faces), bs):
        x = torch.from_numpy(np.ascontiguousarray(faces[i:i + bs])).permute(0, 3, 1, 2).float()
        out.append(model(prep(x)).numpy())
    return np.concatenate(out).astype(np.float32)
