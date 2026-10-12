"""Shared settings and helpers for the partial-face recognition experiment."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WEIGHTS = ROOT / "weights"
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"

SEED = 42
MIN_IMGS = 20          # people with at least this many images in LFW
MAX_IMGS = 50          # cap per person to reduce class imbalance
IMG = 224              # aligned face size (VGG-Face input)

# 5-point alignment template for a 224x224 crop (ArcFace template, loosened x1.6)
TEMPLATE = np.array([
    [82.9, 94.7],    # left eye
    [139.2, 94.4],   # right eye
    [112.0, 126.8],  # nose tip
    [88.9, 165.4],   # left mouth corner
    [135.6, 165.1],  # right mouth corner
], dtype=np.float32)

# Partial-face variants. "keep" = only this box is visible, "hide" = this box is blacked out.
# Boxes are (x0, y0, x1, y1) on the aligned 224x224 face.
VARIANTS = {
    "full":        ("keep", None),
    "upper_half":  ("keep", (0, 0, 224, 124)),       # see eyes + forehead (mouth hidden)
    "lower_half":  ("keep", (0, 124, 224, 224)),     # see nose tip + mouth (eyes hidden)
    "eyes_only":   ("keep", (30, 66, 194, 120)),
    "nose_mouth":  ("keep", (56, 108, 168, 192)),
    "mask":        ("hide", (24, 116, 200, 224)),    # surgical mask: nose down hidden
    "sunglasses":  ("hide", (44, 72, 180, 116)),     # dark glasses over both eyes
}

VARIANT_TH = {
    "full": "ใบหน้าเต็ม",
    "upper_half": "ครึ่งบน",
    "lower_half": "ครึ่งล่าง",
    "eyes_only": "เฉพาะดวงตา",
    "nose_mouth": "เฉพาะจมูก+ปาก",
    "mask": "สวมหน้ากากอนามัย",
    "sunglasses": "สวมแว่นกันแดด",
}


def apply_variant(img, name):
    """Return a copy of an aligned face (H, W, 3 uint8) with the variant's occlusion."""
    mode, box = VARIANTS[name]
    if box is None:
        return img.copy()
    x0, y0, x1, y1 = box
    if mode == "keep":
        out = np.zeros_like(img)
        out[y0:y1, x0:x1] = img[y0:y1, x0:x1]
    else:
        out = img.copy()
        out[y0:y1, x0:x1] = 0
    return out
