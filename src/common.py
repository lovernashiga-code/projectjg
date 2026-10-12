"""Shared settings and helpers for the partial-face recognition experiment."""
import re
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WEIGHTS = ROOT / "weights"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
FIG = RESULTS / "figures"

SEED = 42
MIN_IMGS = 20          # people with at least this many images in LFW
MAX_IMGS = 50          # cap per person to reduce class imbalance
IMG = 224              # aligned face size (VGG-Face input)
TAU = 0.5              # confidence threshold: below it the system answers "unknown" (slide 9)
UNKNOWN = "Unknown"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
N_STRANGERS = 500      # LFW people outside the training set used for the open-set ("Unknown") test

# Which images each dataset contributes. LFW follows slide 7 (>= 20 images, at most 50 per person).
# FEI has 14 images for each of 200 people and the custom set is small, so they keep everything.
# fallback_crop: if MTCNN finds no face, use a centre crop (LFW images are centred on the face);
# otherwise the image is skipped and counted in dataset_stats.json.
DATASETS = {
    "lfw":    {"min_imgs": MIN_IMGS, "max_imgs": MAX_IMGS, "fallback_crop": True},
    "fei":    {"min_imgs": 2, "max_imgs": None, "fallback_crop": False},
    "custom": {"min_imgs": 2, "max_imgs": None, "fallback_crop": False},
}

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
OCCLUDED = [v for v in VARIANTS if v != "full"]

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


def augment(img, rng):
    """One random training copy of an aligned face (slide 6): rotation within +/-15 degrees, random crop
    (zoom in up to 15 % and shift up to 5 %), horizontal flip, brightness / contrast and camera noise.
    The occlusion variant is applied afterwards, so the occluder lands slightly off the template, as a
    real mask or pair of glasses would."""
    M = cv2.getRotationMatrix2D((IMG / 2, IMG / 2), rng.uniform(-15, 15), rng.uniform(1.0, 1.15))
    M[:, 2] += rng.uniform(-0.05, 0.05, 2) * IMG
    out = cv2.warpAffine(img, M, (IMG, IMG), borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    if rng.random() < 0.5:
        out = out[:, ::-1]                                   # the template is left-right symmetric
    out = (out - 128) * rng.uniform(0.75, 1.25) + 128 + rng.uniform(-30, 30)   # contrast, brightness
    out += rng.normal(0, rng.uniform(0, 8), out.shape)                          # sensor noise
    return np.clip(out, 0, 255).astype(np.uint8)


def align(img, pts):
    """Similarity-transform a face so its 5 MTCNN landmarks land on TEMPLATE (224x224 output)."""
    M, _ = cv2.estimateAffinePartial2D(np.asarray(pts, np.float32), TEMPLATE, method=cv2.LMEDS)
    return cv2.warpAffine(img, M, (IMG, IMG), borderMode=cv2.BORDER_REFLECT)


def yaw_score(pts):
    """Head-turn estimate from the landmarks: 0 = frontal, towards -1 / +1 as the nose
    moves towards one eye (profile views go past +/-1)."""
    (lx, _), (rx, _), (nx, _) = np.asarray(pts, np.float32)[:3]
    dl, dr = nx - lx, rx - nx
    return float((dl - dr) / (abs(dl) + abs(dr) + 1e-6))


def out_dir(dataset):
    """LFW keeps the original results/ layout; other datasets get results/<dataset>/."""
    d = RESULTS if dataset == "lfw" else RESULTS / dataset
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


def find_lfw(base=DATA):
    """LFW folder with one sub-folder per person. Accepts scikit-learn's data/lfw_home/lfw and the
    Kaggle jessicali9530/lfw-dataset layout (lfw-deepfunneled/lfw-deepfunneled)."""
    legacy = base / "lfw_home" / "lfw"
    if legacy.is_dir():
        return legacy
    for marker in sorted(base.rglob("George_W_Bush")):   # every LFW copy has this person
        if marker.is_dir():
            return marker.parent
    raise FileNotFoundError(f"LFW not found under {base}. Run download_data.py or see README.")


FEI_NAME = re.compile(r"^(\d+)-(\d+)$")   # original FEI file names: <person>-<image 01..14>.jpg


def list_fei(base=DATA / "fei"):
    """{person: [(file, image number 1-14), ...]} for the FEI Face Database.

    Uses the original "<person>-<nn>.jpg" names wherever they appear under base (Kaggle copies nest
    them in originalimages_part1..4); other files such as the normalised "1a.jpg" crops are ignored.
    If no file has that name pattern, falls back to one sub-folder per person."""
    files = sorted(f for f in base.rglob("*") if f.suffix.lower() in IMG_EXT)
    if not files:
        raise FileNotFoundError(f"FEI not found under {base}. Run download_data.py or see README.")
    people, seen = {}, set()
    named = [(f, FEI_NAME.match(f.stem)) for f in files]
    if any(m for _, m in named):
        for f, m in named:
            if m and (key := (int(m[1]), int(m[2]))) not in seen:   # skip duplicate copies
                seen.add(key)
                people.setdefault(f"FEI_{key[0]:03d}", []).append((f, key[1]))
        ignored = len(files) - len(seen)
        if ignored:
            print(f"FEI: using {len(seen)} original images, ignoring {ignored} other files")
    else:
        for f in files:
            people.setdefault(f.parent.name, []).append((f, -1))
    return {p: sorted(v, key=lambda t: (t[1], str(t[0]))) for p, v in sorted(people.items())}
