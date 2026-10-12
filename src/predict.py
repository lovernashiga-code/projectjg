"""Recognise every face in one or more photos (or folders of photos).

  python predict.py photo.jpg                     # default model: models/lfw_M2.joblib
  python predict.py my_photos/ --model ../models/custom_M2.joblib
  python predict.py photo.jpg --variant mask --tau 0.6

Prints the prediction and top-5 for every face and writes annotated copies to results/predict/.
--variant puts the same synthetic occlusion as the experiment on the aligned face first.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from common import IMG_EXT, RESULTS, VARIANTS
from recognizer import Recognizer, draw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="image files or folders")
    ap.add_argument("--model", help="a models/*.joblib file from step 3")
    ap.add_argument("--variant", default="full", choices=list(VARIANTS))
    ap.add_argument("--tau", type=float, help="confidence threshold (default: the model's, 0.5)")
    ap.add_argument("--out", default=str(RESULTS / "predict"))
    a = ap.parse_args()

    rec = Recognizer(a.model, a.tau)
    print(rec.describe())
    files = []
    for x in map(Path, a.images):
        files += sorted(f for f in x.rglob("*") if f.suffix.lower() in IMG_EXT) if x.is_dir() else [x]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        rgb = np.array(ImageOps.exif_transpose(Image.open(f)).convert("RGB"))
        faces = rec(rgb, a.variant)
        print(f"\n{f}: {len(faces)} face(s)")
        for i, r in enumerate(faces, 1):
            print(f"  {i}. {r.name}  p={r.prob:.3f}   top-5: " + ", ".join(f"{n} {p:.2f}" for n, p in r.top))
        cv2.imwrite(str(out / f"{f.stem}_{a.variant}.jpg"), draw(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), faces))
    print(f"\nannotated images in {out}")


if __name__ == "__main__":
    main()
