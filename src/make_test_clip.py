"""Build a short test video from LFW photos of the TEST split, to check video.py before the members' clips exist.

  python make_test_clip.py                 # -> results/video/lfw_test_clip.mp4 (about 12 s, 3 scenes)
  python video.py ../results/video/lfw_test_clip.mp4 --model ../models/lfw_M2.joblib

Every scene puts three different people side by side on a 1280x720 canvas. The photos drift slowly (a
fake camera pan) and some faces get a real-looking occluder drawn on the photo itself, from the MTCNN
landmarks: a light-blue surgical mask from the nose down, dark glasses over the eyes, or a hand-like
block over one side. Unlike the experiment's variants these are not aligned boxes, so this checks the
whole video pipeline. Only test-split photos are used, so a model saved WITHOUT --fit-all has never seen them.
"""
import json

import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

from common import RESULTS, ROOT, out_dir

W, H, FPS, SECONDS = 1280, 720, 25, 4


def occlude(img, pts, kind):
    """Draw an occluder on a raw photo (RGB) using its 5 landmarks (eyes, nose, mouth corners)."""
    out = img.copy()
    le, re, nose, lm, rm = pts.astype(np.float32)
    eye_d = np.linalg.norm(re - le)
    if kind == "mask":
        top = nose[1] - 0.15 * eye_d
        chin = max(lm[1], rm[1]) + 0.9 * eye_d
        poly = np.array([[le[0] - 0.75 * eye_d, top], [re[0] + 0.75 * eye_d, top],
                         [rm[0] + 0.55 * eye_d, chin - 0.2 * eye_d], [(lm[0] + rm[0]) / 2, chin],
                         [lm[0] - 0.55 * eye_d, chin - 0.2 * eye_d]], np.int32)
        cv2.fillPoly(out, [poly], (170, 205, 230))
        for k in (0.3, 0.55, 0.8):                       # pleats
            y = int(top + k * (chin - top) * 0.8)
            cv2.line(out, (int(le[0] - 0.6 * eye_d), y), (int(re[0] + 0.6 * eye_d), y), (140, 175, 205), 2)
    elif kind == "glasses":
        r = int(0.38 * eye_d)
        for c in (le, re):
            cv2.ellipse(out, (int(c[0]), int(c[1])), (r, int(0.75 * r)), 0, 0, 360, (25, 25, 30), -1)
        cv2.line(out, (int(le[0] + r), int(le[1])), (int(re[0] - r), int(re[1])), (25, 25, 30), 3)
    elif kind == "hand":
        x0 = int(nose[0])
        cv2.rectangle(out, (x0, int(le[1] - 0.6 * eye_d)), (int(re[0] + 1.2 * eye_d), int(rm[1] + 0.9 * eye_d)),
                      (198, 160, 135), -1)
    return out


def main():
    out = out_dir("lfw")
    d = np.load(out / "faces.npz")
    split = np.load(out / "split.npz")
    names, labels, files = [str(n) for n in d["names"]], d["labels"], d["files"]
    rng = np.random.default_rng(5)
    test = split["test"]
    people = rng.choice(len(names), 9, replace=False)
    kinds = [["none", "mask", "glasses"], ["mask", "hand", "none"], ["glasses", "mask", "mask"]]
    mtcnn = MTCNN(keep_all=True, device="cpu")
    dst = RESULTS / "video"
    dst.mkdir(parents=True, exist_ok=True)
    path = dst / "lfw_test_clip.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    truth = []
    for s in range(3):
        tiles = []
        for c in range(3):
            k = people[3 * s + c]
            i = rng.choice([j for j in test if labels[j] == k])
            img = np.array(Image.open(ROOT / str(files[i])).convert("RGB"))
            boxes, _, lms = mtcnn.detect(img, landmarks=True)
            boxes, lms = np.asarray(boxes, np.float32), np.asarray(lms, np.float32)
            centres = (boxes[:, :2] + boxes[:, 2:]) / 2
            pts = lms[np.argmin(np.linalg.norm(centres - np.array(img.shape[1::-1]) / 2, axis=1))]
            if kinds[s][c] != "none":
                img = occlude(img, pts, kinds[s][c])
            tiles.append(cv2.resize(img, (380, 380), interpolation=cv2.INTER_CUBIC))
            truth.append({"scene": s + 1, "position": ["left", "middle", "right"][c], "true": names[k],
                          "occluder": kinds[s][c], "file": str(files[i])})
        for f in range(FPS * SECONDS):
            canvas = np.full((H, W, 3), 236, np.uint8)
            dx = int(30 * np.sin(2 * np.pi * f / (FPS * SECONDS)))
            for c, t in enumerate(tiles):
                x = 40 + c * 410 + dx
                y = 170 + int(12 * np.sin(2 * np.pi * f / 50 + c))
                canvas[y:y + 380, x:x + 380] = t
            cv2.putText(canvas, f"test clip from LFW test photos - scene {s + 1}/3", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 60), 2, cv2.LINE_AA)
            vw.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    vw.release()
    (dst / "lfw_test_clip_truth.json").write_text(json.dumps(truth, indent=2))
    print(f"-> {path}\n" + "\n".join(f"scene {t['scene']} {t['position']:6} {t['true']:28} {t['occluder']}"
                                     for t in truth))


if __name__ == "__main__":
    main()
