"""Record face photos of one person from a webcam for the custom dataset (slide 6).

  python capture_faces.py --name Worasak                 # -> data/custom/Worasak/0001.jpg, 0002.jpg, ...
  python capture_faces.py --name Worasak --count 40 --interval 0.4

SPACE = start / pause recording, q or Esc = quit. A frame is saved only while exactly one face is
visible, at most once per --interval seconds. Turn your head a little, move closer and further away,
and record in two or three lighting conditions (slide 6: "หลายแสง หลายมุม"). Use a plain English name
without spaces: it becomes the folder name and the label that webcam.py draws.

Then: python step1_detect_align.py --dataset custom   (or --dataset lfw+custom), step 2, step 3.
"""
import argparse
import time

import cv2
from facenet_pytorch import MTCNN

from common import DATA


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()

    out = DATA / "custom" / a.name
    out.mkdir(parents=True, exist_ok=True)
    n = len(list(out.glob("*.jpg")))
    mtcnn = MTCNN(keep_all=True, device="cpu")
    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {a.camera}")
    recording, last, saved = False, 0.0, 0
    while saved < a.count:
        ok, frame = cap.read()
        if not ok:
            break
        boxes, _ = mtcnn.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        n_faces = 0 if boxes is None else len(boxes)
        if recording and n_faces == 1 and time.time() - last >= a.interval:
            n += 1
            saved += 1
            last = time.time()
            cv2.imwrite(str(out / f"{n:04d}.jpg"), frame)
        view = frame.copy()
        for b in [] if boxes is None else boxes.astype(int):
            color = (70, 170, 60) if n_faces == 1 else (60, 60, 220)
            cv2.rectangle(view, tuple(map(int, b[:2])), tuple(map(int, b[2:])), color, 2)
        state = "REC" if recording else "paused (SPACE)"
        cv2.putText(view, f"{a.name}: {saved}/{a.count} {state} | faces: {n_faces}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("capture", view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            recording = not recording
    cap.release()
    cv2.destroyAllWindows()
    print(f"saved {saved} photos to {out} ({n} in total)")


if __name__ == "__main__":
    main()
