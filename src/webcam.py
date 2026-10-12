"""Live recognition from a webcam (slide 3: testing pipeline with webcam support).

  python webcam.py                                     # default model: models/lfw_M2.joblib, camera 0
  python webcam.py --model ../models/custom_M2.joblib --camera 1

Keys: v = next synthetic occlusion (full, upper_half, ..., sunglasses)   + / - = raise / lower tau
      s = save the current frame to results/webcam/                      q or Esc = quit
The right edge shows the aligned faces exactly as the classifier sees them.
"""
import argparse
import time

import cv2

from common import RESULTS, VARIANTS
from recognizer import Recognizer, draw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="a models/*.joblib file from step 3")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--variant", default="full", choices=list(VARIANTS))
    ap.add_argument("--tau", type=float)
    ap.add_argument("--every", type=int, default=2, help="recognise every N-th frame (CPU is slow)")
    a = ap.parse_args()

    rec = Recognizer(a.model, a.tau)
    print(rec.describe())
    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {a.camera}")
    variants = list(VARIANTS)
    vi, faces, n, fps, t = variants.index(a.variant), [], 0, 0.0, time.time()
    save_dir = RESULTS / "webcam"
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % a.every == 0:
            faces = rec(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), variants[vi])
            now = time.time()
            fps, t = 0.8 * fps + 0.2 / max(now - t, 1e-3), now
        n += 1
        draw(frame, faces)
        for i, f in enumerate(faces[:4]):
            y0 = 40 + i * 100
            if y0 + 96 <= frame.shape[0]:
                frame[y0:y0 + 96, -96:] = cv2.cvtColor(cv2.resize(f.face, (96, 96)), cv2.COLOR_RGB2BGR)
        cv2.putText(frame, f"{variants[vi]} | tau {rec.tau:.2f} | {fps:.1f} rec/s | v s +/- q",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow("Face recognition from partial face", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("v"):
            vi = (vi + 1) % len(variants)
        elif key in (ord("+"), ord("=")):
            rec.tau = min(0.99, rec.tau + 0.05)
        elif key == ord("-"):
            rec.tau = max(0.0, rec.tau - 0.05)
        elif key == ord("s"):
            save_dir.mkdir(parents=True, exist_ok=True)
            path = save_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{variants[vi]}.jpg"
            cv2.imwrite(str(path), frame)
            print("saved", path)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
