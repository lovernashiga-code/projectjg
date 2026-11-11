"""Record face photos of one person from a webcam for the custom dataset (slide 6).

  python capture_faces.py --name Worasak                    # -> data/custom/Worasak/<today>/0001.jpg ...
  python capture_faces.py --name Worasak --session lamp     # a second session, e.g. other lighting
  python capture_faces.py --name Somchai --stranger         # someone outside the group -> data/unknown/
  python capture_faces.py --name Worasak --video worasak_s2.mp4 --session phone --count 40
                                                            # frames from a video clip instead of a webcam

SPACE = start / pause recording, q or Esc = quit. A frame is saved only while exactly one face is
visible, at most once per --interval seconds. With --video the clip is read from the file, recording
starts at once and --interval counts video time, spread so that --count frames cover the whole clip. Turn your head a little and move closer and further away.

Record every member in at least two sessions (another day, room or lighting; slide 6: "หลายแสง หลายมุม").
Frames from one session are near-duplicates, so step 3 tests each member on a whole session the model
never saw. Strangers (--stranger) are never trained on: step 3 counts how many of them still get a
member's name, which shows how high tau must be. Use a plain English name without spaces: it becomes
the folder name and the label that webcam.py draws.

Then: python step1_detect_align.py --dataset custom   (or --dataset lfw+custom), step 2, step 3.
"""
import argparse
import time

import cv2
import torch
from facenet_pytorch import MTCNN

from common import DATA


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", help="read this video file instead of the webcam (no window needed)")
    ap.add_argument("--session", default=time.strftime("%Y%m%d"), help="sub-folder name (default: today's date)")
    ap.add_argument("--stranger", action="store_true", help="someone outside the group: save to data/unknown/")
    a = ap.parse_args()

    out = DATA / ("unknown" if a.stranger else "custom") / a.name / a.session
    out.mkdir(parents=True, exist_ok=True)
    n = len(list(out.glob("*.jpg")))
    mtcnn = MTCNN(keep_all=True, device="cuda" if torch.cuda.is_available() else "cpu")
    cap = cv2.VideoCapture(a.video if a.video else a.camera)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.video or f'camera {a.camera}'}")
    recording, last, saved, frame_no = bool(a.video), -1e9, 0, 0
    if a.video:     # spread the saved frames over the whole clip
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        length = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        a.interval = max(a.interval, 0.9 * length / a.count) if length > 0 else a.interval
    while saved < a.count:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        now = frame_no / fps if a.video else time.time()
        if a.video and now - last < a.interval:
            continue                                       # skip detection between saved frames
        boxes, probs = mtcnn.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        n_faces = 0 if boxes is None else int((probs >= 0.95).sum())
        if recording and n_faces == 1 and now - last >= a.interval:
            n += 1
            saved += 1
            last = now
            cv2.imwrite(str(out / f"{n:04d}.jpg"), frame)
        if a.video:
            continue
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
    if not a.video:
        cv2.destroyAllWindows()
    print(f"saved {saved} photos to {out} ({n} in total)")


if __name__ == "__main__":
    main()
