"""Recognise every face in a video file and write an annotated copy: box + arrow + name + confidence.

  python video.py clip.mp4                                         # default model: models/lfw_M2.joblib
  python video.py clip.mp4 --model "../models/lfw+custom_M2+aug.joblib" --tau 0.6
  python video.py clip.mp4 --variant mask        # add the experiment's synthetic mask before recognising
  python video.py 0                              # webcam 0 instead of a file (q = quit)

Output (default results/video/):
  <clip>_annotated.mp4   every frame with a box around each face, an arrow from a name tag to the face,
                         the name (or "Unknown") and the Softmax probability
  <clip>_tracks.csv      one row per frame x face: frame, time, track id, box, raw prediction, p, shown name
  <clip>_summary.txt     for each track: who it was, for how many frames, mean p

Pipeline per frame (same as predict.py / webcam.py): MTCNN -> 5-point alignment -> frozen VGG-Face or
FaceNet -> Softmax -> name, or "Unknown" when p < tau. Real masks and glasses are already in the frame, so
the default --variant is "full": the M2 / M4 models learned partial faces during training (step 3).

A video adds one thing to the photo pipeline: the same person is seen in many frames. Faces are linked
from frame to frame by box overlap (IoU), and the name shown for a track is a vote over its last --window
frames weighted by p. One bad frame (motion blur, a hand in front of the face) then no longer flips the
label. A track keeps its label for --keep frames when MTCNN misses the face, e.g. when it is covered, and all
tracks are closed at a scene cut. Name tags are moved so they cover no face and no other tag.
"""
import argparse
import csv
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch

from common import RESULTS, UNKNOWN, VARIANTS
from recognizer import Recognizer

PALETTE = [(214, 120, 42), (38, 89, 217), (60, 170, 70), (170, 70, 170), (30, 160, 200),
           (200, 60, 90), (120, 120, 30), (90, 60, 200)]        # BGR, one colour per known person
GREY = (110, 110, 110)


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Track:
    def __init__(self, tid, box, window):
        self.id, self.box, self.missed, self.frames = tid, box, 0, 0
        self.votes = deque(maxlen=window)          # (name, p) of the recent frames
        self.all_votes = []                        # every frame, for the summary

    def update(self, box, name, p):
        self.box = 0.6 * box + 0.4 * self.box      # smooth the box so the arrow does not jitter
        self.missed = 0
        self.frames += 1
        self.votes.append((name, p))
        self.all_votes.append((name, p))

    def label(self, tau):
        """Name with the largest sum of p over the window, and its mean p; Unknown below tau."""
        score, count = defaultdict(float), defaultdict(int)
        for n, p in self.votes:
            score[n] += p
            count[n] += 1
        best = max(score, key=score.get)
        mean_p = score[best] / count[best]
        if best == UNKNOWN or mean_p < tau:
            return UNKNOWN, mean_p
        return best, mean_p


class Tracker:
    def __init__(self, window=15, keep=10, min_iou=0.3):
        self.tracks, self.next_id = [], 1
        self.window, self.keep, self.min_iou = window, keep, min_iou

    def step(self, faces):
        """Match this frame's faces to the tracks (greedy, highest IoU first). Returns [(track, face)]."""
        pairs = sorted(((iou(t.box, f.box), ti, fi) for ti, t in enumerate(self.tracks)
                        for fi, f in enumerate(faces)), reverse=True)
        used_t, used_f, matched = set(), set(), []
        for score, ti, fi in pairs:
            if score < self.min_iou or ti in used_t or fi in used_f:
                continue
            used_t.add(ti); used_f.add(fi)
            matched.append((self.tracks[ti], faces[fi]))
        for fi, f in enumerate(faces):
            if fi not in used_f:
                t = Track(self.next_id, f.box.astype(np.float32), self.window)
                self.next_id += 1
                self.tracks.append(t)
                matched.append((t, f))
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.missed += 1
        self.finished = [t for t in self.tracks if t.missed > self.keep]
        self.tracks = [t for t in self.tracks if t.missed <= self.keep]
        return matched


def colour_for(name, names):
    return GREY if name == UNKNOWN else PALETTE[names.index(name) % len(PALETTE)]


def overlaps(a, b, pad=4):
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])


def draw_callout(img, box, text, colour, taken, faded=False):
    """Name tag above (or below) the face with an arrow pointing at the box. `taken` holds the tags and
    boxes already drawn in this frame; the tag moves up / down until it covers none of them."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box.astype(int)
    cx = (x0 + x1) // 2
    scale = max(0.55, min(1.1, (x1 - x0) / 160))
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    tx = int(np.clip(cx - tw // 2, 4, w - tw - 12))
    step = th + 22
    gap = int(max(24, 0.3 * (y1 - y0)))
    tries = [y0 - gap - k * step for k in range(6)] + [y1 + gap + th + 14 + k * step for k in range(6)]
    tag = None
    for ty in tries:                                   # ty = baseline of the tag text
        cand = (tx - 6, ty - th - 8, tx + tw + 6, ty + 8)
        if cand[1] < 0 or cand[3] > h:
            continue
        if not any(overlaps(cand, o) for o in taken):
            tag = cand
            break
    if tag is None:                                    # no free spot: plain placement above / below
        ty = y0 - gap if y0 - gap - th - 8 > 0 else min(h - 10, y1 + gap + th + 14)
        tag = (tx - 6, ty - th - 8, tx + tw + 6, ty + 8)
    above = tag[3] <= y0
    line = 1 if faded else 2
    cv2.rectangle(img, (x0, y0), (x1, y1), colour, line + 1)
    cv2.rectangle(img, tag[:2], tag[2:], colour, -1)
    cv2.putText(img, text, (tag[0] + 6, tag[3] - 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick,
                cv2.LINE_AA)
    start = ((tag[0] + tag[2]) // 2, tag[3] if above else tag[1])
    end = (cx, y0 - 3) if above else (cx, y1 + 3)
    cv2.arrowedLine(img, start, end, colour, line + 1, cv2.LINE_AA, tipLength=min(0.3, 14 / max(1, abs(end[1] - start[1]))))
    taken.append(tag)


class CutDetector:
    """A cut in an edited clip: the mean absolute difference between small grey copies of two frames jumps
    far above its recent level (camera motion raises it slowly, a cut in one frame). The tracks are then
    closed, so a new person never inherits another person's votes."""

    def __init__(self, min_diff=15.0, ratio=5.0):
        self.prev, self.recent = None, deque(maxlen=12)
        self.min_diff, self.ratio = min_diff, ratio

    def __call__(self, frame):
        small = cv2.cvtColor(cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        cut = False
        if self.prev is not None:
            d = float(np.mean(cv2.absdiff(self.prev, small)))
            level = np.mean(self.recent) if self.recent else d
            cut = d > self.min_diff and d > self.ratio * level
            if not cut:
                self.recent.append(d)
        self.prev = small
        return cut


def h264(path):
    """OpenCV writes MPEG-4 Part 2 ("mp4v"), which the Windows player and PowerPoint may refuse. Re-encode to
    H.264 with the ffmpeg that ships with imageio-ffmpeg (pip install imageio-ffmpeg), if it is installed."""
    try:
        import subprocess
        import imageio_ffmpeg
    except ImportError:
        print("imageio-ffmpeg not installed: the mp4 stays mp4v (pip install imageio-ffmpeg for H.264)")
        return
    tmp = path.with_name(path.stem + "_h264.mp4")
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(tmp)]
    if subprocess.run(cmd).returncode == 0:
        tmp.replace(path)


def open_source(src):
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not src.isdigit() else 0
    return cap, fps if 1 <= fps <= 120 else 25.0, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="video file, or a camera index such as 0")
    ap.add_argument("--model", help="a models/*.joblib file from step 3")
    ap.add_argument("--tau", type=float,
                    help="confidence threshold (default: the model's tau that lets <= 5%% of strangers through)")
    ap.add_argument("--variant", default="full", choices=list(VARIANTS),
                    help="synthetic occlusion added before recognition (default: none, use the real frame)")
    ap.add_argument("--every", type=int, default=1, help="recognise every N-th frame (others reuse the boxes)")
    ap.add_argument("--window", type=int, default=15, help="frames in the label vote")
    ap.add_argument("--keep", type=int, default=10, help="frames a track survives without a detection")
    ap.add_argument("--min-detect", type=float, default=0.90, help="MTCNN face probability to keep a box")
    ap.add_argument("--min-face", type=int, default=40, help="smallest face (pixels) MTCNN looks for")
    ap.add_argument("--max-width", type=int, default=1280, help="frames wider than this are scaled down")
    ap.add_argument("--show", action="store_true", help="also show the frames in a window")
    ap.add_argument("--out", default=str(RESULTS / "video"))
    a = ap.parse_args()

    rec = Recognizer(a.model, a.tau, min_face=a.min_face, min_detect=a.min_detect)
    if a.tau is None and rec.bundle.get("tau_strangers_5pct") is not None:
        # a video of the group also shows people outside it: use the tau at which step 3 measured that at
        # most 5 % of strangers get a name, instead of the fixed 0.5
        rec.tau = float(rec.bundle["tau_strangers_5pct"])
    print(rec.describe())
    names = [str(n) for n in rec.names]
    cap, fps, total = open_source(a.video)
    stem = "camera" if a.video.isdigit() else Path(a.video).stem
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    writer = None
    tracker = Tracker(a.window, a.keep * a.every)
    done, log = [], []
    t0, i, cuts, is_cut = time.time(), 0, 0, CutDetector()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] > a.max_width:
            s = a.max_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if writer is None:
            writer = cv2.VideoWriter(str(out / f"{stem}_annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, frame.shape[1::-1])
        if is_cut(frame):
            done += tracker.tracks
            tracker.tracks = []
            cuts += 1
        if i % a.every == 0:
            faces = rec(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), a.variant, tau=0.0)  # tau applied per track
            for t, f in tracker.step(faces):
                t.update(f.box.astype(np.float32), f.top[0][0], f.prob)
                shown, p = t.label(rec.tau)
                log.append([i, f"{i / fps:.2f}", t.id, *np.round(f.box).astype(int), f.top[0][0],
                            f"{f.prob:.3f}", shown, f"{p:.3f}"])
            done += tracker.finished
        shown_tracks = [t for t in tracker.tracks if t.votes]
        taken = [tuple(int(v) for v in t.box) for t in shown_tracks]        # tags must not cover any face
        for t in sorted(shown_tracks, key=lambda t: -(t.box[2] - t.box[0])):   # biggest face first
            shown, p = t.label(rec.tau)
            text = f"{shown.replace('_', ' ')} {p:.0%}"
            draw_callout(frame, t.box, text, colour_for(shown, names), taken, faded=t.missed > 0)
        info = f"{rec.path.stem} | tau {rec.tau:.2f} | frame {i}"
        (iw, ih), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (4, frame.shape[0] - ih - 18), (iw + 16, frame.shape[0] - 4), (30, 30, 30), -1)
        cv2.putText(frame, info, (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        writer.write(frame)
        if a.show:
            cv2.imshow("video recognition", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
        i += 1
        if i % 50 == 0:
            rate = i / (time.time() - t0)
            print(f"frame {i}" + (f"/{total}" if total else "") + f"  {rate:.1f} frames/s", flush=True)
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    done += tracker.tracks
    h264(out / f"{stem}_annotated.mp4")

    with open(out / f"{stem}_tracks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "track", "x0", "y0", "x1", "y1", "predicted", "p", "shown", "shown_p"])
        w.writerows(log)
    lines = [f"{stem}: {i} frames, {cuts} scene cuts, model {rec.path.name}, tau {rec.tau:.2f}, variant {a.variant}", ""]
    for t in sorted(done, key=lambda t: t.id):
        if t.frames < 3:          # flickers and false detections
            continue
        score = defaultdict(float)
        for n, p in t.all_votes:
            score[n] += p
        best = max(score, key=score.get)
        ps = [p for n, p in t.all_votes if n == best]
        agree = len(ps) / len(t.all_votes)
        shown = best if np.mean(ps) >= rec.tau else UNKNOWN
        lines.append(f"track {t.id:3d}: {shown:28s} {t.frames:5d} frames  mean p {np.mean(ps):.2f}  "
                     f"same top-1 in {agree:.0%} of frames")
    (out / f"{stem}_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    secs = time.time() - t0
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    (out / f"{stem}_run.json").write_text(json.dumps({"frames": i, "seconds": round(secs, 1),
                                                      "fps": round(i / max(secs, 1e-6), 2), "device": gpu,
                                                      "tau": rec.tau}))
    print("\n".join(lines))
    print(f"\n-> {out / (stem + '_annotated.mp4')}")


if __name__ == "__main__":
    main()
