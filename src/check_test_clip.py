"""Score video.py on the LFW test clip from make_test_clip.py and save frames for the slides.

  python make_test_clip.py
  python video.py ../results/video/lfw_test_clip.mp4 --model ../models/lfw_M2.joblib
  python check_test_clip.py          # -> results/video/video_check.json, frame_scene1..3.png

Each track's name is the p-weighted vote over all its frames (as in <clip>_summary.txt); a track is matched
to the person who stood at that position in that scene.
"""
import csv
import json
from collections import defaultdict

import cv2

from common import RESULTS, UNKNOWN

FPS, SCENE = 25, 100       # make_test_clip.py: 25 fps, 4 s per scene


def main(d, model):
    run = json.loads((d / "lfw_test_clip_run.json").read_text())
    tau = run.get("tau", 0.5)                 # the tau video.py used
    truth = json.loads((RESULTS / "video" / "lfw_test_clip_truth.json").read_text())
    votes = defaultdict(lambda: defaultdict(float))
    where = defaultdict(list)
    with open(d / "lfw_test_clip_tracks.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            scene = int(r["frame"]) // SCENE + 1
            cx = (int(r["x0"]) + int(r["x1"])) / 2
            pos = ["left", "middle", "right"][min(2, int(cx // (1280 / 3)))]
            votes[(scene, pos, r["track"])][r["predicted"]] += float(r["p"])
            where[(scene, pos, r["track"])].append(float(r["p"]))
    rows = []
    for t in truth:
        tracks = [k for k in votes if k[0] == t["scene"] and k[1] == t["position"]]
        if not tracks:
            rows.append({**t, "shown": "not detected", "p": 0.0})
            continue
        k = max(tracks, key=lambda k: len(where[k]))              # the longest track at that spot
        best = max(votes[k], key=votes[k].get)
        n = sum(1 for _ in where[k])
        p = votes[k][best] / n
        rows.append({**t, "shown": best if p >= tau else UNKNOWN, "p": round(p, 3), "frames": n})
    cap = cv2.VideoCapture(str(d / "lfw_test_clip_annotated.mp4"))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for s in range(3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, s * SCENE + 60)
        ok, fr = cap.read()
        if ok:
            cv2.imwrite(str(d / f"frame_scene{s + 1}.png"), fr)
    info = {"model": model, "total": len(rows), "correct": sum(r["shown"] == r["true"] for r in rows),
            "frames": frames, "rows": rows}
    info.update({k: v for k, v in run.items() if k in ("fps", "device", "tau")})   # measured by video.py
    (d / "video_check.json").write_text(json.dumps(info, indent=2))
    for r in rows:
        print(f"scene {r['scene']} {r['position']:6} {r['occluder']:8} {r['true']:28} -> {r['shown']:28} p={r['p']}")
    print(f"{info['correct']}/{info['total']} correct")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(RESULTS / "video"), help="the --out folder given to video.py")
    ap.add_argument("--model", default="lfw_M2.joblib", help="model name to record in video_check.json")
    a = ap.parse_args()
    from pathlib import Path
    main(Path(a.dir), a.model)
