"""Group-member demo: numbers and figures for the member slides (run after step 1-5 on lfw+custom and video.py).

  python members_eval.py                       # -> results/lfw+custom/members_accuracy.json, operating_point.json,
                                               #    video_scores.json, figures/demo/*.png
  python members_eval.py --truth ../VDO/test/truth.json --models M2 M4

Expects
  results/lfw+custom/            from step1-5 --dataset lfw+custom (data/custom/<member>/s1, s2; data/unknown/<stranger>/)
  models/lfw+custom_<M>.joblib   from step 3 --fit-all (the demo models)
  results/video/members_<M>/     video.py <clip> --model ../models/lfw+custom_<M>.joblib --out ../results/video/members_<M>
  VDO/test/truth.json            who is where in each test clip (see that file)

What it computes
  members_accuracy.json  per method x variant: accuracy on the members' held-out session only (argmax, no tau)
  operating_point.json   at each model's own tau (tau_strangers_5pct): members right AND named / members -> Unknown
                         (held-out session, train-split classifier from probs.npz), and the share of the real stranger's
                         frames and of the LFW strangers that get a name (deployed fit-all model; strangers are never
                         trained on). Also the tau re-chosen without the real stranger, to show he did not set it.
  video_scores.json      every track of >= 10 rows in every test clip: its most frequent shown label vs the truth.
                         A member is right when named correctly; the stranger is right when shown as Unknown.
  figures/demo/          members_enrol.png (enrolment faces) and annotated frames from the demo videos
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize

import step4_figures  # noqa: F401  (fonts)
import step3_train_evaluate as s3
from common import ROOT, SEED, VARIANTS, out_dir

DATASET = "lfw+custom"
V3 = ["full", "mask", "sunglasses"]


def members_accuracy(out, members):
    d, P = np.load(out / "faces.npz"), np.load(out / "probs.npz")
    names = [str(n) for n in d["names"]]
    y = P["y_test"]
    rows = np.isin(y, [names.index(n) for n in members])
    keys = sorted({k.split("|")[0] for k in P.files if "|" in k})
    res = {m: {v: float((P[f"{m}|{v}"].argmax(1)[rows] == y[rows]).mean()) for v in VARIANTS} for m in keys if m != "M0"}
    (out / "members_accuracy.json").write_text(json.dumps(res, indent=1))
    return res, P, y, rows


def operating_point(out, models, members, stranger, P, y, rows):
    un = np.load(out / "unknown.npz")["names"]
    real = np.array([stranger in str(n) for n in un])
    d = np.load(out / "faces.npz")
    names = [str(n) for n in d["names"]]
    feats = {bb: {v: normalize(np.load(out / f"features_{bb}.npz")[v]) for v in VARIANTS} for bb in ("vggface", "facenet")}
    idx_tr, idx_te, _ = s3.make_split(d["labels"], d["sessions"], names, SEED)
    res = {}
    for m in models:
        b = joblib.load(ROOT / "models" / f"{DATASET}_{m}.joblib")
        tau, bb = b["tau_strangers_5pct"], b["backbone"]
        F = np.load(out / f"unknown_features_{bb}.npz")
        split_clf = s3.fit_deep(m, feats, {}, d["labels"], idx_tr)        # same classifier step 3 chose tau with
        U = normalize(F["full"])
        tau_lfw = s3.tau_for(s3.open_set(split_clf, feats[bb]["full"][idx_te], d["labels"][idx_te], U[~real]))
        res[m] = {"tau": tau, "tau_without_real_stranger": tau_lfw["tau"] if tau_lfw else None}
        if tau is None:
            continue
        for v in V3:
            p = P[f"{m}|{v}"][rows]
            ok, conf = p.argmax(1) == y[rows], p.max(1)
            pu = b["classifier"].predict_proba(normalize(F[v]))
            cu = pu.max(1)
            nm = np.array([b["names"][b["classifier"].classes_[k]] for k in pu.argmax(1)])
            res[m][v] = {"members_argmax": float(ok.mean()),
                         "members_right_at_tau": float((ok & (conf >= tau)).mean()),
                         "members_unknown_at_tau": float((conf < tau).mean()),
                         "stranger_as_member_at_tau": float(((cu >= tau) & np.isin(nm, members))[real].mean()),
                         "lfw_strangers_named_at_tau": float((cu >= tau)[~real].mean())}
    (out / "operating_point.json").write_text(json.dumps(res, indent=1))
    return res


def truth_of(spec, t, x):
    for row in spec.get("by_x", []):
        lo, hi, who = row[:3]
        if lo <= x < hi and (len(row) < 5 or row[3] <= t <= row[4]):
            return who
    for lo, hi, who in spec.get("windows", []):
        if lo <= t <= hi:
            return who
    return None


def score_videos(models, truth, stranger, min_rows=10):
    res = {}
    for m in models:
        res[m] = []
        for clip, spec in truth.items():
            if not isinstance(spec, dict):
                continue
            f = ROOT / "results" / "video" / f"members_{m}" / f"{clip}_tracks.csv"
            if not f.exists():
                continue
            by = defaultdict(list)
            for r in csv.DictReader(open(f, encoding="utf-8")):
                by[r["track"]].append(r)
            for tid, rs in by.items():
                if len(rs) < min_rows:
                    continue
                tr = Counter(truth_of(spec, float(r["time_s"]), (int(r["x0"]) + int(r["x1"])) / 2) for r in rs).most_common(1)[0][0]
                if tr is None:
                    continue
                shown = Counter(r["shown"] for r in rs).most_common(1)[0][0]
                res[m].append({"clip": clip, "track": int(tid), "rows": len(rs), "t0": float(rs[0]["time_s"]),
                               "t1": float(rs[-1]["time_s"]), "truth": tr, "shown": shown,
                               "ok": shown == "Unknown" if tr == stranger else shown == tr})
    return res


def track_rows(m, clip, spec):
    """Rows of a clip's tracks.csv plus, per track: its truth (majority of the per-row truth rule) and its majority
    shown label. Per-row truth can be wrong for a moment (people shift when someone walks in); per track it is not."""
    f = ROOT / "results" / "video" / f"members_{m}" / f"{clip}_tracks.csv"
    if not f.exists():
        return [], {}
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    by = defaultdict(list)
    for r in rows:
        by[r["track"]].append(r)
    info = {}
    for tid, rs in by.items():
        tr = Counter(truth_of(spec, float(r["time_s"]), (int(r["x0"]) + int(r["x1"])) / 2) for r in rs).most_common(1)[0][0]
        info[tid] = {"truth": tr, "shown": Counter(r["shown"] for r in rs).most_common(1)[0][0], "rows": len(rs),
                     "first": min(int(r["frame"]) for r in rs)}
    return rows, info


def raw_guesses(models, truth):
    """Per model x person: share of that person's boxes whose top-1 guess (before tau) is right, and the mean
    confidence of those guesses. Shows when a person is recognised but shown as Unknown only because p < tau."""
    res = {}
    for m in models:
        tot, hit, conf = Counter(), Counter(), defaultdict(list)
        for clip, spec in truth.items():
            if not isinstance(spec, dict):
                continue
            rows, info = track_rows(m, clip, spec)
            for r in rows:
                who = info[r["track"]]["truth"]
                if who is None or who == truth.get("stranger"):
                    continue
                tot[who] += 1
                if r["predicted"] == who:
                    hit[who] += 1
                    conf[who].append(float(r["p"]))
        res[m] = {w: {"boxes": tot[w], "top1_right": hit[w] / tot[w], "mean_p_when_right": float(np.mean(conf[w])) if conf[w] else None}
                  for w in tot}
    return res


def enrol_figure(out, members, stranger, dst):
    """10 enrolment faces per member spread over all their capture sessions (each tile tagged with its session),
    plus the stranger's frames."""
    d, u = np.load(out / "faces.npz"), np.load(out / "unknown.npz")
    names, ses, lab = [str(n) for n in d["names"]], d["sessions"], d["labels"]
    rows = []
    for who in members:
        idx = np.where(lab == names.index(who))[0]
        ss = sorted(set(ses[idx]))
        per = [10 // len(ss) + (k < 10 % len(ss)) for k in range(len(ss))]
        pick = []
        for s_, n_ in zip(ss, per):
            i = idx[ses[idx] == s_]
            pick += [(j, s_) for j in i[np.linspace(0, len(i) - 1, n_).astype(int)]]
        rows.append((who + "\n" + f"(สมาชิก · {len(ss)} session)", [d["faces"][j] for j, _ in pick], [s_ for _, s_ in pick]))
    wi = np.where([stranger in str(n) for n in u["names"]])[0]
    if len(wi):
        rows.append((stranger + "\n(คนแปลกหน้า)", [u["faces"][i] for i in wi[np.linspace(0, len(wi) - 1, 10).astype(int)]],
                     ["ไม่ใช้เทรน"] * 10))
    fig, ax = plt.subplots(len(rows), 10, figsize=(15, 1.65 * len(rows)), squeeze=False)
    for r, (lbl, ims, tags) in enumerate(rows):
        for c in range(10):
            a = ax[r, c]; a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
            if c < len(ims):
                a.imshow(ims[c])
                a.text(4, 216, tags[c], fontsize=8.5, color="white", va="bottom",
                       bbox=dict(boxstyle="square,pad=0.15", fc="#1f2430", ec="none", alpha=0.75))
        ax[r, 0].set_ylabel(lbl, fontsize=12, rotation=0, ha="right", va="center", labelpad=8,
                            color="#d03b3b" if "แปลก" in lbl else "#1f2430")
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    fig.savefig(dst / "members_enrol.png", dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def demo_frames(models, truth, stranger, dst, min_rows=10, settle=12):
    """Annotated frames for the slides. A frame qualifies only if every box in it belongs to a track of >= min_rows
    rows and shows that track's majority label, so the picture agrees with the per-track scores. demo_frames.json
    records who is in each saved frame (track truth) and what the video showed; the slide captions come from it."""
    info = {}
    cx = lambda r: (int(r["x0"]) + int(r["x1"])) / 2  # noqa: E731

    def frames(m, clip):
        rows, tinfo = track_rows(m, clip, truth[clip])
        by = defaultdict(list)
        for r in rows:
            by[int(r["frame"])].append(r)
        # every track at least `settle` frames old, so no box of an ended track is still fading out (video.py keeps a
        # lost track for 10 frames) and no duplicate box from a track that is just starting
        good = {f: rs for f, rs in by.items()
                if all(tinfo[r["track"]]["rows"] >= min_rows and r["shown"] == tinfo[r["track"]]["shown"]
                       and tinfo[r["track"]]["truth"] and f - tinfo[r["track"]]["first"] >= settle for r in rs)
                and not any(f - t["first"] < settle and t["first"] <= f for t in tinfo.values())}
        return good, tinfo

    def grab(m, clip, fr, rs, tinfo, name):
        c = cv2.VideoCapture(str(ROOT / "results" / "video" / f"members_{m}" / f"{clip}_annotated.mp4"))
        c.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, im = c.read()
        if not ok:
            return
        cv2.imwrite(str(dst / f"{name}.png"), cv2.resize(im, (1280, int(1280 * im.shape[0] / im.shape[1]))))
        people = [{"truth": tinfo[r["track"]]["truth"], "shown": r["shown"], "x": cx(r)} for r in rs]
        info[name] = {"model": m, "clip": clip, "frame": fr, "time_s": float(rs[0]["time_s"]),
                      "people": sorted(people, key=lambda p: p["x"])}

    def pick(good, tinfo, want):
        """First qualifying frame whose set of track truths equals `want` (a sorted list); None if there is none."""
        return next((f for f, rs in sorted(good.items()) if sorted(tinfo[r["track"]]["truth"] for r in rs) == want), None)

    for m in models:                                     # all three people in view, stranger included
        good, tinfo = frames(m, "group_1")
        fr = pick(good, tinfo, sorted(["Rapepong", "Krit", stranger]))
        if fr is not None:
            grab(m, "group_1", fr, good[fr], tinfo, f"{m}_group_1_stranger")
    m = models[-1]
    good, tinfo = frames(m, "group_2")
    fr = pick(good, tinfo, ["Rapepong", "Threeraphat"])
    if fr is not None:
        grab(m, "group_2", fr, good[fr], tinfo, f"{m}_group_2_both")
    good, tinfo = frames(m, "group_1")                   # the two members alone, preferring Krit named
    cand = [f for f, rs in sorted(good.items()) if sorted(tinfo[r["track"]]["truth"] for r in rs) == ["Krit", "Rapepong"]]
    named = [f for f in cand if all(r["shown"] == tinfo[r["track"]]["truth"] for r in good[f])]
    fr = (named or cand or [None])[0]
    if fr is not None:
        grab(m, "group_1", fr, good[fr], tinfo, f"{m}_group_1_krit")
    (dst / "demo_frames.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", default=str(ROOT / "VDO" / "test" / "truth.json"))
    ap.add_argument("--models", nargs="+", default=["M2", "M4"])
    a = ap.parse_args()
    truth = json.loads(Path(a.truth).read_text(encoding="utf-8"))
    members, stranger = truth["members"], truth["stranger"]
    out = out_dir(DATASET)
    dst = out / "figures" / "demo"
    dst.mkdir(parents=True, exist_ok=True)

    acc, P, y, rows = members_accuracy(out, members)
    op = operating_point(out, a.models, members, stranger, P, y, rows)
    vs = score_videos(a.models, truth, stranger)
    (out / "video_scores.json").write_text(json.dumps(vs, indent=1))
    raw = raw_guesses(a.models, truth)
    (out / "video_raw.json").write_text(json.dumps(raw, indent=1))
    enrol_figure(out, members, stranger, dst)
    demo_frames(a.models, truth, stranger, dst)

    print("members, held-out session (argmax):")
    for m in a.models:
        print(f"  {m}: " + "  ".join(f"{v} {100 * acc[m][v]:.1f}%" for v in V3))
    for m in a.models:
        r = op[m]
        print(f"{m}: tau {r['tau']} (re-chosen without {stranger}: {r['tau_without_real_stranger']})")
        for v in V3:
            if v in r:
                x = r[v]
                print(f"   {v:10s} members right+named {100 * x['members_right_at_tau']:5.1f}%  Unknown {100 * x['members_unknown_at_tau']:5.1f}%"
                      f"  {stranger}->member {100 * x['stranger_as_member_at_tau']:5.1f}%  LFW strangers named {100 * x['lfw_strangers_named_at_tau']:4.1f}%")
        t = vs[m]
        mem = [x for x in t if x["truth"] != stranger]
        st = [x for x in t if x["truth"] == stranger]
        print(f"   video: member tracks right {sum(x['ok'] for x in mem)}/{len(mem)}, {stranger} -> Unknown {sum(x['ok'] for x in st)}/{len(st)}")
        print("   raw top-1 per member: " + "  ".join(f"{w} {100 * x['top1_right']:.0f}% (p {x['mean_p_when_right']:.2f})"
                                                   for w, x in sorted(raw[m].items()) if x["mean_p_when_right"] is not None))
    print(f"\n-> {out}/members_accuracy.json, operating_point.json, video_scores.json and {dst}")


if __name__ == "__main__":
    main()
