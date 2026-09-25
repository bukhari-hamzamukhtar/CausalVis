"""
v7/report.py  —  score a v7 run under every detector and extension length

    python v7/report.py v7/runs/val_A                 # the grid, pick on VAL
    python v7/report.py v7/runs/test_A rule 30        # one setting, full breakdown
"""

import glob
import json
import os
import sys
from collections import defaultdict

EXTENDS = (0, 10, 20, 30, 45, 60)
DETECTORS = [("rule", None), ("cal", None), ("kick2", None), ("kick4", None),
             ("voxel", 1.0), ("voxel", 3.0), ("voxel", 6.0)]


def happens(r, det, tol, ext):
    if r["participant_removed"]:
        return False
    limit = r["T"] + ext
    if det != "voxel":
        return any(f < limit for f in r.get(det, []))
    return any(f < limit and f != r["first_both"] and g <= tol for f, g in r["near"])


def score(recs, det, tol, ext):
    ok = 0
    byq = defaultdict(lambda: True)
    for r in recs:
        h = happens(r, det, tol, ext)
        good = (("correct" if h != r["negate"] else "wrong") == r["truth"])
        ok += good
        byq[(r["video"], r["qid"])] &= good
    return ok, len(recs), sum(byq.values()), len(byq)


def pct(a, b):
    return "%5.1f%% (%d/%d)" % (100.0 * a / b, a, b) if b else "--"


def main():
    d = sys.argv[1]
    recs = [json.loads(l) for f in sorted(glob.glob(os.path.join(d, "shard*.jsonl")))
            for l in open(f, encoding="utf-8")]
    print("run %s   choices %d" % (d, len(recs)))
    if not recs:
        raise SystemExit("no records in " + d)
    if len(sys.argv) == 2:
        print("\nper-choice accuracy (per-question in brackets)")
        print("%-14s" % "detector" + "".join("%18s" % ("+%d frames" % e) for e in EXTENDS))
        best = None
        for det, tol in DETECTORS:
            name = det if tol is None else "voxel<=%d" % tol
            row = "%-14s" % name
            for e in EXTENDS:
                ok, n, qok, qn = score(recs, det, tol, e)
                row += "%11.1f%% [%4.1f]" % (100.0 * ok / n, 100.0 * qok / qn)
                if best is None or ok > best[0]:
                    best = (ok, name, e)
            print(row)
        print("\nbest on this run: %s  +%d frames  (%d/%d)" % (best[1], best[2], best[0], len(recs)))
        return

    det = sys.argv[2]; ext = int(sys.argv[3])
    tol = float(det.split("<=")[1]) if det.startswith("voxel") else None
    det = "voxel" if det.startswith("voxel") else det
    ok, n, qok, qn = score(recs, det, tol, ext)
    print("detector %s  extension +%d\nPER-CHOICE   %s\nPER-QUESTION %s" % (sys.argv[2], ext, pct(ok, n), pct(qok, qn)))

    def grp(title, groups):
        print("\n" + title)
        for name, rs in groups:
            if not rs:
                continue
            o = sum((("correct" if happens(r, det, tol, ext) != r["negate"] else "wrong") == r["truth"]) for r in rs)
            fn = sum(1 for r in rs if r["truth_collides"] and not happens(r, det, tol, ext))
            fp = sum(1 for r in rs if not r["truth_collides"] and happens(r, det, tol, ext))
            print("   %-52s %-20s missed %4d  invented %3d" % (name, pct(o, len(rs)), fn, fp))

    def aff(r):
        # simulated because the intervention reached it -- not merely handed to
        # the simulator when the recording ran out
        return any(x is not None and x[1] != "video_ended" for x in (r["taint_i"], r["taint_j"]))
    grp("BY TRUTH", [("pair really collides", [r for r in recs if r["truth_collides"]]),
                     ("pair really does not", [r for r in recs if not r["truth_collides"]])])
    grp("BY WHAT THE WORLD MODEL DID WITH THE PAIR", [
        ("a removed object is in the pair", [r for r in recs if r["participant_removed"]]),
        ("both followed the recording throughout", [r for r in recs if not r["participant_removed"] and not aff(r)]),
        ("at least one was simulated", [r for r in recs if not r["participant_removed"] and aff(r)])])
    grp("THE THREE v6 FAILURE BUCKETS, NOW (annotations used only to find them)", [
        ("pair collided in video before being simulated", [r for r in recs if r["diag_obs_frames"] and aff(r) and
            all(f < min(x[0] for x in (r["taint_i"], r["taint_j"]) if x) for f in r["diag_obs_frames"])]),
        ("no video collision, truth collides", [r for r in recs if not r["diag_obs_frames"] and r["truth_collides"]]),
        ("collision after the video ends (event frame >= T)", [r for r in recs if r["truth_collides"] and
            any(f >= r["T"] for f in r[det if det != "voxel" else "rule"])])])
    grp("BY POLARITY", [("will happen", [r for r in recs if not r["negate"]]),
                        ("will NOT happen", [r for r in recs if r["negate"]])])


if __name__ == "__main__":
    main()
