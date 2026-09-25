"""Score each v7_3 VAL run against v7.1's own VAL records (paired, same options).

    python v7_3/compare.py
"""
import glob
import json
import math
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = [("v7.1 as it is (baseline)", "v7_1/runs/val"),
        ("physics radius x1.2, detector unchanged (one size for both)", "v7_3/runs/val_phys12"),
        ("hand-off 6 frames before the lost contact", "v7_3/runs/val_lb6"),
        ("hand-off 20 frames before the lost contact", "v7_3/runs/val_lb20"),
        ("hand-off 30 frames before the lost contact", "v7_3/runs/val_lb30")]


def load(d):
    out = {}
    for f in glob.glob(os.path.join(ROOT, d, "shard*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            out[(r["video"], str(r["qid"]), r["choice"])] = r
    return out


def happens(r, det, ext):
    return any(f < r["T"] + ext for f in r[det])


def right(r, det, ext):
    return ("correct" if happens(r, det, ext) != r["negate"] else "wrong") == r["truth"]


def main():
    base = load(RUNS[0][1])
    for name, d in RUNS:
        recs = load(d)
        if not recs:
            print("%-62s (not run)" % name)
            continue
        keys = [k for k in recs if k in base]
        print("\n" + name + "   [%d options matched]" % len(keys))
        for det in ("cal", "rule"):
            for ext in (30, 45, 60):
                ok = {k: right(recs[k], det, ext) for k in keys}
                ref = {k: right(base[k], "cal", 30) for k in keys}
                q = defaultdict(lambda: True)
                for k in keys:
                    q[(k[0], k[1])] &= ok[k]
                miss = sum(recs[k]["truth_collides"] and not happens(recs[k], det, ext) for k in keys)
                inv = sum(happens(recs[k], det, ext) and not recs[k]["truth_collides"] for k in keys)
                x = sum(ok[k] and not ref[k] for k in keys)
                y = sum(ref[k] and not ok[k] for k in keys)
                z = (x - y) / math.sqrt(x + y) if x + y else 0.0
                print("  %-4s +%d  options %4d (%.1f%%)  questions %3d/%d (%.1f%%)  missed %3d  invented %3d"
                      "   vs v7.1 cal+30: fixes %3d breaks %3d z=%+.2f"
                      % (det, ext, sum(ok.values()), 100.0 * sum(ok.values()) / len(keys), sum(q.values()),
                         len(q), 100.0 * sum(q.values()) / len(q), miss, inv, x, y, z))


if __name__ == "__main__":
    main()
