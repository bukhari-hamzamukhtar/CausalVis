"""
paper_exp/score.py  —  one scorer for every paper number
=======================================================

Detectors: rule, cal, kick2, kick4 and union (= cal OR kick2, v7.3's). Extensions
past the video end: 0, 10, 20, 30, 45, 60 frames. Every option in the question file
is scored (the evaluator records all of them; checked against the question file).

    python paper_exp/score.py grid  RUN                      # every detector x extension
    python paper_exp/score.py pick  RUN1 RUN2 ...            # the usual VAL rule: most options
    python paper_exp/score.py one   RUN DET EXT              # one setting + 95% CI + subsets
    python paper_exp/score.py pair  RUN_A DET EXT RUN_B DET EXT   # paired McNemar

RUN is a folder name under paper_exp/runs or any path holding shard*.jsonl.
Confidence intervals resample VIDEOS (a question's options, and the questions of one
video, are not independent), 2,000 resamples, fixed seed.
"""

import glob
import json
import math
import os
import random
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTENDS = (0, 10, 20, 30, 45, 60)
DETECTORS = ("rule", "cal", "kick2", "kick4", "union")


def run_dir(name):
    p = name if os.path.isdir(name) else os.path.join(ROOT, "paper_exp", "runs", name)
    if not os.path.isdir(p):
        raise SystemExit("no run folder %s" % name)
    return p


def load(name):
    """Records keyed by (video, question id, choice text, occurrence). A few questions list
    the same choice text twice; the occurrence number keeps both (3,391 VAL options, not
    3,387)."""
    recs = [json.loads(l) for f in sorted(glob.glob(os.path.join(run_dir(name), "shard*.jsonl")))
            for l in open(f, encoding="utf-8")]
    out, seen = {}, defaultdict(int)
    for r in recs:
        k = (r["video"], str(r["qid"]), r["choice"])
        out[k + (seen[k],)] = r
        seen[k] += 1
    return out


def happens(r, det, ext):
    if r["participant_removed"]:
        return False
    lim = r["T"] + ext
    fr = r["cal"] + r["kick2"] if det == "union" else r.get(det, [])
    return any(f < lim for f in fr)


def right(r, det, ext):
    return ("correct" if happens(r, det, ext) != r["negate"] else "wrong") == r["truth"]


def tally(recs, det, ext, keys=None):
    keys = list(recs) if keys is None else keys
    ok = {k: right(recs[k], det, ext) for k in keys}
    q = defaultdict(lambda: True)
    for k in keys:
        q[(k[0], k[1])] &= ok[k]
    return ok, q


def summary(recs, det, ext, keys=None):
    ok, q = tally(recs, det, ext, keys)
    return {"options": sum(ok.values()), "n_options": len(ok), "questions": sum(q.values()), "n_questions": len(q)}


def boot(recs, det, ext, n=2000, seed=0):
    ok, q = tally(recs, det, ext)
    byv_o, byv_q = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for k, g in ok.items():
        byv_o[k[0]][0] += g; byv_o[k[0]][1] += 1
    for k, g in q.items():
        byv_q[k[0]][0] += g; byv_q[k[0]][1] += 1
    vids = sorted(byv_o)
    rng = random.Random(seed)
    so, sq = [], []
    for _ in range(n):
        pick = [rng.choice(vids) for _ in vids]
        a = sum(byv_o[v][0] for v in pick); b = sum(byv_o[v][1] for v in pick)
        c = sum(byv_q[v][0] for v in pick); d = sum(byv_q[v][1] for v in pick)
        so.append(100.0 * a / b); sq.append(100.0 * c / d)
    so.sort(); sq.sort()
    lo, hi = int(0.025 * n), int(0.975 * n) - 1
    return (so[lo], so[hi]), (sq[lo], sq[hi])


def paired(ra, da, ea, rb, db, eb):
    keys = [k for k in ra if k in rb]
    a, qa = tally(ra, da, ea, keys)
    b, qb = tally(rb, db, eb, keys)
    x = sum(a[k] and not b[k] for k in keys); y = sum(b[k] and not a[k] for k in keys)
    qx = sum(qa[k] and not qb[k] for k in qa); qy = sum(qb[k] and not qa[k] for k in qa)
    z = (x - y) / math.sqrt(x + y) if x + y else 0.0
    zq = (qx - qy) / math.sqrt(qx + qy) if qx + qy else 0.0
    return {"matched": len(keys), "a_only": x, "b_only": y, "z": z, "q_a_only": qx, "q_b_only": qy, "z_q": zq,
            "a": summary(ra, da, ea, keys), "b": summary(rb, db, eb, keys)}


def grid(recs):
    out = []
    for det in DETECTORS:
        for e in EXTENDS:
            s = summary(recs, det, e)
            out.append((s["options"], s["questions"], det, e))
    return out


def pick(names):
    """The usual rule: the (run, detector, extension) with the most VAL options right;
    ties broken by more questions right, then by the order given."""
    best = None
    for i, n in enumerate(names):
        for o, q, det, e in grid(load(n)):
            key = (o, q, -i)
            if best is None or key > best[0]:
                best = (key, n, det, e)
    return best[1], best[2], best[3]


def fmt(s):
    return "options %5.1f%% (%d/%d)  questions %5.1f%% (%d/%d)" % (
        100.0 * s["options"] / s["n_options"], s["options"], s["n_options"],
        100.0 * s["questions"] / s["n_questions"], s["questions"], s["n_questions"])


def main():
    cmd = sys.argv[1]
    if cmd == "grid":
        recs = load(sys.argv[2])
        for o, q, det, e in grid(recs):
            print("%-6s +%2d  options %4d  questions %3d" % (det, e, o, q))
    elif cmd == "pick":
        n, det, e = pick(sys.argv[2:])
        print(json.dumps({"run": n, "det": det, "ext": e, **summary(load(n), det, e)}))
    elif cmd == "one":
        recs = load(sys.argv[2]); det, e = sys.argv[3], int(sys.argv[4])
        s = summary(recs, det, e); (ol, oh), (ql, qh) = boot(recs, det, e)
        print(fmt(s) + "   95%% CI options %.1f-%.1f  questions %.1f-%.1f" % (ol, oh, ql, qh))
    elif cmd == "pair":
        ra, rb = load(sys.argv[2]), load(sys.argv[5])
        p = paired(ra, sys.argv[3], int(sys.argv[4]), rb, sys.argv[6], int(sys.argv[7]))
        print("matched %d | A %s | B %s" % (p["matched"], fmt(p["a"]), fmt(p["b"])))
        print("options: A only %d, B only %d, z = %+.2f | questions: A only %d, B only %d, z = %+.2f"
              % (p["a_only"], p["b_only"], p["z"], p["q_a_only"], p["q_b_only"], p["z_q"]))


if __name__ == "__main__":
    main()
