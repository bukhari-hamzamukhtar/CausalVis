"""
v7_2/decide.py  —  pick each arm's setting on VAL, apply the bar, pair on TEST
==============================================================================
    python v7_2/decide.py choose   -> v7_2/runs/choice.json and a printed table
    python v7_2/decide.py paired   -> printed paired McNemar comparisons

Bar, fixed before training started: an arm is run on TEST only if its best VAL
score beats v7.1's best VAL score, 3024/3391 (cal, +30), found with the same
rule. If any long-simulation arm is tested, the control is tested too, so a win
can be split into "trained on longer simulations" versus "just trained more".
"""

import glob
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "v7"))
import report as R                                                        # noqa: E402

BAR = 3024
ARMS = ("fixed", "curriculum", "control")
RUNS = os.path.join(ROOT, "v7_2", "runs")


def shard_records(d):
    return [json.loads(l) for f in sorted(glob.glob(os.path.join(d, "shard*.jsonl")))
            for l in open(f, encoding="utf-8")]


def keyed(d):
    return {(r["video"], str(r["qid"]), r["choice"]): r for r in shard_records(d)}


def right(r, det, tol, ext):
    return ("correct" if R.happens(r, det, tol, ext) != r["negate"] else "wrong") == r["truth"]


def name(det, tol):
    return det if tol is None else "%s<=%d" % (det, tol)


def choose():
    out = {"bar_v7_1_val": BAR, "arms": {}, "test": []}
    for arm in ARMS:
        recs = shard_records(os.path.join(RUNS, "val_" + arm))
        best = None
        for det, tol in R.DETECTORS:
            for e in R.EXTENDS:
                ok, n, qok, qn = R.score(recs, det, tol, e)
                if best is None or ok > best[0]:
                    best = (ok, n, det, tol, e, qok, qn)
        ok, n, det, tol, e, qok, qn = best
        out["arms"][arm] = {"det": det, "tol": tol, "ext": e, "val_ok": ok, "val_n": n,
                            "val_q_ok": qok, "val_q_n": qn, "beats_bar": ok > BAR}
        print("%-10s VAL best %s +%d: %d/%d (%.1f%%), questions %d/%d (%.1f%%), beats v7.1's %d: %s"
              % (arm, name(det, tol), e, ok, n, 100.0 * ok / n, qok, qn, 100.0 * qok / qn, BAR, ok > BAR))
    long_ok = [a for a in ("fixed", "curriculum") if out["arms"][a]["beats_bar"]]
    if long_ok:
        out["test"] = long_ok + ["control"]
    elif out["arms"]["control"]["beats_bar"]:
        out["test"] = ["control"]
    json.dump(out, open(os.path.join(RUNS, "choice.json"), "w"), indent=1)
    print("TEST:", " ".join(out["test"]) or "nothing -- no arm beat v7.1 on VAL")


def mcnemar(a, b, keys):
    x = sum(a[k] and not b[k] for k in keys)
    y = sum(b[k] and not a[k] for k in keys)
    return x, y, ((x - y) / math.sqrt(x + y) if x + y else 0.0)


def paired():
    ch = json.load(open(os.path.join(RUNS, "choice.json")))
    base = keyed(os.path.join(ROOT, "v7_1", "runs", "test"))
    ctl = keyed(os.path.join(RUNS, "test_control")) if "control" in ch["test"] else None
    for arm in ch["test"]:
        c = ch["arms"][arm]
        det, tol, ext = c["det"], c["tol"], c["ext"]
        new = keyed(os.path.join(RUNS, "test_" + arm))
        keys = [k for k in new if k in base]
        a = {k: right(new[k], det, tol, ext) for k in keys}
        b = {k: right(base[k], "cal", None, 30) for k in keys}
        x, y, z = mcnemar(a, b, keys)
        print("%s (%s +%d) vs v7.1 (cal +30), %d options: %d vs %d correct; %s right & v7.1 wrong %d, "
              "v7.1 right & %s wrong %d, McNemar z = %.2f"
              % (arm, name(det, tol), ext, len(keys), sum(a.values()), sum(b.values()), arm, x, arm, y, z))
        if arm != "control" and ctl is not None:
            keys = [k for k in new if k in ctl]
            a = {k: right(new[k], det, tol, ext) for k in keys}
            b = {k: right(ctl[k], det, tol, ext) for k in keys}      # control read at the SAME setting
            x, y, z = mcnemar(a, b, keys)
            print("%s vs control, both %s +%d, %d options: %d vs %d correct; %s right & control wrong %d, "
                  "control right & %s wrong %d, McNemar z = %.2f"
                  % (arm, name(det, tol), ext, len(keys), sum(a.values()), sum(b.values()), arm, x, arm, y, z))


if __name__ == "__main__":
    {"choose": choose, "paired": paired}[sys.argv[1]]()
