"""
paper_exp/predictive_common.py  —  does the predictive reversal survive a common setting?
=========================================================================================

The paper reports that on predictive questions the ranking of simulators reverses: straight
lines score highest and the textbook laws, which have the lowest rollout error, score lowest.
Each simulator's detector and future extension were chosen on VAL-A, and straight lines ended
up with fut_ext 45 where the other three got 60. That is a small difference, but the extension
is the single most valuable setting in the counterfactual evaluation, so the reversal has to be
shown at ONE setting shared by all four before it can be believed.

This re-scores the stored predictive runs across the whole grid. No simulation is re-run.

    python paper_exp/predictive_common.py
"""

import glob
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from executor import Scene, ERR                                          # noqa: E402
import score_qtypes as Q                                                 # noqa: E402

SIMS = ["straight", "strfric", "learned", "laws"]
RUN = {"learned": "%s", "laws": "%s_pred_laws", "straight": "%s_pred_straight",
       "strfric": "%s_pred_straight_friction"}
DETS = ["rule", "cal", "kick2", "union"]
EXTS = [10, 20, 30, 45, 60]


def load(sim, splits):
    vids = []
    for sp in splits:
        d = os.path.join(HERE, "qruns", RUN[sim] % sp)
        for f in sorted(glob.glob(os.path.join(d, "shard*.pkl"))):
            vids += pickle.load(open(f, "rb"))["videos"]
    return vids


def predictive(vids, det, ext):
    """Options correct and questions correct, CLEVRER predictive semantics."""
    cfg = dict(Q.DEFAULT, det=det, fut_ext=ext)
    o_ok = o_n = q_ok = q_n = 0
    for rec in vids:
        qs = [q for q in rec["questions"] if q["type"] == "predictive"]
        if not qs:
            continue
        sc = None
        if "obs" in rec:
            sc = Scene(rec["keys"], rec["presence"], rec["speed"], rec["obs"], rec.get("fut"),
                       det=cfg["det"], moving_th=cfg["moving_th"], end_frame=cfg["end_frame"],
                       refine_span=cfg["refine_span"], fut_ext=cfg["fut_ext"],
                       min_run=cfg["min_run"], smooth=cfg["smooth"], gap_th=cfg["gap_th"],
                       kick_th=cfg["kick_th"], tiebreak=cfg["tiebreak"])
        for q in qs:
            allgood = True
            for c in q["choices"]:
                pred = sc.run(c["program"] + q["program"]) if sc else ERR
                lab = {"yes": "correct", "no": "wrong"}.get(pred, ERR)
                good = lab == c["answer"]
                o_ok += good; o_n += 1; allgood &= good
            q_ok += allgood; q_n += 1
    return o_ok, o_n, q_ok, q_n


def table(splits, title):
    data = {s: load(s, splits) for s in SIMS}
    n = predictive(data["learned"], "rule", 60)[1]
    print("\n=== %s  (%d predictive options)" % (title, n))
    print("    %-9s %-6s " % ("detector", "ext") + "".join("%11s" % s for s in SIMS) + "   best")
    rows = {}
    for det in DETS:
        for ext in EXTS:
            v = [100.0 * predictive(data[s], det, ext)[0] / n for s in SIMS]
            rows[(det, ext)] = v
            print("    %-9s %-6d " % (det, ext) + "".join("%11.1f" % x for x in v)
                  + "   %s" % SIMS[v.index(max(v))])
    order_ok = sum(1 for v in rows.values() if v[0] >= v[3])
    laws_last = sum(1 for v in rows.values() if v[3] == min(v))
    straight_first = sum(1 for v in rows.values() if v[0] == max(v))
    print("    straight lines >= textbook laws in %d of %d common settings" % (order_ok, len(rows)))
    print("    laws last in %d, straight lines first in %d" % (laws_last, straight_first))
    return rows


if __name__ == "__main__":
    val = table(["val_A"], "VAL-A")
    test = table(["test_A", "test_B"], "TEST-A+B")
    json.dump({"val": {"%s_%d" % k: v for k, v in val.items()},
               "test": {"%s_%d" % k: v for k, v in test.items()},
               "sims": SIMS},
              open(os.path.join(HERE, "predictive_common.json"), "w"), indent=1)
