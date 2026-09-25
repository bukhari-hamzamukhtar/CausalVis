"""
paper_exp/sim_agreement.py  —  on how many options does the choice of simulator matter at all?
=============================================================================================

For every TEST A+B counterfactual option, the answers of the four simulators (learned, laws,
straight lines, straight + friction) at their VAL-chosen settings. Options where all four
agree cannot be moved by better physics; the score difference between simulators comes only
from the options where they disagree.

    python paper_exp/sim_agreement.py
"""

import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score as S                  # noqa: E402
import final_report as F           # noqa: E402

SIMS = ("learned", "laws", "straight", "strfric")


def main():
    chosen = json.load(open(os.path.join(HERE, "chosen.json")))
    recs = {n: F.pooled(n) for n in SIMS + ("nosim",)}
    keys = [k for k in recs["learned"] if all(k in recs[n] for n in SIMS + ("nosim",))]
    ans, ok = {}, {}
    for n in SIMS + ("nosim",):
        d, e = chosen[n]["det"], chosen[n]["ext"]
        ans[n] = {k: S.happens(recs[n][k], d, e) for k in keys}
        ok[n] = {k: S.right(recs[n][k], d, e) for k in keys}
    agree = [k for k in keys if len({ans[n][k] for n in SIMS}) == 1]
    dis = [k for k in keys if k not in set(agree)]
    print("options %d; the four simulators agree on %d (%.1f%%), disagree on %d (%.1f%%)"
          % (len(keys), len(agree), 100.0 * len(agree) / len(keys), len(dis), 100.0 * len(dis) / len(keys)))
    a_right = sum(ok["learned"][k] for k in agree)
    print("  where they agree, all four are right on %d (%.1f%%)" % (a_right, 100.0 * a_right / len(agree)))
    for n in SIMS:
        r = sum(ok[n][k] for k in dis)
        print("  where they disagree, %-8s right on %d / %d (%.1f%%)" % (n, r, len(dis), 100.0 * r / len(dis)))
    # does no physics agree with the simulators too?
    same_np = sum(1 for k in agree if ans["nosim"][k] == ans["learned"][k])
    print("  of the agreed options, no physics gives the same answer on %d (%.1f%%)" % (same_np, 100.0 * same_np / len(agree)))
    # how many options have simulated evidence at all (any object simulated before the deciding frame)?
    sim_touched = sum(1 for k in keys if (recs["learned"][k]["taint_i"] and recs["learned"][k]["taint_i"][1] != "video_ended")
                      or (recs["learned"][k]["taint_j"] and recs["learned"][k]["taint_j"][1] != "video_ended"))
    print("  options where at least one asked object is simulated before the video ends: %d (%.1f%%)"
          % (sim_touched, 100.0 * sim_touched / len(keys)))
    out = {"options": len(keys), "agree": len(agree), "disagree": len(dis), "agree_all_right": a_right,
           "disagree_right": {n: sum(ok[n][k] for k in dis) for n in SIMS},
           "agree_nosim_same": same_np, "asked_object_simulated_in_video": sim_touched}
    json.dump(out, open(os.path.join(HERE, "sim_agreement.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
