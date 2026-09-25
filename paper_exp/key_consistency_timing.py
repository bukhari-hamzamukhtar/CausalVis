"""
paper_exp/key_consistency_timing.py  —  is the "responsible, yet still collides" pair the SAME collision?
=====================================================================================================

key_consistency.py found explanatory options where CLEVRER's key calls Z responsible for the
collision of X and Y, while its counterfactual key says X and Y still collide without Z.
CLEVRER's counterfactual answer carries no time, so the collision without Z could be a
different, later collision of the same pair. Two checks:

  1. model-free: how often X and Y collide more than once in the recorded video
     (annotation-based detector frames stored in the counterfactual records)
  2. with the engine: in the counterfactual world without Z (learned engine, TEST settings),
     when X and Y first collide, relative to their recorded collision

    python paper_exp/key_consistency_timing.py
"""

import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score as S                   # noqa: E402

RUNS = {"VAL-A": "learned_lb3", "TEST-A": "A_learned", "TEST-B": "B_learned"}


def main():
    kc = json.load(open(os.path.join(HERE, "key_consistency.json")))
    chosen = json.load(open(os.path.join(HERE, "chosen.json")))["learned"]
    det, ext = chosen["det"], chosen["ext"]
    out = {}
    tot = Counter()
    for name, run in RUNS.items():
        recs = S.load(run)
        idx = {}
        for r in recs.values():
            if len(r["removed"]) == 1 and not r["unresolved"]:
                idx.setdefault((r["video"], r["removed"][0], tuple(sorted((r["i"], r["j"])))), r)
        c = Counter()
        for row in kc[name]["rows"]:
            if not (row["responsible"] and row["collides_without_Z"]):
                continue
            r = idx.get((row["video"], row["Z"], tuple(row["pair"])))
            if r is None:
                c["no record"] += 1
                continue
            obs = sorted(r["diag_obs_frames"])
            c["video: pair collides %s" % ("once" if len(obs) == 1 else ("twice or more" if obs else "never (detector)"))] += 1
            fr = sorted(f for f in (r["cal"] + r["kick2"] if det == "union" else r[det]) if f < r["T"] + ext)
            if not fr:
                c["engine: no collision without Z"] += 1
            elif obs and abs(fr[0] - obs[0]) <= 10:
                c["engine: same collision (within 10 frames of the recorded one)"] += 1
            elif obs and fr[0] > obs[0] + 10:
                c["engine: later collision (%s the video)" % ("inside" if fr[0] < r["T"] else "after")] += 1
            else:
                c["engine: earlier or other"] += 1
        out[name] = dict(c)
        tot.update(c)
        print("\n=== %s" % name)
        for k in sorted(c):
            print("   %-70s %d" % (k, c[k]))
    print("\n=== all")
    for k in sorted(tot):
        print("   %-70s %d" % (k, tot[k]))
    out["all"] = dict(tot)
    json.dump(out, open(os.path.join(HERE, "key_consistency_timing.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
