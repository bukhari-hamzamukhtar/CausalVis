"""
paper_exp/comphy_audit_score.py  —  are ComPhy predictive answers right for the right reason?
===========================================================================================

Reads comphy_runs/audit_pred_*.json (paper_exp/comphy.py audit). ComPhy's annotation holds the
real future (frames 125-174), so every simulated collision that decides a predictive answer
can be checked against a real future collision of the same pair.

  answer says collide:     evidence valid if a real future collision of the pair lies within
                           TOL frames of our first simulated one
  answer says no collision: evidence valid if the real future has no collision of the pair

    python paper_exp/comphy_audit_score.py
"""

import glob
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    recs = []
    for f in sorted(glob.glob(os.path.join(HERE, "comphy_runs", "audit_pred_*.json"))):
        recs += json.load(open(f))
    out = {"options": len(recs), "scenes": len({r["scene"] for r in recs})}
    for tol in (3, 5, 10):
        c = Counter()
        for r in recs:
            yes = bool(r["ours"])
            right = ("correct" if yes != r["negate"] else "wrong") == r["answer"]
            if yes:
                valid = any(abs(g - r["ours"][0]) <= tol for g in r["real"])
                kind = "collides"
            else:
                valid = not r["real"]
                kind = "no collision"
            c[("right" if right else "wrong", kind, valid)] += 1
        out[str(tol)] = {"|".join(map(str, k)): v for k, v in c.items()}
        rv = sum(v for k, v in c.items() if k[0] == "right" and k[2])
        ra = sum(v for k, v in c.items() if k[0] == "right")
        print("tolerance +-%d frames: correct answers %d, evidence valid %d (%.1f%%)" % (tol, ra, rv, 100.0 * rv / ra))
        for k in sorted(c):
            print("   ", k, c[k])
    # event precision / recall over the future window, same pair within 5 frames
    tp = fp = fn = 0
    for r in recs:
        used = set()
        for t in r["ours"][:1]:
            k = [i for i, g in enumerate(r["real"]) if abs(g - t) <= 5 and i not in used]
            if k:
                used.add(k[0]); tp += 1
            else:
                fp += 1
        fn += (1 if r["real"] else 0) - (1 if used else 0)
    out["first_event"] = {"tp": tp, "fp": fp, "fn": fn, "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn)}
    print("first future collision per asked pair (+-5 frames): precision %.3f recall %.3f (tp %d fp %d fn %d)"
          % (out["first_event"]["precision"], out["first_event"]["recall"], tp, fp, fn))
    json.dump(out, open(os.path.join(HERE, "comphy_audit_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
