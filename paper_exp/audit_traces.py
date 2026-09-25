"""
paper_exp/audit_traces.py  —  are CausalVis answers right for the right reason?
==============================================================================

Every CausalVis counterfactual answer is decided by an event in its stored world: the first
collision of the asked pair (a frame), or the absence of any collision. When both objects
are still following the recording at the deciding moment, that event can be checked against
CLEVRER's collision annotation, which is never used to answer.

  collides, recorded evidence   valid if an annotated collision of the pair lies within
                                TOL frames of the deciding frame
  no collision, both objects    valid if the annotation has no collision of the pair in the
  replayed for the whole video  video (the part after the video end is simulated, so it is
                                not checked)
  everything else               the deciding event is simulated: no ground-truth trajectory
                                exists for a counterfactual world in CLEVRER

Reported for correct and wrong answers separately. Bourigault (2026) measures the same
property for vision-language models (35% of correct answers from mid-tier models rest on
physically invalid traces).

    python paper_exp/audit_traces.py
"""

import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score as S                                                        # noqa: E402

TOLS = (3, 6, 10)


def taint_frame(t):
    return None if not t else int(t[0])


def classify(r, det, ext, tol):
    """-> (evidence kind, verifiable, valid)"""
    T = r["T"]
    frames = sorted(f for f in (r["cal"] + r["kick2"] if det == "union" else r.get(det, [])) if f < T + ext)
    if r.get("unresolved") or r["participant_removed"]:
        return "no object / removed", False, None
    ti, tj = taint_frame(r["taint_i"]), taint_frame(r["taint_j"])
    first_sim = min([x for x in (ti, tj) if x is not None] or [10 ** 9])
    if frames:
        f0 = frames[0]
        if f0 < T and f0 < first_sim:
            ok = any(abs(g - f0) <= tol for g in r["diag_obs_frames"])
            return "collides: recorded", True, ok
        return ("collides: simulated, in video" if f0 < T else "collides: after video end"), False, None
    if first_sim >= T - 1:
        return "no collision: recorded", True, not r["diag_obs_frames"]
    return "no collision: simulated", False, None


def main():
    chosen = json.load(open(os.path.join(HERE, "chosen.json")))
    out = {}
    for sysname in ("learned", "laws"):
        det, ext = chosen[sysname]["det"], chosen[sysname]["ext"]
        recs = {}
        for tag in ("A", "B"):
            recs.update({(tag,) + k: v for k, v in S.load(os.path.join(HERE, "runs", "%s_%s" % (tag, sysname))).items()})
        res = {}
        for tol in TOLS:
            c = Counter()
            for k, r in recs.items():
                good = S.right(r, det, ext)
                kind, ver, valid = classify(r, det, ext, tol)
                c[("right" if good else "wrong", kind, valid)] += 1
            res[tol] = c
        out[sysname] = res
        n = len(recs)
        print("\n=== %s  (TEST A+B, %d options, detector %s +%d)" % (sysname, n, det, ext))
        c6 = res[6]
        for outcome in ("right", "wrong"):
            tot = sum(v for (o, _, _), v in c6.items() if o == outcome)
            ver = sum(v for (o, k, val), v in c6.items() if o == outcome and val is not None)
            print("  %s answers: %d; evidence checkable against the annotation: %d (%.1f%%)" % (outcome, tot, ver, 100.0 * ver / max(1, tot)))
            for tol in TOLS:
                c = res[tol]
                val = sum(v for (o, k, vv), v in c.items() if o == outcome and vv is True)
                inv = sum(v for (o, k, vv), v in c.items() if o == outcome and vv is False)
                print("     tolerance +-%2d frames: evidence valid %d, invalid %d -> %.1f%% valid" % (tol, val, inv, 100.0 * val / max(1, val + inv)))
            kinds = Counter()
            for (o, k, vv), v in c6.items():
                if o == outcome:
                    kinds[k] += v
            print("     by evidence kind:", dict(kinds))
    json.dump({s: {str(t): {"|".join(map(str, k)): v for k, v in c.items()} for t, c in r.items()} for s, r in out.items()},
              open(os.path.join(HERE, "audit_traces.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
