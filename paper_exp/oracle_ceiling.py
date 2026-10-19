"""
paper_exp/oracle_ceiling.py  —  what could the engine score with a perfect event detector?
==========================================================================================

The referee question this answers: the engine's largest checkable error source is the event
detector (188 of 305 checkable wrong answers are collisions it reports that the annotation
does not contain). So how much of the gap to the published leaders could a better detector
close, and how much is left for the simulator?

A pair whose two objects both follow the recording for the whole video lives in a
counterfactual world that is IDENTICAL to the recording for that pair. For those pairs a
perfect detector is simply CLEVRER's collision annotation, which the engine never reads.
Substituting it gives an upper bound on the whole pipeline that involves no simulator at all.

Three systems are reported:

  engine        the paper's system, unchanged
  oracle-rec    annotated collisions decide every pair that is never simulated; the engine's
                own events still decide the part after the video ends and every simulated pair
  oracle-rec0   the same, but the oracle sees only the video: no post-video events at all.
                The gap between the two prices CLEVRER's post-video answers.

Nothing is selected on these numbers; they are a diagnostic, reported for VAL and TEST.

    python paper_exp/oracle_ceiling.py
"""

import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score as S                                                        # noqa: E402


def taint(t):
    return None if not t else int(t[0])


def simulated_from(r):
    """First frame at which either asked object stops following the recording."""
    fs = [x for x in (taint(r["taint_i"]), taint(r["taint_j"])) if x is not None]
    return min(fs) if fs else 10 ** 9


def recording_only(r):
    """True when neither asked object is ever handed to the simulator inside the video."""
    return simulated_from(r) >= r["T"] - 1


def engine_says(r, det, ext):
    return S.happens(r, det, ext)


def oracle_says(r, det, ext, post):
    """Perfect detection on the recorded part; the engine still supplies the extension."""
    if r["participant_removed"]:
        return False
    if not recording_only(r):
        return engine_says(r, det, ext)              # simulated: no ground truth exists
    if r["diag_obs_frames"]:
        return True
    if not post:
        return False
    fr = r["cal"] + r["kick2"] if det == "union" else r.get(det, [])
    return any(r["T"] <= f < r["T"] + ext for f in fr)


def score_with(recs, fn):
    ok = {k: ("correct" if fn(r) != r["negate"] else "wrong") == r["truth"] for k, r in recs.items()}
    q = {}
    for k, g in ok.items():
        key = (k[0], k[1], k[2])
        q[key] = q.get(key, True) and g
    return ok, sum(ok.values()), len(ok), sum(q.values()), len(q)


def mcnemar(a, b):
    bb = sum(1 for k in a if a[k] and not b[k])
    cc = sum(1 for k in a if b[k] and not a[k])
    z = (bb - cc) / (bb + cc) ** 0.5 if bb + cc else 0.0
    return bb, cc, z


def errors(recs, fn):
    """Missed against invented, counted on whether the PAIR COLLIDES, not on whether the option
    is labelled correct. Half the questions ask which event will NOT happen, so the option label
    is the inverse of the event there and counting it would swap the two columns."""
    c = Counter()
    for r in recs.values():
        said, truth = fn(r), r["truth_collides"]
        if said == truth:
            continue
        c["invented" if said else "missed"] += 1
    return c


def run(tag, runs, det, ext):
    recs = {}
    for name in runs:
        recs.update({(name,) + k: v for k, v in S.load(os.path.join(HERE, "runs", name)).items()})
    n = len(recs)
    rec_only = sum(recording_only(r) for r in recs.values())
    print("\n=== %s  (%d options, detector %s +%d)" % (tag, n, det, ext))
    print("    never simulated inside the video: %d options (%.1f%%)" % (rec_only, 100.0 * rec_only / n))

    rows = [("engine", lambda r: engine_says(r, det, ext)),
            ("oracle-rec", lambda r: oracle_says(r, det, ext, True)),
            ("oracle-rec0", lambda r: oracle_says(r, det, ext, False))]
    res = {}
    print("    %-13s %-18s %-16s  missed  invented" % ("system", "options", "questions"))
    for name, fn in rows:
        ok, o, no, qn, nq = score_with(recs, fn)
        e = errors(recs, fn)
        res[name] = ok
        print("    %-13s %5d/%d (%5.1f%%)  %4d/%d (%5.1f%%)  %6d  %8d"
              % (name, o, no, 100.0 * o / no, qn, nq, 100.0 * qn / nq, e["missed"], e["invented"]))
    for name in ("oracle-rec", "oracle-rec0"):
        b, c, z = mcnemar(res[name], res["engine"])
        print("    %s vs engine: %d better, %d worse, McNemar z = %.2f" % (name, b, c, z))
    return res


def main():
    chosen = json.load(open(os.path.join(HERE, "chosen.json")))
    out = {}
    for sysname in ("learned", "laws"):
        det, ext = chosen[sysname]["det"], chosen[sysname]["ext"]
        print("\n" + "#" * 78 + "\n# %s" % sysname)
        run("VAL-A", ["%s_lb%d" % (sysname, chosen[sysname]["lookback"])], det, ext)
        res = run("TEST-A+B", ["A_%s" % sysname, "B_%s" % sysname], det, ext)
        out[sysname] = {"det": det, "ext": ext}
    json.dump(out, open(os.path.join(HERE, "oracle_ceiling.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
