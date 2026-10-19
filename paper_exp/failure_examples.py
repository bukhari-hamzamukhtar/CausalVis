"""
paper_exp/failure_examples.py  —  real wrong answers, with the evidence that produced them
==========================================================================================

The paper claims every answer carries the event that decided it. The clearest way to show
that is to print the evidence behind answers that are WRONG, where it says exactly what went
astray. This selects representative cases of each error kind from the test records and writes

  paper_exp/figures/data/fig7_failures.csv   gap-vs-frame curves for the drawn figure
  paper_exp/failure_examples.txt             the same cases as text

Error kinds, the same four the paper reports:
  invented-in-video   the detector fired on a pair the annotation never has colliding
  missed-visible      an annotated collision the engine never produced
  invented-after-end  fired only in the extension past the last frame
  missed-after-end    the key needs a collision after the last frame and we do not reach it

    python paper_exp/failure_examples.py
"""

import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score as S                                                        # noqa: E402
import oracle_ceiling as O                                               # noqa: E402

N_PER_KIND = 3


def kind_of(r, det, ext):
    T = r["T"]
    fired = sorted(f for f in (r["cal"] + r["kick2"] if det == "union" else r.get(det, []))
                   if f < T + ext)
    # judge on whether the PAIR COLLIDES. Half the questions ask which event will NOT happen,
    # so the option label is the inverse of the event and would swap invented with missed.
    said, truth = bool(fired), r["truth_collides"]
    if said == truth:
        return None
    invented = said and not truth
    after = bool(fired) and fired[0] >= T
    if invented:
        return "invented-after-end" if after else "invented-in-video"
    return "missed-after-end" if not r["diag_obs_frames"] else "missed-visible"


def main():
    chosen = json.load(open(os.path.join(HERE, "chosen.json")))
    det, ext = chosen["learned"]["det"], chosen["learned"]["ext"]
    recs = {}
    for tag in ("A", "B"):
        recs.update({(tag,) + k: v for k, v in
                     S.load(os.path.join(HERE, "runs", "A_learned" if tag == "A" else "B_learned")).items()})

    buckets, totals = {}, {}
    for k, r in recs.items():
        kd = kind_of(r, det, ext)
        if kd is None:
            continue
        totals[kd] = totals.get(kd, 0) + 1
        if r["near"]:                      # only these can be drawn as a gap curve
            buckets.setdefault(kd, []).append((k, r))

    rows, lines = [], []
    for kd in ("invented-in-video", "missed-visible", "invented-after-end", "missed-after-end"):
        got = buckets.get(kd, [])
        # prefer cases where the pair is still on the recording, so the annotation can be quoted
        got.sort(key=lambda kr: (not O.recording_only(kr[1]), kr[1]["video"]))
        n = len(got)
        for k, r in got[:N_PER_KIND]:
            T = r["T"]
            fired = sorted(f for f in (r["cal"] + r["kick2"]) if f < T + ext)
            lines.append(
                "%-19s video %s q%-4s pair %d-%d  removed %s  %s\n"
                "    truth %-7s detector fired %s  annotation %s  %s\n"
                "    %s\n"
                % (kd, r["video"], r["qid"], r["i"], r["j"], r["removed"],
                   "on the recording" if O.recording_only(r) else "simulated",
                   r["truth"], fired or "never", r["diag_obs_frames"] or "no collision",
                   "negated question" if r["negate"] else "plain question",
                   r["choice"]))
            for f, gap in r["near"]:
                rows.append({"kind": kd, "video": r["video"], "qid": r["qid"],
                             "pair": "%d-%d" % (r["i"], r["j"]), "frame": f,
                             "gap_voxels": gap,
                             "annotated": 1 if f in r["diag_obs_frames"] else 0,
                             "fired": 1 if f in fired else 0})
        print("%-19s %4d wrong options, %d with a gap curve, %d written"
              % (kd, totals.get(kd, 0), n, min(n, N_PER_KIND)))

    out_csv = os.path.join(HERE, "figures", "data", "fig7_failures.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["kind", "video", "qid", "pair", "frame",
                                           "gap_voxels", "annotated", "fired"])
        w.writeheader()
        w.writerows(rows)
    open(os.path.join(HERE, "failure_examples.txt"), "w").write("\n".join(lines))
    print("\nwrote %s (%d points) and failure_examples.txt" % (out_csv, len(rows)))


if __name__ == "__main__":
    main()
