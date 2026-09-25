"""
paper_exp/figures/fig_data.py  —  every number the figures plot, as CSV files
============================================================================

Reads finished runs only (DONE markers) through paper_exp/score.py and final_report.py,
so a figure and the tables can never disagree. Writes paper_exp/figures/data/*.csv.

    python paper_exp/figures/fig_data.py
"""

import csv
import json
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PX = os.path.dirname(HERE)
sys.path.insert(0, PX)
import score as S                   # noqa: E402
import final_report as F            # noqa: E402
import audit_traces as A            # noqa: E402

OUT = os.path.join(HERE, "data")
CHOSEN = json.load(open(os.path.join(PX, "chosen.json")))


def write(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote", name, len(rows), "rows")


def setting(name):
    return F.setting(CHOSEN, name)


def diff_ci(ra, da, ea, rb, db, eb, n=2000, seed=0):
    """Paired video bootstrap of (b - a) in percentage points, options and questions."""
    keys = [k for k in ra if k in rb]
    a, qa = S.tally(ra, da, ea, keys)
    b, qb = S.tally(rb, db, eb, keys)
    vo, vq = defaultdict(lambda: [0, 0, 0]), defaultdict(lambda: [0, 0, 0])
    for k in keys:
        vo[k[0]][0] += a[k]; vo[k[0]][1] += b[k]; vo[k[0]][2] += 1
    for k in qa:
        vq[k[0]][0] += qa[k]; vq[k[0]][1] += qb[k]; vq[k[0]][2] += 1
    vids = sorted(vo)
    rng = random.Random(seed)
    do, dq = [], []
    for _ in range(n):
        pick = [rng.choice(vids) for _ in vids]
        o = [sum(vo[v][i] for v in pick) for i in range(3)]
        q = [sum(vq[v][i] for v in pick) for i in range(3)]
        do.append(100.0 * (o[1] - o[0]) / o[2]); dq.append(100.0 * (q[1] - q[0]) / q[2])
    do.sort(); dq.sort()
    lo, hi = int(0.025 * n), int(0.975 * n) - 1
    full_o = 100.0 * (sum(b.values()) - sum(a.values())) / len(keys)
    full_q = 100.0 * (sum(qb.values()) - sum(qa.values())) / len(qa)
    return full_o, do[lo], do[hi], full_q, dq[lo], dq[hi]


def main():
    os.makedirs(OUT, exist_ok=True)
    rep = open(os.path.join(PX, "final_report.txt"), encoding="utf-8").read()
    ref = F.pooled("learned")
    det, ext = CHOSEN["learned"]["det"], CHOSEN["learned"]["ext"]

    # Figure 2a: physics quality vs counterfactual score, 13 simulators
    rows = []
    for line in rep.split("=== AUDIT")[1].split("===")[0].splitlines():
        p = line.split()
        if len(p) > 8 and p[1] == "err@40":
            rows.append([p[0], p[2], p[4], p[7].rstrip("%"), p[9].rstrip("%")])
    write("fig2a_audit_counterfactual.csv", ["simulator", "err40_val", "replay30_val", "cf_options_testAB", "cf_questions_testAB"], rows)

    # Figure 2b: predictive questions, 4 simulators (RESULTS.md, TEST A+B)
    write("fig2b_audit_predictive.csv", ["simulator", "err40_val", "pred_options_testAB", "pred_questions_testAB", "z_vs_learned"],
          [["straight", 0.1125, 91.4, 84.4, 1.94], ["strfric", 0.0945, 90.9, 83.4, 0.82],
           ["learned", 0.0629, 90.5, 82.6, 0.0], ["laws", 0.0502, 88.1, 77.8, -3.80]])

    # Figure 3: effect of each change, paired bootstrap CI of the difference, TEST A+B
    changes = [("no physics", "nosim"), ("straight lines", "straight"), ("straight lines + friction", "strfric"),
               ("textbook laws", "laws"), ("no extension past the video end", "abl_noext"),
               ("12-frame hand-off", "abl_lb12"), ("distance rule only", "abl_calonly"),
               ("no voxel contact impulse", "abl_novoxel"), ("no learned pair energy", "abl_noforce"),
               ("no learned friction", "abl_nodrag"), ("no entry correction", "abl_noentry"),
               ("seed 1", "seed1"), ("seed 2", "seed2")]
    rows = []
    for label, name in changes:
        if name == "abl_noext":
            r, d, e = ref, det, 0
        elif name == "abl_calonly":
            r, d, e = ref, "cal", ext
        else:
            r = F.pooled(name)
            if not r:
                print("missing", name); continue
            d, e = setting(name)
            if name.startswith("abl_") or name.startswith("seed"):
                d, e = det, ext
        o, olo, ohi, q, qlo, qhi = diff_ci(ref, det, ext, r, d, e)
        rows.append([label, round(o, 2), round(olo, 2), round(ohi, 2), round(q, 2), round(qlo, 2), round(qhi, 2)])
    write("fig3_effects.csv", ["change", "d_options_pp", "ci_lo", "ci_hi", "d_questions_pp", "q_ci_lo", "q_ci_hi"], rows)

    # Figure 4: subsets E / M / H, five systems
    qs, groups = F.subsets(ref, None)
    rows = []
    for name, label in (("nosim", "no physics"), ("straight", "straight lines"), ("strfric", "straight + friction"),
                        ("learned", "learned (CausalVis)"), ("laws", "textbook laws")):
        r = F.pooled(name); d, e = setting(name)
        for g in ("E", "M", "H"):
            ks = [k for q in groups[g] for k in qs[q] if k in r]
            s = S.summary(r, d, e, ks)
            rows.append([label, g, round(100.0 * s["options"] / s["n_options"], 1), round(100.0 * s["questions"] / s["n_questions"], 1),
                         s["n_options"], s["n_questions"]])
    write("fig4_subsets.csv", ["system", "subset", "options_pct", "questions_pct", "n_options", "n_questions"], rows)

    # Figure 5b: timing of the deciding event against the annotation (recorded evidence, right answers)
    offs = []
    for k, r in ref.items():
        if not S.right(r, det, ext):
            continue
        kind, ver, ok = A.classify(r, det, ext, 6)
        if kind != "collides: recorded" or not r["diag_obs_frames"]:
            continue
        fr = sorted(f for f in r["cal"] + r["kick2"] if f < r["T"] + ext)
        f0 = fr[0]
        g = min(r["diag_obs_frames"], key=lambda x: abs(x - f0))
        offs.append(f0 - g)
    hist = defaultdict(int)
    for o in offs:
        hist[max(-15, min(15, o))] += 1
    write("fig5b_offset_hist.csv", ["offset_frames_clipped_pm15", "count"], sorted(hist.items()))
    offs.sort()
    print("offsets n %d median %d p10 %d p90 %d" % (len(offs), offs[len(offs) // 2], offs[len(offs) // 10], offs[9 * len(offs) // 10]))

    # Figure 6: VAL-A look-ahead sweep and extension sweep
    sw = json.load(open(os.path.join(PX, "lookback_sweep_val.json")))
    rows = []
    for k, v in sw.items():
        sim, lb = k.rsplit("_", 1)
        s = v["union_chosen_ext"]
        rows.append([sim, int(lb), round(100.0 * s["options"] / s["n_options"], 1), round(100.0 * s["questions"] / s["n_questions"], 1)])
    write("fig6a_lookahead_val.csv", ["simulator", "lookahead_frames", "options_pct", "questions_pct"], rows)
    rows = []
    for name, run in (("learned", "learned_lb3"), ("laws", "laws_lb3"), ("straight", "straight_lb1"), ("strfric", "strfric_lb1")):
        r = S.load(run)
        for e in S.EXTENDS:
            s = S.summary(r, "union", e)
            rows.append([name, e, round(100.0 * s["options"] / s["n_options"], 1), round(100.0 * s["questions"] / s["n_questions"], 1)])
    write("fig6b_extension_val.csv", ["simulator", "extension_frames", "options_pct", "questions_pct"], rows)


if __name__ == "__main__":
    main()
