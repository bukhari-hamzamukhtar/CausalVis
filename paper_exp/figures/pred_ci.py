"""
paper_exp/figures/pred_ci.py  —  95% video-bootstrap intervals for predictive questions per simulator
===================================================================================================

Same scorer as paper_exp/score_qtypes.py (settings: VAL-A choice per simulator), TEST A+B pooled.
Writes paper_exp/figures/data/fig2b_predictive_ci.csv.
"""

import csv
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PX = os.path.dirname(HERE)
sys.path.insert(0, PX)
import score_qtypes as Q            # noqa: E402

RUNS = {"learned": ("test_A", "test_B"), "laws": ("test_A_pred_laws", "test_B_pred_laws"),
        "straight": ("test_A_pred_straight", "test_B_pred_straight"),
        "strfric": ("test_A_pred_straight_friction", "test_B_pred_straight_friction")}


def main():
    Q.PROG = "program_parsed"
    base = json.load(open(os.path.join(PX, "qruns", "val_A", "chosen_settings_v2.json")))["predictive"]
    per = json.load(open(os.path.join(PX, "qruns", "predictive_per_sim_choice.json")))
    rows = []
    for sim, runs in RUNS.items():
        cfg = dict(base); cfg.update(per[sim])
        byv = []
        for tag, run in zip("AB", runs):
            for rec in Q.load(os.path.join(PX, "qruns", run)):
                s = Q.score([rec], "predictive", cfg)
                if s["n_options"]:
                    byv.append((s["options"], s["n_options"], s["questions"], s["n_questions"]))
        tot = [sum(x[i] for x in byv) for i in range(4)]
        rng = random.Random(0)
        so, sq = [], []
        for _ in range(2000):
            pick = [rng.choice(byv) for _ in byv]
            t = [sum(x[i] for x in pick) for i in range(4)]
            so.append(100.0 * t[0] / t[1]); sq.append(100.0 * t[2] / t[3])
        so.sort(); sq.sort()
        row = [sim, round(100.0 * tot[0] / tot[1], 1), round(so[50], 1), round(so[1949], 1),
               round(100.0 * tot[2] / tot[3], 1), round(sq[50], 1), round(sq[1949], 1), tot[1], tot[3]]
        print(row, flush=True)
        rows.append(row)
    with open(os.path.join(HERE, "data", "fig2b_predictive_ci.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["simulator", "options_pct", "ci_lo", "ci_hi", "questions_pct", "q_ci_lo", "q_ci_hi", "n_options", "n_questions"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
