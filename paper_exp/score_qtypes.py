"""
paper_exp/score_qtypes.py  —  run the executor over stored scenes; choose settings on VAL
=======================================================================================

    python paper_exp/score_qtypes.py grid paper_exp/qruns/val_A          # every setting, per type
    python paper_exp/score_qtypes.py one  paper_exp/qruns/test_A SETTINGS.json

Scores: descriptive = exact answer match per question; explanatory and predictive =
per option and per question (every option right), as in CLEVRER. A video whose tracks
could not be built still counts: all its answers are wrong.
"""

import glob
import itertools
import json
import os
import pickle
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from executor import Scene, ERR                                         # noqa: E402

GRID = {"det": ["rule", "cal", "kick2", "union"], "moving_th": [0.001, 0.0015, 0.002, 0.003, 0.004, 0.006],
        "end_frame": [125, 127], "refine_span": [0, 6], "fut_ext": [10, 20, 30, 45, 60]}
USES = {"descriptive": ("det", "moving_th", "end_frame", "refine_span"),
        "explanatory": ("det", "refine_span"),
        "predictive": ("det", "fut_ext")}
DEFAULT = {"det": "union", "moving_th": 0.002, "end_frame": 125, "refine_span": 0, "fut_ext": 30,
           "min_run": 1, "smooth": 0, "gap_th": 0.01, "kick_th": 0.003, "tiebreak": False}


def load(run):
    vids = []
    for f in sorted(glob.glob(os.path.join(run, "shard*.pkl"))):
        vids += pickle.load(open(f, "rb"))["videos"]
    return vids


def norm(a):
    return str(a).strip().lower().rstrip(".")


PROG = "program"          # or "program_parsed" (set by --parsed)


def score(vids, qtype, cfg):
    o_ok = o_n = q_ok = q_n = 0
    for rec in vids:
        qs = [q for q in rec["questions"] if q["type"] == qtype]
        if not qs:
            continue
        sc = None
        if "obs" in rec:
            sc = Scene(rec["keys"], rec["presence"], rec["speed"], rec["obs"], rec.get("fut"),
                       det=cfg["det"], moving_th=cfg["moving_th"], end_frame=cfg["end_frame"],
                       refine_span=cfg["refine_span"], fut_ext=cfg["fut_ext"],
                       min_run=cfg.get("min_run", 1), smooth=cfg.get("smooth", 0), gap_th=cfg.get("gap_th", 0.01),
                       kick_th=cfg.get("kick_th", 0.003), tiebreak=cfg.get("tiebreak", False))
        for q in qs:
            if qtype == "descriptive":
                pred = sc.run(q[PROG]) if sc else ERR
                good = norm(pred) == norm(q["answer"])
                q_ok += good; q_n += 1; o_ok += good; o_n += 1
                continue
            allgood = True
            for c in q["choices"]:
                pred = sc.run(c[PROG] + q[PROG]) if sc else ERR
                lab = {"yes": "correct", "no": "wrong"}.get(pred, ERR)
                good = lab == c["answer"]
                o_ok += good; o_n += 1; allgood &= good
            q_ok += allgood; q_n += 1
    return {"options": o_ok, "n_options": o_n, "questions": q_ok, "n_questions": q_n}


def grid(vids):
    best = {}
    for qtype, keys in USES.items():
        rows = []
        for vals in itertools.product(*[GRID[k] for k in keys]):
            cfg = dict(DEFAULT); cfg.update(dict(zip(keys, vals)))
            s = score(vids, qtype, cfg)
            rows.append((s["questions"], s["options"], cfg, s))
        rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
        best[qtype] = rows[0][2]
        print("== %s: best %s -> %s" % (qtype, {k: rows[0][2][k] for k in keys}, fmt(rows[0][3])))
        for r in rows[:5]:
            print("   ", {k: r[2][k] for k in keys}, fmt(r[3]))
    return best


def fmt(s):
    return "options %.1f%% (%d/%d)  questions %.1f%% (%d/%d)" % (
        100.0 * s["options"] / max(1, s["n_options"]), s["options"], s["n_options"],
        100.0 * s["questions"] / max(1, s["n_questions"]), s["questions"], s["n_questions"])


def main():
    global PROG
    if "--parsed" in sys.argv:
        sys.argv.remove("--parsed")
        PROG = "program_parsed"
    cmd, run = sys.argv[1], sys.argv[2]
    vids = load(run)
    if cmd == "grid":
        best = grid(vids)
        out = os.path.join(run, "chosen_settings.json")
        json.dump(best, open(out, "w"), indent=1)
        print("wrote", out)
    elif cmd == "one":
        best = json.load(open(sys.argv[3]))
        for qtype, cfg in best.items():
            print("%-12s %s   %s" % (qtype, fmt(score(vids, qtype, cfg)), {k: cfg[k] for k in USES[qtype]}))


if __name__ == "__main__":
    main()
