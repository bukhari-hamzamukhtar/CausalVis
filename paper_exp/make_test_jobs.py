"""
paper_exp/make_test_jobs.py  —  apply the pre-registered VAL choices, write the TEST job list
===========================================================================================

Reads the VAL-A selection runs, picks each simulator's (look-ahead, detector,
extension) by the usual rule (paper_exp/score.py pick), writes paper_exp/chosen.json,
and writes paper_exp/jobs_test.txt with every system of PREREGISTRATION.md for TEST-A
(CLEVRER-validation videos of our test split) and TEST-B (CLEVRER-train videos of our
test split). Seeds are added when their checkpoints exist.

    python paper_exp/make_test_jobs.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import score                                                             # noqa: E402

D = "--data data/trajectories_3d_det"
SIMS = {"learned": ("learned", ""), "laws": ("laws", "SIM=laws"), "straight": ("straight", "SIM=straight"),
        "strfric": ("strfric", "SIM=straight_friction")}


def main():
    chosen = {}
    for name, (prefix, _) in SIMS.items():
        runs = ["%s_lb%d" % (prefix, lb) for lb in (1, 3, 6, 12)]
        run, det, ext = score.pick(runs)
        chosen[name] = {"run": run, "lookback": int(run.split("_lb")[1]), "det": det, "ext": ext,
                        **score.summary(score.load(run), det, ext)}
    run, det, ext = score.pick(["nosim"])
    chosen["nosim"] = {"run": run, "lookback": None, "det": det, "ext": 0, **score.summary(score.load(run), det, 0)}
    json.dump(chosen, open(os.path.join(HERE, "chosen.json"), "w"), indent=1)
    for k, v in chosen.items():
        print("%-9s %s" % (k, v))

    lb = chosen["learned"]["lookback"]
    L = "PARSED=1 CF_LOOKBACK=%d" % lb
    jobs = []
    for tag, qs in (("A", "cf-val"), ("B", "cf-train")):
        def add(name, env, args=""):
            jobs.append("%s_%s | %s | %s %s | test | %s" % (tag, name, env, args, D, qs))
        for name, (prefix, env) in SIMS.items():
            add(name, "PARSED=1 CF_LOOKBACK=%d %s" % (chosen[name]["lookback"], env))
        add("nosim", "PARSED=1", "--no-sim")
        add("abl_novoxel", L + " NO_VOXEL=1")
        add("abl_noforce", L + " NO_FORCE=1")
        add("abl_nodrag", L + " DRAG_SCALE=0")
        add("abl_noentry", L, "--no-entry-fix")
        add("abl_lb12", "PARSED=1 CF_LOOKBACK=12")
        for m in ("v3_6_3d", "v3_7_3d_frozen", "v3_8_impulse", "v5_noyaw", "v5b_noyaw", "v6_control"):
            add("ckpt_" + m, L, "--model %s.pt" % m)
        for m in ("fixed", "curriculum", "control"):
            add("ckpt_v72_" + m, L, "--model v7_2/ckpt/%s.pt" % m)
        for s in (1, 2):
            if os.path.exists(os.path.join(ROOT, "v6_voxel_seed%d.pt" % s)):
                add("seed%d" % s, L, "--model v6_voxel_seed%d.pt" % s)
    open(os.path.join(HERE, "jobs_test.txt"), "w").write("# TEST-A / TEST-B, pre-registered (PREREGISTRATION.md)\n" + "\n".join(jobs) + "\n")
    print("wrote %d jobs to paper_exp/jobs_test.txt" % len(jobs))


if __name__ == "__main__":
    main()
