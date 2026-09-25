"""
paper_exp/runtime.py  —  seconds per counterfactual question, one process, idle CPU
=================================================================================

Runs the evaluator on the first 5 and the first 45 TEST-A questions for each simulator and
reports (t45 - t5) / 40, which removes start-up cost (loading the model and parsing).
torch uses one thread, as in every test run.

    python paper_exp/runtime.py
"""

import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
Q = "zechennlp/counterfactual/validation-00000-of-00001.json"
SIMS = {"learned": ({"CF_LOOKBACK": "3"}, []),
        "laws": ({"CF_LOOKBACK": "3", "SIM": "laws"}, []),
        "straight": ({"CF_LOOKBACK": "1", "SIM": "straight"}, []),
        "no physics": ({}, ["--no-sim"])}


def run(env_extra, extra, n):
    env = dict(os.environ, PARSED="1", **env_extra)
    out = tempfile.mkdtemp()
    t = time.perf_counter()
    subprocess.run([sys.executable, "paper_exp/evaluate.py", "--data", "data/trajectories_3d_det", "--which", "test",
                    "--questions", Q, "--limit", str(n), "--out", out] + extra,
                   cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return time.perf_counter() - t


def main():
    res = {}
    for name, (env, extra) in SIMS.items():
        t5, t45 = run(env, extra, 5), run(env, extra, 45)
        res[name] = {"start_up_plus_5": round(t5, 1), "start_up_plus_45": round(t45, 1), "seconds_per_question": round((t45 - t5) / 40, 2)}
        print(name, res[name], flush=True)
    res["machine"] = "Intel i5-8350U, 4 cores, 7.9 GB, one process, torch 1 thread, nothing else running"
    json.dump(res, open(os.path.join(HERE, "runtime.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
