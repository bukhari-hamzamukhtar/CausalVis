"""
paper_exp/runq.py  —  run benchmark jobs from a job file, a few at a time
=========================================================================

Each line of the job file:   name | ENV=a ENV2=b | extra evaluate args | which | questions
(questions may be 'cf-val' = CLEVRER validation counterfactuals, 'cf-train' = CLEVRER
train counterfactuals; the video split decides which videos are used). Each job runs
paper_exp/evaluate.py in 4 shards into paper_exp/runs/<name>/ and writes DONE there.
A job whose DONE exists is skipped, so the queue can be restarted safely.

    python paper_exp/runq.py paper_exp/jobs_val.txt --parallel 2
"""

import argparse
import os
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QFILES = {"cf-val": "zechennlp/counterfactual/validation-00000-of-00001.json",
          "cf-train": "zechennlp/counterfactual/train-00000-of-00001.json"}


def parse(line):
    parts = [p.strip() for p in line.split("|")]
    name, envs, args, which, qs = (parts + [""] * 5)[:5]
    env = dict(kv.split("=", 1) for kv in envs.split()) if envs else {}
    return {"name": name, "env": env, "args": shlex.split(args), "which": which or "val",
            "questions": QFILES.get(qs or "cf-val", qs)}


def launch(job):
    out = os.path.join(ROOT, "paper_exp", "runs", job["name"])
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ)
    env.update(job["env"])
    procs = []
    for k in range(4):
        cmd = [sys.executable, "-u", os.path.join(ROOT, "paper_exp", "evaluate.py"), "--which", job["which"],
               "--questions", job["questions"], "--shard", str(k), "--nshards", "4", "--out", out] + job["args"]
        log = open(os.path.join(out, "log%d.txt" % k), "w")
        procs.append((subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    open(os.path.join(out, "job.txt"), "w").write(repr(job) + "\n")
    return procs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--parallel", type=int, default=2)
    a = ap.parse_args()
    jobs = [parse(l) for l in open(a.jobs, encoding="utf-8") if l.strip() and not l.startswith("#")]
    todo = [j for j in jobs if not os.path.exists(os.path.join(ROOT, "paper_exp", "runs", j["name"], "DONE"))]
    print("%d jobs, %d to run" % (len(jobs), len(todo)), flush=True)
    running = []
    while todo or running:
        while todo and len(running) < a.parallel:
            j = todo.pop(0)
            print(time.strftime("%H:%M"), "start", j["name"], j["env"], j["args"], j["which"], flush=True)
            running.append((j, launch(j), time.time()))
        time.sleep(15)
        still = []
        for j, procs, t0 in running:
            if all(p.poll() is not None for p, _ in procs):
                for p, log in procs:
                    log.close()
                codes = [p.returncode for p, _ in procs]
                out = os.path.join(ROOT, "paper_exp", "runs", j["name"])
                if all(c == 0 for c in codes):
                    open(os.path.join(out, "DONE"), "w").write("%.1f min\n" % ((time.time() - t0) / 60))
                    print(time.strftime("%H:%M"), "done ", j["name"], "%.1f min" % ((time.time() - t0) / 60), flush=True)
                else:
                    print(time.strftime("%H:%M"), "FAILED", j["name"], codes, flush=True)
            else:
                still.append((j, procs, t0))
        running = still
    print("queue finished", flush=True)


if __name__ == "__main__":
    main()
