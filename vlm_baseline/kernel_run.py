"""Kaggle GPU job: answer the counterfactual questions with a vision-language model.

Kaggle mounts the code dataset read-only under /kaggle/input. This script copies it to
/tmp/proj (the job writes a small frame folder next to it), installs the packages the
model needs, picks up any answers from earlier runs that were attached as datasets, and
runs vlm_baseline/job.py. The videos are pulled out of CLEVRER's remote zip by byte range,
so the session downloads about 1.6 MB per video instead of the 6.2 GB archive.

Filled in by vlm_baseline/prepare.py.
"""

import glob
import os
import shutil
import subprocess
import sys

JOB = "__JOB__"
QUESTIONS = "__QUESTIONS__"
CONDITIONS = "__CONDITIONS__"
FRAMES = "__FRAMES__"
LIMIT = "__LIMIT__"
HOURS = "__HOURS__"
MODEL = "__MODEL__"
DTYPE = "__DTYPE__"
SEED = "__SEED__"

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers==4.51.3", "accelerate", "qwen-vl-utils"], check=True)

hits = glob.glob("/kaggle/input/**/job.py", recursive=True)
if not hits:
    raise SystemExit("code dataset not found under /kaggle/input")
code = os.path.dirname(hits[0])

work = "/tmp/proj"          # outside /kaggle/working, so Kaggle does not save copies of the inputs
shutil.rmtree(work, ignore_errors=True)
shutil.copytree(code, work)

# answers from earlier sessions, attached as extra datasets: nothing is asked twice
caches = sorted(p for p in glob.glob("/kaggle/input/**/answers*.jsonl", recursive=True))
print("code:", code, "| caches:", caches, flush=True)

out_dir = "/kaggle/working/out/%s" % JOB
os.makedirs(out_dir, exist_ok=True)
subprocess.run(["nvidia-smi"], check=False)

cmd = [sys.executable, "job.py",
       "--questions", os.path.join(work, QUESTIONS),
       "--out", os.path.join(out_dir, "answers.jsonl"),
       "--frame-dir", "/tmp/frames",
       "--frames", FRAMES, "--hours", HOURS, "--model", MODEL, "--dtype", DTYPE,
       "--backend", "hf", "--conditions", CONDITIONS, "--seed", SEED]
if LIMIT:
    cmd += ["--limit", LIMIT]
if caches:
    cmd += ["--cache", ",".join(caches)]
print("running:", " ".join(cmd), flush=True)
env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
subprocess.run(cmd, cwd=work, env=env, check=False)

n = sum(1 for _ in open(os.path.join(out_dir, "answers.jsonl"), encoding="utf-8")) \
    if os.path.exists(os.path.join(out_dir, "answers.jsonl")) else 0
print("answers written this session:", n, flush=True)
