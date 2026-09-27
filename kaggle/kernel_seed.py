"""Kaggle CPU job: one more training seed of the voxel fine-tune.

The two input datasets are mounted read-only under /kaggle/input: the code and checkpoint in
one, the training clips in the other. This script copies the code to /tmp/proj, links the
clips in at the path the trainer expects, and runs v6/train_voxel.py with the same recipe as
the production checkpoint. Only the seed differs.

The trainer saves whenever the validation error improves, so a session that is cut off still
leaves the best checkpoint reached. No GPU: the model is 43,460 parameters and the work is in
the voxel contact check.

Filled in by kaggle/prepare_seeds.py.
"""

import glob
import os
import shutil
import subprocess
import sys

SEED = "__SEED__"
EPOCHS = "__EPOCHS__"
HORIZON = "__HORIZON__"
BATCH = "__BATCH__"
LR = "__LR__"
THREADS = "__THREADS__"
VOXEL = "__VOXEL__"
DATA = "__DATA__"

hits = glob.glob("/kaggle/input/**/v6/train_voxel.py", recursive=True)
if not hits:
    raise SystemExit("code dataset not found under /kaggle/input")
code = os.path.dirname(os.path.dirname(hits[0]))

# Kaggle extracts the uploaded archive but drops its top "data/" folder, so look for the
# clip folder by name wherever it ended up.
clips = glob.glob("/kaggle/input/**/%s/*.npz" % os.path.basename(DATA), recursive=True)
if not clips:
    raise SystemExit("clip dataset not found under /kaggle/input")
clip_dir = os.path.dirname(clips[0])
print("code:", code, "| clips:", clip_dir, len(clips), flush=True)

work = "/tmp/proj"          # outside /kaggle/working, so Kaggle does not save copies of the inputs
shutil.rmtree(work, ignore_errors=True)
shutil.copytree(code, work)
os.makedirs(os.path.join(work, "data"), exist_ok=True)
link = os.path.join(work, DATA)
if not os.path.exists(link):
    os.symlink(clip_dir, link)

out = "/kaggle/working/v6_voxel_seed%s.pt" % SEED
cmd = [sys.executable, "-u", "v6/train_voxel.py",
       "--data", DATA, "--split", "split3d_yaw.json", "--init", "v5b_noyaw.pt",
       VOXEL, "--epochs", EPOCHS, "--horizon", HORIZON, "--batch", BATCH, "--lr", LR,
       "--threads", THREADS, "--seed", SEED, "--out", out]
print("running:", " ".join(cmd), flush=True)
subprocess.run(cmd, cwd=work, check=False)
print("checkpoint:", out, os.path.exists(out), flush=True)
