"""
kaggle/prepare_seeds.py  —  more training seeds of the voxel fine-tune, on Kaggle CPU
===================================================================================

The paper reports the production checkpoint and two extra seeds, and both of those were
stopped after two of four epochs because the laptop had no time for more. A free Kaggle CPU
session has twelve hours, which is enough for the full four epochs, so the seed spread can be
measured on the recipe as written instead of a truncated version of it.

One variable: the seed. Data, split, init, epochs, horizon, batch, learning rate and the
voxel contacts are the same as `paper_exp/stage3_seeds.sh` used.

    python kaggle/prepare_seeds.py --user bukharihamzamukhtar --seeds 3,4,5,6
    kaggle datasets create -p kaggle/build/code  --dir-mode zip      # 0.5 MB, first time
    kaggle datasets create -p kaggle/build/clips --dir-mode zip      # 178 MB, first time
    KAGGLE_API_TOKEN=... bash kaggle/run_kaggle.sh causalvis-train-seed3

No GPU is asked for: the model has 43,460 parameters and the time goes into the voxel contact
check, which is CPU work. That leaves the GPU quota for the VLM baseline.
"""

import argparse
import glob
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src2"))

CODE_DIRS = ("src2", "v5", "v6")
CODE_FILES = ("split3d_yaw.json", "camera_fit.json", "v5b_noyaw.pt")
CODE_SLUG = "causalvis-train-code"
CLIPS_SLUG = "causalvis-train-clips"
DATA = "data/trajectories_3d_yaw"
VAL_CLIPS = 200                      # what train_voxel.py selects on


def needed_clips(split):
    from splits import load_split
    have = {os.path.basename(f) for f in glob.glob(os.path.join(ROOT, DATA, "*.npz"))}
    train = [os.path.basename(f) for f in load_split(os.path.join(ROOT, split), "train")
             if os.path.basename(f) in have]
    val = [os.path.basename(f) for f in load_split(os.path.join(ROOT, split), "val")
           if os.path.basename(f) in have][:VAL_CLIPS]
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--seeds", default="3,4,5,6")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", default="3e-4")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--split", default="split3d_yaw.json")
    ap.add_argument("--no-voxel", action="store_true", help="the control arm instead")
    ap.add_argument("--clips", action="store_true", help="also copy the clips (slow, do it once)")
    a = ap.parse_args()

    build = os.path.join(HERE, "build")
    code_dir = os.path.join(build, "code")
    clip_dir = os.path.join(build, "clips", DATA)
    os.makedirs(code_dir, exist_ok=True)
    os.makedirs(clip_dir, exist_ok=True)

    # --- code and checkpoint -------------------------------------------------
    for d in CODE_DIRS:
        dst = os.path.join(code_dir, d)
        shutil.rmtree(dst, ignore_errors=True)
        os.makedirs(dst)
        for f in sorted(glob.glob(os.path.join(ROOT, d, "*.py"))):   # code only, no records
            shutil.copy2(f, os.path.join(dst, os.path.basename(f)))
    for f in CODE_FILES:
        shutil.copy2(os.path.join(ROOT, f), os.path.join(code_dir, f))
    json.dump({"title": CODE_SLUG, "id": "%s/%s" % (a.user, CODE_SLUG),
               "licenses": [{"name": "other"}]},
              open(os.path.join(code_dir, "dataset-metadata.json"), "w"), indent=2)
    size = sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(code_dir) for f in fs) / 1e6
    print("code dataset: %.1f MB" % size)

    # --- the clips the trainer reads -----------------------------------------
    train, val = needed_clips(a.split)
    names = train + val
    json.dump({"title": CLIPS_SLUG, "id": "%s/%s" % (a.user, CLIPS_SLUG),
               "licenses": [{"name": "other"}]},
              open(os.path.join(build, "clips", "dataset-metadata.json"), "w"), indent=2)
    have = set(os.listdir(clip_dir))
    if a.clips:
        copied = 0
        for f in names:
            if f not in have:
                shutil.copy2(os.path.join(ROOT, DATA, f), os.path.join(clip_dir, f))
                copied += 1
        print("clips: %d train + %d val, %d newly copied" % (len(train), len(val), copied))
    else:
        print("clips: %d train + %d val needed, %d already in the build folder "
              "(pass --clips to copy them)" % (len(train), len(val), len(have & set(names))))

    # --- one kernel per seed --------------------------------------------------
    template = open(os.path.join(HERE, "kernel_seed.py"), encoding="utf-8").read()
    sources = ["%s/%s" % (a.user, CODE_SLUG), "%s/%s" % (a.user, CLIPS_SLUG)]
    for s in [x for x in a.seeds.split(",") if x]:
        name = "causalvis-train-seed%s" % s
        kdir = os.path.join(build, "kernels", name)
        os.makedirs(kdir, exist_ok=True)
        code = (template.replace("__SEED__", s).replace("__EPOCHS__", str(a.epochs))
                .replace("__HORIZON__", str(a.horizon)).replace("__BATCH__", str(a.batch))
                .replace("__LR__", a.lr).replace("__THREADS__", str(a.threads))
                .replace("__VOXEL__", "--no-voxel" if a.no_voxel else "--voxel")
                .replace("__DATA__", DATA))
        open(os.path.join(kdir, "run.py"), "w", encoding="utf-8").write(code)
        meta = {"id": "%s/%s" % (a.user, name), "title": name, "code_file": "run.py",
                "language": "python", "kernel_type": "script", "is_private": True,
                "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
                "dataset_sources": sources, "competition_sources": [],
                "kernel_sources": [], "model_sources": []}
        json.dump(meta, open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=2)
        print("kernel %s | seed %s | %d epochs | horizon %d | %s"
              % (name, s, a.epochs, a.horizon, "no voxels" if a.no_voxel else "voxels"))
    print("built", build)


if __name__ == "__main__":
    main()
