"""
vlm_baseline/prepare.py  —  build the Kaggle dataset and kernels for the VLM baseline
===================================================================================

Everything the GPU session needs is a few small files: the prompter, the job, the remote
zip reader and the question sets. The videos are fetched inside the session, so nothing
large is uploaded.

    python vlm_baseline/prepare.py --user <kaggle-username>
    kaggle datasets create  -p vlm_baseline/kaggle/build/code --dir-mode zip     # first time
    kaggle datasets version -p vlm_baseline/kaggle/build/code -m "update" --dir-mode zip
    kaggle kernels push -p vlm_baseline/kaggle/build/kernels/smoke

Jobs:
  smoke  12 validation videos, both conditions, to check the prompt and the parser
  val    150 validation videos, where the frame count is chosen if it needs choosing
  test   the full test set, run once
A session that runs out of time can be continued: download its output, upload it as a
dataset, and pass --resume-from <that slug>. Answers already in a cache are never re-asked.
"""

import argparse
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_FILES = ("job.py", "vlm.py", "remote_zip.py",
              "questions_testA.json", "questions_valA.json")
CODE_SLUG = "causalvis-vlm-code"

JOBS = {
    # name: questions file, conditions, videos (0 = all), hours
    "smoke": ("questions_valA.json", "video,blind", 12, 1.0),
    "val": ("questions_valA.json", "video,blind", 150, 4.0),
    "test": ("questions_testA.json", "video,blind", 0, 10.5),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--model", default="Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--dtype", default="mixed",
                    choices=["float16", "mixed", "float32", "bfloat16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--machine", default="NvidiaTeslaT4")
    ap.add_argument("--resume-from", default="", help="dataset slugs holding earlier answers")
    a = ap.parse_args()

    build = os.path.join(HERE, "kaggle", "build")
    code_dir = os.path.join(build, "code")
    os.makedirs(code_dir, exist_ok=True)
    for f in CODE_FILES:
        src = os.path.join(HERE, f)
        if not os.path.exists(src):
            raise SystemExit("missing %s: run make_questions.py first" % f)
        shutil.copy2(src, os.path.join(code_dir, f))
    json.dump({"title": CODE_SLUG, "id": "%s/%s" % (a.user, CODE_SLUG),
               "licenses": [{"name": "other"}]},
              open(os.path.join(code_dir, "dataset-metadata.json"), "w"), indent=2)
    size = sum(os.path.getsize(os.path.join(code_dir, f)) for f in CODE_FILES) / 1e6
    print("code dataset: %d files, %.1f MB" % (len(CODE_FILES), size))

    sources = ["%s/%s" % (a.user, CODE_SLUG)]
    sources += [s if "/" in s else "%s/%s" % (a.user, s)
                for s in a.resume_from.split(",") if s]

    template = open(os.path.join(HERE, "kernel_run.py"), encoding="utf-8").read()
    for job, (qfile, conds, limit, hours) in JOBS.items():
        kdir = os.path.join(build, "kernels", job)
        os.makedirs(kdir, exist_ok=True)
        code = (template.replace("__JOB__", job).replace("__QUESTIONS__", qfile)
                .replace("__CONDITIONS__", conds).replace("__FRAMES__", str(a.frames))
                .replace("__LIMIT__", "" if not limit else str(limit))
                .replace("__HOURS__", str(hours)).replace("__MODEL__", a.model)
                .replace("__DTYPE__", a.dtype).replace("__SEED__", str(a.seed)))
        open(os.path.join(kdir, "run.py"), "w", encoding="utf-8").write(code)
        meta = {"id": "%s/causalvis-vlm-%s" % (a.user, job),
                "title": "causalvis-vlm-%s" % job, "code_file": "run.py",
                "language": "python", "kernel_type": "script", "is_private": True,
                "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
                "machine_shape": a.machine, "dataset_sources": sources,
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
        json.dump(meta, open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=2)
        n = len(json.load(open(os.path.join(HERE, qfile), encoding="utf-8")))
        print("kernel %-6s %s | %s | videos %s | %d questions in the file | %.1f h"
              % (job, qfile, conds, limit or "all", n, hours))
    print("built", build)


if __name__ == "__main__":
    main()
