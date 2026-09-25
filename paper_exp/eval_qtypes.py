"""
paper_exp/eval_qtypes.py  —  descriptive, explanatory and predictive CLEVRER questions
======================================================================================

Pass 1 (this script, slow): per video, build the factual world twice with the SAME
builder as the counterfactual benchmark (paper_exp/cf_world.py, knobs by environment):
  obs  nothing removed, no extension  -> events seen in the video (every detector)
  fut  nothing removed, +60 frames    -> events after the video ends (world model)
and store what the executor needs, plus every question of the three types with its
program and answer. Nothing is answered here.

Pass 2 (score_qtypes.py, fast): run the executor under each setting, choose on VAL.

    python paper_exp/eval_qtypes.py --which val --qset val --shard 0 --nshards 4 --out paper_exp/qruns/val_A
"""

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for d in ("src2", "v4", "v5", "v6", "v7"):
    sys.path.insert(0, os.path.join(ROOT, d))
from benchmark_eval import load_questions, load_split_set, _vid_num      # noqa: E402
from world import load_spatial                                           # noqa: E402
import importlib.util                                                    # noqa: E402
_spec = importlib.util.spec_from_file_location("paper_cf_world", os.path.join(HERE, "cf_world.py"))
_cfw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfw)
sys.path.insert(0, HERE)
import qparser as _qp                                                    # noqa: E402

QFILES = {
    "val": {"descriptive": "zechennlp/descriptive/validation-00000-of-000012.json",
            "explanatory": "zechennlp/explanatory/validation-00000-of-00001 (2).json",
            "predictive": "zechennlp/predictive/validation-00000-of-00001 (3).json"},
    "train": {"descriptive": "zechennlp/descriptive/train-00000-of-000012.json",
              "explanatory": "zechennlp/explanatory/train-00000-of-00001 (2).json",
              "predictive": "zechennlp/predictive/train-00000-of-00001 (3).json"}}


def answer_of(row):
    conv = row.get("conversations")
    if isinstance(conv, dict) and len(conv.get("value", [])) > 1:
        return str(conv["value"][1]).strip()
    return None


def lean(w):
    return {"T": w["T"], "events": {k: {tuple(p): list(v) for p, v in d.items()} for k, d in w["events"].items()},
            "positions": w["positions"].astype(np.float32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v6_voxel.pt")
    ap.add_argument("--data", default="data/trajectories_3d_det")
    ap.add_argument("--split", default="split3d_yaw.json")
    ap.add_argument("--which", default="val")
    ap.add_argument("--qset", default="val", choices=["val", "train"])
    ap.add_argument("--extend", type=int, default=60)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-predictive", action="store_true",
                    help="only videos with predictive questions (to compare simulators on them)")
    a = ap.parse_args()
    torch.set_num_threads(1)
    entry = json.load(open(os.path.join(HERE, "entry_profile.json")))
    model = load_spatial(os.path.join(ROOT, a.model), yaw_known=False)
    files = sorted(glob.glob(os.path.join(ROOT, a.data, "*.npz")))
    by_num = {_vid_num(f): f for f in files if _vid_num(f) is not None}
    allowed = load_split_set(os.path.join(ROOT, a.split), a.which)
    qs = {}
    for t, f in QFILES[a.qset].items():
        if a.only_predictive and t != "predictive":
            continue
        for r in load_questions(os.path.join(ROOT, f)):
            v = _vid_num(str(r.get("video", "")))
            if v in allowed:
                qs.setdefault(v, []).append((t, r))
    vids = sorted(qs)[a.shard::a.nshards][:a.limit]
    missing = [v for v in vids if v not in by_num]
    os.makedirs(a.out, exist_ok=True)
    out = []
    for n, v in enumerate(vids):
        rows = qs[v]
        rec = {"video": v, "questions": [], "missing_track": v not in by_num}
        for t, r in rows:
            ch = r.get("choices")
            choices = []
            if isinstance(ch, dict) and ch.get("program"):
                for c, p, ans in zip(ch["choice"], ch["program"], ch.get("answer") or [None] * len(ch["program"])):
                    choices.append({"choice": c, "program": p, "answer": ans,
                                    "program_parsed": _qp.parse(c) or []})
            # both program sources are stored: the dataset's, and the one parsed from the
            # question text by qparser.py (fitted on TRAIN questions); scoring picks one
            rec["questions"].append({"type": t, "qid": str(r.get("question_id")), "question": r.get("question"),
                                     "program": r.get("program", []), "answer": answer_of(r), "choices": choices,
                                     "program_parsed": _qp.parse(r.get("question", "")) or []})
        if v in by_num:
            z = np.load(by_num[v], allow_pickle=True)
            obs = _cfw.build_world(model, z, set(), extend=0, entry_fix=entry)
            rec.update({"keys": [str(k) for k in z["obj_keys"]], "presence": z["presence"].astype(np.float32),
                        "speed": np.linalg.norm(z["velocities"], axis=-1).astype(np.float32), "obs": lean(obs)})
            if any(t == "predictive" for t, _ in rows):
                rec["fut"] = lean(_cfw.build_world(model, z, set(), extend=a.extend, entry_fix=entry))
        out.append(rec)
        print("shard %d video %d (%d/%d)" % (a.shard, v, n + 1, len(vids)), flush=True)
    with open(os.path.join(a.out, "shard%d.pkl" % a.shard), "wb") as fh:
        pickle.dump({"videos": out, "missing": missing, "args": vars(a),
                     "knobs": {k: os.environ.get(k) for k in ("SIM", "CF_LOOKBACK", "NO_VOXEL", "NO_FORCE", "DRAG_SCALE")}}, fh)
    print("done", len(out), "videos; missing tracks:", missing, flush=True)


if __name__ == "__main__":
    main()
