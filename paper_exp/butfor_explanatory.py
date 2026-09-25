"""
paper_exp/butfor_explanatory.py  —  CLEVRER explanatory questions by counterfactual test
=======================================================================================

"Which of the following is (not) responsible for the collision between X and Y?"
CLEVRER's answer key follows a chain rule: an event is responsible if it precedes the
target in some object's chain of events (the official executor's filter_ancestor). This
script answers the same questions with a counterfactual test instead:

  remove what the candidate cause adds, simulate that world with the CausalVis engine
  (paper_exp/cf_world.py, knobs by environment), and call the candidate responsible if
  the target collision no longer happens.

  candidate "the presence of Z"            -> remove Z
  candidate "Z's entrance"                  -> remove Z
  candidate "the collision of A and B"      -> remove whichever of A, B is not in the
                                               target (both outside: remove both)
  a candidate that shares both objects with the target cannot be removed without removing
  the target itself; it keeps the chain-rule answer (counted separately).

Target and candidates are found by running the parsed programs on the observed scene
(paper_exp/executor.py, VAL-A settings). Records keep both answers per option, so the two
rules are scored on the same options.

    PARSED=1 CF_LOOKBACK=3 python paper_exp/butfor_explanatory.py --which val --qset val --shard 0 --nshards 4 --out paper_exp/butfor/val_A
"""

import argparse
import glob
import json
import os
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
_spec = importlib.util.spec_from_file_location("paper_cf_world_bf", os.path.join(HERE, "cf_world.py"))
CFW = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CFW)
sys.path.insert(0, HERE)
import qparser                                                           # noqa: E402
from executor import Scene, ERR, pair_frames                             # noqa: E402
from eval_qtypes import lean                                             # noqa: E402

QF = {"val": "zechennlp/explanatory/validation-00000-of-00001 (2).json",
      "train": "zechennlp/explanatory/train-00000-of-00001 (2).json"}
EXTS = (0, 10, 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="val")
    ap.add_argument("--qset", default="val")
    ap.add_argument("--data", default="data/trajectories_3d_det")
    ap.add_argument("--model", default="v6_voxel.pt")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    entry = json.load(open(os.path.join(HERE, "entry_profile.json")))
    cfg = json.load(open(os.path.join(HERE, "qruns", "val_A", "chosen_settings_v2.json")))["explanatory"]
    model = load_spatial(os.path.join(ROOT, a.model), yaw_known=False)
    files = {_vid_num(f): f for f in glob.glob(os.path.join(ROOT, a.data, "*.npz"))}
    allowed = load_split_set(os.path.join(ROOT, "split3d_yaw.json"), a.which)
    byv = {}
    for r in load_questions(os.path.join(ROOT, QF[a.qset])):
        v = _vid_num(str(r.get("video", "")))
        if v in allowed:
            byv.setdefault(v, []).append(r)
    vids = sorted(byv)[a.shard::a.nshards][:a.limit]
    os.makedirs(a.out, exist_ok=True)
    fout = open(os.path.join(a.out, "shard%d.jsonl" % a.shard), "w", encoding="utf-8")
    for n, v in enumerate(vids):
        rows = byv[v]
        z = np.load(files[v], allow_pickle=True) if v in files else None
        sc = None
        if z is not None:
            obs = CFW.build_world(model, z, set(), extend=0, entry_fix=entry)
            sc = Scene([str(k) for k in z["obj_keys"]], z["presence"].astype(np.float32),
                       np.linalg.norm(z["velocities"], axis=-1).astype(np.float32), lean(obs), None,
                       det=cfg["det"], refine_span=cfg["refine_span"])
        worlds = {}
        for r in rows:
            qprog = qparser.parse(r.get("question", "")) or []
            ch = r["choices"]
            neg = "negate" in qprog
            tgt = ERR
            if sc is not None and "filter_ancestor" in qprog:
                tgt = sc.run_raw(qprog[:qprog.index("filter_ancestor")])
            for ci, (ctext, ans) in enumerate(zip(ch["choice"], ch["answer"])):
                cprog = qparser.parse(ctext) or []
                trace = sc.run(cprog + qprog) if sc is not None else ERR
                rec = {"video": v, "qid": str(r["question_id"]), "ci": ci, "choice": ctext, "truth": ans, "negate": neg,
                       "trace": {"yes": "correct", "no": "wrong"}.get(trace, "error")}
                cand = sc.run_raw(cprog) if sc is not None else ERR
                removal, kind = None, "unresolved"
                if isinstance(tgt, dict) and tgt.get("type") == "collision":
                    tset = set(tgt["object"])
                    if isinstance(cand, int):
                        removal, kind = {cand}, "presence"
                    elif isinstance(cand, dict) and cand.get("type") in ("in", "out"):
                        removal, kind = {cand["object"][0]}, "entrance"
                    elif isinstance(cand, dict) and cand.get("type") == "collision":
                        extra = set(cand["object"]) - tset
                        removal, kind = (extra, "collision") if extra else (None, "same_pair")
                if removal is not None and removal & tset:
                    removal, kind = None, "removes_target"
                rec["kind"] = kind
                if removal is not None:
                    key = frozenset(removal)
                    if key not in worlds:
                        w = CFW.build_world(model, z, set(removal), extend=max(EXTS), entry_fix=entry)
                        worlds[key] = {"T": w["T"], "events": {d: {p: list(f) for p, f in w["events"][d].items()} for d in ("cal", "kick2", "rule")}}
                    w = worlds[key]
                    pair = tuple(sorted(tgt["object"]))
                    fr = pair_frames(w["events"], cfg["det"], pair)
                    rec["target_after"] = {str(e): any(f < w["T"] + e for f in fr) for e in EXTS}
                fout.write(json.dumps(rec) + "\n")
        fout.flush()
        print("shard %d video %d (%d/%d) worlds %d" % (a.shard, v, n + 1, len(vids), len(worlds)), flush=True)
    fout.close()


if __name__ == "__main__":
    main()
