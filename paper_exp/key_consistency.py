"""
paper_exp/key_consistency.py  —  is CLEVRER's "responsible" a counterfactual notion?
===================================================================================

Model-free check of the answer key against itself. In the same video:

  explanatory:     "Z (its presence or its entrance) is responsible for the collision of X and Y"
  counterfactual:  "without Z, X and Y collide"            (CLEVRER's own simulated answer)

Under a but-for reading of causation, Z is responsible exactly when X and Y do NOT collide
without Z. This script finds every pair of options that ask about the same (Z, X, Y) and
counts how often the two keys agree with that reading. No simulator is used; object
references are resolved on the observed scene (paper_exp/executor.py, VAL-A settings) and
matched to the counterfactual records' resolved indices.

    python paper_exp/key_consistency.py
"""

import glob
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for d in ("src2", "v4", "v5", "v6", "v7"):
    sys.path.insert(0, os.path.join(ROOT, d))
from benchmark_eval import load_questions, _vid_num                     # noqa: E402
from world import load_spatial                                           # noqa: E402
import importlib.util                                                    # noqa: E402
_spec = importlib.util.spec_from_file_location("paper_cf_world_kc", os.path.join(HERE, "cf_world.py"))
CFW = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CFW)
sys.path.insert(0, HERE)
import qparser                                                           # noqa: E402
import score as S                                                        # noqa: E402
from executor import Scene, ERR                                          # noqa: E402
from eval_qtypes import lean                                             # noqa: E402

QF = {"val": "zechennlp/explanatory/validation-00000-of-00001 (2).json",
      "train": "zechennlp/explanatory/train-00000-of-00001 (2).json"}
SETS = (("VAL-A", "learned_lb3", "val"), ("TEST-A", "A_learned", "val"), ("TEST-B", "B_learned", "train"))


def main():
    entry = json.load(open(os.path.join(HERE, "entry_profile.json")))
    cfg = json.load(open(os.path.join(HERE, "qruns", "val_A", "chosen_settings_v2.json")))["explanatory"]
    model = load_spatial(os.path.join(ROOT, "v6_voxel.pt"), yaw_known=False)
    files = {_vid_num(f): f for f in glob.glob(os.path.join(ROOT, "data", "trajectories_3d_det", "*.npz"))}
    report = {}
    for name, cfrun, qset in SETS:
        cf = {}
        for r in S.load(cfrun).values():
            if r["unresolved"] or r["removed_unresolved"] or len(r["removed"]) != 1:
                continue
            cf.setdefault((r["video"], r["removed"][0], tuple(sorted((r["i"], r["j"])))), []).append(r["truth_collides"])
        vids = {k[0] for k in cf}
        rows = {}
        for r in load_questions(os.path.join(ROOT, QF[qset])):
            v = _vid_num(str(r.get("video", "")))
            if v in vids:
                rows.setdefault(v, []).append(r)
        c = Counter()
        examples = []
        matched = []
        for v in sorted(rows):
            if v not in files:
                continue
            z = np.load(files[v], allow_pickle=True)
            obs = CFW.build_world(model, z, set(), extend=0, entry_fix=entry)
            sc = Scene([str(k) for k in z["obj_keys"]], z["presence"].astype(np.float32),
                       np.linalg.norm(z["velocities"], axis=-1).astype(np.float32), lean(obs), None,
                       det=cfg["det"], refine_span=cfg["refine_span"])
            for q in rows[v]:
                qprog = qparser.parse(q.get("question", "")) or []
                if "filter_ancestor" not in qprog:
                    continue
                tgt = sc.run_raw(qprog[:qprog.index("filter_ancestor")])
                if not (isinstance(tgt, dict) and tgt.get("type") == "collision"):
                    continue
                pair = tuple(sorted(tgt["object"]))
                neg = "negate" in qprog
                for ctext, ans in zip(q["choices"]["choice"], q["choices"]["answer"]):
                    cand = sc.run_raw(qparser.parse(ctext) or [])
                    if isinstance(cand, int):
                        zobj, kind = cand, "presence"
                    elif isinstance(cand, dict) and cand.get("type") in ("in", "out"):
                        zobj, kind = cand["object"][0], "entrance"
                    else:
                        continue
                    if zobj in pair or (v, zobj, pair) not in cf:
                        continue
                    responsible = (ans == "correct") != neg
                    for collides in cf[(v, zobj, pair)]:
                        agree = responsible != collides
                        matched.append({"video": v, "qid": str(q["question_id"]), "choice": ctext, "kind": kind, "Z": int(zobj),
                                        "pair": [int(p) for p in pair], "responsible": responsible, "collides_without_Z": bool(collides)})
                        c[(kind, "responsible" if responsible else "not responsible",
                           "collides without Z" if collides else "no collision without Z")] += 1
                        if responsible and collides and len(examples) < 15:
                            examples.append({"video": v, "qid": q["question_id"], "question": q["question"], "choice": ctext,
                                             "Z": str(z["obj_keys"][zobj]), "pair": [str(z["obj_keys"][p]) for p in pair]})
        n = sum(c.values())
        agree = sum(v for k, v in c.items() if (k[1] == "responsible") != (k[2] == "collides without Z"))
        resp = sum(v for k, v in c.items() if k[1] == "responsible")
        resp_coll = sum(v for k, v in c.items() if k[1] == "responsible" and k[2] == "collides without Z")
        notr = n - resp
        notr_nocoll = sum(v for k, v in c.items() if k[1] == "not responsible" and k[2] == "no collision without Z")
        print("\n=== %s: %d matched (explanatory option, counterfactual option) pairs" % (name, n))
        for k in sorted(c):
            print("   %-9s %-16s %-24s %d" % (k + (c[k],)))
        print("   keys agree with the but-for reading: %d / %d (%.1f%%)" % (agree, n, 100.0 * agree / max(1, n)))
        print("   'responsible' but X and Y still collide without Z: %d / %d (%.1f%%)" % (resp_coll, resp, 100.0 * resp_coll / max(1, resp)))
        print("   'not responsible' but X and Y do not collide without Z: %d / %d (%.1f%%)" % (notr_nocoll, notr, 100.0 * notr_nocoll / max(1, notr)))
        report[name] = {"counts": {"|".join(k): v for k, v in c.items()}, "n": n, "agree": agree,
                        "responsible": resp, "responsible_but_collides": resp_coll,
                        "not_responsible": notr, "not_responsible_but_no_collision": notr_nocoll, "examples": examples, "rows": matched}
    json.dump(report, open(os.path.join(HERE, "key_consistency.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
