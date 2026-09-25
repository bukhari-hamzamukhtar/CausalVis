"""
paper_exp/perception_parity.py  —  is perception equally good on every evaluation set?
=====================================================================================

TEST-B's videos come from CLEVRER's training portion, and NS-DR's detector was trained
on frames from that portion. If its detections were better there, TEST-B would be
easier for reasons unrelated to our method. This compares, per set, the detection-only
tracks (data/trajectories_3d_det) with the scene annotation: object inventory recall /
precision, clips with an exact inventory, entry / exit frame error, and the share of
frames with a detection while the annotation says the object is in view.

    python paper_exp/perception_parity.py
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src2"))
from benchmark_eval import load_questions, load_split_set, _vid_num      # noqa: E402

PROP = os.path.join(ROOT, "legacy", "v1", "data", "processed_proposals")
SETS = {"VAL-A": ("val", "zechennlp/counterfactual/validation-00000-of-00001.json"),
        "TEST-A": ("test", "zechennlp/counterfactual/validation-00000-of-00001.json"),
        "TEST-B": ("test", "zechennlp/counterfactual/train-00000-of-00001.json"),
        "VAL-B": ("val", "zechennlp/counterfactual/train-00000-of-00001.json")}


def main():
    res = {}
    for name, (which, qf) in SETS.items():
        allowed = load_split_set(os.path.join(ROOT, "split3d_yaw.json"), which)
        vids = sorted(set(_vid_num(r["video"]) for r in load_questions(os.path.join(ROOT, qf))) & allowed)
        g = o = m = exact = 0
        d_in, d_out, cover, missing = [], [], [], 0
        for v in vids:
            p = os.path.join(ROOT, "data", "trajectories_3d_det", "sim_%05d.npz" % v)
            if not os.path.exists(p):
                missing += 1
                continue
            z = np.load(p, allow_pickle=True)
            gt = json.load(open(os.path.join(PROP, "sim_%05d.json" % v)))["ground_truth"]
            gk = {(ob["color"], ob["material"], ob["shape"]): ob["id"] for ob in gt["objects"]}
            ok = {tuple(str(k).split("_")[:3]): i for i, k in enumerate(z["obj_keys"])}
            mt = set(gk) & set(ok)
            g += len(gk); o += len(ok); m += len(mt); exact += int(len(gk) == len(ok) == len(mt))
            T = z["presence"].shape[0]
            ins = {e["object"]: e["frame"] for e in gt["in_outs"] if e["type"] == "in"}
            outs = {e["object"]: e["frame"] for e in gt["in_outs"] if e["type"] == "out"}
            for key in mt:
                gi, oi = gk[key], ok[key]
                pr = z["presence"][:, oi] > 0
                on = np.where(pr)[0]
                if on.size == 0:
                    continue
                a, b = ins.get(gi, 0), min(outs.get(gi, T), T)
                d_in.append(abs(int(on[0]) - a)); d_out.append(abs(int(on[-1]) + 1 - b))
                if b > a:
                    cover.append(float(pr[a:b].mean()))
        res[name] = {"videos": len(vids), "missing_tracks": missing, "recall": round(m / max(1, g), 4),
                     "precision": round(m / max(1, o), 4), "exact_inventory": round(exact / max(1, len(vids) - missing), 4),
                     "entry_err_mean": round(float(np.mean(d_in)), 3), "exit_err_mean": round(float(np.mean(d_out)), 3),
                     "coverage_in_view": round(float(np.mean(cover)), 4)}
        print(name, json.dumps(res[name]), flush=True)
    json.dump(res, open(os.path.join(HERE, "perception_parity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
