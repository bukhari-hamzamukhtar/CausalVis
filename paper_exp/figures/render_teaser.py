"""
paper_exp/figures/render_teaser.py  —  source frames and track CSVs for Figure 1
================================================================================

TEST-A video 10880, question 14: "What will happen if the cube is removed?"
Builds the observed world and the counterfactual world (blue metal cube removed) with the
final CausalVis engine (learned world model, look-ahead 3, +30 frames), then writes
  figures/src/fig1_obs_tNNN.png   recorded tracks, video camera, 960 x 640
  figures/src/fig1_cf_tNNN.png    counterfactual world, same camera and frames
  figures/src/fig1_tracks.csv     frame, world, object, x, y (table units), handed_off
  figures/src/fig1_events.json    events of both worlds and the hand-off frames

    CF_LOOKBACK=3 python paper_exp/figures/render_teaser.py
"""

import csv
import importlib.util
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PX = os.path.dirname(HERE)
ROOT = os.path.dirname(PX)
for d in ("src2", "v4", "v5", "v6", "v7", "app"):
    sys.path.insert(0, os.path.join(ROOT, d))
spec = importlib.util.spec_from_file_location("cfw_fig", os.path.join(PX, "cf_world.py"))
CFW = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CFW)
from world import load_spatial                      # noqa: E402
from render_worker import render_one                # noqa: E402
from PIL import Image                               # noqa: E402

VIDEO, WORLD, W, H = 10880, 6.0, 960, 640
FRAMES = (0, 15, 27, 34, 47, 54, 60, 66, 73, 80, 100, 126, 140, 157)
VIEW = {"az": -69.17, "el": 34.9, "dist": 8.29}


def objects(keys, attrs, p):
    out = []
    for k, key in enumerate(keys):
        if not np.isfinite(p[k]).all():
            continue
        col, mat, shp = key.split("_")[:3]
        r = float(attrs[k, 15]) * WORLD
        out.append({"shape": shp, "size": r / (4.0 / math.pi) if shp == "cube" else r,
                    "x": float(p[k, 0]) * WORLD, "y": float(p[k, 1]) * WORLD, "yaw": 0.0, "color": col, "material": mat})
    return out


def main():
    out = os.path.join(HERE, "src")
    os.makedirs(out, exist_ok=True)
    z = np.load(os.path.join(ROOT, "data", "trajectories_3d_det", "sim_%05d.npz" % VIDEO), allow_pickle=True)
    keys = [str(k) for k in z["obj_keys"]]
    model = load_spatial(os.path.join(ROOT, "v6_voxel.pt"), yaw_known=False)
    entry = json.load(open(os.path.join(PX, "entry_profile.json")))
    cube = [k for k, s in enumerate(keys) if s.endswith("cube")]
    worlds = {"obs": CFW.build_world(model, z, set(), extend=30, entry_fix=entry),
              "cf": CFW.build_world(model, z, set(cube), extend=30, entry_fix=entry)}
    ev = {}
    rows = []
    for name, w in worlds.items():
        P = w["positions"]
        ev[name] = {"T": int(w["T"]), "taint": {keys[k]: list(v) for k, v in w["taint"].items()},
                    "events": {d: {"%s|%s" % (keys[p[0]], keys[p[1]]): [int(f) for f in fr] for p, fr in w["events"][d].items()}
                               for d in ("cal", "kick2")}}
        for t in range(P.shape[0]):
            for k in range(len(keys)):
                if np.isfinite(P[t, k]).all():
                    ho = name == "cf" and k in w["taint"] and t >= w["taint"][k][0]
                    rows.append([t, name, keys[k], round(float(P[t, k, 0]) * WORLD, 4), round(float(P[t, k, 1]) * WORLD, 4), int(ho)])
        for t in FRAMES:
            if t >= P.shape[0]:
                continue
            _, img = render_one((t, VIEW, objects(keys, z["attrs"], P[t]), [], W, H))
            Image.fromarray(img).save(os.path.join(out, "fig1_%s_t%03d.png" % (name, t)))
            print(name, t, flush=True)
    with open(os.path.join(out, "fig1_tracks.csv"), "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["frame", "world", "object", "x", "y", "handed_off_to_simulator"])
        cw.writerows(rows)
    ev["objects"] = keys
    ev["removed"] = [keys[k] for k in cube]
    ev["annotated_video_collisions"] = [[int(c[0]), keys[int(c[1])], keys[int(c[2])]] for c in z["collisions"]]
    json.dump(ev, open(os.path.join(out, "fig1_events.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
