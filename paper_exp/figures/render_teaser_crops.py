"""
paper_exp/figures/render_teaser_crops.py  —  paper-ready panels for Figure 1
===========================================================================

Same worlds as render_teaser.py (video 10880, cube removed), redrawn with a soft floor
(tones 234 / 226 instead of the app's 208 / 176) and cropped with ONE fixed box so every
panel shares the same camera and scale. Also prints the pixel position of every event
marker inside the crop, so the markers can be drawn by hand at the right place.

    CF_LOOKBACK=3 python paper_exp/figures/render_teaser_crops.py
"""

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
spec = importlib.util.spec_from_file_location("cfw_fig2", os.path.join(PX, "cf_world.py"))
CFW = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CFW)
import render3d as R                                  # noqa: E402
import render_worker as RW                            # noqa: E402
from world import load_spatial                        # noqa: E402
from PIL import Image                                 # noqa: E402

VIDEO, WORLD, W, H = 10880, 6.0, 1440, 960
FRAMES = (15, 34, 54, 70)
VIEW = {"az": -69.17, "el": 34.9, "dist": 8.29}


def soft_ground(buf, zbuf, cam, half=6.0, step=0.4):
    n = int(2 * half / step)
    quads, depths, shades = [], [], []
    for i in range(n):
        for j in range(n):
            x0, y0 = -half + i * step, -half + j * step
            c = np.array([(x0, y0, 0), (x0 + step, y0, 0), (x0 + step, y0 + step, 0), (x0, y0 + step, 0)])
            uv, d, fr = R.project_pts(cam, c)
            if not fr.all():
                continue
            quads.append(uv); depths.append(float(d.mean()))
            t = 234 if (i + j) % 2 == 0 else 226
            shades.append((t, t, t + 2))
    for k in np.argsort(depths)[::-1]:
        R.fill_poly(buf, zbuf, quads[k], depths[k] + 0.05, shades[k])


R.draw_ground = soft_ground


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
    z = np.load(os.path.join(ROOT, "data", "trajectories_3d_det", "sim_%05d.npz" % VIDEO), allow_pickle=True)
    keys = [str(k) for k in z["obj_keys"]]
    model = load_spatial(os.path.join(ROOT, "v6_voxel.pt"), yaw_known=False)
    entry = json.load(open(os.path.join(PX, "entry_profile.json")))
    cube = [k for k, s in enumerate(keys) if s.endswith("cube")]
    worlds = {"obs": CFW.build_world(model, z, set(), extend=30, entry_fix=entry),
              "cf": CFW.build_world(model, z, set(cube), extend=30, entry_fix=entry)}
    cam = RW.view_camera(VIEW, W, H)
    # one crop box: every object centre over frames 10..80 in both worlds, +-0.5 table units
    pts = []
    for w in worlds.values():
        P = w["positions"][10:81].reshape(-1, 2)
        P = P[np.isfinite(P).all(1)] * WORLD
        for dx in (-0.5, 0.5):
            for dy in (-0.5, 0.5):
                for zz in (0.0, 0.8):
                    pts.append(np.c_[P[:, 0] + dx, P[:, 1] + dy, np.full(len(P), zz)])
    uv, _, _ = R.project_pts(cam, np.concatenate(pts))
    u0, v0 = np.floor(uv.min(0)).astype(int)
    u1, v1 = np.ceil(uv.max(0)).astype(int)
    # widen to 4:3
    cw, ch = u1 - u0, v1 - v0
    if cw / ch < 4 / 3:
        e = int((ch * 4 / 3 - cw) / 2); u0 -= e; u1 += e
    else:
        e = int((cw * 3 / 4 - ch) / 2); v0 -= e; v1 += e
    u0, v0 = max(0, u0), max(0, v0)
    u1, v1 = min(W, u1), min(H, v1)
    print("crop box (pixels of the %dx%d render): u %d..%d  v %d..%d  -> %d x %d" % (W, H, u0, u1, v0, v1, u1 - u0, v1 - v0))
    for name, w in worlds.items():
        for t in FRAMES:
            _, img = RW.render_one((t, VIEW, objects(keys, z["attrs"], w["positions"][t]), [], W, H))
            Image.fromarray(img[v0:v1, u0:u1]).save(os.path.join(out, "fig1_panel_%s_t%03d.png" % (name, t)))
    # marker positions: midpoint of the two centres at the event frame, at sphere-centre height
    marks = []
    def mark(label, world, a, b, f):
        P = worlds[world]["positions"]
        pa = P[f, keys.index(a)] if np.isfinite(P[f, keys.index(a)]).all() else None
        pb = P[f, keys.index(b)]
        if pa is None:  # removed object: take the recorded position
            pa = worlds["obs"]["positions"][f, keys.index(a)]
        m = (pa + pb) / 2 * WORLD
        uv, _, _ = R.project_pts(cam, np.array([[m[0], m[1], 0.35]]))
        marks.append({"label": label, "world": world, "frame": f, "pair": [a, b],
                      "pixel_in_crop": [round(float(uv[0, 0]) - u0, 1), round(float(uv[0, 1]) - v0, 1)],
                      "crop_size": [int(u1 - u0), int(v1 - v0)]})
    mark("recorded collision", "obs", "blue_metal_cube", "red_rubber_sphere", 34)
    mark("recorded collision", "obs", "red_rubber_sphere", "brown_metal_cylinder", 54)
    mark("recorded collision", "obs", "blue_metal_cube", "gray_rubber_sphere", 70)
    mark("prevented collision (drawn where the video had it)", "obs", "red_rubber_sphere", "brown_metal_cylinder", 54)
    mark("new collision", "cf", "red_rubber_sphere", "gray_rubber_sphere", 66)
    # where each object is at the panel frames, for labels
    for name, w in worlds.items():
        for t in FRAMES:
            for k, key in enumerate(keys):
                p = w["positions"][t, k]
                if np.isfinite(p).all():
                    uv, _, _ = R.project_pts(cam, np.array([[p[0] * WORLD, p[1] * WORLD, 0.0]]))
                    marks.append({"label": "object base", "world": name, "frame": t, "object": key,
                                  "pixel_in_crop": [round(float(uv[0, 0]) - u0, 1), round(float(uv[0, 1]) - v0, 1)]})
    # closest approach of the red sphere and the cylinder in the counterfactual world
    P = worlds["cf"]["positions"]
    a, b = keys.index("red_rubber_sphere"), keys.index("brown_metal_cylinder")
    ok = np.isfinite(P[:, a]).all(1) & np.isfinite(P[:, b]).all(1)
    d = np.where(ok, np.linalg.norm(P[:, a] - P[:, b], axis=1), np.inf)
    ra, rb = float(z["attrs"][a, 15]), float(z["attrs"][b, 15])
    info = {"cf_red_cylinder_min_gap_table_units": round(float(d.min() - ra - rb) * WORLD, 3),
            "cf_red_cylinder_min_gap_frame": int(d.argmin()),
            "contact_distance_table_units": round((ra + rb) * WORLD, 3)}
    print(info)
    json.dump({"crop": [int(u0), int(u1), int(v0), int(v1)], "render": [W, H], "marks": marks, "info": info},
              open(os.path.join(out, "fig1_panel_marks.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
