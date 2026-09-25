"""
probe/edge_fix_test.py  —  are the bad entry states caused by objects cut off at the image edge?
================================================================================================

src2/build_trajectories_3d.py puts each object at the CENTRE OF ITS VISIBLE MASK,
back-projected onto the plane at the object's mid-height. While an object slides
into view it is only partly inside the picture, so that centre sits too far
inside and moves faster than the object -- a wrong position AND a wrong speed at
exactly the moment the simulator has to start the object.

The renderer inversion in src2/fit3d.py already handles this correctly: the
rendered silhouette is clipped to the frame, so matching it against a cut-off
mask recovers the whole object's true position. It was only ever used for
cube rotation.

This test refits ONLY the frames where an entering object's mask touches the
image border, and measures entry errors before and after, on VAL clips.

    python probe/edge_fix_test.py --clips 150
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src2"))
from benchmark_eval import load_split_set, _vid_num      # noqa: E402
from lift3d import Camera, lsq_velocity                  # noqa: E402
from fit3d import load_one, fit_pose                     # noqa: E402

WORLD_SCALE = 6.0
PROPOSALS = os.path.join(ROOT, "legacy", "v1", "data", "processed_proposals")


def touches_border(m):
    return bool(m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any())


def slope(P, ts):
    ts = np.asarray(ts, float); X = ts - ts.mean(); Y = P - P.mean(0)
    return (X[:, None] * Y).sum(0) / (X ** 2).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=150)
    a = ap.parse_args()
    blob = json.load(open(os.path.join(ROOT, "camera_fit.json")))
    cam, sizes = Camera(blob["camera"]), blob["sizes"]
    files = {_vid_num(f): f for f in glob.glob(os.path.join(ROOT, "data", "trajectories_3d_yaw", "*.npz"))}
    val = sorted(k for k in load_split_set(os.path.join(ROOT, "split3d_yaw.json"), "val") if k in files)[:a.clips]

    stats = {name: {"spd": [], "ang": [], "pos": []} for name in ("base", "fix")}
    n_entry = n_border = fits = 0
    t0 = time.time()
    for n, v in enumerate(val):
        p = os.path.join(PROPOSALS, "sim_%05d.json" % v)
        if not os.path.exists(p):
            continue
        z = np.load(files[v], allow_pickle=True)
        pos = z["positions"].astype(np.float64); pres = z["presence"] > 0
        T, N, _ = pos.shape
        c = load_one(p)
        if c["N"] != N:
            continue
        cols = {}
        for f, i, j in z["collisions"]:
            for k in (int(i), int(j)):
                cols.setdefault(k, []).append(int(f))
        fixed = pos.copy()
        cut = np.zeros((T, N), bool)
        entering = []
        for k in range(N):
            on = np.where(pres[:, k])[0]
            if not on.size:
                continue
            e = int(on[0])
            if e == 0 or e + 16 >= T or not pres[e:e + 16, k].all():
                continue
            if any(e <= f <= e + 16 for f in cols.get(k, [])):
                continue
            entering.append((k, e))
            shape = c["objects"][k].get("shape")
            size = sizes.get(shape, 0.35)
            prev = None
            for t in range(e, e + 16):
                m, d = c["raw"][t][k], c["det"][t][k]
                if m is None or d is None or not touches_border(m):
                    continue
                cut[t, k] = True
                r = fit_pose(cam, shape, size, m, d[0], d[1], iters=3, init=prev)
                prev = (r["x"], r["y"], r["yaw"])
                fixed[t, k] = (r["x"] / WORLD_SCALE, r["y"] / WORLD_SCALE)
                fits += 1
        if not entering:
            continue
        vel_base = lsq_velocity(pos.copy(), pres)
        vel_fix = lsq_velocity(fixed.copy(), pres)
        for k, e in entering:
            n_entry += 1
            n_border += bool(cut[e, k])
            clean = [t for t in range(e + 5, e + 16) if not cut[t, k]]
            if len(clean) < 5:
                continue
            for name, P, Vf in (("base", pos, vel_base), ("fix", fixed, vel_fix)):
                vt = slope(P[clean, k], clean)
                sp = float(np.linalg.norm(vt))
                if sp < 0.0015:
                    continue
                vu = Vf[e, k]
                stats[name]["spd"].append(float(np.linalg.norm(vu - vt)) / sp)
                cosv = float(vu @ vt) / (float(np.linalg.norm(vu)) * sp + 1e-12)
                stats[name]["ang"].append(float(np.degrees(np.arccos(np.clip(cosv, -1, 1)))))
                line = P[clean, k].mean(0) + vt * (e - np.mean(clean))
                stats[name]["pos"].append(float(np.linalg.norm(P[e, k] - line)))
        if (n + 1) % 25 == 0:
            print("  %d/%d clips  %d shape fits  %.0fs" % (n + 1, len(val), fits, time.time() - t0), flush=True)

    print("\n" + "=" * 74)
    print("ENTERING OBJECTS, %d VAL clips: %d entries, %d (%.0f%%) cut off by the image edge at entry"
          % (len(val), n_entry, n_border, 100 * n_border / max(n_entry, 1)))
    print("%d edge frames refitted by shape, %.1f s total" % (fits, time.time() - t0))
    print("=" * 74)
    print("                                   mask centre (now)   shape fit (fix)")
    for key, label, fmt in (("spd", "entry speed error, median", "%.0f%%"), ("ang", "entry direction error, median", "%.1f deg"),
                            ("pos", "entry position off clean path", "%.4f")):
        b, f = np.array(stats["base"][key]), np.array(stats["fix"][key])
        scale = 100 if key == "spd" else 1
        print("   %-32s %-19s %s" % (label, fmt % (np.median(b) * scale), fmt % (np.median(f) * scale)))
    print("   (compared on %d entries with enough clean frames)" % len(stats["fix"]["spd"]))


if __name__ == "__main__":
    main()
