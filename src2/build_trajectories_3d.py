"""
build_trajectories_3d.py  —  metric 3D trajectories from CLEVRER proposals
==========================================================================

Replaces the pixel-space trajectory builder. Same output schema, so
dynamics.py and every diagnostic train and evaluate on it unchanged -- but
every quantity is now metric and geometrically exact instead of being a
perspective projection.

WHAT CHANGES, AND WHY IT MATTERS
--------------------------------
pixel pipeline (build_trajectories.py)      this file
-------------------------------------      -------------------------------
position = mask centroid / frame size       ray-plane intersection, metres
radius   = sqrt(area/pi), SHRINKS with      one constant metric radius per
           distance from the camera         object for the whole clip
contact  = varies 0.087..0.148 depending     constant, set by real object size
           on where on the table you are
mass     = a per-material guess              volume x material density
occlusion= object silently disappears        flagged as PRESENT BUT HIDDEN
orientation = does not exist                 recoverable (noisy -- see below)

The occlusion one is a genuine supervision bug, not a nicety. Presence is
currently (inside the in/out window) AND (detected). An object that passes
behind another is not detected, so presence goes to 0 and the world model is
taught that objects vanish and reappear. In 3D we can tell "hidden" from
"gone", which is exactly the kind of spatial reasoning the 2D pipeline cannot
do at all.

SCALE
-----
World units are divided by WORLD_SCALE = 6.0. That is not arbitrary: it puts
contact distance at 0.112 against the pixel pipeline's measured 0.1156, and
radius at 0.056 against 0.0585. So every downstream constant -- collision
threshold 0.02, hysteresis 0.02, chain margin 0.06 -- stays valid and nothing
else has to be retuned. ONE constant for the whole dataset; per-clip
normalisation would destroy the cross-clip consistency that is the point.

ORIENTATION: MEASURED, NOT TRUSTED
----------------------------------
Cube yaw can be recovered by silhouette fitting, but measured on sim_00000 it
jumps 76.9 -> 72.3 -> 74.6 -> 56.2 -> 49.4 -> 67.7 degrees between sampled
frames. A sliding cube rotates smoothly, so that is fit noise: a cube's
silhouette from this camera is weakly sensitive to yaw. Orientation is
therefore written out only under --with-yaw (slow, and off by default) and is
flagged noisy. Do not build on it without validating it first.

RUN
---
    # try a handful and read the report
    python src2/build_trajectories_3d.py --limit 20 --output data/traj3d_test

    # the real thing (~7 h for 20k; run it when the CPU is free)
    python src2/build_trajectories_3d.py --output data/trajectories_3d

    # then a split over the new folder, and train exactly as before
    python src2/make_split.py --data data/trajectories_3d --out split3d.json
    python src2/dynamics.py --data data/trajectories_3d --split split3d.json ...
"""

import os
import sys
import glob
import json
import math
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lift3d import (Camera, read_clip, MASS_PROXY, RESTITUTION,
                    SHAPES, COLORS, MATERIALS, VEL_HALF_WINDOW, MAX_GAP_FILL,
                    lsq_velocity, fill_short_gaps, object_key)
from fit3d import load_one, fit_yaw_track, exact_contact

WORLD_SCALE = 6.0          # see the SCALE note above -- one constant, dataset wide

# Density ratio only; the absolute value is irrelevant because the dynamics
# uses mass ratios. Keeps metal heavier than rubber, as the old proxy did.
DENSITY = {"metal": 1.5, "rubber": 1.0}


def volume_of(shape, s):
    """Metric volume of each primitive as fit3d defines it (base on the table,
    height 2s)."""
    if shape == "sphere":
        return 4.0 / 3.0 * math.pi * s ** 3
    if shape == "cylinder":
        return math.pi * s ** 2 * (2 * s)
    return (2 * s) ** 3                       # cube of half-side s


def occlusion_flags(pos3, rad3, depth, present):
    """Which objects are hidden behind a nearer one this frame.

    Each object is approximated by a disc of its own angular radius at its own
    depth; object A counts as occluded when a NEARER object's disc covers most
    of it. Approximate on purpose -- exact silhouette compositing costs ~0.8 s
    per clip and this costs microseconds, and the flag only has to be right
    about 'mostly hidden'.
    """
    T, N = present.shape
    occ = np.zeros((T, N), np.float32)
    for t in range(T):
        live = [k for k in np.where(present[t])[0] if np.isfinite(depth[t, k])]
        for i in live:
            ri = rad3[i] / max(depth[t, i], 1e-6)
            if ri <= 0:
                continue
            covered = 0.0
            for j in live:
                if i == j or depth[t, j] >= depth[t, i]:
                    continue                   # j must be NEARER than i
                rj = rad3[j] / max(depth[t, j], 1e-6)
                # separation projected onto the image plane, at i's depth
                d = np.linalg.norm(pos3[t, i] - pos3[t, j]) / max(depth[t, i], 1e-6)
                # standard circle-circle overlap. An earlier version only
                # checked full containment (d < rj - ri), which never fires
                # here because every CLEVRER object is nearly the same size --
                # so it reported zero occlusion across 14k object-frames.
                if d >= ri + rj:
                    continue
                if d <= abs(rj - ri):
                    inter = math.pi * min(ri, rj) ** 2
                else:
                    a1 = math.acos(np.clip((d*d + ri*ri - rj*rj) / (2*d*ri), -1, 1))
                    a2 = math.acos(np.clip((d*d + rj*rj - ri*ri) / (2*d*rj), -1, 1))
                    inter = (ri*ri*(a1 - math.sin(2*a1)/2)
                             + rj*rj*(a2 - math.sin(2*a2)/2))
                covered += inter / (math.pi * ri * ri)
            occ[t, i] = min(covered, 1.0)
    return occ


def build_one(path, cam, sizes, with_yaw=False):
    # the yaw fitter needs the decoded masks, not just centroids
    clip = load_one(path) if with_yaw else read_clip(path)
    T, N = clip["T"], clip["N"]
    objs = clip["objects"]

    pos = np.full((T, N, 2), np.nan, np.float64)
    dep = np.full((T, N), np.nan, np.float64)
    rad_frames = [[] for _ in range(N)]

    for t in range(T):
        for k in range(N):
            d = clip["det"][t][k]
            if d is None or not clip["inside"][t, k]:
                continue
            u, v, area, _ = d
            shape = objs[k].get("shape")
            s = sizes.get(shape, 0.35)
            # the object's centre sits at height s; back-project the mask
            # centroid onto THAT plane, not onto the table
            X, Y, depth = cam.intersect_plane(u, v, s)
            if not (np.isfinite(X) and np.isfinite(Y) and np.isfinite(depth)):
                continue
            if abs(X) > cam.max_extent or abs(Y) > cam.max_extent:
                continue                       # grazing ray, not a measurement
            pos[t, k] = (X, Y)
            dep[t, k] = depth
            rad_frames[k].append(s)

    rad = np.array([sizes.get(o.get("shape"), 0.35) for o in objs], np.float64)

    posn = pos / WORLD_SCALE
    posn, radn, present = fill_short_gaps(posn, np.tile(rad / WORLD_SCALE, (T, 1)),
                                          clip["inside"])
    if present.sum() == 0:
        return None
    posn[~present] = 0.0
    vel = lsq_velocity(posn, present)

    occ = occlusion_flags(pos, rad, np.nan_to_num(dep, nan=1e9), present)

    # ---- orientation, and the exact contact geometry it unlocks ------------
    # r_i + r_j is EXACT for spheres and cylinders: their horizontal cross
    # section is a circle, so contact distance does not depend on direction.
    # It is wrong only for cubes -- understating contact by up to 19% against a
    # sphere and 41% against another cube. So orientation is worth recovering
    # for exactly one shape, and only cubes get the expensive treatment.
    yaw = np.zeros((T, N), np.float32)
    if with_yaw:
        for k, o in enumerate(objs):
            if o.get("shape") != "cube":
                continue
            xy = [tuple(pos[t, k]) if np.isfinite(pos[t, k]).all() else None
                  for t in range(T)]
            track = fit_yaw_track(clip, cam, sizes, k, xy)
            for t in range(T):
                if track[t] is not None:
                    yaw[t, k] = track[t]

    # per-pair exact contact distance, in the SAME normalised units as
    # positions, so it can be dropped straight into the physics
    contact = np.zeros((T, N, N), np.float32)
    for t in range(T):
        for i in range(N):
            if not present[t, i]:
                continue
            for j in range(i + 1, N):
                if not present[t, j]:
                    continue
                dx = posn[t, j, 0] - posn[t, i, 0]
                dy = posn[t, j, 1] - posn[t, i, 1]
                if dx == 0.0 and dy == 0.0:
                    dx = 1.0
                yi = float(yaw[t, i]) if with_yaw else None
                yj = float(yaw[t, j]) if with_yaw else None
                cij = exact_contact(objs[i].get("shape"), rad[i], yi,
                                    objs[j].get("shape"), rad[j], yj,
                                    dx, dy) / WORLD_SCALE
                contact[t, i, j] = contact[t, j, i] = cij

    at = np.zeros((N, 16), np.float32)
    for k, o in enumerate(objs):
        shape, material = o.get("shape"), o.get("material")
        vol = volume_of(shape, rad[k])
        at[k, 0] = DENSITY.get(material, 1.0) * vol / 0.1        # mass, scaled
        at[k, 1] = RESTITUTION.get(material, 0.5)
        if shape in SHAPES:
            at[k, 2 + SHAPES.index(shape)] = 1.0
        if o.get("color") in COLORS:
            at[k, 5 + COLORS.index(o["color"])] = 1.0
        if material in MATERIALS:
            at[k, 13 + MATERIALS.index(material)] = 1.0
        # METRIC radius. For a cube store the direction-AVERAGED support
        # (s * 4/pi), so r_i + r_j reproduces true contact on average and the
        # support function can recover the half-side as r / (4/pi). Storing
        # the raw half-side understated cube contact by 21% (see
        # fix_cube_radius.py, which this makes unnecessary for new runs).
        at[k, 15] = rad[k] / WORLD_SCALE * (4.0 / math.pi if shape == "cube" else 1.0)

    return {
        "positions": posn.astype(np.float32),
        "velocities": vel.astype(np.float32),
        "presence": present.astype(np.float32),
        "attrs": at,
        "collisions": np.asarray(clip["collisions"], np.int64).reshape(-1, 3),
        "exits": np.asarray(clip["exits"], np.int64).reshape(-1, 2),
        "obj_keys": np.asarray([object_key(o) for o in objs]),
        "video_name": clip["video_name"],
        # --- 3D extras, ignored by dynamics.py, available to everything else -
        "size_metric": rad.astype(np.float32),
        "volume": np.array([volume_of(o.get("shape"), rad[k])
                            for k, o in enumerate(objs)], np.float32),
        "height": (2 * rad).astype(np.float32),
        "depth_from_camera": np.nan_to_num(dep).astype(np.float32),
        "occluded": occ,
        "yaw": yaw,                      # cubes only; zero for round shapes
        "contact_exact": contact,        # [T,N,N] shape-aware, beats r_i+r_j
        "world_scale": np.float32(WORLD_SCALE),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="legacy/v1/data/processed_proposals")
    ap.add_argument("--output", default="data/trajectories_3d")
    ap.add_argument("--camera", default="camera_fit.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=None, help="first clip number (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="last clip number (exclusive)")
    ap.add_argument("--with-yaw", action="store_true",
                    help="also fit cube orientation (SLOW, and noisy -- see docstring)")
    a = ap.parse_args()

    with open(a.camera, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    cam, sizes = Camera(blob["camera"]), blob["sizes"]
    print(f"camera: {a.camera}  (median silhouette IoU "
          f"{blob.get('median_iou', float('nan')):.4f})")
    print(f"sizes : {sizes}")
    print(f"scale : world / {WORLD_SCALE}  -> contact "
          f"{2*np.mean(list(sizes.values()))/WORLD_SCALE:.4f} "
          f"(pixel pipeline measured 0.1156)")

    files = sorted(glob.glob(os.path.join(a.input, "*.json")))
    if a.start is not None or a.end is not None:
        import re
        def num(f): return int(re.search(r"(\d+)", os.path.basename(f)).group(1))
        files = [f for f in files if (a.start is None or num(f) >= a.start)
                 and (a.end is None or num(f) < a.end)]
        print(f"clip range [{a.start}, {a.end}) -> {len(files)} files")
    if a.limit:
        files = files[:a.limit]
    os.makedirs(a.output, exist_ok=True)

    written = skipped = 0
    occ_total = occ_frames = 0
    for n, p in enumerate(files):
        # resumable: a 17-hour yaw run must survive an interruption
        out_path = os.path.join(a.output, os.path.basename(p).replace(".json", ".npz"))
        if os.path.exists(out_path):
            written += 1
            continue
        try:
            rec = build_one(p, cam, sizes, a.with_yaw)
            if rec is None:
                skipped += 1
                continue
            occ_total += int((rec["occluded"] > 0.5).sum())
            occ_frames += int(rec["presence"].sum())
            name = str(rec["video_name"])
            final = os.path.join(a.output, f"{name}.npz")
            tmp = final + f".{os.getpid()}.tmp"
            np.savez_compressed(tmp, **rec)          # savez appends .npz to the name
            os.replace(tmp + ".npz" if not tmp.endswith(".npz") else tmp, final)
            written += 1
        except Exception as ex:
            skipped += 1
            if skipped <= 5:
                print(f"   skipped {os.path.basename(p)}: {ex}")
        if (n + 1) % 200 == 0:
            print(f"   {n+1}/{len(files)}  written {written}  skipped {skipped}",
                  flush=True)

    print(f"\nwrote {written} clips -> {a.output}  (skipped {skipped})")
    if occ_frames:
        print(f"occluded object-frames: {occ_total}/{occ_frames} "
              f"({100*occ_total/occ_frames:.2f}%) -- these were previously "
              f"indistinguishable from an object having left the scene")


if __name__ == "__main__":
    main()
