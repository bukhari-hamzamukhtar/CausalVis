"""
build_trajectories.py  —  CausalVis V3, Phase 1
================================================

Turns CLEVRER `processed_proposals` JSON files into per-object TRAJECTORY
sequences. This is the input the V3 world model (dynamics.py) trains on.

WHY THIS FILE EXISTS
--------------------
V1/V2 turned each collision into one yes/no training example (a classifier).
V3 is a world model: it must watch objects MOVE and learn to roll motion
forward. So instead of (collision -> label) rows, we emit per video:

    positions   [T, N, 2]   where each object is at every frame  (normalised 0-1)
    velocities  [T, N, 2]   how fast it moves (least-squares fit, see below)
    attrs       [N, A]      fixed facts per object (mass proxy, material, shape...)
    presence    [T, N]      1 if the object is really on screen that frame, else 0
    collisions  [K, 3]      ground-truth (frame, i, j)
    exits       [K2, 2]     ground-truth (frame, i) for objects that leave
    obj_keys    [N]         readable id, e.g. "blue_rubber_sphere"

T = frames (128 in CLEVRER), N = objects in that video.

REAL-DATA QUIRKS THIS FILE HANDLES  (found by inspecting sim_00000.json)
-----------------------------------------------------------------------
1. RLE `counts` arrives as a STRING. Some pycocotools builds demand bytes.
   We encode it defensively so it never crashes.

2. THE DETECTOR FLIPS ATTRIBUTES BETWEEN FRAMES.
   In sim_00000 the cyan cube is detected as ("cyan","rubber","cube") at
   frame 9, then ("cyan","metal","cube") from frame 10 on. Ground truth says
   metal. A strict (color, material, shape) lookup would silently DROP that
   frame. So we match in three passes:
       exact (color, material, shape) -> (color, shape) -> (shape) 
   and if several detections claim the same object in one frame we keep the
   one with the highest detection score.

3. DETECTIONS STOP BEFORE THE GROUND-TRUTH "out" FRAME.
   Object 0 in sim_00000 was last detected ~frame 70 but `in_outs` says it
   exits at frame 74. If we trusted in/out alone we would freeze the object
   in place for those frames and invent a stationary object where one is
   actually leaving. So:
       presence = (inside the in/out window) AND (actually detected)
   with only SHORT internal gaps (occlusions) interpolated.

4. VELOCITY BY LEAST SQUARES, NOT FRAME DIFFERENCE.
   A single-frame difference is very noisy on centroids. We fit a straight
   line over a small window around each frame and take its slope. This must
   stay IDENTICAL between training and inference, or the model sees skewed
   inputs at test time (a lesson already paid for in V2).

RUN
---
    pip install numpy pycocotools tqdm
    python build_trajectories.py \
        --input  /path/to/processed_proposals \
        --output /path/to/trajectories_out \
        --limit 20          # start small, then drop --limit for all 20k
"""

import os
import glob
import json
import argparse
import numpy as np

try:
    from pycocotools import mask as coco_mask
except ImportError:
    coco_mask = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):
        return x


# ---------------------------------------------------------------- constants
FRAME_H, FRAME_W = 320, 480        # CLEVRER masks are [320, 480]

MASS_PROXY  = {"metal": 1.5, "rubber": 1.0}   # metal is heavier
RESTITUTION = {"metal": 0.8, "rubber": 0.5}   # metal bounces, rubber absorbs

SHAPES    = ["cube", "sphere", "cylinder"]
COLORS    = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
MATERIALS = ["metal", "rubber"]

VEL_HALF_WINDOW = 2      # least-squares velocity uses frames [t-2 .. t+2]
MAX_GAP_FILL    = 5      # only interpolate occlusion gaps up to this many frames


def _one(path):
        try:
            with open(path) as fh:
                data = json.load(fh)
            out = build_one_video(data)
            return (path, out)
        except Exception as e:
            return (path, None)
        
        
def _one_hot(value, vocab):
    v = [0.0] * len(vocab)
    if value in vocab:
        v[vocab.index(value)] = 1.0
    return v


def centroid_from_rle(rle):
    """Decode an RLE mask and return (cx, cy, area_px), or None.
    area_px = number of pixels in the mask -> a REAL measure of object size."""
    counts = rle["counts"]
    if isinstance(counts, str):                 # QUIRK 1: force bytes
        counts = counts.encode("utf-8")
    m = coco_mask.decode({"size": rle["size"], "counts": counts})
    ys, xs = np.where(m > 0)                    # rows = y, cols = x
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean()), float(len(xs))


def match_detection(det, exact, by_color_shape, by_shape):
    """
    QUIRK 2: the detector sometimes gets `material` (or colour) wrong.
    Try progressively looser matches. Returns an object index, or None.
    """
    k = (det["color"], det["material"], det["shape"])
    if k in exact:
        return exact[k]
    k2 = (det["color"], det["shape"])
    if k2 in by_color_shape:                    # ignore material
        return by_color_shape[k2]
    k3 = det["shape"]
    if k3 in by_shape:                          # last resort: unique shape
        return by_shape[k3]
    return None


def lstsq_velocity(positions, presence):
    """
    QUIRK 4: velocity as the slope of a straight line fitted over a small
    window, instead of a noisy one-frame difference.

    positions: [T, N, 2]   presence: [T, N]   ->  velocities [T, N, 2]
    Units: normalised position change per frame.
    """
    T, N, _ = positions.shape
    vel = np.zeros_like(positions)
    for i in range(N):
        for t in range(T):
            if presence[t, i] == 0:
                continue
            lo = max(0, t - VEL_HALF_WINDOW)
            hi = min(T, t + VEL_HALF_WINDOW + 1)
            idx = [u for u in range(lo, hi) if presence[u, i] > 0]
            if len(idx) < 2:
                continue
            ts = np.asarray(idx, dtype=np.float64)
            ts_c = ts - ts.mean()
            denom = (ts_c ** 2).sum()
            if denom < 1e-9:
                continue
            for d in range(2):
                ys = positions[idx, i, d].astype(np.float64)
                # slope of the least-squares line = cov(t, y) / var(t)
                vel[t, i, d] = float((ts_c * (ys - ys.mean())).sum() / denom)
    return vel


def build_one_video(data):
    """Parse one loaded JSON dict into trajectory arrays. None if unusable."""
    gt = data.get("ground_truth", {})
    registry = gt.get("objects", [])
    frames = data.get("frames", [])
    if not registry or len(frames) < 3:
        return None

    N = len(registry)
    T = max(f["frame_index"] for f in frames) + 1

    # ---- lookup tables for matching detections to ground-truth objects ----
    exact, by_color_shape, by_shape = {}, {}, {}
    dup_cs, dup_s = set(), set()
    for i, o in enumerate(registry):
        exact[(o["color"], o["material"], o["shape"])] = i
        cs = (o["color"], o["shape"])
        if cs in by_color_shape:
            dup_cs.add(cs)
        by_color_shape[cs] = i
        if o["shape"] in by_shape:
            dup_s.add(o["shape"])
        by_shape[o["shape"]] = i
    # a loose key is only safe if it identifies exactly ONE object
    for k in dup_cs:
        by_color_shape.pop(k, None)
    for k in dup_s:
        by_shape.pop(k, None)

    # ---- read centroids, keeping the highest-scoring detection per object --
    positions = np.full((T, N, 2), np.nan, dtype=np.float32)
    areas_px   = np.full((T, N), np.nan, dtype=np.float32)   # mask pixel count per frame
    detected  = np.zeros((T, N), dtype=bool)
    best_score = np.full((T, N), -1.0, dtype=np.float32)

    for f in frames:
        t = f["frame_index"]
        for det in f.get("objects", []):
            i = match_detection(det, exact, by_color_shape, by_shape)
            if i is None:
                continue
            s = float(det.get("score", 1.0))
            if s <= best_score[t, i]:          # QUIRK 2b: keep best duplicate
                continue
            c = centroid_from_rle(det["mask"])
            if c is None:
                continue
            positions[t, i, 0], positions[t, i, 1] = c[0], c[1]
            areas_px[t, i] = c[2]                       # measured size for this frame
            detected[t, i] = True
            best_score[t, i] = s

    # ---- presence = inside the in/out window AND actually detected ---------
    in_window = np.ones((T, N), dtype=bool)
    exits = []
    for i, o in enumerate(registry):
        oid = o.get("id", i)
        in_f, out_f = 0, T - 1
        for e in gt.get("in_outs", []):
            if e["object"] != oid:
                continue
            if e["type"] == "in":
                in_f = max(in_f, int(e["frame"]))
            elif e["type"] == "out":
                out_f = min(out_f, int(e["frame"]))
                exits.append((int(e["frame"]), i))
        in_window[:in_f, i] = False
        in_window[out_f + 1:, i] = False

    presence = (in_window & detected).astype(np.float32)   # QUIRK 3

    # ---- fill only SHORT gaps (brief occlusion) inside the visible span ----
    for i in range(N):
        vis = np.where(presence[:, i] > 0)[0]
        if len(vis) < 2:
            continue
        first, last = vis[0], vis[-1]
        span = np.arange(first, last + 1)
        missing = span[presence[span, i] == 0]
        if len(missing) == 0:
            continue
        # split the missing frames into consecutive runs
        runs, run = [], [missing[0]]
        for m in missing[1:]:
            if m == run[-1] + 1:
                run.append(m)
            else:
                runs.append(run); run = [m]
        runs.append(run)
        for r in runs:
            if len(r) > MAX_GAP_FILL:     # a long absence is NOT an occlusion
                continue
            for d in range(2):
                positions[r, i, d] = np.interp(r, vis, positions[vis, i, d])
            presence[r, i] = 1.0

    # ---- normalise to [0, 1] and zero out absent objects ------------------
    positions = np.nan_to_num(positions, nan=0.0)
    positions[:, :, 0] /= FRAME_W
    positions[:, :, 1] /= FRAME_H
    positions *= presence[:, :, None]

    velocities = lstsq_velocity(positions, presence)        # QUIRK 4

    # ---- fixed per-object attributes --------------------------------------
    attrs = []
    for o in registry:
        mat = o["material"]
        attrs.append(
            [MASS_PROXY.get(mat, 1.0), RESTITUTION.get(mat, 0.5)]
            + _one_hot(o["shape"], SHAPES)
            + _one_hot(o["color"], COLORS)
            + _one_hot(mat, MATERIALS)
        )
    attrs = np.asarray(attrs, dtype=np.float32)

    # ---- MEASURED object size (Q7): from mask pixel area, not learned --------
    # sphere/cylinder: r = sqrt(area/pi);  cube: half-side = sqrt(area)/2.
    # Take the median over frames the object was really seen (robust to
    # perspective and to occlusion). Normalise by sqrt(W*H) so an AREA in
    # pixels becomes a LENGTH in the same 0-1 scale the positions use.
    NORM = float(np.sqrt(FRAME_W * FRAME_H))            # 392.0
    meas_radius = np.zeros((N, 1), dtype=np.float32)
    for i, o in enumerate(registry):
        seen = (detected[:, i]) & np.isfinite(areas_px[:, i])
        if seen.sum() == 0:
            meas_radius[i, 0] = 0.05                    # fallback if never cleanly seen
            continue
        a_med = float(np.median(areas_px[seen, i]))
        if o["shape"] == "cube":
            r_px = np.sqrt(max(a_med, 1.0)) / 2.0
        else:                                          # sphere, cylinder
            r_px = np.sqrt(max(a_med, 1.0) / np.pi)
        meas_radius[i, 0] = float(np.clip(r_px / NORM, 0.01, 0.20))
    attrs = np.concatenate([attrs, meas_radius], axis=1)   # attrs is now [N, 16]

    # ---- ground-truth events ---------------------------------------------
    id_to_idx = {o.get("id", i): i for i, o in enumerate(registry)}
    collisions = []
    for c in gt.get("collisions", []):
        a, b = c["object"]
        if a in id_to_idx and b in id_to_idx:
            collisions.append((int(c["frame"]), id_to_idx[a], id_to_idx[b]))
    collisions = (np.asarray(collisions, dtype=np.int64)
                  if collisions else np.zeros((0, 3), np.int64))
    exits = (np.asarray(exits, dtype=np.int64)
             if exits else np.zeros((0, 2), np.int64))

    keys = ["_".join((o["color"], o["material"], o["shape"])) for o in registry]

    return {
        "positions":  positions,
        "velocities": velocities,
        "attrs":      attrs,
        "presence":   presence,
        "collisions": collisions,
        "exits":      exits,
        "obj_keys":   np.array(keys),
        "video_name": data.get("video_name", "unknown"),
    }


def load_trajectory(npz_path):
    """Load one saved trajectory back into a dict (used by dynamics.py)."""
    z = np.load(npz_path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to processed proposals")
    ap.add_argument("--output", required=True, help="Path for the output data")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if coco_mask is None:
        raise SystemExit("Missing pycocotools.  pip install pycocotools")

    os.makedirs(args.output, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input, "*.json")))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"No JSON files found in {args.input}")

    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    # replace the `for path in tqdm(files...)` loop with:
    ok, skipped = 0, 0
    workers = max(1, mp.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for path, out in tqdm(ex.map(_one, files, chunksize=8),
                              total=len(files), desc="Building trajectories"):
            if out is None:
                skipped += 1
                continue
            np.savez_compressed(
                os.path.join(args.output, f"{out['video_name']}.npz"), **out)
            ok += 1

    print(f"\nDone. Saved {ok}, skipped {skipped}.  ->  {args.output}")
    if ok:
        s = sorted(glob.glob(os.path.join(args.output, "*.npz")))[0]
        z = np.load(s, allow_pickle=True)
        print(f"\nSanity check on {os.path.basename(s)}")
        print("  positions ", z["positions"].shape, "(T, N, 2)")
        print("  velocities", z["velocities"].shape)
        print("  attrs     ", z["attrs"].shape, "(N, features)")
        print("  presence  ", z["presence"].shape)
        print("  collisions", z["collisions"].shape, "(K, 3)")
        print("  objects   ", list(z["obj_keys"]))


if __name__ == "__main__":
    main()
