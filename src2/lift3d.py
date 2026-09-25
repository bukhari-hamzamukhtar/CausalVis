"""
lift3d.py  —  CausalVis: 2D masks -> metric 3D positions on the table plane
===========================================================================

WHAT THIS FIXES
---------------
Everything upstream of the world model works in PIXEL space: positions are
mask centroids divided by the frame size, and "radius" is sqrt(pixel_area/pi).
That is a perspective projection of a 3D scene, so it lies in three ways that
matter to a physics model:

  1. The same object has a SMALLER radius when it is further from the camera.
     The world model is handed an object whose size changes as it moves.
  2. An object moving at constant speed in the world appears to SLOW DOWN as
     it recedes. The model is asked to learn Newtonian motion from data where
     free objects are visibly accelerating.
  3. Two objects touch at a pixel distance that depends on where on the table
     they happen to meet. Contact distance is not a constant.

Point 3 is measurable in the existing data: contact = r_i + r_j ranges from
0.087 to 0.148 (p5-p95) in trajectories_v2. Some of that is genuine object
size variation; the rest is perspective, and the rest is pure noise injected
into the one quantity the collision model depends on most.

THE GEOMETRY (why this needs no neural network and no GPU)
----------------------------------------------------------
CLEVRER's camera never moves, and every object rests on the same flat table.
That makes the lift a closed-form ray-plane intersection, not an inference
problem:

    pixel (u,v)  ->  ray from the camera through that pixel
                 ->  intersect the plane z = h
                 ->  metric (X, Y)

For an object whose centre sits at height h above the table, back-projecting
its mask CENTROID onto the plane z = h gives its metric position. h depends on
the object's radius, and the radius estimate depends on depth, so the two are
solved together by a short fixed-point iteration (3 passes is plenty).

Centroid, not lowest-pixel: the bottom of a sphere's silhouette is a tangent
point, not the contact point, and the extreme pixel is noisy under occlusion
and antialiasing. The centroid is robust and the height correction is exact
for it.

HONEST STATUS OF THE CAMERA NUMBERS
-----------------------------------
CAMERA_DEFAULT below holds the published CLEVR base-scene camera. TREAT IT AS
A STARTING ESTIMATE, NOT AS FACT. It is not in your data and I could not
verify it against this dataset. That is exactly why --calibrate exists: it
refines the camera against the data using three self-consistency criteria that
need no ground truth at all, and --report prints them before and after so you
can see whether the lift actually helped rather than taking it on faith.

If the report does not improve, the camera is wrong and the lift is worthless.
Check the report before using the output for anything.

THE THREE SELF-CONSISTENCY TESTS (no 3D ground truth required)
--------------------------------------------------------------
  SIZE   an object's metric radius must not change as it moves around the
         table. Reported as median coefficient of variation per object.
  SPEED  between two collisions an object travels in a straight line at
         constant speed. Reported as median CV of per-frame speed within
         collision-free intervals. This is the sharpest test: a straight 3D
         line still looks straight in 2D, so straightness proves nothing, but
         constant 3D speed does NOT look constant in 2D.
  TOUCH  at an annotated collision frame the two objects are touching, so
         their separation should equal r_i + r_j exactly. Reported as the
         spread of separation / (r_i + r_j) about 1.0.

All three should improve markedly in 3D. SPEED and TOUCH are the ones to
believe; SIZE can be gamed by a camera that pushes everything to infinity.

RUN
---
    # 1. look at the numbers on a sample, with the default camera
    python src2/lift3d.py --report --limit 300

    # 2. refine the camera against the data, then look again
    python src2/lift3d.py --calibrate --limit 300 --save-camera camera3d.json

    # 3. build the lifted dataset (same npz schema as trajectories_v2)
    python src2/lift3d.py --build --camera camera3d.json \
        --output data/trajectories_3d

Then train on it exactly as before:
    python src2/dynamics.py --data data/trajectories_3d --split split3d.json ...
(make a fresh split.json for the new folder; the clip names are unchanged so
 the existing split.json works if you point data_dir at the new folder.)
"""

import os
import glob
import json
import math
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# constants shared with build_trajectories.py -- keep these IDENTICAL
# ---------------------------------------------------------------------------
FRAME_H, FRAME_W = 320, 480

MASS_PROXY  = {"metal": 1.5, "rubber": 1.0}
RESTITUTION = {"metal": 0.8, "rubber": 0.5}
SHAPES    = ["cube", "sphere", "cylinder"]
COLORS    = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
MATERIALS = ["metal", "rubber"]

VEL_HALF_WINDOW = 2
MAX_GAP_FILL    = 5

# Published CLEVR base-scene camera. UNVERIFIED against this dataset -- see the
# module docstring. Blender convention: the camera looks down its own -Z axis.
CAMERA_DEFAULT = {
    "loc":        [7.4811, -6.5076, 5.3437],   # camera position, world metres
    "target":     [0.0, 0.0, 0.0],             # it points at the origin
    "lens_mm":    35.0,                        # Blender default lens
    "sensor_mm":  32.0,                        # Blender default sensor width
    "width":      FRAME_W,
    "height":     FRAME_H,
    # sqrt(silhouette_area / pi) is the radius of a CIRCLE, which is only the
    # right answer for a sphere. A cube of the same size covers far more
    # pixels, so without a per-shape factor every cube is handed to the physics
    # engine as an oversized object and its contact distance is wrong. Measured
    # on 25 clips with the default camera: sphere 0.483, cube 0.622, cylinder
    # 0.578, where CLEVR's two real size classes are 0.35 and 0.70. These are
    # first estimates -- --calibrate fits them against collision geometry,
    # which is the signal that actually constrains r_i + r_j.
    "shape_factor": {"sphere": 1.00, "cube": 0.85, "cylinder": 0.90},
    # Rays that graze the table (objects near the horizon) turn small pixel
    # errors into enormous position errors. Anything landing outside the table
    # is a degenerate ray, not a real observation, and is dropped.
    "max_extent": 9.0,
}

# One fixed scale for the WHOLE dataset, applied after the lift, so that metric
# coordinates land in roughly the same numeric range the pixel pipeline used
# (~0..1). It must be a constant: normalising per clip would destroy the
# cross-clip consistency that is the entire point of doing this.
WORLD_HALF_EXTENT = 6.0        # table half-width in world units -> X/12 + 0.5


# ---------------------------------------------------------------------------
#                     COCO RLE decoding, in pure numpy
# ---------------------------------------------------------------------------
# pycocotools is not installed here and its Windows wheels are unreliable on
# new Pythons, so the format is implemented directly. This is the exact
# algorithm from cocoapi's rleFrString(): each run length is written in 5-bit
# groups, bit 0x20 continues the value, bit 0x10 of the final group sign-
# extends it, and from the third run onward the value is a DELTA against the
# run two positions back. Runs alternate background/foreground starting with
# background, laid out in column-major (Fortran) order.

def rle_decode(counts, size):
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    h, w = int(size[0]), int(size[1])
    cnts, p, n = [], 0, len(counts)
    while p < n:
        x, k, more = 0, 0, True
        while more:
            c = ord(counts[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[len(cnts) - 2]
        cnts.append(x)
    flat = np.zeros(h * w, dtype=np.uint8)
    pos, val = 0, 0
    for run in cnts:
        run = int(run)
        if run < 0:
            raise ValueError("negative run length in RLE")
        if val:
            flat[pos:pos + run] = 1
        pos += run
        val ^= 1
        if pos > h * w:
            raise ValueError("RLE overruns the declared mask size")
    return flat.reshape((h, w), order="F")


def mask_stats(rle):
    """-> (centroid_u, centroid_v, pixel_area) or None for an empty mask."""
    m = rle_decode(rle["counts"], rle["size"])
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean()), float(xs.size)


# ---------------------------------------------------------------------------
#                              camera model
# ---------------------------------------------------------------------------
class Camera:
    """Pinhole camera with a Blender-style look-at pose."""

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        loc = np.asarray(cfg["loc"], dtype=np.float64)
        target = np.asarray(cfg["target"], dtype=np.float64)
        W, H = float(cfg["width"]), float(cfg["height"])

        # look-at: camera looks along -Z_cam, +Y_cam is up in the image
        fwd = target - loc
        fwd /= np.linalg.norm(fwd)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)

        self.loc = loc
        self.R = np.stack([right, up, -fwd], axis=1)   # camera -> world
        # Blender fits the sensor to the WIDTH by default
        self.fx = self.fy = cfg["lens_mm"] / cfg["sensor_mm"] * W
        self.cx, self.cy = W / 2.0, H / 2.0
        self.shape_factor = dict(cfg.get("shape_factor",
                                         CAMERA_DEFAULT["shape_factor"]))
        self.max_extent = float(cfg.get("max_extent", CAMERA_DEFAULT["max_extent"]))

    def rays(self, u, v):
        """Pixel arrays -> unit ray directions in world coordinates [...,3]."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        x = (u - self.cx) / self.fx
        y = -(v - self.cy) / self.fy        # image v grows downward
        d_cam = np.stack([x, y, -np.ones_like(x)], axis=-1)
        d = d_cam @ self.R.T
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def intersect_plane(self, u, v, h):
        """Back-project pixels onto the horizontal plane z = h.

        Returns (X, Y, depth). depth is the distance from the camera to the
        intersection, which is what converts an apparent pixel size into a
        metric size. Rays parallel to or pointing away from the plane give NaN
        rather than a silently wrong huge coordinate."""
        d = self.rays(u, v)
        h = np.asarray(h, dtype=np.float64)
        denom = d[..., 2]
        t = np.where(np.abs(denom) < 1e-9, np.nan, (h - self.loc[2]) / denom)
        t = np.where(t > 0, t, np.nan)               # must be in FRONT of the camera
        p = self.loc + t[..., None] * d
        return p[..., 0], p[..., 1], t

    def to_json(self):
        return dict(self.cfg)


def lift_object(cam, u, v, pixel_area, shape=None, iters=3):
    """Metric (X, Y) and metric radius for one detection.

    The object's centre sits at height h above the table and h equals its
    radius, but the radius can only be recovered once the depth is known, and
    the depth depends on h. Fixed-point iteration: start on the table, then
    re-solve. It converges in two or three passes because h is small next to
    the camera distance.
    """
    r_pix = math.sqrt(max(pixel_area, 1.0) / math.pi)
    r_pix *= cam.shape_factor.get(shape, 1.0)     # a cube is not a circle
    h = 0.0
    X = Y = depth = np.nan
    for _ in range(iters):
        X, Y, depth = cam.intersect_plane(u, v, h)
        if not np.isfinite(depth):
            return np.nan, np.nan, np.nan
        # apparent size -> metric size: r_metric = r_pixels * depth / focal
        r_metric = r_pix * float(depth) / cam.fx
        h = r_metric
    # a grazing ray puts the object off the table; that is a degenerate
    # observation, not a measurement, so drop it rather than pass on a wild
    # coordinate that would dominate every average downstream
    if abs(X) > cam.max_extent or abs(Y) > cam.max_extent:
        return np.nan, np.nan, np.nan
    return float(X), float(Y), float(h)


# ---------------------------------------------------------------------------
#                    reading one proposals file into tracks
# ---------------------------------------------------------------------------
def object_key(o):
    return f"{o['color']}_{o['material']}_{o['shape']}"


def read_clip(path):
    """Parse one processed_proposals json into per-frame detections.

    Object identity resolution mirrors build_trajectories.py: match on
    (color, material, shape), then (color, shape), then (shape), keeping the
    highest-scoring detection when several claim the same slot. The detector is
    known to flip `material` between frames, so a strict match silently drops
    frames."""
    with open(path, "r", encoding="utf-8") as fh:
        z = json.load(fh)
    gt = z["ground_truth"]
    objs = gt["objects"]
    N = len(objs)
    keys_exact = {(o["color"], o["material"], o["shape"]): o["id"] for o in objs}
    keys_cs = {}
    keys_s = {}
    for o in objs:
        keys_cs.setdefault((o["color"], o["shape"]), []).append(o["id"])
        keys_s.setdefault(o["shape"], []).append(o["id"])

    frames = z["frames"]
    T = len(frames)
    det = [[None] * N for _ in range(T)]        # (u, v, area, score)
    for f in frames:
        t = int(f["frame_index"])
        if not (0 <= t < T):
            continue
        for d in f.get("objects", []):
            oid = keys_exact.get((d.get("color"), d.get("material"), d.get("shape")))
            if oid is None:
                cand = keys_cs.get((d.get("color"), d.get("shape")), [])
                if len(cand) == 1:
                    oid = cand[0]
            if oid is None:
                cand = keys_s.get(d.get("shape"), [])
                if len(cand) == 1:
                    oid = cand[0]
            if oid is None:
                continue
            st = mask_stats(d["mask"])
            if st is None:
                continue
            u, v, area = st
            score = float(d.get("score", 1.0))
            prev = det[t][oid]
            if prev is None or score > prev[3]:
                det[t][oid] = (u, v, area, score)

    # in/out windows
    inside = np.zeros((T, N), dtype=bool)
    for oid in range(N):
        t_in, t_out = 0, T
        for e in gt.get("in_outs", []):
            if e.get("object") == oid:
                if e.get("type") == "in":
                    t_in = max(t_in, int(e["frame"]))
                elif e.get("type") == "out":
                    t_out = min(t_out, int(e["frame"]))
        inside[t_in:t_out, oid] = True

    collisions = [(int(c["frame"]), int(c["object"][0]), int(c["object"][1]))
                  for c in gt.get("collisions", []) if len(c.get("object", [])) == 2]
    exits = [(int(e["frame"]), int(e["object"]))
             for e in gt.get("in_outs", []) if e.get("type") == "out"]

    return {
        "video_name": z.get("video_name", os.path.basename(path).split(".")[0]),
        "objects": objs, "det": det, "inside": inside,
        "collisions": collisions, "exits": exits, "T": T, "N": N,
    }


def build_tracks(clip, cam, mode):
    """Detections -> positions/radii arrays.

    mode '2d' reproduces the existing pixel pipeline (normalised centroid,
    radius from sqrt(area/pi)); mode '3d' does the metric lift. Same code path
    otherwise, so the report compares like with like."""
    T, N = clip["T"], clip["N"]
    pos = np.full((T, N, 2), np.nan, np.float64)
    rad = np.full((T, N), np.nan, np.float64)
    shapes = [o.get("shape") for o in clip["objects"]]
    for t in range(T):
        for k in range(N):
            d = clip["det"][t][k]
            if d is None or not clip["inside"][t, k]:
                continue
            u, v, area, _ = d
            if mode == "2d":
                pos[t, k] = (u / FRAME_W, v / FRAME_H)
                rad[t, k] = math.sqrt(area / math.pi) / FRAME_W
            else:
                X, Y, r = lift_object(cam, u, v, area, shapes[k])
                if not (np.isfinite(X) and np.isfinite(Y) and np.isfinite(r)):
                    continue
                pos[t, k] = (X, Y)
                rad[t, k] = r
    return pos, rad


# ---------------------------------------------------------------------------
#            the three self-consistency tests (no ground truth needed)
# ---------------------------------------------------------------------------
def _cv(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 3:
        return np.nan
    mu = a.mean()
    return np.nan if abs(mu) < 1e-12 else float(a.std() / abs(mu))


def size_consistency(pos, rad):
    """An object's radius must not change as it moves. Lower is better."""
    out = []
    for k in range(rad.shape[1]):
        c = _cv(rad[:, k])
        if np.isfinite(c):
            out.append(c)
    return out


def speed_consistency(pos, collisions, exits, T):
    """Between collisions, position must be a LINEAR function of time.

    An earlier version of this took the coefficient of variation of per-frame
    step lengths. That was useless: objects move about 1.2 pixels per frame and
    mask-centroid noise is around a pixel, so the statistic measured detector
    jitter and nothing else (it read ~0.35 in both 2D and 3D). Worse, the lift
    multiplies that jitter by depth/focal, so 3D scored WORSE purely from noise
    amplification.

    Fitting a straight line in TIME fixes both problems. It averages the jitter
    down over the whole segment instead of differencing it up, and it is the
    correct discriminator: a constant-velocity 3D path projects to a path that
    is still straight in space but is traversed at a VARYING rate, because an
    object covers fewer pixels per frame as it recedes. So residual-from-linear
    -in-time is large in 2D and should be small in 3D. Straightness alone
    proves nothing; linearity in time is the whole test.

    Reported as RMS residual divided by the distance travelled, so it is
    dimensionless and comparable between pixel and metric units.
    """
    N = pos.shape[1]
    events = {k: [] for k in range(N)}
    for (f, i, j) in collisions:
        events[i].append(f); events[j].append(f)
    for (f, i) in exits:
        if i in events:
            events[i].append(f)
    out = []
    for k in range(N):
        bounds = sorted(set([0] + events[k] + [T]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            # stay clear of the collision frames themselves
            a2, b2 = a + 3, b - 3
            if b2 - a2 < 8:
                continue
            seg = pos[a2:b2, k]
            good = np.isfinite(seg).all(axis=1)
            if good.sum() < 8:
                continue
            ts = np.arange(a2, b2, dtype=np.float64)[good]
            seg = seg[good]
            span = float(np.linalg.norm(seg[-1] - seg[0]))
            if span < 1e-6:
                continue                      # stationary: nothing to test
            A = np.stack([np.ones_like(ts), ts], axis=1)
            resid = 0.0
            for c in range(2):
                coef, *_ = np.linalg.lstsq(A, seg[:, c], rcond=None)
                resid += float(((seg[:, c] - A @ coef) ** 2).sum())
            rms = math.sqrt(resid / len(ts))
            out.append(rms / span)
    return out


def touch_consistency(pos, rad, collisions):
    """At a collision the objects touch, so separation / (r_i + r_j) == 1."""
    out = []
    for (f, i, j) in collisions:
        if not (0 <= f < pos.shape[0]):
            continue
        pi, pj = pos[f, i], pos[f, j]
        ri, rj = rad[f, i], rad[f, j]
        if not (np.isfinite(pi).all() and np.isfinite(pj).all()
                and np.isfinite(ri) and np.isfinite(rj)):
            continue
        contact = ri + rj
        if contact < 1e-9:
            continue
        out.append(float(np.linalg.norm(pi - pj) / contact))
    return out


def evaluate(clips, cam, mode):
    S, V, Tt = [], [], []
    for clip in clips:
        pos, rad = build_tracks(clip, cam, mode)
        S += size_consistency(pos, rad)
        V += speed_consistency(pos, clip["collisions"], clip["exits"], clip["T"])
        Tt += touch_consistency(pos, rad, clip["collisions"])
    Tt = np.asarray(Tt, dtype=np.float64)
    return {
        "size_cv":   float(np.median(S)) if S else np.nan,
        "speed_cv":  float(np.median(V)) if V else np.nan,
        "touch_med": float(np.median(Tt)) if Tt.size else np.nan,
        "touch_iqr": float(np.subtract(*np.percentile(Tt, [75, 25]))) if Tt.size else np.nan,
        "n": (len(S), len(V), int(Tt.size)),
    }


def objective(stats):
    """One number for the calibrator to minimise.

    Deliberately weights SPEED and TOUCH above SIZE. Size constancy alone is
    gameable: a camera that pushes the scene towards infinity flattens every
    apparent size difference and scores well while being badly wrong. Speed
    constancy and a touch ratio pinned at 1.0 are not gameable that way.
    """
    s, v = stats["size_cv"], stats["speed_cv"]
    tm, ti = stats["touch_med"], stats["touch_iqr"]
    if not all(np.isfinite(x) for x in (s, v, tm, ti)):
        return 1e9
    return 1.0 * s + 3.0 * v + 3.0 * abs(tm - 1.0) + 3.0 * ti


def report(before, after=None):
    def row(name, st):
        return (f"   {name:<22} size_cv {st['size_cv']:.4f}   "
                f"linres  {st["speed_cv"]:.4f}   "
                f"touch {st['touch_med']:.3f} (iqr {st['touch_iqr']:.3f})   "
                f"J {objective(st):.4f}")
    print("\n" + "=" * 78)
    print("SELF-CONSISTENCY  (lower size_cv / speed_cv is better; touch -> 1.000)")
    print("=" * 78)
    print(row("2D pixel pipeline", before["2d"]))
    print(row("3D lift (default cam)", before["3d"]))
    if after is not None:
        print(row("3D lift (calibrated)", after))
    print(f"\n   samples: sizes/speeds/collisions = {before['2d']['n']}")
    b, a = before["2d"], (after if after is not None else before["3d"])
    if not np.isfinite(objective(a)) or objective(a) >= objective(b):
        print("\n   VERDICT: the lift did NOT beat the 2D pipeline. The camera is")
        print("            wrong, or the geometry assumption does not hold here.")
        print("            Do not use the lifted data until this improves.")
    else:
        print(f"\n   VERDICT: speed_cv {b['speed_cv']:.4f} -> {a['speed_cv']:.4f}, "
              f"touch {b['touch_med']:.3f}+-{b['touch_iqr']:.3f} -> "
              f"{a['touch_med']:.3f}+-{a['touch_iqr']:.3f}.")
        print("            The lift is doing real work.")
    print("=" * 78)


# ---------------------------------------------------------------------------
#                              calibration
# ---------------------------------------------------------------------------
def calibrate(clips, cam0, rounds=3, verbose=True):
    """Coordinate descent over camera location and focal length.

    Plain local search on purpose: scipy is not a dependency here, the space is
    4-dimensional, and the objective is cheap. It refines a good starting
    guess; it will not rescue a wildly wrong one.
    """
    cfg = dict(cam0.cfg)
    cfg.setdefault("shape_factor", dict(CAMERA_DEFAULT["shape_factor"]))
    cfg["shape_factor"] = dict(cfg["shape_factor"])
    best = objective(evaluate(clips, Camera(cfg), "3d"))
    if verbose:
        print(f"   start J = {best:.4f}   loc={np.round(cfg['loc'], 3).tolist()} "
              f"lens={cfg['lens_mm']:.2f}")
    knobs = ["loc0", "loc1", "loc2", "lens_mm",
             "sf_sphere", "sf_cube", "sf_cylinder"]
    steps = {"loc0": 0.8, "loc1": 0.8, "loc2": 0.8, "lens_mm": 3.0,
             "sf_sphere": 0.10, "sf_cube": 0.10, "sf_cylinder": 0.10}
    for rnd in range(rounds):
        improved = False
        for knob in knobs:
            step = steps[knob]
            while step > 0.02:
                moved = False
                for sign in (+1.0, -1.0):
                    trial = dict(cfg)
                    trial["loc"] = list(cfg["loc"])
                    trial["shape_factor"] = dict(cfg["shape_factor"])
                    if knob.startswith("loc"):
                        trial["loc"][int(knob[-1])] += sign * step
                    elif knob == "lens_mm":
                        trial["lens_mm"] = cfg["lens_mm"] + sign * step
                        if trial["lens_mm"] <= 1.0:
                            continue
                    else:
                        s = knob[3:]
                        trial["shape_factor"][s] = cfg["shape_factor"][s] + sign * step
                        if trial["shape_factor"][s] <= 0.05:
                            continue
                    j = objective(evaluate(clips, Camera(trial), "3d"))
                    if j < best - 1e-6:
                        best, cfg, moved, improved = j, trial, True, True
                        break
                if not moved:
                    step *= 0.5
            steps[knob] = max(0.02, steps[knob] * 0.5)
        if verbose:
            sf = {k: round(v, 3) for k, v in cfg["shape_factor"].items()}
            print(f"   round {rnd}: J = {best:.4f}   "
                  f"loc={np.round(cfg['loc'], 3).tolist()} "
                  f"lens={cfg['lens_mm']:.2f} shape={sf}")
        if not improved:
            break
    return Camera(cfg), best


# ---------------------------------------------------------------------------
#              emit npz in the trajectories_v2 schema
# ---------------------------------------------------------------------------
def lsq_velocity(pos, present, half=VEL_HALF_WINDOW):
    """Least-squares slope over a small window. Identical in spirit to
    build_trajectories.py -- this MUST match between training and inference."""
    T, N, _ = pos.shape
    vel = np.zeros_like(pos)
    for k in range(N):
        for t in range(T):
            a, b = max(0, t - half), min(T, t + half + 1)
            idx = [s for s in range(a, b) if present[s, k]]
            if len(idx) < 2:
                continue
            ts = np.asarray(idx, dtype=np.float64)
            ts = ts - ts.mean()
            denom = (ts ** 2).sum()
            if denom < 1e-12:
                continue
            for c in range(2):
                ys = pos[idx, k, c]
                vel[t, k, c] = float((ts * (ys - ys.mean())).sum() / denom)
    return vel


def fill_short_gaps(pos, rad, inside, max_gap=MAX_GAP_FILL):
    """Interpolate occlusion gaps only. A long gap means the object really is
    not there; filling it would invent a stationary object, which is the bug
    build_trajectories.py documents."""
    T, N, _ = pos.shape
    present = np.isfinite(pos).all(axis=2) & inside
    for k in range(N):
        t = 0
        while t < T:
            if present[t, k]:
                t += 1
                continue
            a = t
            while t < T and not present[t, k]:
                t += 1
            b = t
            if a > 0 and b < T and present[a - 1, k] and present[b, k] \
                    and (b - a) <= max_gap:
                for s in range(a, b):
                    w = (s - (a - 1)) / float(b - (a - 1))
                    pos[s, k] = (1 - w) * pos[a - 1, k] + w * pos[b, k]
                    rad[s, k] = (1 - w) * rad[a - 1, k] + w * rad[b, k]
                    present[s, k] = True
    return pos, rad, present


def make_attrs(objs, rad, present, normalise):
    """attrs [N,16] with the SAME column layout build_trajectories.py emits.
    Column 15 is the measured radius -- now metric, not pixels."""
    N = len(objs)
    at = np.zeros((N, 16), np.float32)
    for k, o in enumerate(objs):
        at[k, 0] = MASS_PROXY.get(o["material"], 1.0)
        at[k, 1] = RESTITUTION.get(o["material"], 0.5)
        if o["shape"] in SHAPES:
            at[k, 2 + SHAPES.index(o["shape"])] = 1.0
        if o["color"] in COLORS:
            at[k, 5 + COLORS.index(o["color"])] = 1.0
        if o["material"] in MATERIALS:
            at[k, 13 + MATERIALS.index(o["material"])] = 1.0
        col = rad[present[:, k], k] if present[:, k].any() else np.array([])
        col = col[np.isfinite(col)]
        # median over the track: one radius per object, robust to bad frames
        at[k, 15] = float(np.median(col)) / normalise if col.size else 0.0
    return at


def build_dataset(files, cam, out_dir, normalise, verbose_every=200):
    os.makedirs(out_dir, exist_ok=True)
    written = skipped = 0
    for n, path in enumerate(files):
        try:
            clip = read_clip(path)
            pos, rad = build_tracks(clip, cam, "3d")
            pos, rad, present = fill_short_gaps(pos, rad, clip["inside"])
            if present.sum() == 0:
                skipped += 1
                continue
            posn = pos / normalise
            posn[~present] = 0.0
            vel = lsq_velocity(posn, present)
            at = make_attrs(clip["objects"], rad, present, normalise)
            np.savez_compressed(
                os.path.join(out_dir, f"{clip['video_name']}.npz"),
                positions=posn.astype(np.float32),
                velocities=vel.astype(np.float32),
                presence=present.astype(np.float32),
                attrs=at,
                collisions=np.asarray(clip["collisions"], np.int64).reshape(-1, 3),
                exits=np.asarray(clip["exits"], np.int64).reshape(-1, 2),
                obj_keys=np.asarray([object_key(o) for o in clip["objects"]]),
                video_name=clip["video_name"],
            )
            written += 1
        except Exception as ex:                     # one bad clip must not stop 20k
            skipped += 1
            if skipped <= 5:
                print(f"   skipped {os.path.basename(path)}: {ex}")
        if verbose_every and (n + 1) % verbose_every == 0:
            print(f"   {n + 1}/{len(files)}  written {written}  skipped {skipped}",
                  flush=True)
    print(f"\nwrote {written} clips to {out_dir}  (skipped {skipped})")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="legacy/v1/data/processed_proposals")
    ap.add_argument("--output", default="data/trajectories_3d")
    ap.add_argument("--camera", default=None, help="camera json from --save-camera")
    ap.add_argument("--save-camera", default=None)
    ap.add_argument("--limit", type=int, default=300,
                    help="clips used for --report / --calibrate")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--normalise", type=float, default=2 * WORLD_HALF_EXTENT,
                    help="one fixed divisor for the whole dataset")
    a = ap.parse_args()

    if not (a.report or a.calibrate or a.build):
        ap.error("pick at least one of --report / --calibrate / --build")

    files = sorted(glob.glob(os.path.join(a.input, "*.json")))
    if not files:
        raise SystemExit(f"no .json proposals found in {a.input}")
    print(f"proposals: {len(files)} clips in {a.input}")

    cfg = CAMERA_DEFAULT
    if a.camera:
        with open(a.camera, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        print(f"camera loaded from {a.camera}")
    cam = Camera(cfg)

    cam_final = cam
    if a.report or a.calibrate:
        sample = files[:a.limit]
        print(f"parsing {len(sample)} clips for the consistency report...")
        clips = []
        for p in sample:
            try:
                clips.append(read_clip(p))
            except Exception as ex:
                print(f"   skipped {os.path.basename(p)}: {ex}")
        print(f"parsed {len(clips)} clips")
        before = {"2d": evaluate(clips, cam, "2d"),
                  "3d": evaluate(clips, cam, "3d")}
        after = None
        if a.calibrate:
            print("\ncalibrating the camera against the data...")
            cam_final, _ = calibrate(clips, cam)
            after = evaluate(clips, cam_final, "3d")
        report(before, after)
        if a.save_camera:
            with open(a.save_camera, "w", encoding="utf-8") as fh:
                json.dump(cam_final.to_json(), fh, indent=2)
            print(f"\ncamera written to {a.save_camera}")

    if a.build:
        print(f"\nbuilding lifted dataset -> {a.output}")
        print(f"   scale divisor {a.normalise:g} (one constant for every clip)")
        build_dataset(files, cam_final, a.output, a.normalise)


if __name__ == "__main__":
    main()
