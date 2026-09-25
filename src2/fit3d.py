"""
fit3d.py  —  CausalVis: recover exact 3D scenes by inverting the renderer
=========================================================================

THE IDEA
--------
CLEVRER is synthetic. Every object is one of three known shapes (cube, sphere,
cylinder), at ONE known size (the ground truth carries no size attribute, and
the 3x spread in silhouette area is entirely depth), resting on one flat table.

So recovering 3D is not a depth-estimation problem. It is a POSE FITTING
problem against a known CAD library, and the pose has almost no freedom left:

    sphere    2 numbers  (X, Y on the table)
    cylinder  2 numbers  (X, Y; spin about its own axis is invisible)
    cube      3 numbers  (X, Y, yaw)

That is why this can be exact where monocular depth is fuzzy. We render a
candidate silhouette and compare it to the observed mask; the pose that
maximises intersection-over-union IS the answer. No network, no training, no
GPU, and the result is a real mesh at a real pose -- so the scene can be
rendered from any viewpoint, which is the only honest way to check it.

WHY SILHOUETTES ARE CHEAP HERE
------------------------------
All three shapes are CONVEX. The silhouette of a convex body is exactly the
convex hull of its projected vertices, so a full rasteriser is unnecessary:
project the vertices, take the 2D hull, fill it. That is a few hundred
floating point operations per object per frame.

WHAT THIS FIXES
---------------
The pixel pipeline hands the world model a radius that shrinks with distance
and a position that is a perspective projection. Measured on this data,
per-object radius varies by 9.5% as an object moves around the table, and
contact distance r_i+r_j spans 0.087 to 0.148. Both are geometry errors, and
both feed straight into the collision behaviour the model is graded on.

ORDER OF OPERATIONS
-------------------
    # 1. recover the camera + the per-shape object size (do this ONCE)
    python src2/fit3d.py --recover-camera --limit 30 --save camera_fit.json

    # 2. look at how well the fitted silhouettes match the real masks
    python src2/fit3d.py --check --camera camera_fit.json --limit 20

Step 1 must reach a high median IoU. If it does not, the geometry assumption
is wrong and nothing downstream is trustworthy -- stop and say so rather than
building on it.
"""

import os
import sys
import glob
import json
import math
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lift3d import read_clip, rle_decode, Camera, CAMERA_DEFAULT, FRAME_W, FRAME_H


# ---------------------------------------------------------------------------
#                        the known object library
# ---------------------------------------------------------------------------
# CLEVR/CLEVRER primitives, expressed in the object's own frame with the BASE
# on z = 0 (objects rest on the table). `s` is the one free size scalar per
# shape, recovered in step 1 rather than assumed.

def mesh_cube(s, n_unused=None):
    """Axis-aligned cube of half-side s, sitting on the table."""
    c = [(x, y, z) for x in (-s, s) for y in (-s, s) for z in (0.0, 2 * s)]
    return np.asarray(c, dtype=np.float64)


def mesh_cylinder(s, n=24):
    """Upright cylinder, radius s, height 2s."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([s * np.cos(a), s * np.sin(a)], axis=1)
    bot = np.concatenate([ring, np.zeros((n, 1))], axis=1)
    top = np.concatenate([ring, np.full((n, 1), 2 * s)], axis=1)
    return np.concatenate([bot, top], axis=0)


def mesh_sphere(s, n=80):
    """Sphere of radius s resting on the table, so its centre is at z = s.

    A Fibonacci sphere gives an even point spread, and the convex hull of the
    projected points is the silhouette to within the sampling density."""
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    p = np.stack([np.cos(theta) * np.sin(phi),
                  np.sin(theta) * np.sin(phi),
                  np.cos(phi)], axis=1) * s
    p[:, 2] += s
    return p


MESH = {"cube": mesh_cube, "cylinder": mesh_cylinder, "sphere": mesh_sphere}
YAW_DOF = {"cube": True, "cylinder": False, "sphere": False}


def place(verts, x, y, yaw=0.0):
    """Object frame -> world: rotate about the vertical axis, then translate."""
    if yaw:
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        verts = verts @ R.T
    out = verts.copy()
    out[:, 0] += x
    out[:, 1] += y
    return out


# ---------------------------------------------------------------------------
#                       projection and silhouette
# ---------------------------------------------------------------------------
def project(cam, pts_world):
    """World points -> pixel coordinates. Points behind the camera are dropped."""
    rel = pts_world - cam.loc
    cam_pts = rel @ cam.R                      # world -> camera (R is orthonormal)
    z = cam_pts[:, 2]
    front = z < -1e-6                          # camera looks down its own -Z
    if not front.any():
        return None
    cam_pts = cam_pts[front]
    d = -cam_pts[:, 2]
    u = cam.cx + cam.fx * cam_pts[:, 0] / d
    v = cam.cy - cam.fy * cam_pts[:, 1] / d
    return np.stack([u, v], axis=1)


def convex_hull(pts):
    """Andrew's monotone chain. Implemented here so scipy is not a dependency."""
    p = np.unique(np.round(pts, 6), axis=0)
    if len(p) < 3:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(points):
        out = []
        for q in points:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) <= 0:
                    out.pop()
                else:
                    break
            out.append(q)
        return out

    lower = half(p)
    upper = half(p[::-1])
    return np.asarray(lower[:-1] + upper[:-1])


def fill_hull(hull, H=FRAME_H, W=FRAME_W):
    """Rasterise a convex polygon by half-plane tests inside its bounding box.

    Convexity is what makes this valid and fast: a point is inside iff it is on
    the same side of every edge. Only the bbox is tested, so cost scales with
    the object, not the frame."""
    m = np.zeros((H, W), dtype=bool)
    if hull is None or len(hull) < 3:
        return m
    u0 = max(int(np.floor(hull[:, 0].min())), 0)
    u1 = min(int(np.ceil(hull[:, 0].max())) + 1, W)
    v0 = max(int(np.floor(hull[:, 1].min())), 0)
    v1 = min(int(np.ceil(hull[:, 1].max())) + 1, H)
    if u1 <= u0 or v1 <= v0:
        return m
    uu, vv = np.meshgrid(np.arange(u0, u1) + 0.5, np.arange(v0, v1) + 0.5)
    inside = np.ones(uu.shape, dtype=bool)
    n = len(hull)
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        cross = (b[0] - a[0]) * (vv - a[1]) - (b[1] - a[1]) * (uu - a[0])
        inside &= cross >= -1e-9
    m[v0:v1, u0:u1] = inside
    return m


def render_object(cam, shape, size, x, y, yaw=0.0):
    verts = place(MESH[shape](size), x, y, yaw)
    pts = project(cam, verts)
    if pts is None:
        return None
    return fill_hull(convex_hull(pts))


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


# ---------------------------------------------------------------------------
#                              pose fitting
# ---------------------------------------------------------------------------
def init_xy(cam, u, v, size):
    """Back-project the mask centroid onto the plane at the object's mid-height.
    Only a starting guess; the IoU search does the real work."""
    X, Y, _ = cam.intersect_plane(u, v, size)
    if not (np.isfinite(X) and np.isfinite(Y)):
        return 0.0, 0.0
    return float(X), float(Y)


def fit_pose(cam, shape, size, obs, u0, v0, iters=3, init=None):
    """Find (x, y, yaw) maximising silhouette IoU against the observed mask.

    Local pattern search: cheap, derivative-free, and the initial guess from
    back-projection is already close. Yaw is only searched for cubes -- a
    sphere has no orientation and an upright cylinder's spin is invisible, so
    searching it would be fitting noise."""
    # Warm start: consecutive frames move about 1.2 pixels, so the previous
    # frame's pose is already almost right. Starting there and taking a small
    # step turns a full search into a couple of refinements -- the difference
    # between minutes and seconds per clip.
    if init is not None:
        x, y, best_yaw = init
        step = 0.06
    else:
        x, y = init_xy(cam, u0, v0, size)
        best_yaw = 0.0
        if YAW_DOF[shape]:
            cand = np.linspace(0, math.pi / 2, 8, endpoint=False)  # 4-fold symmetry
            scored = []
            for yw in cand:
                r = render_object(cam, shape, size, x, y, yw)
                scored.append((iou(r, obs) if r is not None else 0.0, yw))
            best_yaw = max(scored)[1]
        step = 0.35
    r = render_object(cam, shape, size, x, y, best_yaw)
    best = iou(r, obs) if r is not None else 0.0
    for _ in range(iters):
        while step > 0.01:
            moved = False
            for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
                r = render_object(cam, shape, size, x + dx, y + dy, best_yaw)
                s = iou(r, obs) if r is not None else 0.0
                if s > best + 1e-6:
                    best, x, y, moved = s, x + dx, y + dy, True
                    break
            if not moved:
                step *= 0.5
        step = 0.08
        if YAW_DOF[shape]:
            for dw in (0.12, -0.12, 0.04, -0.04):
                r = render_object(cam, shape, size, x, y, best_yaw + dw)
                s = iou(r, obs) if r is not None else 0.0
                if s > best + 1e-6:
                    best, best_yaw = s, best_yaw + dw
    return {"x": x, "y": y, "yaw": best_yaw, "iou": best}


# ---------------------------------------------------------------------------
#                     observations pulled from the proposals
# ---------------------------------------------------------------------------
def gather(clips, per_clip=6, stride=16):
    """A flat list of (shape, observed_mask, centroid_u, centroid_v) samples."""
    obs = []
    for c in clips:
        taken = 0
        for t in range(0, c["T"], stride):
            for k in range(c["N"]):
                d = c["det"][t][k]
                if d is None or not c["inside"][t, k]:
                    continue
                shape = c["objects"][k].get("shape")
                if shape not in MESH:
                    continue
                m = c["raw"][t][k]
                if m is None or m.sum() < 200:
                    continue
                obs.append((shape, m, d[0], d[1]))
                taken += 1
            if taken >= per_clip:
                break
    return obs


def mean_iou(cam, sizes, obs, subsample=None):
    xs = obs if subsample is None else obs[:subsample]
    out = []
    for shape, m, u, v in xs:
        f = fit_pose(cam, shape, sizes[shape], m, u, v, iters=2)
        out.append(f["iou"])
    return float(np.median(out)) if out else 0.0


# ---------------------------------------------------------------------------
#                          camera + size recovery
# ---------------------------------------------------------------------------
def recover(obs, cam0, sizes0, rounds=3, verbose=True, save_path=None):
    """Coordinate descent over camera pose, focal length and the three object
    sizes, maximising median silhouette IoU.

    Derivative-free on purpose: IoU of a rasterised hull is piecewise constant,
    so gradients do not exist and a pattern search is the honest tool."""
    cfg = dict(cam0.cfg)
    cfg["loc"] = list(cfg["loc"])
    sizes = dict(sizes0)
    best = mean_iou(Camera(cfg), sizes, obs)
    if verbose:
        print(f"   start  median IoU {best:.4f}   loc={np.round(cfg['loc'],2).tolist()} "
              f"lens={cfg['lens_mm']:.1f} sizes={ {k: round(v,3) for k,v in sizes.items()} }")
    knobs = ["loc0", "loc1", "loc2", "lens", "s_cube", "s_sphere", "s_cylinder"]
    steps = {"loc0": 0.6, "loc1": 0.6, "loc2": 0.6, "lens": 2.0,
             "s_cube": 0.06, "s_sphere": 0.06, "s_cylinder": 0.06}
    for rnd in range(rounds):
        improved = False
        for knob in knobs:
            step = steps[knob]
            while step > (0.01 if knob.startswith("s_") else 0.03):
                moved = False
                for sign in (1.0, -1.0):
                    t_cfg = dict(cfg); t_cfg["loc"] = list(cfg["loc"])
                    t_sz = dict(sizes)
                    if knob.startswith("loc"):
                        t_cfg["loc"][int(knob[-1])] += sign * step
                    elif knob == "lens":
                        t_cfg["lens_mm"] = cfg["lens_mm"] + sign * step
                        if t_cfg["lens_mm"] <= 5.0:
                            continue
                    else:
                        s = knob[2:]
                        t_sz[s] = sizes[s] + sign * step
                        if t_sz[s] <= 0.05:
                            continue
                    v = mean_iou(Camera(t_cfg), t_sz, obs)
                    if v > best + 1e-5:
                        best, cfg, sizes, moved, improved = v, t_cfg, t_sz, True, True
                        break
                if not moved:
                    step *= 0.5
            steps[knob] = max(steps[knob] * 0.5,
                              0.01 if knob.startswith("s_") else 0.03)
        if verbose:
            print(f"   round {rnd}  median IoU {best:.4f}   "
                  f"loc={np.round(cfg['loc'],2).tolist()} lens={cfg['lens_mm']:.1f} "
                  f"sizes={ {k: round(v,3) for k,v in sizes.items()} }", flush=True)
        # checkpoint every round: this search takes tens of minutes and losing
        # it to a timeout once was enough
        if save_path:
            with open(save_path, "w", encoding="utf-8") as fh:
                json.dump({"camera": Camera(cfg).to_json(), "sizes": sizes,
                           "median_iou": best, "round": rnd}, fh, indent=2)
        if not improved:
            break
    return Camera(cfg), sizes, best


# ---------------------------------------------------------------------------
def load_one(p):
    """Parse one proposals file AND keep the decoded masks. read_clip only
    returns centroid summaries; the silhouette fitter needs the masks."""
    c = read_clip(p)
    with open(p, "r", encoding="utf-8") as fh:
        z = json.load(fh)
    raw = [[None] * c["N"] for _ in range(c["T"])]
    keys = {(o["color"], o["material"], o["shape"]): o["id"] for o in c["objects"]}
    for f in z["frames"]:
        t = int(f["frame_index"])
        if not (0 <= t < c["T"]):
            continue
        for d in f.get("objects", []):
            oid = keys.get((d.get("color"), d.get("material"), d.get("shape")))
            if oid is None:
                continue
            raw[t][oid] = rle_decode(d["mask"]["counts"], d["mask"]["size"]).astype(bool)
    c["raw"] = raw
    return c


def load_clips(input_dir, limit):
    files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    if not files:
        raise SystemExit(f"no proposals in {input_dir}")
    return [load_one(p) for p in files[:limit]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="legacy/v1/data/processed_proposals")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--save", default=None)
    ap.add_argument("--recover-camera", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    print(f"loading {a.limit} clips (decoding masks)...")
    clips = load_clips(a.input, a.limit)
    obs = gather(clips)
    print(f"observations: {len(obs)}  "
          f"({sum(1 for o in obs if o[0]=='sphere')} sphere / "
          f"{sum(1 for o in obs if o[0]=='cube')} cube / "
          f"{sum(1 for o in obs if o[0]=='cylinder')} cylinder)")

    cfg = CAMERA_DEFAULT
    sizes = {"cube": 0.35, "sphere": 0.35, "cylinder": 0.35}
    if a.camera:
        with open(a.camera, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        cfg, sizes = blob["camera"], blob["sizes"]
        print(f"camera loaded from {a.camera}")
    cam = Camera(cfg)

    if a.recover_camera:
        print("\nrecovering camera and object sizes by silhouette matching...")
        cam, sizes, best = recover(obs, cam, sizes, save_path=a.save)
        print(f"\nfinal median silhouette IoU: {best:.4f}")
        if best < 0.75:
            print("VERDICT: the fit is POOR. The geometry assumption or the mask")
            print("         quality does not support exact 3D recovery. Do not")
            print("         build on this until it improves.")
        else:
            print("VERDICT: silhouettes match. The recovered scene is trustworthy.")
        if a.save:
            with open(a.save, "w", encoding="utf-8") as fh:
                json.dump({"camera": cam.to_json(), "sizes": sizes}, fh, indent=2)
            print(f"saved -> {a.save}")

    if a.check:
        print("\nper-shape silhouette IoU with the current camera:")
        for shape in ("sphere", "cube", "cylinder"):
            sub = [o for o in obs if o[0] == shape][:40]
            if not sub:
                continue
            v = [fit_pose(cam, shape, sizes[shape], m, u, vv)["iou"]
                 for _, m, u, vv in sub]
            print(f"   {shape:9s} median {np.median(v):.4f}   "
                  f"p10 {np.percentile(v,10):.4f}   n={len(v)}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
#              yaw as a physical trajectory, not a per-frame guess
# ---------------------------------------------------------------------------
MAX_YAW_FRAMES = 10       # frames scored per collision-free segment


def fit_yaw_track(clip, cam, sizes, k, xy_track):
    """Recover cube orientation over a whole clip using a physics prior.

    Per-frame yaw fitting does not work here. A dense sweep of the full 90
    degrees moves silhouette IoU by only 0.09-0.19, so against imperfect
    detector masks the per-frame estimate is dominated by noise -- measured on
    sim_00000 it jumped 76.9 -> 72.3 -> 74.6 -> 56.2 -> 49.4 degrees, which no
    sliding cube does.

    The fix is to stop treating yaw as free. A rigid body on a table spins at
    CONSTANT angular velocity between collisions, so yaw(t) is piecewise
    linear with breakpoints only at annotated collisions. Fitting two numbers
    per segment (yaw0, omega) against the summed IoU of every frame in that
    segment averages the noise down instead of chasing it.

    Only cubes need this: a sphere has no visible orientation and an upright
    cylinder's spin is invisible, so their horizontal cross-section is a circle
    and r_i + r_j is already the exact contact distance. Cubes are the only
    shape where orientation changes contact geometry (by up to 41% for a
    cube-cube pair).

    Returns yaw per frame, or None where the object is not observed.
    """
    shape = clip["objects"][k].get("shape")
    if shape != "cube":
        return [0.0] * clip["T"]

    T = clip["T"]
    breaks = sorted({0, T} | {int(f) for (f, i, j) in clip["collisions"]
                              if i == k or j == k})
    size = sizes["cube"]
    yaw = [None] * T

    for a, b in zip(breaks[:-1], breaks[1:]):
        frames = [t for t in range(a, b)
                  if clip["raw"][t][k] is not None and clip["inside"][t, k]
                  and xy_track[t] is not None]
        if not frames:
            continue
        # Score a SUBSET of the segment, not every frame. The whole point of
        # the segment fit is to average detector noise out of a 2-parameter
        # estimate; 10 well-spread frames do that as well as 90 and cost a
        # tenth as much (11.6 s/clip -> ~2 s/clip over 20k clips).
        fit_frames = frames if len(frames) <= MAX_YAW_FRAMES else [
            frames[i] for i in np.linspace(0, len(frames) - 1,
                                           MAX_YAW_FRAMES).astype(int)]

        def score(y0, om):
            s = 0.0
            for t in fit_frames:
                x, y = xy_track[t]
                r = render_object(cam, "cube", size, x, y, y0 + om * (t - a))
                s += iou(r, clip["raw"][t][k]) if r is not None else 0.0
            return s / len(fit_frames)

        # coarse over the cube's 4-fold symmetry, then refine both parameters
        best = (-1.0, 0.0, 0.0)
        for y0 in np.linspace(0, math.pi / 2, 12, endpoint=False):
            v = score(y0, 0.0)
            if v > best[0]:
                best = (v, y0, 0.0)
        _, y0, om = best
        for step_y, step_o in ((0.10, 0.010), (0.04, 0.004), (0.015, 0.0015)):
            improved = True
            while improved:
                improved = False
                for dy, do in ((step_y, 0), (-step_y, 0), (0, step_o), (0, -step_o)):
                    v = score(y0 + dy, om + do)
                    if v > best[0] + 1e-5:
                        best = (v, y0 + dy, om + do)
                        y0, om = y0 + dy, om + do
                        improved = True
                        break
        for t in frames:
            yaw[t] = (y0 + om * (t - a)) % (math.pi / 2)
    return yaw


# Mean of s*(|cos a| + |sin a|) over all a. When orientation is UNKNOWN this
# is the honest answer -- assuming yaw=0 would assert a specific orientation we
# did not measure, which is worse than admitting ignorance.
CUBE_MEAN_SUPPORT = 4.0 / math.pi        # ~1.273


def support_radius(shape, s, yaw, phi):
    """How far the surface reaches from the centre in horizontal direction phi.

    Exact for these primitives. A sphere or upright cylinder has a circular
    cross-section, so this is just s regardless of direction -- which is why
    r_i + r_j is already exact for those pairs. A cube's square cross-section
    reaches s at a face and s*sqrt(2) at a corner.
    """
    if shape != "cube":
        return s
    if yaw is None:                       # orientation not recovered
        return s * CUBE_MEAN_SUPPORT
    a = phi - yaw
    return s * (abs(math.cos(a)) + abs(math.sin(a)))


def exact_contact(shape_i, s_i, yaw_i, shape_j, s_j, yaw_j, dx, dy):
    """True surface-to-surface contact distance along the centre line.

    Replaces the sphere approximation r_i + r_j, which understates contact by
    up to 19% for a cube-sphere pair and 41% for cube-cube.
    """
    phi = math.atan2(dy, dx)
    return (support_radius(shape_i, s_i, yaw_i, phi)
            + support_radius(shape_j, s_j, yaw_j, phi + math.pi))


# ---------------------------------------------------------------------------
#          per-object size recovery -- no size prior, no global constant
# ---------------------------------------------------------------------------
def fit_object_size(clip, cam, k, n_frames=8, lo=0.10, hi=0.80):
    """Recover ONE object's metric size from its own silhouettes.

    recover() finds a single size per SHAPE for the whole dataset. That is a
    valid shortcut for CLEVRER, which genuinely has one size class, but it is a
    dataset-specific assumption: hand the pipeline a video whose objects differ
    in size and it would apply the wrong geometry to every one of them.

    This derives size per object instead, from nothing but that object's masks
    and the camera. Size and position are coupled -- a bigger object sits with
    its centre higher, which moves where its centroid back-projects to -- so
    position is re-solved inside the search rather than held fixed.

    Golden-section over a unimodal IoU-vs-size curve: bigger than the truth
    overshoots the silhouette, smaller undershoots, so there is a single peak.
    """
    shape = clip["objects"][k].get("shape")
    if shape not in MESH:
        return None
    frames = [t for t in range(clip["T"])
              if clip["raw"][t][k] is not None and clip["inside"][t, k]]
    if len(frames) < 3:
        return None
    if len(frames) > n_frames:
        frames = [frames[i] for i in
                  np.linspace(0, len(frames) - 1, n_frames).astype(int)]

    def score(s):
        if s <= 0.02:
            return 0.0
        tot = 0.0
        for t in frames:
            d = clip["det"][t][k]
            x, y = init_xy(cam, d[0], d[1], s)   # position follows size
            r = render_object(cam, shape, s, x, y, 0.0)
            tot += iou(r, clip["raw"][t][k]) if r is not None else 0.0
        return tot / len(frames)

    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = score(c), score(d)
    for _ in range(24):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = score(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = score(d)
        if b - a < 1e-3:
            break
    s = (a + b) / 2
    return {"size": s, "iou": score(s), "shape": shape, "n_frames": len(frames)}
