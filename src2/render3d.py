"""
render3d.py  —  CausalVis: turn a 2D CLEVRER clip into a real 3D scene you can
                fly a camera around
=============================================================================

WHY THIS IS THE VALIDATION, NOT JUST A DEMO
-------------------------------------------
`fit3d.py` recovers a pose for every object in every frame by matching
rendered silhouettes against the observed masks (median IoU 0.95). That number
is convincing but abstract. This script makes it checkable by eye: it renders
the recovered scene from viewpoints the original video never had. If the
geometry were wrong, objects would float, sink through the table, interpenetrate,
or jitter as the camera moves. If it is right, the scene holds together from
every angle.

Nothing here is generated or hallucinated. Every frame is COMPUTED from the
recovered poses, so a novel view is exactly as accurate as the fit that
produced it -- which is the whole reason this beats a video model for the job.

WHAT IT DRAWS
-------------
Real geometry with shaded faces, not flat silhouettes. That matters: a
silhouette of a cube looks the same at any yaw, so a silhouette render would
hide the one quantity the current 2D pipeline cannot measure at all --
ORIENTATION. Shaded faces make recovered yaw visible, which is the point.

Rendering is a painter's algorithm: back-face cull, sort remaining faces far
to near, fill each as a convex polygon with Lambertian shading. No GPU, no
graphics library, a few thousand polygon fills per frame.

RUN
---
    # reconstruct one clip and orbit the camera around it
    python src2/render3d.py --clip sim_00000 --mode orbit --out out3d

    # side by side: observed masks vs the reprojected fit, original viewpoint
    python src2/render3d.py --clip sim_00000 --mode validate --out out3d

    # a fixed novel viewpoint, e.g. straight down
    python src2/render3d.py --clip sim_00000 --mode topdown --out out3d

Each writes PNG frames plus an animated GIF.
"""

import os
import sys
import glob
import json
import math
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lift3d import Camera, CAMERA_DEFAULT
from fit3d import load_one, fit_pose, MESH, YAW_DOF, place

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Pillow is required:  pip install pillow")


# CLEVRER's palette, approximated in RGB for display only. The physics engine
# never sees colour (see the colour rule in dynamics.py); this is purely so a
# human can tell the objects apart.
COLOR_RGB = {
    "gray":   (140, 140, 140),
    "red":    (196,  60,  50),
    "blue":   ( 55,  95, 190),
    "green":  ( 60, 155,  75),
    "brown":  (135,  95,  55),
    "purple": (140,  75, 175),
    "cyan":   ( 60, 175, 190),
    "yellow": (220, 195,  60),
}
METAL_BOOST = 1.18      # metal reads brighter/specular; a display cue only


# ---------------------------------------------------------------------------
#                     meshes WITH FACES (not just vertices)
# ---------------------------------------------------------------------------
def faces_cube(s):
    v = np.array([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0),
                  (-s, -s, 2 * s), (s, -s, 2 * s), (s, s, 2 * s), (-s, s, 2 * s)],
                 dtype=np.float64)
    f = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
         (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    return v, f


def faces_cylinder(s, n=20):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    bot = np.stack([s * np.cos(a), s * np.sin(a), np.zeros(n)], axis=1)
    top = np.stack([s * np.cos(a), s * np.sin(a), np.full(n, 2 * s)], axis=1)
    v = np.concatenate([bot, top], axis=0)
    f = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    f.append(tuple(range(n, 2 * n)))                 # top disc
    f.append(tuple(range(n - 1, -1, -1)))            # bottom disc
    return v, f


def faces_sphere(s, nlat=10, nlon=16):
    """Latitude-longitude quads. Centre sits at z = s so the base touches z=0."""
    verts, idx = [], {}
    for i in range(nlat + 1):
        th = math.pi * i / nlat
        for j in range(nlon):
            ph = 2 * math.pi * j / nlon
            idx[(i, j)] = len(verts)
            verts.append((s * math.sin(th) * math.cos(ph),
                          s * math.sin(th) * math.sin(ph),
                          s * math.cos(th) + s))
    v = np.asarray(verts, dtype=np.float64)
    f = []
    for i in range(nlat):
        for j in range(nlon):
            j2 = (j + 1) % nlon
            f.append((idx[(i, j)], idx[(i, j2)], idx[(i + 1, j2)], idx[(i + 1, j)]))
    return v, f


FACE_MESH = {"cube": faces_cube, "cylinder": faces_cylinder, "sphere": faces_sphere}


# ---------------------------------------------------------------------------
#                          a free-flying camera
# ---------------------------------------------------------------------------
def make_camera(loc, target, lens_mm, width, height, sensor_mm=32.0):
    return Camera({"loc": list(loc), "target": list(target), "lens_mm": lens_mm,
                   "sensor_mm": sensor_mm, "width": width, "height": height})


def project_pts(cam, pts):
    """World -> pixels. Returns (uv, depth, in_front) with no points dropped,
    so face indices stay valid."""
    rel = pts - cam.loc
    cp = rel @ cam.R
    z = cp[:, 2]
    in_front = z < -1e-6
    d = np.where(in_front, -z, 1e-6)
    u = cam.cx + cam.fx * cp[:, 0] / d
    v = cam.cy - cam.fy * cp[:, 1] / d
    return np.stack([u, v], axis=1), d, in_front


# ---------------------------------------------------------------------------
#                          polygon rasteriser
# ---------------------------------------------------------------------------
def fill_poly(buf, zbuf, poly, depth, rgb):
    """Fill a convex polygon with a flat colour, painter's-algorithm style.

    zbuf holds one depth per pixel so later (nearer) faces overwrite earlier
    ones correctly even when the far-to-near sort is ambiguous for
    interpenetrating geometry."""
    H, W = zbuf.shape
    u0 = max(int(np.floor(poly[:, 0].min())), 0)
    u1 = min(int(np.ceil(poly[:, 0].max())) + 1, W)
    v0 = max(int(np.floor(poly[:, 1].min())), 0)
    v1 = min(int(np.ceil(poly[:, 1].max())) + 1, H)
    if u1 <= u0 or v1 <= v0:
        return
    uu, vv = np.meshgrid(np.arange(u0, u1) + 0.5, np.arange(v0, v1) + 0.5)
    inside = np.ones(uu.shape, dtype=bool)
    n = len(poly)
    # Convex polygon: a point is inside iff it is on the same side of every
    # edge. Which side that is comes from the polygon's WINDING, so take it
    # from the signed area. Deriving it from a sample pixel instead (as this
    # did) picks the wrong side whenever the first pixel of the bounding box
    # lies outside the polygon -- which is most of the time for small quads,
    # and is why spheres rendered as crescents.
    sa = 0.0
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        sa += a[0] * b[1] - b[0] * a[1]
    sign = 1.0 if sa > 0 else -1.0
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        cr = (b[0] - a[0]) * (vv - a[1]) - (b[1] - a[1]) * (uu - a[0])
        inside &= (cr * sign) >= -1e-9
    if not inside.any():
        return
    sub = zbuf[v0:v1, u0:u1]
    win = inside & (depth < sub)
    if not win.any():
        return
    sub[win] = depth
    tile = buf[v0:v1, u0:u1]
    tile[win] = rgb


LIGHT = np.array([0.35, -0.5, 0.78])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def shade(base, normal, face_centre=None, cam_loc=None, material="rubber"):
    """Lambertian for rubber, Blinn-Phong specular for metal.

    Material has to be VISIBLE, not decorative. CLEVRER's objects are unique
    on (colour, material, shape), not on colour -- 21% of clips contain two
    objects of the same colour and shape that differ only by material (e.g.
    sim_00000 has a blue rubber sphere AND a blue metal sphere). The earlier
    1.18x brightness nudge made those two indistinguishable, so a correct
    reconstruction looked like a duplicated object. A specular highlight makes
    the distinction obvious at a glance.

    It is also physically meaningful here: material sets the mass prior
    (metal 1.5 / rubber 1.0) that the dynamics uses.
    """
    n = normal / (np.linalg.norm(normal) + 1e-12)
    diff = max(float(np.dot(n, LIGHT)), 0.0)
    if material == "metal" and face_centre is not None and cam_loc is not None:
        view = cam_loc - face_centre
        view = view / (np.linalg.norm(view) + 1e-12)
        half = LIGHT + view
        half = half / (np.linalg.norm(half) + 1e-12)
        spec = max(float(np.dot(n, half)), 0.0) ** 48
        k = 0.40 + 0.52 * diff
        return tuple(int(np.clip(c * k + 235 * 0.6 * spec, 0, 255)) for c in base)
    k = 0.36 + 0.60 * diff                      # matte
    return tuple(int(np.clip(c * k, 0, 255)) for c in base)


def draw_ground(buf, zbuf, cam, half=6.0, step=0.4):
    """A checkerboard table. Without it a novel viewpoint has no reference and
    the reconstruction is impossible to read."""
    n = int(2 * half / step)
    quads, depths, shades = [], [], []
    for i in range(n):
        for j in range(n):
            x0, y0 = -half + i * step, -half + j * step
            corners = np.array([(x0, y0, 0), (x0 + step, y0, 0),
                                (x0 + step, y0 + step, 0), (x0, y0 + step, 0)])
            uv, d, fr = project_pts(cam, corners)
            if not fr.all():
                continue
            quads.append(uv)
            depths.append(float(d.mean()))
            tone = 208 if (i + j) % 2 == 0 else 176
            shades.append((tone, tone, tone + 6))
    # Push the table very slightly away from the camera. Objects rest with
    # their base EXACTLY on z=0, so at the contact point the base faces and the
    # ground quad have identical depth and z-fighting lets the ground win --
    # which reads as objects sinking into a marsh. The geometry is right; the
    # tie-break was not.
    # Each tile is filled at ONE depth -- the mean of its corners. Viewed at a
    # shallow angle a large tile spans a big depth range, so its mean badly
    # understates the far half and the tile gets drawn in FRONT of objects
    # standing behind it. Smaller tiles (step 0.4 instead of 1.0) shrink that
    # error to well under an object's size, and the bias breaks the remaining
    # tie at the contact point where an object's base and the floor coincide.
    GROUND_BIAS = 0.05
    for k in np.argsort(depths)[::-1]:
        fill_poly(buf, zbuf, quads[k], depths[k] + GROUND_BIAS, shades[k])


def render_scene(cam, objects, width, height, bg=(248, 248, 250)):
    """objects: list of dicts {shape, size, x, y, yaw, color, material}."""
    buf = np.zeros((height, width, 3), dtype=np.uint8)
    buf[:, :] = bg
    zbuf = np.full((height, width), np.inf)
    draw_ground(buf, zbuf, cam)

    faces = []
    for ob in objects:
        v, fl = FACE_MESH[ob["shape"]](ob["size"])
        v = place(v, ob["x"], ob["y"], ob.get("yaw", 0.0))
        uv, d, fr = project_pts(cam, v)
        base = COLOR_RGB.get(ob.get("color", "gray"), (150, 150, 150))
        mat = ob.get("material", "rubber")
        # every one of these primitives is convex and centred here, so the
        # OUTWARD normal is the one pointing away from the centre. Deriving it
        # that way is winding-order independent -- relying on winding gave the
        # sphere inward normals, which culled the near hemisphere and drew
        # crescents.
        centre = np.array([ob["x"], ob["y"], ob["size"]])
        for f in fl:
            idx = list(f)
            if not fr[idx].all():
                continue
            p3 = v[idx]
            nrm = np.cross(p3[1] - p3[0], p3[2] - p3[0])
            nn = np.linalg.norm(nrm)
            if nn < 1e-12:
                continue                       # degenerate (sphere pole quads)
            nrm = nrm / nn
            fc = p3.mean(axis=0)
            if np.dot(nrm, fc - centre) < 0:
                nrm = -nrm                     # force it to point outward
            # back-face cull: keep faces whose outward normal faces the camera
            if np.dot(nrm, cam.loc - fc) <= 0:
                continue
            # A sphere's pole rows are quads with two coincident vertices, i.e.
            # triangles. Left as duplicated points they measure zero area and
            # got dropped, leaving a hole at each pole. Dedupe first, then
            # judge the area.
            keep = [idx[0]]
            for w in idx[1:]:
                if not np.allclose(v[w], v[keep[-1]], atol=1e-9):
                    keep.append(w)
            if len(keep) > 2 and np.allclose(v[keep[0]], v[keep[-1]], atol=1e-9):
                keep = keep[:-1]
            if len(keep) < 3:
                continue
            poly = uv[keep]
            area = 0.5 * abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1]
                                 - poly[(i + 1) % len(poly)][0] * poly[i][1]
                                 for i in range(len(poly))))
            if area < 0.01:
                continue                       # genuinely sub-pixel
            idx = keep
            faces.append((float(d[idx].mean()), poly,
                          shade(base, nrm, fc, cam.loc, mat)))

    for dep, poly, rgb in sorted(faces, key=lambda t: -t[0]):
        fill_poly(buf, zbuf, poly, dep, rgb)
    return buf


# ---------------------------------------------------------------------------
#                    fit every frame of one clip
# ---------------------------------------------------------------------------
def fit_clip(clip, cam, sizes, stride=1, verbose=True):
    """Per-frame poses, warm-started from the previous frame."""
    T, N = clip["T"], clip["N"]
    poses = [[None] * N for _ in range(T)]
    last = [None] * N
    ious = []
    for t in range(0, T, stride):
        for k in range(N):
            d = clip["det"][t][k]
            if d is None or not clip["inside"][t, k]:
                last[k] = None
                continue
            shape = clip["objects"][k].get("shape")
            if shape not in MESH:
                continue
            m = clip["raw"][t][k]
            if m is None or m.sum() < 150:
                last[k] = None
                continue
            f = fit_pose(cam, shape, sizes[shape], m, d[0], d[1],
                         iters=2 if last[k] is not None else 3,
                         init=last[k])
            last[k] = (f["x"], f["y"], f["yaw"])
            o = clip["objects"][k]
            poses[t][k] = {"shape": shape, "size": sizes[shape],
                           "x": f["x"], "y": f["y"], "yaw": f["yaw"],
                           "color": o.get("color"), "material": o.get("material"),
                           "iou": f["iou"]}
            ious.append(f["iou"])
        if verbose and t % 16 == 0:
            print(f"   frame {t:3d}/{T}  median IoU so far "
                  f"{np.median(ious) if ious else 0:.3f}", flush=True)
    return poses, (float(np.median(ious)) if ious else 0.0)


# ---------------------------------------------------------------------------
def save_gif(frames, path, fps=12):
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="legacy/v1/data/processed_proposals")
    ap.add_argument("--clip", default="sim_00000")
    ap.add_argument("--camera", default="camera_fit.json")
    ap.add_argument("--out", default="out3d")
    ap.add_argument("--mode", default="orbit",
                    choices=["orbit", "frozen", "topdown", "validate", "original"])
    ap.add_argument("--freeze-at", type=int, default=40,
                    help="frame to hold still in --mode frozen")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--stride", type=int, default=2, help="frame stride")
    ap.add_argument("--frames", type=int, default=64)
    a = ap.parse_args()

    with open(a.camera, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    cam_fit = Camera(blob["camera"])
    sizes = blob["sizes"]

    path = os.path.join(a.input, a.clip + ".json")
    if not os.path.isfile(path):
        raise SystemExit(f"clip not found: {path}")
    print(f"loading {a.clip} ...")
    clip = load_one(path)

    print("fitting 3D poses frame by frame (warm-started)...")
    poses, med = fit_clip(clip, cam_fit, sizes, stride=a.stride)
    print(f"median silhouette IoU over the clip: {med:.4f}")
    if med < 0.75:
        print("WARNING: poor fit -- the render below will not be trustworthy.")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{a.clip}_poses3d.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"clip": a.clip, "median_iou": med, "stride": a.stride,
                   "camera": blob["camera"], "sizes": sizes,
                   "poses": [[p for p in row] for row in poses]}, fh)

    ts = [t for t in range(0, clip["T"], a.stride) if any(poses[t])][:a.frames]
    if a.mode == "frozen":
        # Hold ONE instant and move only the camera. Orbiting a playing clip
        # conflates two motions and is genuinely hard to read -- objects appear
        # to fly in from nowhere when it is really the viewpoint sweeping. This
        # isolates the question the render is meant to answer: is the GEOMETRY
        # of a single reconstructed instant correct from every angle?
        near = min(ts, key=lambda t: abs(t - a.freeze_at)) if ts else 0
        ts = [near] * a.frames
    frames = []
    for n, t in enumerate(ts):
        objs = [p for p in poses[t] if p]
        if a.mode in ("orbit", "frozen"):
            # one full revolution across the clip, held at a constant elevation
            ang = 2 * math.pi * n / max(len(ts) - 1, 1)
            r, z = 9.0, 5.2
            cam = make_camera((r * math.cos(ang), r * math.sin(ang), z),
                              (0, 0, 0.4), 34.2, a.width, a.height)
        elif a.mode == "topdown":
            cam = make_camera((0.01, 0.01, 12.0), (0, 0, 0), 34.2, a.width, a.height)
        else:
            cam = cam_fit
        img = render_scene(cam, objs, a.width, a.height)

        if a.mode == "validate":
            # observed masks on the left, our reprojection on the right
            left = np.full((a.height, a.width, 3), 250, np.uint8)
            for k in range(clip["N"]):
                m = clip["raw"][t][k]
                if m is None or not clip["inside"][t, k]:
                    continue
                c = COLOR_RGB.get(clip["objects"][k].get("color"), (120, 120, 120))
                left[m] = c
            img = np.concatenate([left, img], axis=1)

        frames.append(img)
        Image.fromarray(img).save(os.path.join(a.out, f"{a.clip}_{a.mode}_{n:03d}.png"))

    gif = os.path.join(a.out, f"{a.clip}_{a.mode}.gif")
    save_gif(frames, gif)
    print(f"\nwrote {len(frames)} frames + {gif}")
    if a.mode == "validate":
        print("left = observed detector masks, right = our recovered 3D scene")
    elif a.mode == "orbit":
        print("the camera circles the scene; the original video had ONE fixed view")


if __name__ == "__main__":
    main()
