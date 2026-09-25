"""
app/render_worker.py  —  draws one GIF frame in a separate process

Kept tiny on purpose: worker processes import only numpy and the renderer, never
torch or the world model, so starting them is fast. One frame at 480x320 takes
~0.56 s; three workers draw a GIF about three times faster.
"""

import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src2"))

from render3d import render_scene, make_camera, project_pts, fill_poly  # noqa: E402


def view_camera(view, W, H):
    """The camera the page's 3D player uses: orbit around the table centre at
    azimuth / elevation (degrees) and distance. Same numbers -> same picture."""
    az, el, dist = math.radians(view["az"]), math.radians(view["el"]), float(view["dist"])
    loc = (dist * math.cos(el) * math.cos(az), dist * math.cos(el) * math.sin(az), dist * math.sin(el))
    return make_camera(loc, (0.0, 0.0, 0.0), float(view.get("lens", 34.21875)), W, H)


def render_one(args):
    """args = (frame index t, camera view {az, el, dist}, objects, markers, W, H).
    The user picks the view in the page; every frame of a GIF uses that one view."""
    t, view, objs, markers, W, H = args
    cam = view_camera(view, W, H)
    img = render_scene(cam, objs, W, H)
    for x, y, z, rgb, rad in markers:
        uv, _, front = project_pts(cam, np.array([[x, y, z]], float))
        if not front[0]:
            continue
        u, v = uv[0]
        poly = np.array([[u + rad * math.cos(a), v + rad * math.sin(a)]
                         for a in np.linspace(0, 2 * math.pi, 14, endpoint=False)])
        fill_poly(img, np.full((H, W), np.inf), poly, 0.0, rgb)
    return t, img
