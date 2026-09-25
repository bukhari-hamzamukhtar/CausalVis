"""
v5/contact_point.py  —  WHERE two surfaces meet, not just how far apart
=======================================================================

Everything before this measured contact DISTANCE: "how far apart would their
centres be if they touched". This derives the contact POINT: the actual pair
of closest points on the two surfaces, and the normal AT that point.

The difference is the whole argument. Two objects can touch at a corner that
does not lie on the line joining their centres. Resolving the collision along
the line of centres then sends them off in the wrong direction -- measured at
23.5 deg even for spheres, worse for cubes.

Everything here is derived from the fitted 3D scene (0.95 silhouette IoU), in
the horizontal plane, because every CLEVRER object rests on the same table:

    sphere / cylinder -> circle of radius r
    cube              -> square of half-side r / (4/pi), rotated by its yaw

For each pair we return (point_on_i, point_on_j, normal, distance). The normal
points from j to i along the shortest connection between the surfaces, which
is the direction a real impulse acts along.

Rejected earlier and worth not repeating: snapping the normal to a cube's
nearest FACE (26.8 vs 21.4 deg). A face normal is right only when the contact
projects inside that face; near an edge the true normal comes off the edge.
The closest-point construction below handles face, edge and corner without
special cases, which is the point.
"""

import math

import numpy as np

CUBE_MEAN_SUPPORT = 4.0 / math.pi


def square_corners(cx, cy, s, yaw):
    c, sn = math.cos(yaw), math.sin(yaw)
    return [(cx + dx * c - dy * sn, cy + dx * sn + dy * c)
            for dx, dy in ((-s, -s), (s, -s), (s, s), (-s, s))]


def closest_on_segment(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    L = vx * vx + vy * vy
    t = 0.0 if L < 1e-12 else max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / L))
    return ax + t * vx, ay + t * vy


def closest_on_polygon(px, py, poly):
    """Closest point on the polygon BOUNDARY, and whether (px,py) is inside."""
    best, bd = None, 1e18
    n = len(poly)
    inside = True
    for k in range(n):
        ax, ay = poly[k]
        bx, by = poly[(k + 1) % n]
        if (bx - ax) * (py - ay) - (by - ay) * (px - ax) < 0:
            inside = False
        qx, qy = closest_on_segment(px, py, ax, ay, bx, by)
        d = (qx - px) ** 2 + (qy - py) ** 2
        if d < bd:
            bd, best = d, (qx, qy)
    return best, inside


def shape_of(attrs, k):
    return "cube" if attrs[k, 2] > 0.5 else ("sphere" if attrs[k, 3] > 0.5 else "cyl")


def contact(shape_i, ci, ri, yi, shape_j, cj, rj, yj):
    """-> (p_i, p_j, normal_from_j_to_i, surface_distance).

    normal is a unit vector; distance is negative when the shapes overlap.
    """
    ci = np.asarray(ci, float); cj = np.asarray(cj, float)

    if shape_i != "cube" and shape_j != "cube":
        d = ci - cj
        L = float(np.linalg.norm(d))
        n = d / L if L > 1e-9 else np.array([1.0, 0.0])
        return ci - n * ri, cj + n * rj, n, L - ri - rj

    if shape_i == "cube" and shape_j != "cube":
        poly = square_corners(ci[0], ci[1], ri / CUBE_MEAN_SUPPORT, yi)
        q, inside = closest_on_polygon(cj[0], cj[1], poly)
        d = cj - np.asarray(q, float)                 # from cube surface to centre j
        L = float(np.linalg.norm(d))
        n = -(d / L) if L > 1e-9 else np.array([1.0, 0.0])   # j -> i
        if inside:
            n = -n
        return np.asarray(q, float), cj + n * rj, n, (-L - rj if inside else L - rj)

    if shape_j == "cube" and shape_i != "cube":
        p_j, p_i, n, dist = contact(shape_j, cj, rj, yj, shape_i, ci, ri, yi)
        return p_i, p_j, -n, dist

    # cube - cube: shortest connection between two convex polygons
    Pi = square_corners(ci[0], ci[1], ri / CUBE_MEAN_SUPPORT, yi)
    Pj = square_corners(cj[0], cj[1], rj / CUBE_MEAN_SUPPORT, yj)
    best = (1e18, None, None)
    for (px, py) in Pi:
        q, _ = closest_on_polygon(px, py, Pj)
        d = math.hypot(px - q[0], py - q[1])
        if d < best[0]:
            best = (d, (px, py), q)
    for (px, py) in Pj:
        q, _ = closest_on_polygon(px, py, Pi)
        d = math.hypot(px - q[0], py - q[1])
        if d < best[0]:
            best = (d, q, (px, py))
    dist, p_i, p_j = best
    v = np.asarray(p_i, float) - np.asarray(p_j, float)
    L = float(np.linalg.norm(v))
    if L > 1e-9:
        n = v / L
    else:
        v = ci - cj
        n = v / max(float(np.linalg.norm(v)), 1e-9)
    return np.asarray(p_i, float), np.asarray(p_j, float), n, dist
