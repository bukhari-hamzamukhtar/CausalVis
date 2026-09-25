"""
v6/resolve.py  —  collisions resolved from the voxels that got pressed
=======================================================================

Every voxel is a sensor. An object occupying a voxel presses it. When two
objects press the SAME voxels, that is the contact -- and where they press is
the contact point. Nothing about centres, radii, or contact distance enters.

From the pressed voxels we get three things, all read off the grid:

    POINT   the centroid of the pressed region
    NORMAL  the direction the two occupied sets face each other along,
            taken from the local shape of each set near that point
    IMPULSE the momentum exchange along that normal

Then the vectors: object i arrives with velocity v_i, and at the contact point
the exchange is
        J = -(1+e)(v_rel . n) / (1/m_i + 1/m_j)
        v_i += (J/m_i) n        v_j -= (J/m_j) n
which splits the incoming vector into the part along n (exchanged) and the
part across n (kept). Equal and opposite, so momentum is conserved exactly.

WHY THIS IS DIFFERENT FROM v5/impulse.py
----------------------------------------
v5 used the line joining the two centres as the normal. That is exact only for
spheres. Measured against a flat face struck off-centre it is wrong by up to
20 degrees -- and the truth there is not a matter of opinion: a flat face
pushes straight out along its own normal, whatever the centres are doing.
Here the normal comes from the pressed voxels, so a face pushes square, a
corner pushes diagonally, and no case is special.
"""

import math

import numpy as np
import torch

from voxel import (occupancy, touching, contact_normal, shape_of, VOXEL,
                   shell_local, place)

WORLD = 6.0            # normalised units -> world units


def resolve_voxel(q, p, mass, radius, present, restitution, shapes, yaw,
                  dilate=3, prefilter=0.35):
    """One impulse per pressed pair. q, p in NORMALISED units; the grid works
    in world units, so positions are scaled on the way in.

    `prefilter` skips voxelising pairs that are obviously far apart. It is a
    SPEED optimisation only -- the physics is decided by the voxels, never by
    the centre distance. Set it large enough that no real contact is skipped.
    """
    N = q.shape[1]
    qn = q[0].detach().cpu().numpy()
    out = p.clone()
    occ = {}
    events = []
    for i in range(N):
        if present[0, i] <= 0:
            continue
        for j in range(i + 1, N):
            if present[0, j] <= 0:
                continue
            if float(np.linalg.norm(qn[i] - qn[j])) > prefilter:
                continue
            for k in (i, j):
                if k not in occ:
                    # cached local shell, translated -- 5x faster than
                    # rebuilding, and verified to give identical verdicts
                    occ[k] = place(shell_local(shapes[k], float(radius[0, k]) * WORLD,
                                               float(yaw[k])), qn[k] * WORLD)
            hit, point = touching(occ[i], occ[j], dilate=dilate)
            if not hit:
                continue
            n3 = contact_normal(occ[i], occ[j], point)       # points j -> i
            n = np.array([n3[0], n3[1]], np.float64)
            L = float(np.linalg.norm(n))
            if L < 1e-9:
                continue
            n = n / L
            nt = torch.tensor(n, dtype=torch.float32)
            vi = out[0, i] / mass[0, i]
            vj = out[0, j] / mass[0, j]
            v_n = float(torch.dot(vi - vj, nt))
            if v_n >= 0:                       # already separating: leave alone
                continue
            e = 0.5 * (float(restitution[i]) + float(restitution[j]))
            J = -(1.0 + e) * v_n / (1.0 / float(mass[0, i]) + 1.0 / float(mass[0, j]))
            out[0, i] = out[0, i] + J * nt
            out[0, j] = out[0, j] - J * nt
            events.append({"i": i, "j": j, "point_world": point.tolist(),
                           "normal": n.tolist(), "impulse": J})
    return out, events
