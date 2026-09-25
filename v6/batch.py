"""
v6/batch.py  —  voxel contact resolution for a whole training batch
====================================================================

`v6/resolve.py` handles one scene at a time, which is right for the demo and
for evaluation but useless for training: the loop there runs 16 scenes at once
for 20 steps, so a per-scene Python call would be 320 numpy round-trips per
gradient step.

This is the same physics, restructured so the training loop can carry it:

  1. ONE vectorised prefilter over the whole batch decides which pairs are
     close enough to be worth voxelising. It is a speed gate only -- it is set
     wider than any contact the grid could report, so it never decides physics.
  2. Only the surviving pairs are voxelised. Their shells are cached, so
     placing one is integer addition.
  3. The impulses are applied with ONE scatter over the batch.

WHY IT IS DIFFERENTIABLE, AND EXACTLY WHERE
-------------------------------------------
The contact NORMAL is read off the voxel grid, in numpy, and enters as a
constant. That is correct and not a compromise: a normal is a geometric fact
about where two surfaces meet, not a quantity the potential should be able to
bend to reduce its loss. Letting a gradient flow into it would let the model
learn to point contacts wherever the loss is lowest, which is the soft-spring
failure this whole version exists to remove.

The impulse MAGNITUDE

    J = -(1 + e)(v_rel . n) / (1/m_i + 1/m_j)

is computed in torch from the live momentum, so gradients flow through v_rel
into everything upstream: the potential that set the approach velocity, the
drag that slowed it, the mass and radius heads. The model is therefore trained
knowing that when surfaces meet, momentum is exchanged along the true normal --
so the potential no longer has to imitate a bounce it can never get right, and
can specialise in the approach.

This is the standard arrangement in differentiable rigid-body engines: contact
geometry detached, contact response differentiable.

MOMENTUM
--------
Object i receives +J n, object j receives -J n, in the same scatter. They
cancel exactly, so the project's core guarantee is untouched.
"""

import numpy as np
import torch

from voxel import shell_local, place, touching, contact_normal, keys_sorted

WORLD = 6.0            # normalised units -> world units

# Widest contact the grid can report, in NORMALISED units:
#   a cube's corner reaches sqrt(2)/(4/pi) = 1.111 x its stored radius,
#   two cubes at ~0.06 each give 0.133, plus the dilation band (3 voxels =
#   0.06 world = 0.01 normalised). 0.20 leaves a 40% margin on top of that.
PREFILTER = 0.20


def resolve_voxel_batch(q, p, mass, radius, present, restitution, shapes, yaw,
                        dilate=3, prefilter=PREFILTER, collect=None):
    """Apply one voxel impulse per pressed, approaching pair, across a batch.

    q, p, mass, radius, present : torch, [B,N,2] / [B,N]
    restitution                 : torch [B,N] or numpy [B,N]
    shapes                      : list of B lists of N shape strings
    yaw                         : [B,N] torch or numpy
    collect                     : optional list; contact events are appended

    Returns p with the impulses added. p keeps its graph.
    """
    B, N, _ = q.shape
    qd = q.detach()
    with torch.no_grad():
        d = torch.cdist(qd, qd)                                  # [B,N,N]
        both = present.unsqueeze(2) * present.unsqueeze(1)
        iu = torch.triu(torch.ones(N, N, device=q.device), diagonal=1)
        near = (d < prefilter) & (both > 0) & (iu > 0)
    if not bool(near.any()):
        return p

    qn = qd.cpu().numpy()
    rn = radius.detach().cpu().numpy()
    yn = yaw.detach().cpu().numpy() if torch.is_tensor(yaw) else np.asarray(yaw)
    bs, is_, js, ns = [], [], [], []
    occ = {}
    idx = near.nonzero(as_tuple=False).cpu().numpy()
    for b, i, j in idx:
        b, i, j = int(b), int(i), int(j)
        for k in (i, j):
            if (b, k) not in occ:
                v = place(shell_local(shapes[b][k], float(rn[b, k]) * WORLD,
                                      float(yn[b, k])), qn[b, k] * WORLD)
                # keys depend only on WHERE the object is, so compute them once
                # per object rather than once per pair it takes part in
                occ[(b, k)] = (v, keys_sorted(v))
        (vi_, ki_), (vj_, kj_) = occ[(b, i)], occ[(b, j)]
        hit, point = touching(vi_, vj_, dilate=dilate, keys_i=ki_, keys_j=kj_)
        if not hit:
            continue
        n3 = contact_normal(vi_, vj_, point)                      # points j -> i
        n = np.array([n3[0], n3[1]], np.float64)
        L = float(np.linalg.norm(n))
        if L < 1e-9:
            continue
        bs.append(b); is_.append(i); js.append(j); ns.append(n / L)
        if collect is not None:
            collect.append({"b": b, "i": i, "j": j,
                            "point_world": np.asarray(point).tolist(),
                            "normal": (n / L).tolist()})
    if not bs:
        return p

    bi = torch.as_tensor(bs, dtype=torch.long, device=q.device)
    ii = torch.as_tensor(is_, dtype=torch.long, device=q.device)
    jj = torch.as_tensor(js, dtype=torch.long, device=q.device)
    nrm = torch.as_tensor(np.asarray(ns), dtype=p.dtype, device=q.device)   # [K,2]

    mi = mass[bi, ii]; mj = mass[bi, jj]
    vi = p[bi, ii] / mi.unsqueeze(-1)
    vj = p[bi, jj] / mj.unsqueeze(-1)
    v_n = ((vi - vj) * nrm).sum(-1)                              # closing speed
    rest = restitution if torch.is_tensor(restitution) else \
        torch.as_tensor(np.asarray(restitution), dtype=p.dtype, device=q.device)
    ee = 0.5 * (rest[bi, ii] + rest[bi, jj])
    inv = (1.0 / mi + 1.0 / mj).clamp_min(1e-9)
    # only pairs that are still APPROACHING get an impulse; a separating pair
    # struck again would gain energy from nothing and the rollout would explode
    gate = (v_n < 0).to(p.dtype)
    J = (-(1.0 + ee) * v_n / inv) * gate                         # [K]
    contrib = J.unsqueeze(-1) * nrm                              # [K,2]

    flat_i = bi * N + ii
    flat_j = bi * N + jj
    dp = torch.zeros(B * N, 2, dtype=p.dtype, device=q.device)
    dp = dp.index_add(0, flat_i, contrib).index_add(0, flat_j, -contrib)
    if collect is not None:
        for k in range(len(bs)):
            collect[len(collect) - len(bs) + k]["impulse"] = float(J[k].detach())
    return p + dp.view(B, N, 2) * present.unsqueeze(-1)


def shapes_from_attrs(attrs):
    """[B,N,16] -> list of B lists of shape names, matching voxel.shape_of."""
    a = attrs.detach().cpu().numpy() if torch.is_tensor(attrs) else np.asarray(attrs)
    out = []
    for b in range(a.shape[0]):
        out.append(["cube" if a[b, k, 2] > 0.5 else
                    ("sphere" if a[b, k, 3] > 0.5 else "cylinder")
                    for k in range(a.shape[1])])
    return out
