"""
v5/impulse.py  —  proper collision mechanics: the impulse the model never had
==============================================================================

WHY THIS EXISTS
---------------
Measured on 100 held-out clips, replaying real annotated collisions:

    collision      direction error   outgoing speed
    sphere-sphere       23.5 deg          0.81
    cube-cube           37.9 deg          0.99

For two spheres the line joining their centres IS the exact contact normal --
the geometry cannot be wrong. Being 23.5 degrees off there proves the error is
not about shape. It is that the model has no collision mechanics at all.

What it has is a SOFT SPRING. The learned potential V(distance) produces a
smooth force that builds over many frames. During those frames both objects
keep moving, so the line of centres ROTATES while the force is being applied,
and the accumulated impulse points somewhere between where the normal started
and where it ended. A real collision is impulsive: one exchange, one normal,
one instant.

THE PHYSICS (textbook rigid-body impulse, nothing invented)
-----------------------------------------------------------
At contact with unit normal n, relative velocity v_rel = v_i - v_j, and
restitution e:

    v_n = v_rel . n                       (approach speed along the normal)
    J   = -(1 + e) * v_n / (1/m_i + 1/m_j)
    v_i += (J/m_i) n                      v_j -= (J/m_j) n

Three properties, all structural rather than trained:

  MOMENTUM IS EXACTLY CONSERVED. The two objects receive +J n and -J n. They
  cancel. This is the same guarantee the potential gives, by the same
  mechanism (Newton's third law), so the project's core claim survives intact.

  THE ANGLE IS RIGHT BY CONSTRUCTION. The impulse is along n at the instant of
  contact. No smearing across a rotating normal.

  RESTITUTION FINALLY DOES SOMETHING. attrs[:,1] holds 0.8 for metal and 0.5
  for rubber, and until now it only fed the attribute encoder -- it removed no
  energy. Here e is the pair's restitution and directly sets how much of the
  approach speed comes back. e = 1 is a perfect bounce, e = 0 is fully
  inelastic.

WHAT IS STILL LEARNED
---------------------
Everything that should be. The potential still learns the interaction; mass,
radius and drag are unchanged. Only the CONTACT RESPONSE is now derived from
mechanics instead of hoped for from a spring. Optionally `learn_restitution`
lets a small head adjust e per object around its material prior, so the model
can discover that CLEVRER's metal bounces differently than the prior says.

CONTACT NORMAL
--------------
For round shapes the normal is the line of centres -- exact. For a cube the
true normal is perpendicular to the struck face; `support_normal` derives it
from the same support function used for shape-aware contact, so a cube hit on
a face is pushed square and a cube hit on a corner is pushed diagonally.
"""

import math

import torch

SMOOTH_EPS = 1e-4
CUBE_MEAN_SUPPORT = 4.0 / math.pi


def contact_normals(q, present):
    """Unit vector from j to i for every pair, [B,N,N,2]. Degenerate pairs
    (absent, or the diagonal) get a fixed direction so no NaN reaches the
    gradient -- masking the forward pass does not mask the backward one."""
    B, N, _ = q.shape
    r = q.unsqueeze(2) - q.unsqueeze(1)                       # i - j
    d = torch.sqrt((r * r).sum(-1, keepdim=True) + 1e-12)
    off = (1.0 - torch.eye(N, device=q.device)).unsqueeze(0).unsqueeze(-1)
    valid = (present.unsqueeze(2) * present.unsqueeze(1)).unsqueeze(-1) * off
    safe = torch.zeros_like(r); safe[..., 0] = 1.0
    return torch.where(valid > 0, r / d, safe), d.squeeze(-1)


def cube_normal(n_hat, yaw, is_cube):
    """Snap the normal to a cube's nearest face.

    MEASURED AND REJECTED (2026-09-10). On 200 held-out clips with real
    recovered orientation, snapping to the face made the bounce WORSE:

        cube pairs, line-of-centres normal : 21.4 deg
        cube pairs, face normal            : 26.8 deg

    The reason is that this snaps to a face unconditionally. When a sphere
    strikes a cube near an EDGE the true normal points out of the edge, not
    out of either adjoining face, and forcing a face normal there is a bigger
    error than the line of centres it replaced. A correct version would use the
    true closest-point normal (face when the contact projects inside a face,
    edge or corner otherwise) -- i.e. real GJK/EPA contact generation, not a
    softmax over two axes.

    Left in place, unused by default, because the negative result is worth
    keeping and the correct formulation would start from here.

    A cube struck anywhere on a face is pushed perpendicular to that face, not
    along the line of centres. In the cube's own frame the face normals are the
    axes, so the nearest face is whichever of |cos|, |sin| is larger. Blended
    by is_cube so round shapes keep the exact line-of-centres normal.
    """
    c, s = torch.cos(yaw), torch.sin(yaw)
    # rotate the world normal into the cube's frame
    lx = n_hat[..., 0] * c + n_hat[..., 1] * s
    ly = -n_hat[..., 0] * s + n_hat[..., 1] * c
    # pick the dominant axis, keep the sign, smooth so the gradient survives
    w = torch.softmax(torch.stack([lx.abs(), ly.abs()], -1) / 0.15, dim=-1)
    fx = w[..., 0] * torch.sign(lx)
    fy = w[..., 1] * torch.sign(ly)
    # back to world
    wx = fx * c - fy * s
    wy = fx * s + fy * c
    face = torch.stack([wx, wy], -1)
    face = face / torch.sqrt((face * face).sum(-1, keepdim=True) + 1e-12)
    k = is_cube.unsqueeze(-1)
    out = k * face + (1.0 - k) * n_hat
    return out / torch.sqrt((out * out).sum(-1, keepdim=True) + 1e-12)


def resolve_contacts(q, p, mass, radius, present, restitution,
                     yaw=None, is_cube=None, thresh=0.0, max_pairs_note=None):
    """Apply one impulse per touching, approaching pair. Returns new momentum.

    Only pairs that are BOTH overlapping and closing get an impulse -- objects
    already separating must not be hit again, or they gain energy from nothing
    and the simulation explodes. That approach gate is the same one the
    collision detector uses, for the same reason.
    """
    B, N, _ = q.shape
    n_hat, d = contact_normals(q, present)
    contact = radius.unsqueeze(2) + radius.unsqueeze(1)
    gap = d - contact

    if yaw is not None and is_cube is not None:
        # normal of the face that i presents toward j
        n_hat = cube_normal(n_hat, yaw.unsqueeze(2), is_cube.unsqueeze(2))

    v = p / mass.unsqueeze(-1)
    v_rel = v.unsqueeze(2) - v.unsqueeze(1)                    # v_i - v_j
    v_n = (v_rel * n_hat).sum(-1)                              # closing speed

    both = present.unsqueeze(2) * present.unsqueeze(1)
    iu = torch.triu(torch.ones(N, N, device=q.device), diagonal=1)
    touching = (gap <= thresh).float()
    approaching = (v_n < 0).float()
    active = both * iu * touching * approaching

    e = 0.5 * (restitution.unsqueeze(2) + restitution.unsqueeze(1))
    inv_m = (1.0 / mass).unsqueeze(2) + (1.0 / mass).unsqueeze(1)
    J = -(1.0 + e) * v_n / inv_m.clamp_min(1e-9)
    J = J * active                                             # [B,N,N]

    # +J n on i, -J n on j. Equal and opposite => total momentum unchanged.
    dp = (J.unsqueeze(-1) * n_hat).sum(dim=2) - (J.unsqueeze(-1) * n_hat).sum(dim=1)
    return p + dp * present.unsqueeze(-1)
