"""
v5/world.py  —  a world model that reasons about shape and orientation
=======================================================================

WHAT v3 COULD NOT DO
--------------------
v3's physics sees every object as a disc. Contact happens at r_i + r_j no
matter what the objects are or which way they face. That is exact for
spheres and upright cylinders (circular cross-section) and wrong for cubes:
a cube reaches s at a face and s*sqrt(2) at a corner, so r_i + r_j
understates a cube-cube contact by up to 41%. v3 also has no rotational
state at all -- it cannot represent a cube turning, and measured data shows
they do (74.1 -> 87.4 degrees across one collision in sim_00000).

WHAT THIS ADDS, AND WHAT IT KEEPS
---------------------------------
Kept, unchanged: the learned pairwise potential, forces by autograd,
symplectic leapfrog, exact momentum conservation of the interaction, and the
conformal-symplectic drag that produced the only measured improvement so far
(rollout error -25%, collision error -28%).

Added:
  1. SHAPE-AWARE CONTACT. The gap between two objects is measured along the
     line joining them with the true support function of each body:
         gap = |q_j - q_i| - h_i(phi) - h_j(phi)
     where h is the radius for round shapes and, for a cube of half-side s
     facing direction phi at yaw theta,
         h(phi) = s * (|cos(phi - theta)| + |sin(phi - theta)|).
     So the potential now sees a cube as a square, and a collision depends
     on which face or corner it presents.
  2. ROTATIONAL STATE. Each object carries yaw and angular velocity omega.
     Because the potential depends on yaw, torque is its negative gradient
     with respect to yaw -- by the SAME autograd call that gives the force.
     Nothing is hand-derived, and because V is invariant under a rigid
     rotation of the whole scene, total angular momentum (orbital + spin) is
     conserved by the interaction exactly as linear momentum is.
  3. ROTATIONAL DRAG, learned per object like translational drag, so a
     spinning cube slows the way the data says it does.

HONESTY ABOUT ORIENTATION DATA
------------------------------
`data/trajectories_3d` was built without recovered yaw (it costs ~17 h for the
training split; that run is in progress -> `data/trajectories_3d_yaw`). Until it
lands, yaw is UNKNOWN, and the model must not pretend otherwise: with
yaw_known=False the contact uses each cube's direction-averaged support
(s*4/pi, which is what attrs[:,15] stores), the potential has no yaw
dependence, torque is identically zero, and the rotational state is inert.
Set yaw_known=True only on data that actually contains fitted yaw.

The support function uses sqrt(cos^2 + eps) + sqrt(sin^2 + eps) rather than
absolute values so it stays differentiable at the corners; the kink is
physically real but a non-smooth gradient there produces NaN torques.
"""

import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src2"))
from dynamics import HamiltonianDynamics, strip_colour, RADIUS_COL   # noqa: E402

CUBE_MEAN_SUPPORT = 4.0 / math.pi     # matches fit3d / fix_cube_radius
SMOOTH_EPS = 1e-4
# moment of inertia about the vertical axis, as k * m * r^2 with r the
# stored (averaged-support) radius. sphere 2/5 m r^2; cylinder 1/2 m r^2;
# cube of half-side s: m (2s)^2 / 6 = (2/3) m s^2 with s = r / (4/pi).
INERTIA_K = {"cube": (2.0 / 3.0) / CUBE_MEAN_SUPPORT ** 2,   # ~0.411
             "sphere": 0.4, "cylinder": 0.5}


class SpatialWorld(HamiltonianDynamics):
    """HamiltonianDynamics plus shape-aware contact and rotational dynamics."""

    def __init__(self, hidden=128, emb=64, dissipative=True, yaw_known=False):
        super().__init__(hidden=hidden, emb=emb, dissipative=dissipative)
        self.yaw_known = yaw_known
        # rotational drag, same construction and same measured-physics init
        # as the translational drag head
        self.rot_drag_head = nn.Linear(emb, 1)
        nn.init.zeros_(self.rot_drag_head.weight)
        nn.init.constant_(self.rot_drag_head.bias, -4.76)

    # -- geometry ---------------------------------------------------------
    @staticmethod
    def inertia(phys, mass, radius):
        """[B,N] moment of inertia about the vertical axis."""
        k = (phys[..., 2] * INERTIA_K["cube"]
             + phys[..., 3] * INERTIA_K["sphere"]
             + phys[..., 4] * INERTIA_K["cylinder"])
        k = torch.where(k > 0, k, torch.full_like(k, 0.4))
        return (k * mass * radius ** 2).clamp_min(1e-6)

    def support(self, phys_k, radius_k, yaw_k, phi):
        """Reach of object k's surface in direction phi. Broadcasts over the
        pair grid: phys_k/radius_k/yaw_k are [B,N,1] or [B,1,N], phi [B,N,N]."""
        if not self.yaw_known:
            return radius_k                       # averaged support, no yaw
        is_cube = phys_k[..., 2]
        s = radius_k / CUBE_MEAN_SUPPORT          # half-side from stored radius
        a = phi - yaw_k
        h_cube = s * (torch.sqrt(torch.cos(a) ** 2 + SMOOTH_EPS)
                      + torch.sqrt(torch.sin(a) ** 2 + SMOOTH_EPS))
        return is_cube * h_cube + (1.0 - is_cube) * radius_k

    # -- the learned potential, now shape- and orientation-aware ----------
    def V(self, q, mass, radius, e, present, phys=None, yaw=None):
        B, N, _ = q.shape
        r = q.unsqueeze(2) - q.unsqueeze(1)                  # [B,N,N,2] i->j
        d = torch.sqrt((r * r).sum(-1) + 1e-12)
        if self.yaw_known and phys is not None and yaw is not None:
            # atan2's gradient is (-y, x)/(x^2+y^2): NaN at r = 0. Padded absent
            # objects sit at q = 0, so absent-absent pairs are exactly there,
            # and NaN * 0 from the presence mask is still NaN in the backward.
            # Give every masked pair a fixed unit direction before atan2.
            # ...and the DIAGONAL: pair (i,i) has r = 0 with both 'present', so
            # the presence mask alone lets it through. Exclude it explicitly.
            off = 1.0 - torch.eye(N, device=q.device)
            both_ = (present.unsqueeze(2) * present.unsqueeze(1) * off).unsqueeze(-1)
            safe = torch.zeros_like(r); safe[..., 0] = 1.0
            r_dir = torch.where(both_ > 0, r, safe)
            phi = torch.atan2(r_dir[..., 1], r_dir[..., 0])
            h_i = self.support(phys.unsqueeze(2), radius.unsqueeze(2),
                               yaw.unsqueeze(2), phi)
            h_j = self.support(phys.unsqueeze(1), radius.unsqueeze(1),
                               yaw.unsqueeze(1), phi)      # |cos|,|sin| are pi-periodic
            contact = h_i + h_j
        else:
            contact = radius.unsqueeze(2) + radius.unsqueeze(1)
        gap = d - contact
        feats = torch.stack([
            gap.clamp(-0.5, 0.5),
            torch.tanh(gap / 0.05),
            torch.exp(-(gap.clamp(min=0.0) / 0.08) ** 2),
            torch.sigmoid(-gap / 0.02),
        ], dim=-1)
        pair_e = e.unsqueeze(2) + e.unsqueeze(1)
        v = self.potential(torch.cat([feats, pair_e], dim=-1)).squeeze(-1)
        both = present.unsqueeze(2) * present.unsqueeze(1)
        iu = torch.triu(torch.ones(N, N, device=q.device), diagonal=1)
        return (v * both * iu).sum(dim=(1, 2))

    # -- force AND torque from one autograd call --------------------------
    def forces_torques(self, q, yaw, mass, radius, e, present, phys,
                       create_graph=True):
        q = q.requires_grad_(True)
        need_tau = self.yaw_known and yaw is not None
        if need_tau:
            yaw = yaw.requires_grad_(True)
        Vtot = self.V(q, mass, radius, e, present, phys, yaw).sum()
        if need_tau:
            gq, gy = torch.autograd.grad(Vtot, [q, yaw], create_graph=create_graph,
                                         allow_unused=True)
            tau = -gy if gy is not None else torch.zeros_like(yaw)
        else:
            gq, = torch.autograd.grad(Vtot, q, create_graph=create_graph)
            tau = torch.zeros_like(mass)
        F = -gq * present.unsqueeze(-1)
        tau = tau * present
        # one SHARED rescale keeps every pair equal-and-opposite (see v3 notes)
        CAP = 50.0
        with torch.no_grad():
            maxf = F.abs().flatten(1).max(dim=1).values.clamp_min(1e-9).view(-1, 1, 1)
            scale = (CAP / maxf).clamp_max(1.0)
        return F * scale, tau * scale.view(-1, 1)

    def rot_drag_coeff(self, e, present):
        if not self.dissipative:
            return None
        return torch.nn.functional.softplus(self.rot_drag_head(e).squeeze(-1)) * present

    # -- one conformal-symplectic step over (q, p, yaw, L) ------------------
    def step_spatial(self, q, p, yaw, L, mass, radius, e, present, phys, dt,
                     create_graph=True):
        """L is angular momentum I*omega, so the rotational half mirrors the
        translational one exactly: half drag, half kick, drift, half kick,
        half drag."""
        I = self.inertia(phys, mass, radius)
        c = self.drag_coeff(e, present)
        cr = self.rot_drag_coeff(e, present)
        if c is not None:
            p = p * torch.exp(-c * (0.5 * dt)).unsqueeze(-1)
            L = L * torch.exp(-cr * (0.5 * dt))
        F, tau = self.forces_torques(q, yaw, mass, radius, e, present, phys, create_graph)
        p = p + 0.5 * dt * F
        L = L + 0.5 * dt * tau
        q = q + dt * p / mass.unsqueeze(-1)
        yaw = yaw + dt * L / I
        F2, tau2 = self.forces_torques(q, yaw, mass, radius, e, present, phys, create_graph)
        p = p + 0.5 * dt * F2
        L = L + 0.5 * dt * tau2
        if c is not None:
            p = p * torch.exp(-c * (0.5 * dt)).unsqueeze(-1)
            L = L * torch.exp(-cr * (0.5 * dt))
        return q, p, yaw, L

    # -- keep the v3 signature working for every existing caller ----------
    def step(self, q, p, mass, radius, e, present, dt, F=None, create_graph=True):
        """v3-compatible: no orientation state. Averaged contact, no torque."""
        saved = self.yaw_known
        self.yaw_known = False
        try:
            return super().step(q, p, mass, radius, e, present, dt, F, create_graph)
        finally:
            self.yaw_known = saved

    def rollout_spatial(self, q0, v0, yaw0, omega0, attrs, present, steps,
                        dt=1.0, create_graph=True):
        """Full state rollout. Returns qs, vs, yaws, omegas, each [B,steps,N,...]."""
        phys = strip_colour(torch.nan_to_num(attrs))
        mass, radius, e = self.properties(phys)
        I = self.inertia(phys, mass, radius)
        q = torch.nan_to_num(q0) * present.unsqueeze(-1)
        p = mass.unsqueeze(-1) * torch.nan_to_num(v0) * present.unsqueeze(-1)
        yaw = torch.nan_to_num(yaw0) * present
        L = I * torch.nan_to_num(omega0) * present
        qs, vs, ys, ws = [], [], [], []
        for _ in range(steps):
            q, p, yaw, L = self.step_spatial(q, p, yaw, L, mass, radius, e,
                                             present, phys, dt, create_graph)
            qs.append(q); vs.append(p / mass.unsqueeze(-1))
            ys.append(yaw); ws.append(L / I)
        return (torch.stack(qs, 1), torch.stack(vs, 1),
                torch.stack(ys, 1), torch.stack(ws, 1))


def load_spatial(path, yaw_known=None):
    """Load a SpatialWorld checkpoint (or a v3 checkpoint into one)."""
    sd = torch.load(path, map_location="cpu")
    has_drag = any(k.startswith("drag_head.") for k in sd)
    m = SpatialWorld(dissipative=has_drag)
    if yaw_known is not None:
        m.yaw_known = yaw_known
    own = m.state_dict()
    for k, v in own.items():
        if k not in sd:
            sd[k] = v            # a v3 file lacks the rotational heads; seed them
    m.load_state_dict(sd)
    m.eval()
    return m
