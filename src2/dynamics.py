"""
dynamics.py  —  CausalVis V3, Phase 1: the physics engine
==========================================================

WHAT THIS IS
------------
A learned world model that rolls a CLEVRER scene forward in time.
You give it where the objects are and how fast they are
moving; it tells you where they will be next. Because it can do that, you
can DELETE an object and ask what happens instead -- which is the thing
V1/V2 structurally could not do.

THE ONE IDEA THAT MAKES THIS WORK
---------------------------------
We do not let a neural network predict motion directly. Instead the network
learns ONE thing: the potential energy V between two objects, as a function
of the DISTANCE between them. Everything else is derived from physics:

    force on an object   =  -(slope of V)      <- not learned, derived
    motion from force    =  leapfrog integrator <- not learned, fixed physics

Two guarantees fall out of this for free -- not as loss penalties, but
structurally, so the model literally cannot violate them:

  1. MOMENTUM IS CONSERVED EXACTLY.
     V depends only on distance, so the force on A from B is exactly equal
     and opposite to the force on B from A (Newton's third law). Add them up
     and they cancel. Total momentum cannot change. Machine precision.

  2. ENERGY STAYS BOUNDED.
     Leapfrog is a symplectic integrator. Its energy error oscillates
     forever instead of growing. A plain MLP or Euler step drifts upward and
     silently invents energy over a long rollout.

Importance of CCD
------------------------------
This is what makes Counterfactual Conservation Drift (CCD) a real measurement.
If the model leaks energy when we delete an object, it cannot be blamed on a
weak regulariser -- the structure forbids leaking on normal rollouts. So a
leak means the model genuinely failed to learn OBJECT-LEVEL physics and only
memorised whole scenes. That is exactly the failure CCD is designed to expose.

THE COLOUR RULE (important)
---------------------------
The physics core NEVER sees colour. Colour is physically meaningless -- it
tells you nothing about how something moves. If we fed it in, the network
would get a perfect object ID and could memorise trajectories instead of
learning forces, which is the very cheat we are trying to detect. Colour
stays in the dataset (the UI needs it to say "the red ball"), but it is
stripped before it reaches the engine.

INITIALISATION TRICK
--------------------
The potential is zero-initialised. So at step 0 the model is free particles
travelling in perfectly straight lines -- which is exactly what CLEVRER
objects do between collisions. The network only has to learn the collisions.
That is a large head start and it trains fast.

RUN
---
    python dynamics.py --data path/to/trajectories_out --epochs 30
    python dynamics.py --selftest        # proves the two guarantees, 5 seconds
"""

import os
import glob
import argparse
import math
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# attrs layout produced by build_trajectories.py (15 numbers):
#   [0]    mass proxy
#   [1]    restitution
#   [2:5]  shape   one-hot (cube, sphere, cylinder)
#   [5:13] colour  one-hot   <-- DELIBERATELY DROPPED
#   [13:15] material one-hot (metal, rubber)
#   [15]    MEASURED radius (from mask pixels, build_trajectories src2)  <-- NEW
# The engine only ever sees the physical ones.
PHYS_IDX = [0, 1, 2, 3, 4, 13, 14, 15]  # 8 physical features (incl. measured radius)
N_PHYS = len(PHYS_IDX)
RADIUS_COL = N_PHYS - 1                  # measured radius is the last stripped feature


def strip_colour(attrs):
    """attrs [..., N, 15] -> physical only [..., N, 7]. Colour never enters."""
    return attrs[..., PHYS_IDX]


def mlp(sizes, act=nn.SiLU, zero_last=False):
    layers = []
    for i in range(len(sizes) - 1):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        if zero_last and i == len(sizes) - 2:
            nn.init.zeros_(lin.weight)      # start with V = 0  -> free particles
            nn.init.zeros_(lin.bias)
        layers.append(lin)
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class HamiltonianDynamics(nn.Module):
    """
    Learns a pairwise potential V(distance) and integrates the resulting
    Newtonian motion with a symplectic leapfrog step.
    """

    def __init__(self, hidden=128, emb=64, dissipative=False):
        super().__init__()
        # DISSIPATION. CLEVRER is simulated in Bullet on a tabletop with ground
        # friction, and we measured the consequence directly: objects lose
        # 0.85% of their speed per frame, 70% of collision-free segments
        # decelerate. A conservative field F = -grad V CANNOT represent that --
        # not "struggles to", cannot, as a matter of arithmetic. This is the
        # D != 0 case of the port-Hamiltonian form; the conservative model is
        # the D = 0 special case, so this completes the architecture rather
        # than abandoning it.
        self.dissipative = dissipative
        # symmetric per-object embedding of PHYSICAL attributes only
        self.attr_enc = mlp([N_PHYS, hidden, emb])

        # small learned correction to the mass prior (metal 1.5 / rubber 1.0).
        # zero-init -> at start, mass == the physical prior exactly.
        self.mass_corr = mlp([emb, hidden, 1], zero_last=True)

        # radius is MEASURED in build_trajectories (Q7), not learned.
        #
        # A bounded correction used to sit on top of the measurement, the way
        # mass does. It was a mistake. Training drove tanh() to -1 for EVERY
        # object -- the ratio came out 0.7000 across 200 clips, i.e. pinned at
        # the floor -- so the model quietly shrank every object by 30% and its
        # internal contact distance fell to 0.082 while the world's is 0.117.
        # That is a loophole: shrinking radius weakens collisions without
        # touching the potential, which lowers average loss because only ~7%
        # of pair-frames are actually in contact. It is the same
        # unidentifiable-radius collapse the project already hit once.
        #
        # The module stays (so every existing checkpoint still loads) but
        # train() zero-inits it and keeps it out of the optimiser, and
        # freeze_radius short-circuits it entirely. Both routes give
        # radius == the measured radius, exactly.
        self.radius_corr = mlp([emb, hidden, 1], zero_last=True)
        self.freeze_radius = False   # train() sets this True

        # THE ONLY THING THE PHYSICS LEARNS: V as a function of the gap between
        # two surfaces. zero-init -> V = 0 -> straight-line motion at step 0.
        self.potential = mlp([4 + emb, hidden, hidden, 1], zero_last=True)

        # Per-object drag coefficient c >= 0, read off the attribute embedding.
        # It is a function of ATTRIBUTES ONLY (material, shape, mass, radius --
        # never position or velocity), so it is a material property, which is
        # what a drag coefficient has to be.
        #
        # It lives on the embedding rather than being a free per-material
        # scalar for one specific reason: step() already receives `e`, and
        # step() is called directly from five places outside rollout()
        # (benchmark_eval, collision_check, dynamics_check, readout, selftest).
        # An optional drag= argument would leave every one of those silently
        # running frictionless physics -- exactly the class of bug that hid the
        # radius collapse for weeks. Computing it inside step() from `e` makes
        # that impossible. drag_report() recovers the interpretable
        # per-material numbers for the writeup.
        #
        # Bias init: softplus(-4.76) = 0.0085, the MEASURED per-frame loss. So
        # the model starts at the measured physics and only has to learn how it
        # differs by material -- the same head start the zero-initialised
        # potential gets.
        self.drag_head = nn.Linear(emb, 1)
        nn.init.zeros_(self.drag_head.weight)
        nn.init.constant_(self.drag_head.bias, -4.76)

    # -- per-object physical properties -------------------------------------
    def properties(self, phys):
        """phys [B, N, 7] -> mass [B,N], radius [B,N], embedding [B,N,emb]

        Also caches this call's restitution (phys[...,1]) for the contact
        impulse. It has to be set HERE, not by the caller: step() receives only
        mass/radius/e/present, so an externally-stashed restitution silently
        goes stale whenever a different caller steps a different number of
        objects -- which is exactly how it broke (a 4-object tensor met a
        5-object rollout). Every caller calls properties() before step(), so
        setting it here keeps the shapes matched by construction."""
        self._restitution = phys[..., 1].detach()
        e = self.attr_enc(phys)
        prior = phys[..., 0]                                  # mass proxy
        mass = prior * (1.0 + 0.5 * torch.tanh(self.mass_corr(e).squeeze(-1)))
        mass = mass.clamp_min(0.05)
        r_meas = phys[..., RADIUS_COL]                    # measured, from pixels
        if self.freeze_radius:
            radius = r_meas.clamp_min(0.01)               # the measurement, full stop
        else:
            radius = r_meas * (1.0 + 0.3 * torch.tanh(self.radius_corr(e).squeeze(-1)))
            radius = radius.clamp_min(0.01)
        return mass, radius, e

    # -- the learned potential energy ---------------------------------------
    def V(self, q, mass, radius, e, present):
        """
        q       [B, N, 2]  positions
        present [B, N]     1 if the object is on screen
        returns scalar-per-batch total potential energy [B]

        V depends ONLY on the distance between two objects. That single fact
        is what buys us exact momentum conservation.
        """
        B, N, _ = q.shape
        r = q.unsqueeze(2) - q.unsqueeze(1)                   # [B,N,N,2]
        d = torch.sqrt((r * r).sum(-1) + 1e-12)               # [B,N,N] distance (NaN-safe at r=0)
        contact = radius.unsqueeze(2) + radius.unsqueeze(1)   # r_i + r_j
        gap = d - contact                                     # <0 means overlapping

        # a few smooth, short-range features of the gap
        feats = torch.stack([
            gap.clamp(-0.5, 0.5),                          # bounded raw gap
            torch.tanh(gap / 0.05),                        # smooth sign of gap
            torch.exp(-(gap.clamp(min=0.0) / 0.08) ** 2),  # decays for gap>0, in (0,1]
            torch.sigmoid(-gap / 0.02),                    # ~1 on overlap, ~0 when far
        ], dim=-1)                                         # [B,N,N,4]

        # symmetric pair embedding: e_i + e_j  (so V(i,j) == V(j,i))
        pair_e = e.unsqueeze(2) + e.unsqueeze(1)              # [B,N,N,emb]

        v = self.potential(torch.cat([feats, pair_e], dim=-1)).squeeze(-1)

        # only count each pair once, only if BOTH objects are present
        both = present.unsqueeze(2) * present.unsqueeze(1)    # [B,N,N]
        iu = torch.triu(torch.ones(N, N, device=q.device), diagonal=1)
        return (v * both * iu).sum(dim=(1, 2))                # [B]

    # -- per-pair potential matrix (for the ledger's per-object PE share) -----
    def pair_potentials(self, q, mass, radius, e, present):
        """Same per-pair potential V_ij as V(), but returned as a [B,N,N]
        matrix (not summed). Used only for reporting a per-object PE share."""
        B, N, _ = q.shape
        r = q.unsqueeze(2) - q.unsqueeze(1)
        d = torch.sqrt((r * r).sum(-1) + 1e-12)
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
        return v * both * iu                                  # [B,N,N], upper-tri only

    # -- forces = -dV/dq  (derived, never learned directly) ------------------
    def forces(self, q, mass, radius, e, present, create_graph=True):
        q = q.requires_grad_(True)
        Vtot = self.V(q, mass, radius, e, present).sum()
        F, = torch.autograd.grad(Vtot, q, create_graph=create_graph)
        F = -F
        F = F * present.unsqueeze(-1)                         # absent -> no force
        # Conservation-safe stabiliser. If forces get huge early in training,
        # rescale EVERY force in a sample by ONE shared scalar. A global rescale
        # keeps each pair equal-and-opposite, so net force stays exactly zero and
        # momentum is still conserved. (The OLD per-component clamp did not: it
        # clipped each object's summed force on its own, which broke the symmetry
        # and caused the 0.118 momentum error.)
        CAP = 50.0
        with torch.no_grad():
            maxf = F.abs().flatten(1).max(dim=1).values.clamp_min(1e-9).view(-1, 1, 1)
            scale = (CAP / maxf).clamp_max(1.0)
        F = F * scale
        return F

    # -- per-object drag coefficient ----------------------------------------
    def drag_coeff(self, e, present):
        """c >= 0 per object, [B,N]. None when the model is conservative."""
        if not self.dissipative:
            return None
        c = torch.nn.functional.softplus(self.drag_head(e).squeeze(-1))
        return c * present

    # -- one conformal-symplectic (leapfrog + exact decay) step -------------
    def step(self, q, p, mass, radius, e, present, dt, F=None, create_graph=True):
        """Strang splitting: half a step of drag, the full conservative
        kick-drift-kick, then the other half of the drag.

        The drag half-steps are the EXACT solution of dp/dt = -c*p over dt/2,
        namely p <- p*exp(-c*dt/2). This is not a stylistic choice. A
        symplectic integrator applied to a dissipative system is no longer
        symplectic, and subtracting a drag FORCE inside the existing leapfrog
        would wreck the bounded-energy behaviour the whole architecture rests
        on -- and would then present as "the drag term did not work" when what
        actually failed was the integrator.

        Momentum: drag is an EXTERNAL force from the table, so total object
        momentum is legitimately no longer conserved. The pairwise interaction
        is still exactly equal-and-opposite; only drag removes momentum, and
        ledger() logs how much. Conservation becomes an auditable balance
        sheet rather than a constant, which is a stronger claim, not a weaker
        one.
        """
        c = self.drag_coeff(e, present)
        decay = None
        _imp = getattr(self, "impulse", False)
        if c is not None:
            decay = torch.exp(-c * (0.5 * dt)).unsqueeze(-1)
            p = p * decay                                     # half drag

        # CONTACT IMPULSE, applied BEFORE the drift.
        #
        # The impulse must use the configuration in which contact was
        # DETECTED. Applying it after the drift evaluates the normal one step
        # late, at a configuration the objects have already moved past -- the
        # same smearing that makes the soft potential wrong, compressed into a
        # single step. Measured: a ball struck along +x had its across-normal
        # component corrupted from 0.0200 to 0.0253, and the struck ball
        # picked up a spurious -0.0055 sideways.
        #
        # Before the drift, the normal matches the geometry that produced the
        # contact, so the velocity splits cleanly: the component ALONG the
        # normal is exchanged, the component ACROSS it is untouched.
        if _imp:
            from impulse import resolve_contacts
            rest = getattr(self, "_restitution", None)
            if rest is None or rest.shape != present.shape:
                if not getattr(self, "_rest_warned", False):
                    print("[impulse] restitution shape mismatch; using 0.65 "
                          "(call properties() before step())")
                    self._rest_warned = True
                rest = torch.full_like(present, 0.65)
            p = resolve_contacts(q, p, mass, radius, present, rest)

        if F is None:
            F = self.forces(q, mass, radius, e, present, create_graph)
        p = p + 0.5 * dt * F                                  # half kick
        q = q + dt * p / mass.unsqueeze(-1)                   # drift
        F_new = self.forces(q, mass, radius, e, present, create_graph)
        p = p + 0.5 * dt * F_new                              # half kick

        if decay is not None:
            p = p * decay                                     # half drag

        return q, p, F_new

    # -- roll the world forward ---------------------------------------------
    def rollout(self, q0, v0, attrs, present, steps, dt=1.0, create_graph=True):
        """
        q0      [B, N, 2]   starting positions
        v0      [B, N, 2]   starting velocities
        attrs   [B, N, 15]  raw attributes (colour stripped inside)
        present [B, N]      who is on screen (from the intervention, if any)

        returns qs [B, steps, N, 2], vs [B, steps, N, 2]
        """
        q0 = torch.nan_to_num(q0)
        v0 = torch.nan_to_num(v0)
        phys = strip_colour(attrs)
        mass, radius, e = self.properties(phys)

        q = q0 * present.unsqueeze(-1)
        p = mass.unsqueeze(-1) * v0 * present.unsqueeze(-1)   # momentum = m*v
        if getattr(self, "impulse", False):
            self._restitution = phys[..., 1]      # attrs[:,1]: metal .8 / rubber .5

        qs, vs = [], []
        F = None
        for _ in range(steps):
            q, p, F = self.step(q, p, mass, radius, e, present, dt, F, create_graph)
            qs.append(q)
            vs.append(p / mass.unsqueeze(-1))
        return torch.stack(qs, 1), torch.stack(vs, 1)

    # -- THE AUDIT LEDGER  (this is the paper's signature output) ------------
    @torch.no_grad()
    def ledger(self, q, v, attrs, present):
        """
        The physical account of a single frame: what energy and momentum each
        object is carrying. Run this on the factual rollout and again on the
        counterfactual one, and the DIFFERENCE is the explanation of why the
        outcome changed. This is the thing SlotPi / C-JEPA cannot hand back.
        """
        phys = strip_colour(attrs)
        mass, radius, e = self.properties(phys)
        p = mass.unsqueeze(-1) * v * present.unsqueeze(-1)

        ke_per_obj = (p ** 2).sum(-1) / (2 * mass)            # [B,N]
        mom_per_obj = p                                       # [B,N,2]
        pot = self.V(q, mass, radius, e, present)             # [B]

        # per-object POTENTIAL SHARE: potential energy belongs to a PAIR, not an
        # object, so there is no exact per-object value. We split each pair's V
        # equally between its two objects -- a labelling convention for the UI,
        # not physics. (Sum of shares == total potential.)
        Vmat = self.pair_potentials(q, mass, radius, e, present)   # [B,N,N]
        pe_share = 0.5 * (Vmat.sum(2) + Vmat.sum(1))               # [B,N]

        # momentum the table takes out per frame, so the ledger balances:
        #   momentum_in - momentum_out = drag + numerical noise (~1e-7)
        c = self.drag_coeff(e, present)
        drag_dp = (p * (1.0 - torch.exp(-c.unsqueeze(-1)))).sum(1) \
            if c is not None else torch.zeros_like(p.sum(1))

        return {
            "kinetic_per_object":  ke_per_obj,
            "momentum_per_object": mom_per_obj,
            "momentum_to_table":   drag_dp,       # zero for a conservative model
            "drag_per_object":     c if c is not None else torch.zeros_like(mass),
            "total_kinetic":       ke_per_obj.sum(-1),
            "total_momentum":      mom_per_obj.sum(1),
            "potential":           pot,
            "potential_share_per_object": pe_share,   # UI only; a convention
            "hamiltonian":         ke_per_obj.sum(-1) + pot,  # H = KE + V
            "mass":                mass,
            "radius":              radius,
        }

    # -- interpretable read-out of what the drag head learned ----------------
    @torch.no_grad()
    def drag_report(self):
        """Per-frame speed retention for one canonical object of each type.

        The drag head reads an embedding, so the learned coefficients are not
        directly legible as parameters. Evaluating it on canonical attribute
        vectors recovers the physical numbers, which is what belongs in a
        writeup: 'metal cubes lose x% per frame, rubber spheres y%'.
        """
        if not self.dissipative:
            return {}
        out = {}
        for shape, si in (("cube", 2), ("sphere", 3), ("cylinder", 4)):
            for material, mi, mass_p in (("metal", 13, 1.5), ("rubber", 14, 1.0)):
                at = torch.zeros(1, 1, 16)
                at[..., 0] = mass_p
                at[..., 1] = 0.6
                at[..., si] = 1.0
                at[..., mi] = 1.0
                at[..., 15] = 0.0585            # measured mean radius
                _, _, e = self.properties(strip_colour(at))
                c = float(torch.nn.functional.softplus(
                    self.drag_head(e).squeeze()).item())
                out[f"{material}_{shape}"] = {
                    "c": c,
                    "loss_per_frame": 1.0 - math.exp(-c),
                    "speed_left_after_40": math.exp(-c * 40),
                }
        return out


# ---------------------------------------------------------------------------
#                                DATA
# ---------------------------------------------------------------------------
class TrajectoryDataset(torch.utils.data.Dataset):
    """Serves (start state, future ground truth) windows from the .npz files.

    WHY THE SAMPLING IS NOT UNIFORM
    -------------------------------
    Uniform start frames were measured on this data: at horizon 20, 40.7% of
    windows contain an annotated collision, which sounds fine -- but only
    6.8% of PAIR-FRAMES are actually overlapping. So even inside a "collision
    window" the loss is ~93% straight-line coasting, and the average is what
    gradient descent optimises. Collisions are the rare event that matters and
    they were being drowned.

    collision_frac of the windows are therefore centred on a real annotated
    collision (the frame indices are already in z["collisions"][:,0]).
    windows_per_clip lets one epoch draw several different windows from the
    same clip, so 16k clips still gives a varied epoch after the restriction.
    """
    def __init__(self, folder, horizon=10, max_objects=8, limit=None, files=None,
                 collision_frac=0.0, windows_per_clip=1):
        # files: an explicit list (from the train split). If None, use everything.
        self.files = files if files is not None else sorted(glob.glob(os.path.join(folder, "*.npz")))
        if limit:
            self.files = self.files[:limit]
        if not self.files:
            raise SystemExit(f"no .npz found -- run build_trajectories.py first, and make_split.py")
        self.horizon = horizon
        self.max_objects = max_objects
        self.collision_frac = collision_frac
        self.windows_per_clip = max(1, int(windows_per_clip))

    def __len__(self):
        return len(self.files) * self.windows_per_clip

    def set_horizon(self, h):
        self.horizon = h            # used by the curriculum

    def __getitem__(self, i):
        z = np.load(self.files[i % len(self.files)], allow_pickle=True)
        pos, vel = z["positions"], z["velocities"]
        pres, attrs = z["presence"], z["attrs"]
        T, N, _ = pos.shape
        H = self.horizon

        # pick a start frame where at least 2 objects are on screen for the
        # whole window (otherwise there is no interaction to learn from)
        ok = [t for t in range(1, T - H - 1)
              if (pres[t:t + H + 1].min(axis=0) > 0).sum() >= 2]
        t0 = int(np.random.choice(ok)) if ok else 1

        # ...and with probability collision_frac, insist that a real collision
        # lands inside the supervised window (t0, t0+H]. Falls back to the
        # uniform draw above whenever no such start frame exists.
        if ok and self.collision_frac > 0.0 and np.random.rand() < self.collision_frac:
            col = z["collisions"]
            if len(col):
                tc = int(col[np.random.randint(len(col))][0])
                centred = [t for t in ok if t < tc <= t + H]
                if centred:
                    t0 = int(np.random.choice(centred))

        keep = pres[t0:t0 + H + 1].min(axis=0) > 0     # objects present throughout
        idx = np.where(keep)[0][: self.max_objects]
        M = self.max_objects

        def pad2(a):                                    # [.., n, 2] -> [.., M, 2]
            out = np.zeros(a.shape[:-2] + (M, 2), np.float32)
            out[..., : len(idx), :] = a[..., idx, :]
            return out

        q0 = np.zeros((M, 2), np.float32); q0[: len(idx)] = pos[t0, idx]
        v0 = np.zeros((M, 2), np.float32); v0[: len(idx)] = vel[t0, idx]
        at = np.zeros((M, attrs.shape[1]), np.float32); at[: len(idx)] = attrs[idx]
        pr = np.zeros((M,), np.float32); pr[: len(idx)] = 1.0

        qf = pad2(pos[t0 + 1: t0 + 1 + H])              # [H, M, 2] future truth
        vf = pad2(vel[t0 + 1: t0 + 1 + H])

        return (torch.from_numpy(q0), torch.from_numpy(v0),
                torch.from_numpy(at), torch.from_numpy(pr),
                torch.from_numpy(qf), torch.from_numpy(vf))


# ---------------------------------------------------------------------------
#                                TRAIN
# ---------------------------------------------------------------------------
RADIUS_COL_RAW = PHYS_IDX[RADIUS_COL]        # measured radius in the RAW attrs (15)


def contact_weights(qf, at, pr, lam, sigma):
    """Per-frame loss weight, computed from GROUND TRUTH only (no gradient).

    Returns w [B,H,1,1]: about 1 while everything is coasting, about 1+lam on
    the frames where two objects are actually touching. Only ~7% of pair-frames
    overlap, so without this the collision -- the only part of the trajectory
    the potential is responsible for -- contributes ~7% of the loss and the
    optimiser correctly ignores it. lam=9 lifts contact to roughly 40% of the
    signal, which is where the useful gradient lives.

    This reweights the OBJECTIVE only. It touches no force, so momentum
    conservation is untouched.
    """
    if lam <= 0.0:
        return torch.ones(qf.shape[0], qf.shape[1], 1, 1, device=qf.device)
    with torch.no_grad():
        r = at[..., RADIUS_COL_RAW]                            # [B,M] measured
        contact = r[:, None, :, None] + r[:, None, None, :]    # [B,1,M,M]
        d = torch.cdist(qf, qf)                                # [B,H,M,M]
        gap = d - contact
        M = pr.shape[1]
        valid = (pr[:, None, :, None] * pr[:, None, None, :]
                 * torch.triu(torch.ones(M, M, device=qf.device), diagonal=1))
        gap = torch.where(valid > 0, gap, torch.full_like(gap, 1e3))
        gmin = gap.amin(dim=(-1, -2))                          # [B,H] closest pair
        w = 1.0 + lam * torch.exp(-(gmin.clamp_min(0.0) / sigma) ** 2)
    return w[:, :, None, None]


def rollout_position_error(model, dl, H, dev, max_batches=None):
    """Unweighted mean L2 position error after H frames -- the same quantity
    dynamics_check.py reports, so training numbers and diagnostic numbers are
    directly comparable. NOT wrapped in no_grad: forces are a derivative."""
    model.eval()
    tot, n = 0.0, 0
    for bi, (q0, v0, at, pr, qf, vf) in enumerate(dl):
        if max_batches is not None and bi >= max_batches:
            break
        q0, v0, at, pr = q0.to(dev), v0.to(dev), at.to(dev), pr.to(dev)
        qf = qf.to(dev)
        qs, _ = model.rollout(q0, v0, at, pr, steps=H, create_graph=False)
        err = (qs[:, -1] - qf[:, -1]).norm(dim=-1)             # [B,M]
        live = pr > 0
        if live.any():
            tot += float(err[live].sum().detach()); n += int(live.sum())
    model.train()
    return tot / max(n, 1)


def train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.threads:
        torch.set_num_threads(args.threads)

    train_files = val_files = None
    if args.split:
        from splits import load_split
        train_files = load_split(args.split, "train")
        val_files = load_split(args.split, "val")[:args.val_clips]
        print(f"training on the TRAIN split only: {len(train_files)} clips "
              f"(val/test are held out)")

    ds = TrajectoryDataset(args.data, horizon=3, limit=args.limit, files=train_files,
                           collision_frac=args.collision_frac,
                           windows_per_clip=args.windows_per_clip)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                     num_workers=0, drop_last=True)
    val_ds = val_dl = None
    if val_files:
        val_ds = TrajectoryDataset(args.data, horizon=20, files=val_files)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch,
                                             shuffle=False, drop_last=False)

    model = HamiltonianDynamics(dissipative=args.dissipative).to(dev)
    model.impulse = args.impulse
    if args.impulse:
        print("contact: IMPULSE along the normal (restitution active)")
    if args.dissipative:
        print("dissipation: ON (conformal-symplectic drag, per-object c >= 0)")
    if args.resume and os.path.isfile(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=dev))
        print(f"resumed from {args.resume}")

    # THE RADIUS FIX. Zero the correction and keep it out of the optimiser, so
    # radius is exactly the measured radius for the whole run and cannot drift
    # to a smaller, easier world. freeze_radius short-circuits it as well; the
    # module is left in place only so existing checkpoints still load.
    trainable = list(model.parameters())
    if args.freeze_radius:
        model.freeze_radius = True
        for p_ in model.radius_corr.parameters():
            nn.init.zeros_(p_)
        frozen = {id(p_) for p_ in model.radius_corr.parameters()}
        trainable = [p_ for p_ in model.parameters() if id(p_) not in frozen]
        for p_ in model.radius_corr.parameters():
            p_.requires_grad_(False)
        print("radius: FROZEN at the measured value (radius_corr disabled)")
    else:
        print("radius: learnable correction ACTIVE (this is the old, broken setting)")

    opt = torch.optim.AdamW(trainable, lr=args.lr)

    print(f"device={dev}  threads={torch.get_num_threads()}  windows/epoch={len(ds)}")
    print(f"collision-centred windows: {args.collision_frac:.0%}   "
          f"contact loss weight: {args.contact_weight:g}x")

    best = float("inf")
    for ep in range(args.epochs):
        # CURRICULUM: start by predicting 3 frames, work up to 20.
        # Long rollouts from scratch are unstable; short ones teach the
        # collision response first.
        H = min(3 + ep, 20)
        ds.set_horizon(H)

        tot, n, skipped = 0.0, 0, 0
        for q0, v0, at, pr, qf, vf in dl:
            q0, v0, at, pr = q0.to(dev), v0.to(dev), at.to(dev), pr.to(dev)
            qf, vf = qf.to(dev), vf.to(dev)

            qs, vs = model.rollout(q0, v0, at, pr, steps=H)
            m = pr[:, None, :, None]                     # mask absent objects
            if args.legacy_loss:
                # EXACTLY the objective v3_2 was trained with. Needed to isolate
                # one change at a time: the weighted mean below divides by the
                # LIVE object count instead of all 8 padded slots, which alone
                # makes the loss ~2.3x larger and so changes how hard the fixed
                # grad-clip of 1.0 bites. Comparing against v3_2 without this
                # switch confounds the objective with its normalisation.
                loss = ((((qs - qf) * m) ** 2).mean()
                        + 0.1 * (((vs - vf) * m) ** 2).mean())
            else:
                w = contact_weights(qf, at, pr, args.contact_weight, args.contact_sigma)
                wm = w * m                               # [B,H,M,1]
                denom = (wm.sum() * 2).clamp_min(1e-9)   # *2 for the x,y components
                loss = ((wm * ((qs - qf) * m) ** 2).sum() / denom
                        + 0.1 * (wm * ((vs - vf) * m) ** 2).sum() / denom)
            if not torch.isfinite(loss):
                opt.zero_grad(); skipped += 1; continue   # don't poison weights
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            tot += loss.item(); n += 1

        # report drift on the last batch: this is the CCD instrument, warming up
        with torch.no_grad():
            L0 = model.ledger(q0, v0, at, pr)
            L1 = model.ledger(qs[:, -1], vs[:, -1], at, pr)
            dH = (L1["hamiltonian"] - L0["hamiltonian"]).abs().mean().item()
            dP = (L1["total_momentum"] - L0["total_momentum"]).norm(dim=-1).mean().item()
            # radius sanity: must read 1.0000 every epoch, or the loophole is open
            rr = (L1["radius"][pr > 0]
                  / at[..., RADIUS_COL_RAW][pr > 0].clamp_min(1e-9)).mean().item()

        line = (f"epoch {ep:3d}  horizon {H:2d}  loss {tot/max(n,1):.6f}   "
                f"|dH| {dH:.2e}   |dP| {dP:.2e}   r/r_meas {rr:.4f}")

        if val_dl is not None and (ep % args.val_every == 0 or ep == args.epochs - 1):
            ve = rollout_position_error(model, val_dl, 20, dev, args.val_batches)
            line += f"   val_err@20 {ve:.4f}"
            if ve < best:
                best = ve
                torch.save(model.state_dict(), args.out)
                line += "  <- best, saved"
        else:
            torch.save(model.state_dict(), args.out)
        if skipped:
            line += f"   [{skipped} non-finite batches skipped]"
        print(line, flush=True)

    print(f"\nsaved -> {args.out}"
          + (f"   best val position error @20 frames: {best:.4f}"
             if best < float("inf") else ""))


# ---------------------------------------------------------------------------
#                     SELF-TEST: prove the two guarantees
# ---------------------------------------------------------------------------
def selftest():
    """
    Proves the two structural guarantees with a mild random potential (so real
    forces exist) and a stable step size. This is a physics check of the
    integrator; its dt is independent of the dt used in training.
    """
    torch.manual_seed(0)
    model = HamiltonianDynamics()
    for p_ in model.potential[-1].parameters():
        nn.init.normal_(p_, std=0.1)                 # gentle, not std=0.3

    B, N = 1, 5
    q = torch.rand(B, N, 2) * 0.6 + 0.2
    v = torch.randn(B, N, 2) * 0.05                   # REAL velocity -> H0 not ~0
    attrs = torch.zeros(B, N, 16)
    attrs[..., 0] = torch.tensor([1.5, 1.0, 1.0, 1.5, 1.0])
    attrs[..., 1] = 0.6
    attrs[..., 3] = 1.0
    attrs[..., 13] = 1.0
    pres = torch.ones(B, N)

    phys = strip_colour(attrs)
    mass, radius, e = model.properties(phys)
    p = mass.unsqueeze(-1) * v
    dt = 0.05

    with torch.no_grad():
        H0 = (p ** 2).sum(-1).div(2 * mass).sum(-1) + model.V(q, mass, radius, e, pres)
        P0 = p.sum(1).clone()

    H_hist, dP = [], []
    for _ in range(500):
        q, p, _ = model.step(q.detach(), p.detach(), mass, radius, e, pres,
                             dt=dt, F=None, create_graph=False)
        with torch.no_grad():
            H = (p ** 2).sum(-1).div(2 * mass).sum(-1) + model.V(q, mass, radius, e, pres)
            H_hist.append(H.item())
            dP.append((p.sum(1) - P0).abs().max().item())

    H_hist, dP = np.array(H_hist), np.array(dP)
    rel = np.abs(H_hist - H0.item()) / (np.abs(H_hist).mean() + 1e-12)
    mom_err = dP.max()
    worst = rel.max()
    e1, e2 = rel[:250].mean(), rel[250:].mean()
    mom_ok = mom_err < 1e-3
    # Energy is "bounded" if it never drifts more than 1% of the mean energy.
    # Leapfrog sits near 1e-5 here; a non-symplectic (Euler) step would blow far
    # past 1e-2 and keep climbing. The half-vs-half means are printed for info
    # only -- at the 1e-5 floor their ratio is just oscillation noise.
    eng_ok = worst < 1e-2

    print("=" * 60)
    print("GUARANTEE 1  momentum conserved (Newton's 3rd law, structural)")
    print(f"   worst momentum error over 500 steps : {mom_err:.3e}")
    print(f"   -> {'PASS' if mom_ok else 'FAIL - check forces()'}")
    print()
    print("GUARANTEE 2  energy bounded, no drift (symplectic leapfrog)")
    print(f"   initial energy H0                    : {H0.item():.4f}")
    print(f"   worst energy drift / mean|H|         : {worst:.3e}   (want < 1e-2)")
    print(f"   mean drift, steps 1-250 / 251-500    : {e1:.3e} / {e2:.3e}  (info only)")
    print(f"   -> {'PASS (bounded, ~1e-5 is symplectic)' if eng_ok else 'FAIL - energy drifting'}")
    print("=" * 60)
    print("OVERALL:", "PASS - physics core sound" if (mom_ok and eng_ok)
          else "FAIL - do not proceed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="legacy/v1/data/trajectories_out")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="v3_2_dynamics.pt")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default=None,
                    help="path to split.json from make_split.py; trains on train split only")
    # --- the fix -----------------------------------------------------------
    ap.add_argument("--collision-frac", type=float, default=0.5,
                    help="fraction of training windows centred on a real collision")
    ap.add_argument("--contact-weight", type=float, default=9.0,
                    help="extra loss weight on frames where objects are touching "
                         "(0 = the old uniform average)")
    ap.add_argument("--contact-sigma", type=float, default=0.03,
                    help="gap distance over which the contact weight decays")
    ap.add_argument("--impulse", action="store_true",
                    help="resolve contacts with a textbook impulse along the "
                         "contact normal instead of relying on the potential's "
                         "soft push (fixes the bounce angle; restitution finally "
                         "does something)")
    ap.add_argument("--dissipative", action="store_true",
                    help="learn a per-object drag coefficient and integrate it "
                         "conformally (port-Hamiltonian D != 0)")
    ap.add_argument("--legacy-loss", action="store_true",
                    help="use v3_2's exact loss (plain .mean() over padded slots) "
                         "so a single change can be isolated")
    ap.add_argument("--windows-per-clip", type=int, default=1,
                    help="windows drawn per clip per epoch")
    ap.add_argument("--freeze-radius", dest="freeze_radius", action="store_true",
                    default=True, help="use the measured radius (default, correct)")
    ap.add_argument("--learn-radius", dest="freeze_radius", action="store_false",
                    help="re-enable the collapsing radius correction (the old bug)")
    # --- monitoring --------------------------------------------------------
    ap.add_argument("--val-clips", type=int, default=200)
    ap.add_argument("--val-batches", type=int, default=None)
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0,
                    help="torch CPU threads; 0 leaves the default")
    ap.add_argument("--resume", default=None)
    a = ap.parse_args()
    selftest() if a.selftest else train(a)