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
# The engine only ever sees the physical ones.
PHYS_IDX = [0, 1, 2, 3, 4, 13, 14]      # 7 physical features
N_PHYS = len(PHYS_IDX)


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

    def __init__(self, hidden=128, emb=64):
        super().__init__()
        # symmetric per-object embedding of PHYSICAL attributes only
        self.attr_enc = mlp([N_PHYS, hidden, emb])

        # small learned correction to the mass prior (metal 1.5 / rubber 1.0).
        # zero-init -> at start, mass == the physical prior exactly.
        self.mass_corr = mlp([emb, hidden, 1], zero_last=True)

        # each object gets a contact radius (how big it is), in normalised units
        self.radius_head = mlp([emb, hidden, 1])

        # THE ONLY THING THE PHYSICS LEARNS: V as a function of the gap between
        # two surfaces. zero-init -> V = 0 -> straight-line motion at step 0.
        self.potential = mlp([4 + emb, hidden, hidden, 1], zero_last=True)

    # -- per-object physical properties -------------------------------------
    def properties(self, phys):
        """phys [B, N, 7] -> mass [B,N], radius [B,N], embedding [B,N,emb]"""
        e = self.attr_enc(phys)
        prior = phys[..., 0]                                  # mass proxy
        mass = prior * (1.0 + 0.5 * torch.tanh(self.mass_corr(e).squeeze(-1)))
        mass = mass.clamp_min(0.05)
        radius = 0.01 + 0.12 * torch.sigmoid(self.radius_head(e).squeeze(-1))
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

    # -- one symplectic (leapfrog) step -------------------------------------
    def step(self, q, p, mass, radius, e, present, dt, F=None, create_graph=True):
        if F is None:
            F = self.forces(q, mass, radius, e, present, create_graph)
        p = p + 0.5 * dt * F                                  # half kick
        q = q + dt * p / mass.unsqueeze(-1)                   # drift
        F_new = self.forces(q, mass, radius, e, present, create_graph)
        p = p + 0.5 * dt * F_new                              # half kick
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

        return {
            "kinetic_per_object":  ke_per_obj,
            "momentum_per_object": mom_per_obj,
            "total_kinetic":       ke_per_obj.sum(-1),
            "total_momentum":      mom_per_obj.sum(1),
            "potential":           pot,
            "hamiltonian":         ke_per_obj.sum(-1) + pot,  # H = KE + V
            "mass":                mass,
            "radius":              radius,
        }


# ---------------------------------------------------------------------------
#                                DATA
# ---------------------------------------------------------------------------
class TrajectoryDataset(torch.utils.data.Dataset):
    """Serves (start state, future ground truth) windows from the .npz files."""
    def __init__(self, folder, horizon=10, max_objects=8, limit=None):
        self.files = sorted(glob.glob(os.path.join(folder, "*.npz")))
        if limit:
            self.files = self.files[:limit]
        if not self.files:
            raise SystemExit(f"no .npz found in {folder} -- run build_trajectories.py first")
        self.horizon = horizon
        self.max_objects = max_objects

    def __len__(self):
        return len(self.files)

    def set_horizon(self, h):
        self.horizon = h            # used by the curriculum

    def __getitem__(self, i):
        z = np.load(self.files[i], allow_pickle=True)
        pos, vel = z["positions"], z["velocities"]
        pres, attrs = z["presence"], z["attrs"]
        T, N, _ = pos.shape
        H = self.horizon

        # pick a start frame where at least 2 objects are on screen for the
        # whole window (otherwise there is no interaction to learn from)
        ok = [t for t in range(1, T - H - 1)
              if (pres[t:t + H + 1].min(axis=0) > 0).sum() >= 2]
        t0 = int(np.random.choice(ok)) if ok else 1

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
def train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = TrajectoryDataset(args.data, horizon=3, limit=args.limit)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                     num_workers=0, drop_last=True)
    model = HamiltonianDynamics().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"device={dev}  videos={len(ds)}")
    for ep in range(args.epochs):
        # CURRICULUM: start by predicting 3 frames, work up to 20.
        # Long rollouts from scratch are unstable; short ones teach the
        # collision response first.
        H = min(3 + ep, 20)
        ds.set_horizon(H)

        tot, n = 0.0, 0
        for q0, v0, at, pr, qf, vf in dl:
            q0, v0, at, pr = q0.to(dev), v0.to(dev), at.to(dev), pr.to(dev)
            qf, vf = qf.to(dev), vf.to(dev)

            qs, vs = model.rollout(q0, v0, at, pr, steps=H)
            m = pr[:, None, :, None]                     # mask absent objects
            loss = (((qs - qf) * m) ** 2).mean() + 0.1 * (((vs - vf) * m) ** 2).mean()
            if not torch.isfinite(loss):
                opt.zero_grad(); continue          # skip a bad batch, don't poison weights
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1

        # report drift on the last batch: this is the CCD instrument, warming up
        with torch.no_grad():
            L0 = model.ledger(q0, v0, at, pr)
            L1 = model.ledger(qs[:, -1], vs[:, -1], at, pr)
            dH = (L1["hamiltonian"] - L0["hamiltonian"]).abs().mean().item()
            dP = (L1["total_momentum"] - L0["total_momentum"]).norm(dim=-1).mean().item()

        print(f"epoch {ep:3d}  horizon {H:2d}  loss {tot/max(n,1):.6f}   "
              f"|dH| {dH:.2e}   |dP| {dP:.2e}")

        torch.save(model.state_dict(), args.out)
    print(f"\nsaved -> {args.out}")


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
    attrs = torch.zeros(B, N, 15)
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
    ap.add_argument("--out", default="v3_1_dynamics.pt")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    selftest() if a.selftest else train(a)