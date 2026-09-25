"""
ccd_eval.py  —  CausalVis V3: Counterfactual Conservation Drift
===============================================================

Measures how much the physics (energy, momentum) drifts as we remove more and
more objects from a scene and re-simulate. A model that learned OBJECT-LEVEL
physics stays flat; one that only fakes it drifts as the scene is edited.

Your structured model conserves both quantities BY CONSTRUCTION, so its curve is
flat on the floor. To show what "without the structure" looks like, we run the
SAME forces through two ablations at inference (no retraining):

    symplectic (yours)  : leapfrog + equal-and-opposite forces   -> conserves
    euler   (energy off): plain Euler step, same forces          -> energy drifts
    broken  (momentum off): forces NOT equal-and-opposite        -> momentum leaks

X-axis: number of objects removed (0,1,2,3). Y-axis: drift over the rollout.

    python ccd_eval.py --data data/trajectories_out --model v3_1_dynamics.pt \
           --limit 40 --steps 40
"""

import argparse
import glob
import os
import random
import numpy as np
import torch

from dynamics import HamiltonianDynamics, strip_colour


def roll(model, q0, v0, attrs, present, steps, mode, dt=1.0):
    """Roll forward under one of three physics modes; return per-step energy and
    total momentum. Grad is on (forces need it); state detached each step."""
    phys = strip_colour(attrs.unsqueeze(0))
    mass, radius, e = model.properties(phys)
    mass, radius, e = mass.detach(), radius.detach(), e.detach()
    pres = present.unsqueeze(0)

    # The broken ablation must act on an object that is PRESENT and actually
    # feels a force.  Hardcoding index 0 silently switches the ablation off
    # whenever object 0 is the deleted one, which drags the broken curve down
    # towards the symplectic one and understates the gap.
    _probe = model.forces(q0.unsqueeze(0) * pres.unsqueeze(-1), mass, radius, e,
                          pres, create_graph=False).detach()
    _mag = _probe[0].norm(dim=-1) * present
    break_idx = int(torch.argmax(_mag)) if float(_mag.max()) > 0 else 0

    def forces(qcur, broken):
        F = model.forces(qcur, mass, radius, e, pres, create_graph=False)
        if broken:                       # break equal-and-opposite on one object
            F = F.clone()
            F[:, break_idx, :] = F[:, break_idx, :] * 1.5
        return F.detach()

    q = q0.unsqueeze(0) * pres.unsqueeze(-1)
    p = mass.unsqueeze(-1) * v0.unsqueeze(0) * pres.unsqueeze(-1)
    # baseline is the state BEFORE any step; using the post-first-step state
    # hides whatever leaks during that first step
    H0 = float(((p ** 2).sum(-1).div(2 * mass).sum(-1)
                + model.V(q, mass, radius, e, pres)).item())
    P0 = p.sum(1)[0].detach().clone()
    Hs, Ps = [H0], [P0]
    for _ in range(steps):
        q, p = q.detach(), p.detach()
        if mode == "symplectic":
            F = forces(q, False); p = p + 0.5 * dt * F
            q = q + dt * p / mass.unsqueeze(-1)
            F2 = forces(q, False); p = p + 0.5 * dt * F2
        elif mode == "euler":
            F = forces(q, False)
            q = q + dt * p / mass.unsqueeze(-1)
            p = p + dt * F
        elif mode == "broken":
            F = forces(q, True); p = p + 0.5 * dt * F
            q = q + dt * p / mass.unsqueeze(-1)
            F2 = forces(q, True); p = p + 0.5 * dt * F2
        H = (p ** 2).sum(-1).div(2 * mass).sum(-1) + model.V(q, mass, radius, e, pres)
        Hs.append(H.item())
        Ps.append(p.sum(1)[0].detach())
    Hs = np.array(Hs)
    mom_drift = max((Pt - P0).norm().item() for Pt in Ps)
    eng_drift = float(np.abs(Hs - H0).max() / (np.abs(Hs).mean() + 1e-9))
    return eng_drift, mom_drift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=40, help="scenes to average over")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--max-remove", type=int, default=3)
    ap.add_argument("--out", default="ccd_curve.png")
    a = ap.parse_args()

    random.seed(0)
    model = HamiltonianDynamics()
    model.load_state_dict(torch.load(a.model, map_location="cpu"))
    model.eval()

    files = sorted(glob.glob(os.path.join(a.data, "*.npz")))[:a.limit]
    modes = ["symplectic", "euler", "broken"]
    removes = list(range(a.max_remove + 1))
    eng = {m: {r: [] for r in removes} for m in modes}
    mom = {m: {r: [] for r in removes} for m in modes}

    for fp in files:
        z = np.load(fp, allow_pickle=True)
        pos, vel, pres, attrs = z["positions"], z["velocities"], z["presence"], z["attrs"]
        T, N, _ = pos.shape
        # a frame where >=3 objects are present, so we can remove several
        cand = [t for t in range(T) if (pres[t] > 0).sum() >= 3]
        if not cand:
            continue
        s = cand[len(cand) // 2]
        on = list(np.where(pres[s] > 0)[0])
        q0 = torch.from_numpy(pos[s].astype(np.float32))
        v0 = torch.from_numpy(vel[s].astype(np.float32))
        at = torch.from_numpy(attrs.astype(np.float32))

        for r in removes:
            if r > len(on) - 2:                 # keep at least 2 objects
                continue
            drop = set(random.sample(on, r))
            present = torch.zeros(N)
            for idx in on:
                if idx not in drop:
                    present[idx] = 1.0
            for m in modes:
                ed, md = roll(model, q0, v0, at, present, a.steps, m)
                eng[m][r].append(ed)
                mom[m][r].append(md)

    def mean(d):
        return {r: (np.mean(v) if v else float("nan")) for r, v in d.items()}

    print("=" * 66)
    print("Counterfactual Conservation Drift (mean over scenes)")
    print("=" * 66)
    print("ENERGY drift (relative):   objects removed ->")
    print("   mode         " + "".join(f"{r:>10d}" for r in removes))
    for m in modes:
        mm = mean(eng[m])
        print(f"   {m:<11s} " + "".join(f"{mm[r]:>10.2e}" for r in removes))
    print("\nMOMENTUM drift (abs):      objects removed ->")
    print("   mode         " + "".join(f"{r:>10d}" for r in removes))
    for m in modes:
        mm = mean(mom[m])
        print(f"   {m:<11s} " + "".join(f"{mm[r]:>10.2e}" for r in removes))
    print("=" * 66)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (axE, axP) = plt.subplots(1, 2, figsize=(11, 4.2))
        for m in modes:
            axE.plot(removes, [mean(eng[m])[r] for r in removes], marker="o", label=m)
            axP.plot(removes, [mean(mom[m])[r] for r in removes], marker="o", label=m)
        for ax, ttl in ((axE, "Energy drift"), (axP, "Momentum drift")):
            ax.set_xlabel("objects removed"); ax.set_ylabel("drift")
            ax.set_yscale("log"); ax.set_title(ttl); ax.legend()
        fig.suptitle("Counterfactual Conservation Drift (CCD)")
        fig.tight_layout()
        fig.savefig(a.out, dpi=130)
        print(f"saved plot -> {a.out}")
    except Exception as ex:
        print(f"(plot skipped: {ex}. The table above is the result.)")


if __name__ == "__main__":
    main()