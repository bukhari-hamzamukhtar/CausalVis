"""
collision_check.py  —  does the trained engine reproduce the REAL collisions?
=============================================================================

Before any counterfactual can mean anything, the FACTUAL simulation has to
recreate the bounces that actually happened in the video. This script checks
exactly that, with no guessing about start frames.

Every .npz stores `collisions`: the list of real collisions as (frame, i, j).
For each one, this script:
  1. starts the simulation a few frames BEFORE the collision, at a moment where
     both objects are on screen and still approaching,
  2. rolls the model forward,
  3. reports the closest the two objects got, and whether that counts as a hit.

It prints, per real collision, whether the model reproduced it -- and a final
"reproduced X / K" score. That score is the honest measure of whether the
engine learned collision physics or just straight-line drift.

    python collision_check.py --model v3_1_dynamics.pt --npz data/trajectories_out/sim_00000.npz
"""

import argparse
import numpy as np
import torch

from dynamics import HamiltonianDynamics, strip_colour


def simulate_positions(model, q0, v0, attrs, present, steps, dt=1.0):
    """Roll forward and return positions [steps, N, 2]. Grad on (forces need it),
    but properties and per-step state are detached so nothing accumulates."""
    phys = strip_colour(attrs.unsqueeze(0))
    mass, radius, e = model.properties(phys)
    mass, radius, e = mass.detach(), radius.detach(), e.detach()
    pres = present.unsqueeze(0)
    q = q0.unsqueeze(0) * pres.unsqueeze(-1)
    p = mass.unsqueeze(-1) * v0.unsqueeze(0) * pres.unsqueeze(-1)
    out = []
    for _ in range(steps):
        q, p = q.detach(), p.detach()
        q, p, _ = model.step(q, p, mass, radius, e, pres, dt=dt, F=None, create_graph=False)
        out.append(q.detach())
    return torch.stack(out, 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--lookback", type=int, default=12,
                    help="how many frames before a collision to start the sim")
    ap.add_argument("--extra", type=int, default=20,
                    help="extra steps to run past the collision frame")
    ap.add_argument("--thresh", type=float, default=0.03,
                    help="contact slack; a pair counts as hit if gap <= thresh")
    ap.add_argument("--contact", type=float, default=None,
                    help="override contact distance (from measure_contact.py). "
                         "If unset, uses the model's own radius (often too small).")
    ap.add_argument("--to-end", action="store_true",
                    help="simulate from the start frame to the end of the clip "
                         "instead of just past the collision")
    a = ap.parse_args()

    model = HamiltonianDynamics()
    model.load_state_dict(torch.load(a.model, map_location="cpu"))
    model.eval()

    z = np.load(a.npz, allow_pickle=True)
    pos, vel = z["positions"], z["velocities"]
    pres, attrs = z["presence"], z["attrs"]
    names = [str(x).replace("_", " ") for x in z["obj_keys"]]
    cols = z["collisions"]                         # [K,3] = (frame, i, j)
    T, N, _ = pos.shape

    at = torch.from_numpy(attrs.astype(np.float32))
    with torch.no_grad():
        _, radius, _ = model.properties(strip_colour(at.unsqueeze(0)))
    radius = radius[0]

    def nm(idx):
        return names[idx] if idx < len(names) else f"obj{idx}"

    if len(cols) == 0:
        print(f"{a.npz} has no recorded collisions. Try another sim_XXXXX.npz.")
        return

    print("=" * 70)
    print(f"scene: {a.npz}")
    print(f"real collisions recorded: {len(cols)}")
    print("=" * 70)

    reproduced = 0
    for row in cols:
        f, i, j = int(row[0]), int(row[1]), int(row[2])

        # start at the first frame in [f-lookback, f) where BOTH are present
        start = None
        for s in range(max(1, f - a.lookback), f):
            if pres[s, i] > 0 and pres[s, j] > 0:
                start = s
                break
        if start is None:
            print(f"real frame {f:3d}: {nm(i)} <-> {nm(j)}")
            print(f"   -> skipped (objects not both on screen just before it)\n")
            continue

        steps = (T - start) if a.to_end else (f - start) + a.extra
        present = torch.from_numpy((pres[start] > 0).astype(np.float32))
        q0 = torch.from_numpy(pos[start].astype(np.float32))
        v0 = torch.from_numpy(vel[start].astype(np.float32))

        qs = simulate_positions(model, q0, v0, at, present, steps)

        contact = a.contact if a.contact is not None else (radius[i] + radius[j]).item()
        dmin, hit_frame = 1e9, None
        for t in range(steps):
            d = (qs[t, i] - qs[t, j]).norm().item()
            if d < dmin:
                dmin = d
            if (d - contact) <= a.thresh and hit_frame is None:
                hit_frame = start + t
        ok = hit_frame is not None
        reproduced += int(ok)

        print(f"real frame {f:3d}: {nm(i)} <-> {nm(j)}")
        print(f"   sim from frame {start} for {steps} steps")
        print(f"   closest gap reached : {dmin - contact:+.3f}   (contact distance {contact:.3f})")
        if ok:
            print(f"   -> REPRODUCED  (model bounce near frame {hit_frame})\n")
        else:
            print(f"   -> missed      (objects never got within {a.thresh:.2f} of contact)\n")

    print("=" * 70)
    print(f"reproduced {reproduced} / {len(cols)} real collisions")
    print("=" * 70)
    if reproduced == 0:
        print("If closest gaps are small positive numbers, raise --thresh (e.g. 0.06).")
        print("If they are large, the model needs more training on collision frames.")


if __name__ == "__main__":
    main()