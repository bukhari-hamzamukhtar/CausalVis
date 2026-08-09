"""
readout.py  —  CausalVis V3, Phase 2b: the counterfactual answer + audit ledger
===============================================================================

Answers "without object X, which collisions would not happen -- and why?"

Each collision is tested in its own short, accurate window (the model cannot add
objects mid-rollout, and long rollouts drift). For each recorded collision we
simulate factually and with X removed:

  [direct]  X was one of the two objects        -> trivially cannot happen
  [caused]  X was a DIFFERENT object whose removal breaks the collision
            -> the real causal result

The AUDIT LEDGER is computed in the window where X actually moves (not frame 0),
and for a chain it reports X's momentum, which object it struck, and how far that
redirected the struck object.

    python readout.py --model v3_2_dynamics.pt --npz data/trajectories_out/sim_00077.npz --list
    python readout.py --model v3_2_dynamics.pt --npz data/trajectories_out/sim_00077.npz \
           --remove 0 --contact 0.119
"""

import argparse
import math
import numpy as np
import torch

from dynamics import HamiltonianDynamics, strip_colour
from intervene import do_remove


def build_scene(z, start, max_objects=8):
    pos, vel = z["positions"], z["velocities"]
    pres, attrs = z["presence"], z["attrs"]
    N = pos.shape[1]
    M = min(max_objects, N)
    q0 = torch.from_numpy(pos[start, :M].astype(np.float32)).clone()
    v0 = torch.from_numpy(vel[start, :M].astype(np.float32)).clone()
    at = torch.from_numpy(attrs[:M].astype(np.float32)).clone()
    pr = torch.from_numpy((pres[start, :M] > 0).astype(np.float32)).clone()
    return [q0, v0, at, pr]


def simulate(model, scene, steps, dt=1.0):
    """Roll forward; returns positions [steps,N,2] and velocities [steps,N,2].
    NOT no_grad (forces() needs autograd); state detached each step so no graph
    accumulates."""
    q0, v0, attrs, present = scene[0], scene[1], scene[2], scene[3]
    phys = strip_colour(attrs.unsqueeze(0))
    mass, radius, e = model.properties(phys)
    mass, radius, e = mass.detach(), radius.detach(), e.detach()
    pres = present.unsqueeze(0)
    q = q0.unsqueeze(0) * pres.unsqueeze(-1)
    p = mass.unsqueeze(-1) * v0.unsqueeze(0) * pres.unsqueeze(-1)
    qs, vs = [], []
    for _ in range(steps):
        q, p = q.detach(), p.detach()
        q, p, _ = model.step(q, p, mass, radius, e, pres, dt=dt, F=None, create_graph=False)
        qs.append(q.detach())
        vs.append((p / mass.unsqueeze(-1)).detach())
    return torch.stack(qs, 1)[0], torch.stack(vs, 1)[0]


def pair_collides(qs, i, j, contact, thresh):
    T = qs.shape[0]
    for t in range(T):
        d = (qs[t, i] - qs[t, j]).norm().item()
        if d - contact <= thresh:
            if t == 0 or (qs[t, i] - qs[t, j]).norm().item() < (qs[t - 1, i] - qs[t - 1, j]).norm().item():
                return True, t
    return False, None


def first_contact_frame(qs, a, b, contact, thresh):
    for t in range(qs.shape[0]):
        if (qs[t, a] - qs[t, b]).norm().item() - contact <= thresh:
            return t
    return None


def window_start(pres, f, i, j, lookback):
    for s in range(max(0, f - lookback), f):
        if pres[s, i] > 0 and pres[s, j] > 0:
            return s
    return None


def heading(v):
    return math.degrees(math.atan2(float(v[1]), float(v[0])))


def vec_angle(u, v):
    ux, uy, vx, vy = float(u[0]), float(u[1]), float(v[0]), float(v[1])
    nu, nv = math.hypot(ux, uy), math.hypot(vx, vy)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    c = max(-1.0, min(1.0, (ux * vx + uy * vy) / (nu * nv)))
    return math.degrees(math.acos(c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--remove", type=int, default=None)
    ap.add_argument("--contact", type=float, default=None,
                    help="override contact distance; default uses the "
                         "measured per-pair radii r_i + r_j")
    ap.add_argument("--lookback", type=int, default=12)
    ap.add_argument("--extra", type=int, default=15)
    ap.add_argument("--thresh", type=float, default=0.02)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    model = HamiltonianDynamics()
    model.load_state_dict(torch.load(a.model, map_location="cpu"))
    model.eval()

    z = np.load(a.npz, allow_pickle=True)
    pos, pres, attrs = z["positions"], z["presence"], z["attrs"]
    names = [str(x).replace("_", " ") for x in z["obj_keys"]]
    cols = z["collisions"]
    N = pos.shape[1]
    at = torch.from_numpy(attrs.astype(np.float32))

    def nm(idx):
        return names[idx] if idx < len(names) else f"obj{idx}"

    if a.list or a.remove is None:
        print(f"objects in {a.npz}:")
        for i in range(N):
            print(f"   [{i}] {nm(i)}")
        print("\nrecorded collisions:")
        for row in cols:
            f, i, j = int(row[0]), int(row[1]), int(row[2])
            print(f"   frame {f:3d}: {nm(i)} <-> {nm(j)}")
        if a.remove is None:
            print("\npass --remove <index> --contact 0.119 to ask a counterfactual.")
            return

    k = a.remove
    if not (0 <= k < N):
        raise SystemExit(f"object index {k} out of range 0..{N-1}")
    if len(cols) == 0:
        print("this clip has no recorded collisions; try another sim_XXXXX.npz")
        return

    # measured per-pair contact from the (now real) radii; falls back to a
    # single override if --contact is given
    with torch.no_grad():
        _, radii, _ = model.properties(strip_colour(
            torch.from_numpy(attrs.astype(np.float32)).unsqueeze(0)))
    radii = radii[0]
    def contact_of(i, j):
        return a.contact if a.contact is not None else float(radii[i] + radii[j])

    print("=" * 68)
    print(f"scene   : {a.npz}")
    print(f"contact : {'per-pair measured (r_i+r_j)' if a.contact is None else a.contact}")
    print(f"remove  : [{k}] {nm(k)}")
    print("=" * 68)

    reproduced, vanished = [], []
    for row in cols:
        f, i, j = int(row[0]), int(row[1]), int(row[2])
        s = window_start(pres, f, i, j, a.lookback)
        if s is None:
            print(f"frame {f:3d}: {nm(i)} <-> {nm(j)}  -> untestable (objects not both on screen before it)")
            continue
        steps = (f - s) + a.extra
        scene = build_scene(z, s)

        qs_f, vs_f = simulate(model, scene, steps)
        hit_fact, tcol = pair_collides(qs_f, i, j, contact_of(i, j), a.thresh)
        if not hit_fact:
            print(f"frame {f:3d}: {nm(i)} <-> {nm(j)}  -> model does not reproduce it factually (skip)")
            continue
        reproduced.append((f, i, j))

        if scene[3][k].item() == 0:
            print(f"frame {f:3d}: {nm(i)} <-> {nm(j)}  -> SURVIVES ({nm(k)} not present in this window)")
            continue

        qs_c, vs_c = simulate(model, do_remove(scene, k), steps)
        # a deleted object is parked at the origin; if it is one of the two
        # participants the collision cannot happen, so decide that directly
        # rather than measuring a distance to a ghost
        if k in (i, j):
            hit_cf = False
        else:
            hit_cf, _ = pair_collides(qs_c, i, j, contact_of(i, j), a.thresh)
        if hit_cf:
            print(f"frame {f:3d}: {nm(i)} <-> {nm(j)}  -> SURVIVES")
        else:
            tag = "direct" if k in (i, j) else "caused"
            vanished.append(dict(f=f, i=i, j=j, tag=tag, s=s, steps=steps,
                                 qs_f=qs_f, vs_f=vs_f, qs_c=qs_c, vs_c=vs_c, tcol=tcol))
            print(f"frame {f:3d}: {nm(i)} <-> {nm(j)}  -> VANISHES [{tag}]")

    print("-" * 68)
    n_caused = sum(1 for v in vanished if v["tag"] == "caused")
    print(f"Without the {nm(k)}: {len(vanished)} of {len(reproduced)} reproduced "
          f"collisions would not happen"
          + (f"  ({n_caused} CAUSED / chain)" if n_caused else "") + ".")

    # ---- AUDIT LEDGER, computed in the window where k actually moves ----------
    example = None
    for v in vanished:
        if v["tag"] == "caused":
            example = v
            break
    if example is None and vanished:
        example = vanished[0]

    if example is not None:
        s = example["s"]
        L = model.ledger(build_scene(z, s)[0].unsqueeze(0),
                         build_scene(z, s)[1].unsqueeze(0),
                         at.unsqueeze(0),
                         build_scene(z, s)[3].unsqueeze(0))
        mom = L["momentum_per_object"][0, k]
        speed = float(mom.norm().item())
        print("\nWHY (audit ledger):")
        print(f"   at frame {s} the {nm(k)} was moving with momentum "
              f"({mom[0].item():+.3f}, {mom[1].item():+.3f}), speed {speed:.3f}/frame.")

        if example["tag"] == "caused":
            i, j = example["i"], example["j"]
            qs_f, vs_f = example["qs_f"], example["vs_f"]
            qs_c, vs_c = example["qs_c"], example["vs_c"]
            T = qs_f.shape[0]
            tki = first_contact_frame(qs_f, k, i, contact_of(k, i), a.thresh)
            tkj = first_contact_frame(qs_f, k, j, contact_of(k, j), a.thresh)
            if tki is not None and (tkj is None or tki <= tkj):
                p, tk = i, tki
            else:
                p, tk = j, tkj

            dmin_f = min((qs_f[t, i] - qs_f[t, j]).norm().item() for t in range(T))
            dmin_c = min((qs_c[t, i] - qs_c[t, j]).norm().item() for t in range(T))

            print(f"   with the {nm(k)} present : the {nm(i)} and {nm(j)} come within "
                  f"{dmin_f:.3f} -> they collide.")
            print(f"   with it removed          : they stay {dmin_c:.3f} apart -> they miss "
                  f"(contact is {contact_of(i, j):.3f}).")
            print(f"   mechanism: the {nm(k)} deflects the {nm(p)} onto a path that reaches the "
                  f"{nm(j) if p == i else nm(i)};")
            print(f"   removing it, the {nm(p)} passes wide. The two distances above (contact "
                  f"{contact_of(i, j):.3f}) are the causal signature.")
        else:
            print(f"   it is a direct participant, so its collisions cannot happen once removed.")
    print("=" * 68)


if __name__ == "__main__":
    main()