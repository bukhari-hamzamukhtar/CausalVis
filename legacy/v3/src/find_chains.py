"""
find_chains.py  —  hunt for CAUSED (chain) counterfactuals across the dataset
=============================================================================

The demo that no classifier can produce: a collision between two objects that
DISAPPEARS when you remove a THIRD object that was never in it.

For each recorded collision (i <-> j), this looks for an object k such that:
    1. factually, the model reproduces the i-j collision,
    2. factually, k actually contacts i or j BEFORE that collision
       (so k physically did something to a participant), and
    3. with k removed, the i-j collision no longer happens.

That is a genuine chain: k redirected i or j into the collision, and taking k
away breaks it. The interaction filter (step 2) keeps out coincidences.

    python find_chains.py --data data/trajectories_out --model v3_1_dynamics.pt \
           --contact 0.119 --limit 300
"""

import argparse
import glob
import os
import numpy as np
import torch

from dynamics import HamiltonianDynamics, strip_colour
from intervene import do_remove
from readout import build_scene, simulate, pair_collides, window_start


def first_contact_frame(qs, a, b, contact, thresh):
    T = qs.shape[0]
    for t in range(T):
        d = (qs[t, a] - qs[t, b]).norm().item()
        if d - contact <= thresh:
            return t
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--contact", type=float, default=0.119)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--lookback", type=int, default=12)
    ap.add_argument("--extra", type=int, default=15)
    ap.add_argument("--thresh", type=float, default=0.02)
    ap.add_argument("--max-report", type=int, default=10)
    a = ap.parse_args()

    model = HamiltonianDynamics()
    model.load_state_dict(torch.load(a.model, map_location="cpu"))
    model.eval()

    files = sorted(glob.glob(os.path.join(a.data, "*.npz")))
    if a.limit:
        files = files[:a.limit]

    found = []
    scanned = 0
    for fp in files:
        scanned += 1
        z = np.load(fp, allow_pickle=True)
        pos, pres, cols = z["positions"], z["presence"], z["collisions"]
        names = [str(x).replace("_", " ") for x in z["obj_keys"]]
        N = pos.shape[1]
        if len(cols) == 0:
            continue

        for row in cols:
            f, i, j = int(row[0]), int(row[1]), int(row[2])
            s = window_start(pres, f, i, j, a.lookback)
            if s is None:
                continue
            steps = (f - s) + a.extra
            scene = build_scene(z, s)

            qs_f, _ = simulate(model, scene, steps)
            hit, tcol = pair_collides(qs_f, i, j, a.contact, a.thresh)
            if not hit:
                continue

            for k in range(N):
                if k in (i, j) or scene[3][k].item() == 0:
                    continue
                tki = first_contact_frame(qs_f, k, i, a.contact, a.thresh)
                tkj = first_contact_frame(qs_f, k, j, a.contact, a.thresh)
                interacted = (tki is not None and tki <= tcol) or (tkj is not None and tkj <= tcol)
                if not interacted:
                    continue

                qs_c, _ = simulate(model, do_remove(scene, k), steps)
                hit_c, _ = pair_collides(qs_c, i, j, a.contact, a.thresh)
                if not hit_c:
                    hit_obj = i if (tki is not None and (tkj is None or tki <= tkj)) else j
                    found.append((os.path.basename(fp), f, i, j, k, hit_obj, names))
                    print(f"CHAIN  {os.path.basename(fp)}  frame {f}")
                    print(f"   {names[i]} <-> {names[j]} disappears when the {names[k]} is removed")
                    print(f"   (factually, the {names[k]} first hits the {names[hit_obj]} and redirects it)")
                    print(f"   reproduce: python src/readout.py --model {a.model} "
                          f"--npz {a.data}/{os.path.basename(fp)} --remove {k} --contact {a.contact}\n")

        if len(found) >= a.max_report:
            break

    print("=" * 66)
    print(f"scanned {scanned} clips, found {len(found)} chain example(s)")
    if not found:
        print("no clean chains in this batch -- raise --limit to scan more clips")
    print("=" * 66)


if __name__ == "__main__":
    main()