"""
measure_contact.py  —  what distance counts as a collision, measured from data
==============================================================================

The model's learned radius collapsed to ~0.01, so its idea of "contact" is
useless. But the dataset already knows the answer: at every recorded collision,
the two objects are touching, so the distance between their centres AT THAT
FRAME is the true contact distance -- in the same normalised units the model
uses.

This scans the dataset, measures that distance at every recorded collision, and
prints the distribution. Use a high percentile (p90) as an inclusive collision
threshold for collision_check.py and readout.py.

    python measure_contact.py --data data/trajectories_out --limit 2000
"""

import argparse
import glob
import os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.data, "*.npz")))
    if a.limit:
        files = files[:a.limit]

    dists = []
    files_with_col = 0
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        pos, pres, cols = z["positions"], z["presence"], z["collisions"]
        if len(cols) == 0:
            continue
        files_with_col += 1
        T, N, _ = pos.shape
        for row in cols:
            f, i, j = int(row[0]), int(row[1]), int(row[2])
            if not (0 <= f < T and i < N and j < N):
                continue
            if pres[f, i] <= 0 or pres[f, j] <= 0:
                continue
            dists.append(np.linalg.norm(pos[f, i] - pos[f, j]))

    dists = np.array(dists)
    if len(dists) == 0:
        print("no measurable collisions found -- check the folder")
        return

    p = np.percentile(dists, [5, 10, 25, 50, 75, 90, 95])
    print("=" * 58)
    print(f"files scanned          : {len(files)}")
    print(f"files with collisions  : {files_with_col}")
    print(f"collisions measured    : {len(dists)}")
    print("=" * 58)
    print("centre-to-centre distance at the moment of collision:")
    print(f"   min    {dists.min():.3f}")
    print(f"   p5     {p[0]:.3f}")
    print(f"   p10    {p[1]:.3f}")
    print(f"   p25    {p[2]:.3f}")
    print(f"   median {p[3]:.3f}")
    print(f"   p75    {p[4]:.3f}")
    print(f"   p90    {p[5]:.3f}")
    print(f"   p95    {p[6]:.3f}")
    print(f"   max    {dists.max():.3f}")
    print("=" * 58)
    print(f"suggested collision threshold (p90, inclusive): {p[5]:.3f}")
    print()
    print("then re-run the check with the real contact distance:")
    print(f"   python src/collision_check.py --model v3_1_dynamics.pt \\")
    print(f"          --npz data/trajectories_out/sim_00000.npz --contact {p[5]:.3f}")


if __name__ == "__main__":
    main()