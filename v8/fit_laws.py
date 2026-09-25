"""
v8/fit_laws.py  —  fit the textbook-law constants on TRAIN videos, save v8/laws.json
====================================================================================

The system is given a small library of known laws and chooses for itself:

  motion between collisions, per (shape, material):
    const  constant velocity
    exp    speed shrinks by a fixed percentage each frame   (rolling damping)
    coul   speed shrinks by a fixed amount each frame       (sliding friction)
  collisions:
    textbook bounce along the line of centres, bounciness e per material pair,
    metal/rubber mass ratio, and sliding friction at the contact (mu)

Each constant is fitted by grid search on TRAIN only; each object kind keeps the
law with the lowest TRAIN error. VAL and TEST are never used here.

Measured with the same fits (probe/physics_laws.py, VAL): 40-frame error 0.0502
vs the learned model's 0.0566; head-to-head bounce direction 12.1 deg vs 17.6.

    python v8/fit_laws.py
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "probe"))
import physics_laws as P  # noqa: E402


def main():
    tr = P.free_segments(P.TRAIN)
    C_GRID = np.linspace(0, 0.04, 41)
    A_GRID = np.linspace(0, 0.0002, 41)
    motion = {}
    for g in sorted(set(tr[3])):
        m = tr[3] == g
        P0, V0, F = tr[0][m], tr[1][m], tr[2][m]
        const = P.err(P.pred_exp(P0, V0, 0.0), F, 20).mean()
        ce = [P.err(P.pred_exp(P0, V0, c), F, 20).mean() for c in C_GRID]
        ca = [P.err(P.pred_coul(P0, V0, a), F, 20).mean() for a in A_GRID]
        options = [("const", const, 0.0), ("exp", min(ce), float(C_GRID[int(np.argmin(ce))])),
                   ("coul", min(ca), float(A_GRID[int(np.argmin(ca))]))]
        name, score, param = min(options, key=lambda x: x[1])
        motion[g] = {"chosen": name, "param": param, "train_err20": float(score),
                     "alternatives": {o[0]: {"param": o[2], "train_err20": float(o[1])} for o in options},
                     "n_train": int(m.sum())}
        print("%-18s -> %-5s param %.6f   (train err@20 %.5f; n=%d)" % (g, name, param, score, m.sum()))
    rows = P.collisions(P.TRAIN)
    ratio, E, mu = P.fit_collision_law(rows, "nc")
    print("collisions (%d train): mass ratio metal/rubber %.2f, bounciness %s, contact friction %.2f"
          % (len(rows), ratio, E, mu))
    out = {"motion": motion, "collision": {"mass_ratio": ratio, "e": E, "mu": mu, "normal": "line of centres"},
           "contact_sizes": P.SIZES, "contact_gap": 0.01,
           "fitted_on": "TRAIN: %d clips, %d free-motion segments, %d isolated collisions"
                        % (len(P.TRAIN), len(tr[0]), len(rows))}
    json.dump(out, open(os.path.join(ROOT, "v8", "laws.json"), "w"), indent=1)
    print("wrote v8/laws.json")


if __name__ == "__main__":
    main()
