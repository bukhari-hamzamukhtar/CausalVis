"""
fix_cube_radius.py  —  give cubes their true effective radius
=============================================================

dynamics.py computes contact distance as r_i + r_j, i.e. it treats every
object as a sphere. That is EXACTLY right for spheres and cylinders, whose
horizontal cross-section is a circle. It is wrong for cubes.

A cube of half-side s reaches s at a face and s*sqrt(2) at a corner. Averaged
over all approach directions its support radius is s*4/pi = 1.273*s. Storing
the raw half-side instead understates every cube contact by ~21%, and for a
sphere-cube pair the measured error was 0.1092 stored versus 0.1231 true.

This rewrites attrs[:,15] for cubes only, so r_i + r_j reproduces the true
contact distance on average without any change to the physics engine. The
direction-DEPENDENT part still needs recovered yaw (see fit_yaw_track); this
removes the systematic bias, not the residual +-10% orientation term.

    python src2/fix_cube_radius.py --data data/trajectories_3d
"""
import argparse, glob, math, os
import numpy as np

CUBE_MEAN_SUPPORT = 4.0 / math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/trajectories_3d")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.data, "*.npz")))
    if not files:
        raise SystemExit(f"no .npz in {a.data}")
    print(f"{len(files)} clips in {a.data}")

    changed = cubes = 0
    for n, f in enumerate(files):
        z = dict(np.load(f, allow_pickle=True))
        at = z["attrs"]
        # shape one-hot lives at columns 2:5 as (cube, sphere, cylinder)
        is_cube = at[:, 2] > 0.5
        if not is_cube.any():
            continue
        cubes += int(is_cube.sum())
        at[is_cube, 15] = z["size_metric"][is_cube] / float(z["world_scale"]) \
            * CUBE_MEAN_SUPPORT
        z["attrs"] = at
        if not a.dry_run:
            np.savez_compressed(f, **z)
        changed += 1
        if (n + 1) % 2000 == 0:
            print(f"   {n+1}/{len(files)}  clips touched {changed}", flush=True)

    print(f"\n{'would update' if a.dry_run else 'updated'} {changed} clips "
          f"({cubes} cubes) -- cube radius x{CUBE_MEAN_SUPPORT:.3f}")


if __name__ == "__main__":
    main()
