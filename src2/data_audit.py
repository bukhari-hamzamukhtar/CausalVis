"""
data_audit.py  —  turn the assumptions in build_trajectories.py into measurements
================================================================================

Three questions, answered from your own data instead of by argument:

  1. DETECTOR JITTER. Between collisions a CLEVRER object travels in a straight
     line, so any wobble around a fitted straight line IS detector noise. This
     measures it, and reports whether the 5-frame velocity window is justified
     (and what window size the noise actually implies).

  2. HOW BIG IS THE GAP PROBLEM. Distribution of "object vanished" runs: how
     many are short (interpolated), how many are long (left absent), and what
     fraction of all object-frames they represent. This decides whether a
     physics-based reconstruction is worth building.

  3. NON-COLLIDING OBJECTS. What fraction of objects never appear in the
     collision list -- i.e. how much of the scene the collision array ignores
     but the world model still sees.

    python data_audit.py --data data/trajectories_out --limit 400
"""

import argparse
import glob
import os
import numpy as np

FRAME_W, FRAME_H = 480, 320


def runs_of(mask_true_idx):
    """group a sorted index array into consecutive runs"""
    if len(mask_true_idx) == 0:
        return []
    runs, cur = [], [mask_true_idx[0]]
    for m in mask_true_idx[1:]:
        if m == cur[-1] + 1:
            cur.append(m)
        else:
            runs.append(cur); cur = [m]
    runs.append(cur)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--min-run", type=int, default=8,
                    help="min clean frames needed to measure jitter")
    ap.add_argument("--collision-guard", type=int, default=3,
                    help="frames around a collision to exclude from jitter fit")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.data, "*.npz")))[:a.limit]

    resid = []                 # straight-line residuals (normalised units)
    gap_lens = []              # lengths of absence runs strictly inside a track
    total_objframes = 0
    gap_objframes_short = 0
    gap_objframes_long = 0
    n_objects = 0
    n_objects_no_collision = 0
    n_clips = 0

    for fp in files:
        z = np.load(fp, allow_pickle=True)
        pos, pres, cols = z["positions"], z["presence"], z["collisions"]
        T, N, _ = pos.shape
        n_clips += 1

        involved = set()
        col_frames = {}
        for r in cols:
            f, i, j = int(r[0]), int(r[1]), int(r[2])
            involved.add(i); involved.add(j)
            col_frames.setdefault(i, []).append(f)
            col_frames.setdefault(j, []).append(f)

        for i in range(N):
            on = np.where(pres[:, i] > 0)[0]
            if len(on) == 0:
                continue
            n_objects += 1
            if i not in involved:
                n_objects_no_collision += 1
            total_objframes += len(on)

            # ---- gaps strictly between first and last sighting ----
            span = np.arange(on[0], on[-1] + 1)
            missing = span[pres[span, i] == 0]
            for r in runs_of(list(missing)):
                gap_lens.append(len(r))
                if len(r) <= 5:
                    gap_objframes_short += len(r)
                else:
                    gap_objframes_long += len(r)

            # ---- jitter: straight-line fit on clean stretches ----
            guard = set()
            for f in col_frames.get(i, []):
                for d in range(-a.collision_guard, a.collision_guard + 1):
                    guard.add(f + d)
            clean = [t for t in on if t not in guard]
            for run in runs_of(clean):
                if len(run) < a.min_run:
                    continue
                ts = np.asarray(run, dtype=np.float64)
                tc = ts - ts.mean()
                den = (tc ** 2).sum()
                if den < 1e-9:
                    continue
                for d in range(2):
                    ys = pos[run, i, d].astype(np.float64)
                    slope = (tc * (ys - ys.mean())).sum() / den
                    pred = ys.mean() + slope * tc
                    resid.extend((ys - pred).tolist())

    resid = np.asarray(resid)
    gap_lens = np.asarray(gap_lens)

    print("=" * 68)
    print(f"clips scanned: {n_clips}    object tracks: {n_objects}")
    print("=" * 68)

    print("\n1. DETECTOR JITTER  (wobble around a straight line, between collisions)")
    if len(resid) == 0:
        print("   not enough clean stretches to measure")
    else:
        sd = float(resid.std())
        print(f"   samples                         : {len(resid)}")
        print(f"   position noise (normalised)     : {sd:.5f}")
        print(f"   position noise (pixels, approx) : {sd * FRAME_W:.2f} px in x-scale")
        one_frame = sd * np.sqrt(2)
        lsq5 = sd / np.sqrt(10.0)          # var of centred t over [-2..2] = 10
        print(f"   -> velocity noise, 1-frame difference : {one_frame:.5f} per frame")
        print(f"   -> velocity noise, 5-frame lstsq fit  : {lsq5:.5f} per frame")
        print(f"   -> the 5-frame window reduces velocity noise about "
              f"{one_frame / max(lsq5, 1e-12):.1f}x")
        print("   (jitter is real if the noise is a meaningful fraction of the")
        print("    typical per-frame motion; compare with the speeds you see in readout)")

    print("\n2. GAPS INSIDE A TRACK  (object seen, vanishes, seen again)")
    if len(gap_lens) == 0:
        print("   no interior gaps at all -- interpolation never fires")
    else:
        short = int((gap_lens <= 5).sum())
        long_ = int((gap_lens > 5).sum())
        print(f"   gap events total                : {len(gap_lens)}")
        print(f"   short (<=5, interpolated)       : {short}  ({100*short/len(gap_lens):.1f}%)")
        print(f"   long  (> 5, left absent)        : {long_}  ({100*long_/len(gap_lens):.1f}%)")
        print(f"   longest gap                     : {int(gap_lens.max())} frames")
        denom = total_objframes + gap_objframes_short + gap_objframes_long
        print(f"   frames guessed by interpolation : {gap_objframes_short} "
              f"({100*gap_objframes_short/max(denom,1):.2f}% of all object-frames)")
        print(f"   frames dropped as too long      : {gap_objframes_long} "
              f"({100*gap_objframes_long/max(denom,1):.2f}% of all object-frames)")
        print("   -> if the last two percentages are tiny, physics-based gap")
        print("      reconstruction would change almost nothing.")

    print("\n3. OBJECTS THAT NEVER COLLIDE")
    if n_objects:
        pct = 100 * n_objects_no_collision / n_objects
        print(f"   {n_objects_no_collision} of {n_objects} tracks ({pct:.1f}%) never appear "
              f"in the collision list.")
        print("   The world model still simulates every one of them: they exert and")
        print("   receive forces at every step. The collision list is only the answer")
        print("   key used for grading, never an input.")
    print("=" * 68)


if __name__ == "__main__":
    main()
