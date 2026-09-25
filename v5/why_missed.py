"""
v5/why_missed.py  —  four hypotheses for the missed collisions, measured
========================================================================

821 missed collisions vs 40 invented, and 77% of the misses were nowhere near
(median gap +0.104, most of a contact distance of clear air). Four candidate
causes, each with a decisive test:

  H1 MOMENTUM NOT CONSERVED. Check it directly on the trained model.
  H2 FRICTION OVERSHOOTS. Drag makes objects travel LESS far, so an
     over-damped model would stop short of its target. Test: re-run every
     missed collision with drag switched OFF and count how many appear.
  H3 WRONG DEFLECTION ANGLE. When a collision DOES happen, does the object
     leave in the right direction at the right speed? Measured against the
     recorded trajectory.
  H4 OBJECTS TOO SMALL. Compute, per miss, the radius multiplier that would
     have been needed to make contact. If the answer is ~1.1x it is a sizing
     bug; if it is ~2x the objects were never on a collision course.

These are not mutually exclusive, and the point is the SIZE of each effect,
not a yes/no. Run on the test split so nothing is tuned on it.
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src2"))
from benchmark_eval import load_model, load_split_set, _vid_num, STRIP   # noqa: E402
from dynamics import strip_colour                                        # noqa: E402

PRE = 15          # frames before the annotated collision to start from
POST = 25         # frames to run past it


def roll(model, z, i, j, s, steps, mass, radius, e, dissipative):
    """Whole-scene rollout from observed state at frame s. Returns the
    trajectory of i and j plus the minimum gap they reached."""
    was = model.dissipative
    model.dissipative = dissipative
    try:
        pos = z["positions"].astype(np.float32); vel = z["velocities"].astype(np.float32)
        pres = z["presence"]; T = pos.shape[0]; N = mass.shape[1]
        q = torch.zeros(1, N, 2); p = torch.zeros(1, N, 2); act = torch.zeros(1, N)
        for k in range(N):
            if pres[s, k] > 0:
                act[0, k] = 1.0
                q[0, k] = torch.from_numpy(pos[s, k])
                p[0, k] = mass[0, k] * torch.from_numpy(vel[s, k])
        if act[0, i] <= 0 or act[0, j] <= 0:
            return None, None, None
        contact = float(z["attrs"][i, 15] + z["attrs"][j, 15])
        min_gap = float((q[0, i] - q[0, j]).norm()) - contact
        traj = []
        for t in range(s + 1, min(T, s + 1 + steps)):
            q, p = q.detach(), p.detach()
            q, p, _ = model.step(q, p, mass, radius, e, act, dt=1.0, F=None,
                                 create_graph=False)
            d = float((q[0, i] - q[0, j]).norm())
            min_gap = min(min_gap, d - contact)
            traj.append((q[0, i].detach().numpy().copy(),
                         q[0, j].detach().numpy().copy(),
                         (p[0, i] / mass[0, i]).detach().numpy().copy(),
                         (p[0, j] / mass[0, j]).detach().numpy().copy()))
        return traj, min_gap, contact
    finally:
        model.dissipative = was


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v3_7_3d_frozen.pt")
    ap.add_argument("--data", default="data/trajectories_3d")
    ap.add_argument("--split", default="split3d.json")
    ap.add_argument("--clips", type=int, default=150)
    a = ap.parse_args()

    model = load_model(a.model)
    files = {(_vid_num(f)): f for f in glob.glob(os.path.join(a.data, "*.npz"))}
    test = sorted(k for k in load_split_set(a.split, "test") if k in files)[:a.clips]

    # ---- H1: momentum -----------------------------------------------------
    B, N = 1, 4
    at = torch.zeros(B, N, 16)
    at[..., 0] = 1.0; at[..., 1] = 0.6; at[..., 3] = 1.0; at[..., 13] = 1.0
    at[..., 15] = 0.0583
    pres = torch.ones(B, N)
    m_, r_, e_ = model.properties(strip_colour(at))
    q = torch.rand(B, N, 2) * 0.4 + 0.3
    p = m_.unsqueeze(-1) * torch.randn(B, N, 2) * 0.05
    P0 = p.sum(1).clone()
    was = model.dissipative; model.dissipative = False
    worst = 0.0
    for _ in range(300):
        q, p, _ = model.step(q.detach(), p.detach(), m_, r_, e_, pres, dt=0.05,
                             F=None, create_graph=False)
        worst = max(worst, float((p.sum(1) - P0).abs().max()))
    model.dissipative = was
    print("=" * 72)
    print("H1  MOMENTUM CONSERVED?")
    print("    worst drift over 300 steps (drag off): %.2e" % worst)
    print("    -> %s" % ("CONSERVED. Not the cause." if worst < 1e-4 else "BROKEN."))

    # ---- H2/H3/H4 over real annotated collisions --------------------------
    made_on = made_off = total = 0
    gaps_on, gaps_off, need_scale = [], [], []
    ang_err, spd_ratio = [], []
    for v in test:
        z = np.load(files[v], allow_pickle=True)
        pos, vel, pres_ = z["positions"], z["velocities"], z["presence"]
        T, Nn, _ = pos.shape; Nn = min(Nn, 8)
        att = torch.from_numpy(z["attrs"][:Nn].astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            mass, radius, e = model.properties(STRIP(att))
        for (fr, i, j) in z["collisions"]:
            fr, i, j = int(fr), int(i), int(j)
            if i >= Nn or j >= Nn:
                continue
            s = fr - PRE
            if s < 1 or fr + POST >= T:
                continue
            if not (pres_[s:fr + POST, i] > 0).all() or not (pres_[s:fr + POST, j] > 0).all():
                continue
            total += 1
            tr_on, g_on, contact = roll(model, z, i, j, s, PRE + POST, mass, radius, e, True)
            tr_off, g_off, _ = roll(model, z, i, j, s, PRE + POST, mass, radius, e, False)
            if g_on is None:
                total -= 1
                continue
            gaps_on.append(g_on); gaps_off.append(g_off)
            hit_on, hit_off = g_on <= 0.02, g_off <= 0.02
            made_on += hit_on; made_off += hit_off
            # H4: what radius multiplier would have closed the gap?
            need_scale.append((g_on + contact) / contact if contact > 0 else np.nan)
            # H3: when the model DOES make contact, is the outgoing direction right?
            if hit_on and tr_on is not None:
                k = min(PRE + 10, len(tr_on) - 1)
                for who, idx in ((0, i), (1, j)):
                    pv = tr_on[k][2 + who]
                    tv = vel[min(fr + 10, T - 1), idx]
                    npv, ntv = np.linalg.norm(pv), np.linalg.norm(tv)
                    if npv > 1e-6 and ntv > 1e-6:
                        cos = float(np.clip(np.dot(pv, tv) / (npv * ntv), -1, 1))
                        ang_err.append(math.degrees(math.acos(cos)))
                        spd_ratio.append(npv / ntv)

    g_on = np.array(gaps_on); g_off = np.array(gaps_off); ns = np.array(need_scale)
    print("\n" + "=" * 72)
    print("REAL annotated collisions replayed from %d frames before: %d cases" % (PRE, total))
    print("=" * 72)
    print("\nH2  IS FRICTION KILLING THE COLLISIONS?")
    print("    reproduced WITH drag   : %4d / %d  (%.1f%%)" % (made_on, total, 100.0 * made_on / max(total, 1)))
    print("    reproduced WITHOUT drag: %4d / %d  (%.1f%%)" % (made_off, total, 100.0 * made_off / max(total, 1)))
    print("    median closest gap  with drag: %+.4f   without: %+.4f" % (np.median(g_on), np.median(g_off)))
    d = made_off - made_on
    print("    -> removing friction recovers %d collisions (%.1f pp). %s"
          % (d, 100.0 * d / max(total, 1),
             "FRICTION IS A REAL FACTOR." if d > 0.02 * total else "Not the main cause."))

    print("\nH3  WHEN IT DOES COLLIDE, IS THE BOUNCE RIGHT?")
    if ang_err:
        ae = np.array(ang_err); sr = np.array(spd_ratio)
        print("    outgoing direction error: median %.1f deg   p90 %.1f deg" % (np.median(ae), np.percentile(ae, 90)))
        print("    outgoing speed / true   : median %.2f" % np.median(sr))
        print("    -> %s" % ("deflection is roughly right" if np.median(ae) < 30
                             else "DEFLECTION IS WRONG -- the bounce angle is off"))
    else:
        print("    no successful collisions to measure -- H3 untestable here")

    print("\nH4  ARE THE OBJECTS TOO SMALL?")
    print("    radius multiplier needed to close the gap:")
    for q_ in (50, 75, 90):
        print("        p%-2d  %.2fx" % (q_, np.percentile(ns, q_)))
    frac = float((ns <= 1.2).mean())
    print("    -> %.0f%% of misses need <= 1.2x radius. %s"
          % (100 * frac,
             "SIZING IS A REAL FACTOR." if frac > 0.3 else
             "Most misses need objects nearly DOUBLE size -- they were never on a collision course, so this is a TRAJECTORY error, not a sizing one."))
    print("=" * 72)


if __name__ == "__main__":
    main()
