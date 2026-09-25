"""
dynamics_check.py  (src2)  —  is the world model actually learning physics?
===========================================================================

Every number we have so far (86.1% per choice, 70.6% model decided, 97.5%
collision reproduction) is a QUESTION ANSWERING score.  None of them measure
the one thing the world model is supposed to do: predict where objects go.

This script measures that directly, and compares it against baselines that
use no model at all.  Four things come out:

  1. THE LEARNED POTENTIAL.  V(r) sampled across distances, printed as a
     table and an ASCII plot.  A working model shows a steep wall at small r
     that flattens to ~0 at large r.  A flat line means the model learned
     nothing and every collision it reports is an accident of drift.

  2. ROLLOUT ERROR vs HORIZON.  Start from a real frame, roll forward
     1/5/10/20/40 frames, measure mean position error against the truth.

  3. THE BASELINES.  Same measurement for:
       constant velocity : every object continues in a straight line
       frozen            : nothing moves at all
     If the model does not clearly beat CONSTANT VELOCITY, the learned
     potential is not contributing and the whole pipeline is coasting on
     straight line motion.

  4. TRAIN vs TEST.  The same numbers on clips the model trained on and
     clips it never saw.  Similar scores on both means UNDERFITTING (the
     model is too weak or undertrained), not overfitting.  That distinction
     decides whether we tune training or rebuild the model.

RUN
    python src2/dynamics_check.py --model v3_2_dynamics.pt --split split.json
    python src2/dynamics_check.py --model v3_2_dynamics.pt --split split.json --clips 200
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import torch

from benchmark_eval import load_model


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _vid_num(name):
    m = re.search(r"(\d+)", os.path.basename(str(name)))
    return int(m.group(1)) if m else None


def load_split(path):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    out = {}
    for key in ("train", "val", "test"):
        entries = data.get(key, []) if isinstance(data, dict) else []
        nums = set()
        for e in entries:
            n = e if isinstance(e, int) else _vid_num(e)
            if n is not None:
                nums.add(int(n))
        out[key] = nums
    return out


def ascii_plot(xs, ys, width=58, height=14):
    """Tiny terminal plot so the potential can be eyeballed without matplotlib."""
    lo, hi = min(ys), max(ys)
    if hi - lo < 1e-12:
        return ["   (curve is completely flat over this range)"]
    rows = []
    for r in range(height):
        level = hi - (hi - lo) * r / (height - 1)
        line = ""
        for k in range(len(xs)):
            line += "*" if ys[k] >= level - (hi - lo) / (2 * height) else " "
        rows.append(f"{level:9.3e} | {line}")
    rows.append(" " * 10 + "+" + "-" * len(xs))
    rows.append(" " * 11 + f"r = {xs[0]:.2f}" + " " * max(0, len(xs) - 18) + f"{xs[-1]:.2f}")
    return rows


# ---------------------------------------------------------------------------
# 1. the learned potential
# ---------------------------------------------------------------------------

@torch.no_grad()
def probe_potential(model, z, strip, n_samples=48, rmax=0.6):
    """Sample V(r) and |F(r)| for one representative object pair.

    TWO contact distances are returned and both matter:
      contact_meas  = r_i + r_j from the MEASURED radii (where the world says
                      the objects touch)
      contact_model = r_i + r_j from the radii the model actually integrates
                      with (where the MODEL thinks they touch)
    An earlier version reported only the measured one. When a checkpoint had
    shrunk its radii, that made the force look like it vanished at contact --
    the force was fine, it was just being probed 0.03 away from the model's own
    contact. Reporting both makes that failure visible instead of misleading.
    """
    N = min(8, z["attrs"].shape[0])
    at = torch.from_numpy(z["attrs"][:N].astype(np.float32)).unsqueeze(0)
    mass, radius, e = model.properties(strip(at))
    contact_model = float(radius[0, 0] + radius[0, 1])
    contact = float(z["attrs"][0, 15] + z["attrs"][1, 15]) if z["attrs"].shape[1] >= 16 \
        else contact_model

    rs = np.linspace(0.01, rmax, n_samples)
    Vs, Fs = [], []
    pres = torch.zeros(1, N)
    pres[0, 0] = 1.0
    pres[0, 1] = 1.0
    for r in rs:
        q = torch.zeros(1, N, 2)
        q[0, 1, 0] = float(r)
        Vs.append(float(model.V(q, mass, radius, e, pres).item()))
        with torch.enable_grad():
            qg = q.clone().requires_grad_(True)
            F = model.forces(qg, mass, radius, e, pres, create_graph=False)
        Fs.append(float(F[0, 0].norm().item()))
    return rs, np.array(Vs), np.array(Fs), contact, contact_model


# ---------------------------------------------------------------------------
# 2 & 3. rollout error against baselines
# ---------------------------------------------------------------------------

def rollout_errors(model, z, strip, horizons, n_starts=4, max_objects=8):
    """Mean position error at each horizon for: model, constant velocity, frozen.
    Returns dict mode -> {h: [errors]} plus the count of samples."""
    pos = z["positions"].astype(np.float32)
    vel = z["velocities"].astype(np.float32)
    pres = z["presence"]
    T = pos.shape[0]
    N = min(max_objects, pos.shape[1])
    Hmax = max(horizons)

    at = torch.from_numpy(z["attrs"][:N].astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        mass, radius, e = model.properties(strip(at))
    mass, radius, e = mass.detach(), radius.detach(), e.detach()

    out = {m: {h: [] for h in horizons} for m in ("model", "constvel", "frozen")}

    # start frames spread through the clip, needing Hmax frames of runway
    cands = [t for t in range(T - Hmax) if (pres[t, :N] > 0).sum() >= 2]
    if not cands:
        return out
    step = max(1, len(cands) // n_starts)
    starts = cands[::step][:n_starts]

    for s in starts:
        live = np.where(pres[s, :N] > 0)[0]
        # object must stay on screen for the whole horizon to be scored
        live = [k for k in live if (pres[s:s + Hmax + 1, k] > 0).all()]
        if len(live) < 2:
            continue
        active = torch.zeros(1, N)
        for k in live:
            active[0, k] = 1.0

        q = torch.zeros(1, N, 2)
        p = torch.zeros(1, N, 2)
        for k in live:
            q[0, k] = torch.from_numpy(pos[s, k])
            p[0, k] = mass[0, k] * torch.from_numpy(vel[s, k])

        q0 = pos[s, :N].copy()
        v0 = vel[s, :N].copy()

        for h in range(1, Hmax + 1):
            q, p = q.detach(), p.detach()
            q, p, _ = model.step(q, p, mass, radius, e, active,
                                 dt=1.0, F=None, create_graph=False)
            if h in horizons:
                truth = pos[s + h, :N]
                idx = np.array(live)
                pred = q[0].detach().numpy()
                out["model"][h].append(
                    float(np.linalg.norm(pred[idx] - truth[idx], axis=-1).mean()))
                cv = q0[idx] + v0[idx] * h
                out["constvel"][h].append(
                    float(np.linalg.norm(cv - truth[idx], axis=-1).mean()))
                out["frozen"][h].append(
                    float(np.linalg.norm(q0[idx] - truth[idx], axis=-1).mean()))
    return out


def merge(acc, new):
    for m in new:
        for h in new[m]:
            acc.setdefault(m, {}).setdefault(h, []).extend(new[m][h])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/trajectories_v2")
    ap.add_argument("--split", default=None)
    ap.add_argument("--clips", type=int, default=120,
                    help="clips sampled per partition")
    ap.add_argument("--horizons", default="1,5,10,20,40")
    ap.add_argument("--legacy-dynamics", default=None)
    a = ap.parse_args()

    horizons = [int(x) for x in a.horizons.split(",")]
    model = load_model(a.model, a.legacy_dynamics)
    from benchmark_eval import STRIP
    strip = STRIP

    files = sorted(glob.glob(os.path.join(a.data, "*.npz")))
    if not files:
        raise SystemExit(f"no .npz files in {a.data}")
    by_num = {_vid_num(f): f for f in files if _vid_num(f) is not None}

    if a.split:
        sp = load_split(a.split)
        parts = {"TRAIN (model saw these)": sorted(sp["train"]),
                 "TEST  (never seen)": sorted(sp["test"])}
    else:
        parts = {"ALL CLIPS": sorted(by_num)}

    print("=" * 70)
    print(f"model : {a.model}")
    print(f"data  : {a.data}")
    print("=" * 70)

    # ---- 1. the learned potential -----------------------------------------
    probe_clip = by_num[sorted(by_num)[0]]
    z = np.load(probe_clip, allow_pickle=True)
    rs, Vs, Fs, contact, contact_model = probe_potential(model, z, strip)
    shrink = contact_model / max(contact, 1e-9)
    print("\n1. THE LEARNED POTENTIAL  V(r)")
    print("   (a working model: steep wall at small r, flat ~0 at large r)")
    print(f"   contact distance, MEASURED radii : {contact:.3f}")
    print(f"   contact distance, MODEL's radii  : {contact_model:.3f}  "
          f"({shrink:.3f}x the measurement)")
    if shrink < 0.97:
        print("   WARNING: the model integrates with SMALLER objects than were")
        print("            measured. It will let objects overlap by "
              f"{contact - contact_model:.3f} before")
        print("            reacting, which produces both missed and invented")
        print("            collisions. Retrain with --freeze-radius.")
    print()
    for row in ascii_plot(rs, Vs):
        print("   " + row)
    print()
    print("      r      V(r)        |F(r)|")
    for k in range(0, len(rs), max(1, len(rs) // 10)):
        step_r = rs[1] - rs[0]
        mark = ""
        if abs(rs[k] - contact_model) < step_r:
            mark = "  <- contact (model)"
        if abs(rs[k] - contact) < step_r:
            mark = "  <- contact (measured)"
        print(f"   {rs[k]:5.3f}  {Vs[k]:11.4e}  {Fs[k]:11.4e}{mark}")

    v_span = float(np.max(Vs) - np.min(Vs))
    f_near = float(Fs[:max(1, len(rs) // 6)].mean())
    f_far = float(Fs[-max(1, len(rs) // 6):].mean())
    # force AT the model's own contact -- the number that says whether a
    # collision actually gets pushed apart
    k_c = int(np.argmin(np.abs(rs - contact_model)))
    print(f"\n   V range over r        : {v_span:.4e}")
    print(f"   |F| at model contact  : {Fs[k_c]:.4e}")
    print(f"   mean |F| near contact : {f_near:.4e}")
    print(f"   mean |F| far away     : {f_far:.4e}")
    if v_span < 1e-6 or f_near < 1e-6:
        print("   VERDICT: the potential is essentially FLAT. The model produces")
        print("            no meaningful force, so it cannot be creating collisions.")
    elif f_near < 3 * max(f_far, 1e-12):
        print("   VERDICT: force near contact is not much larger than far away.")
        print("            The interaction is weak and poorly localised.")
    else:
        print(f"   VERDICT: force is {f_near / max(f_far, 1e-12):.0f}x stronger near")
        print("            contact than far away. The potential has real structure.")

    # ---- 2/3/4. rollout error vs baselines, per partition ------------------
    for label, nums in parts.items():
        nums = [n for n in nums if n in by_num][:a.clips]
        if not nums:
            continue
        acc = {}
        for n in nums:
            z = np.load(by_num[n], allow_pickle=True)
            merge(acc, rollout_errors(model, z, strip, horizons))

        print("\n" + "-" * 70)
        print(f"2. ROLLOUT POSITION ERROR  —  {label}   ({len(nums)} clips)")
        print("   mean distance between predicted and true position, scene units")
        print("   (the frame is 1.0 wide, so 0.05 is 5% of the scene)\n")
        print(f"   {'horizon':>8} | {'MODEL':>10} | {'const vel':>10} | "
              f"{'frozen':>10} | {'model vs constvel':>18}")
        print("   " + "-" * 66)
        for h in horizons:
            if not acc.get("model", {}).get(h):
                continue
            m = float(np.mean(acc["model"][h]))
            c = float(np.mean(acc["constvel"][h]))
            f = float(np.mean(acc["frozen"][h]))
            if m < c:
                verdict = f"{c / max(m, 1e-9):.2f}x BETTER"
            else:
                verdict = f"{m / max(c, 1e-9):.2f}x WORSE"
            print(f"   {h:>8} | {m:>10.4f} | {c:>10.4f} | {f:>10.4f} | {verdict:>18}")

        if acc.get("model", {}).get(horizons[-1]):
            hL = horizons[-1]
            m = float(np.mean(acc["model"][hL]))
            c = float(np.mean(acc["constvel"][hL]))
            print()
            if m > c:
                print("   VERDICT: the model is WORSE than assuming straight line motion.")
                print("            The learned dynamics is actively hurting predictions.")
            elif c / max(m, 1e-9) < 1.15:
                print("   VERDICT: the model barely beats straight line motion. Almost all")
                print("            of its apparent accuracy is just objects moving straight.")
            else:
                print(f"   VERDICT: the model is {c / max(m, 1e-9):.2f}x better than straight")
                print("            line motion at the longest horizon. It is learning.")

    print("\n" + "=" * 70)
    print("HOW TO READ TRAIN vs TEST")
    print("  similar errors on both  -> UNDERFITTING. The model is too weak or")
    print("     undertrained. Fix by training longer, raising capacity, or")
    print("     correcting the training signal. This is the good case: fixable.")
    print("  train much better       -> OVERFITTING. Fix with more data or")
    print("     regularisation.")
    print("  both bad AND no better than const velocity -> the potential is not")
    print("     doing useful work and the dynamics needs rebuilding.")
    print("=" * 70)


if __name__ == "__main__":
    main()
