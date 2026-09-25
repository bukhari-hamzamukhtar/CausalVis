"""
probe/shadow_replay.py  —  replay the REAL video with our simulator and find where it breaks
===========================================================================================

Most remaining benchmark mistakes are missed collisions in "what if" worlds,
which have no recording to compare against. But the same physics runs in the
real video. So: for every real (annotated) collision, start the simulator from
the true state L frames before it, run it with no help, and compare it with the
recording frame by frame.

For each collision this records:
  - reproduced?        did the simulated pair touch within +-5 frames of the real one
  - chain depth        how many real collisions of either object happened between
                       the start and this one (bounces that had to come out right first)
  - FIRST DIVERGENCE   the first frame any involved object is more than DEV away from
                       where it really was, and what caused it:
        wrong bounce        a real collision happened just before, the simulation also
                            touched, but the objects left in the wrong direction/speed
        missed collision    a real collision happened just before and the simulation
                            never touched
        invented contact    the simulation touched something the real video did not
        bad entry           the object had just entered the scene
        slow drift          none of the above: friction / speed error while coasting
        never diverged      paths stayed within DEV but the contact was not counted

A TEST, not a version. Annotations are used only to find real collisions to replay.

    python probe/shadow_replay.py --engine v8 --clips 200
    python probe/shadow_replay.py --engine v7 --clips 200
"""

import argparse
import glob
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("src2", "v4", "v5", "v6", "v7"):
    sys.path.insert(0, os.path.join(ROOT, d))
from benchmark_eval import load_split_set, _vid_num      # noqa: E402
from world import load_spatial, strip_colour             # noqa: E402
from resolve import resolve_voxel                        # noqa: E402
from diagnose import shape_name                          # noqa: E402

_spec = importlib.util.spec_from_file_location("v8_cf_world", os.path.join(ROOT, "v8", "cf_world.py"))
V8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V8)

TABLE_EXTENT = 0.95
CAL_SCALE, CAL_GAP = 1.2, 0.03     # the benchmark's collision check
DEV = 0.03                          # "diverged": a quarter of a typical contact distance
VEL_FRAMES = 3                      # 3 = the 3-frame velocity average the benchmark uses
ENTRY_FIX = None                    # v8/entry_profile.json when --entry-fix is on


def simulate(engine, model, laws, z, s, end):
    """Every object present at frame s starts from its true state; later objects
    start when they enter. Nothing follows the recording after that."""
    pos = z["positions"].astype(np.float32); vel = z["velocities"].astype(np.float32)
    pres = z["presence"]; attrs = z["attrs"].astype(np.float32)
    yaw_d = z["yaw"].astype(np.float32) if "yaw" in z else np.zeros(pres.shape, np.float32)
    T = pos.shape[0]; N = min(8, pos.shape[1]); end = min(end, T - 1)
    shapes = [shape_name(attrs, k) for k in range(N)]
    at = torch.from_numpy(attrs[:N]).unsqueeze(0); phys = strip_colour(at)
    with torch.no_grad():
        mass, radius, e = model.properties(phys)
    ctx = V8.make_law_context(z, shapes, N, laws) if engine == "v8" else None
    mode = ["absent"] * N
    q_s = torch.zeros(N, 2); p_s = torch.zeros(N, 2); y_s = torch.zeros(N); L_s = torch.zeros(N)
    M = np.zeros((T, N), np.int8)                       # nobody follows the video
    P = np.full((T, N, 2), np.nan, np.float32)

    def start(k, t):
        vs = [vel[u, k] for u in range(max(0, t - 2), t + 1) if pres[u, k] > 0]
        v0 = np.mean(vs, axis=0) if vs else vel[t, k]
        if VEL_FRAMES > 3:
            # a straight line through the object's last VEL_FRAMES recorded
            # positions -- PAST frames only, so nothing from the future leaks in
            past = [u for u in range(max(0, t - VEL_FRAMES + 1), t + 1) if pres[u, k] > 0]
            if len(past) >= 3:
                ts = np.array(past, float); X = ts - ts.mean(); Y = pos[past, k] - pos[past, k].mean(0)
                v0 = (X[:, None] * Y).sum(0) / (X ** 2).sum()
        p0 = pos[t, k].astype(np.float64)
        if ENTRY_FIX is not None:
            # an object still sliding into view is measured too SLOW and too far
            # AHEAD (the visible part's centre): undo both with the TRAIN profile
            present = np.where(pres[:, k] > 0)[0]
            e0 = int(present[0]) if present.size else 0
            tau = t - e0
            if e0 > 0 and 0 <= tau < len(ENTRY_FIX["ratio"]):
                v0 = np.asarray(v0, np.float64) / max(ENTRY_FIX["ratio"][tau], 0.3)
                sp = float(np.linalg.norm(v0))
                if sp > 1e-9:
                    p0 = p0 - ENTRY_FIX["lag"][tau] * v0
        q_s[k] = torch.from_numpy(p0.astype(np.float32))
        p_s[k] = mass[0, k] * torch.from_numpy(np.asarray(v0, np.float32))
        y_s[k] = float(yaw_d[t, k]); L_s[k] = 0.0
        mode[k] = "sim"

    for t in range(s, end + 1):
        for k in range(N):
            if mode[k] == "absent" and pres[t, k] > 0 and (t == s or pres[t - 1, k] <= 0):
                start(k, t)
        for k in range(N):
            if mode[k] == "sim":
                P[t, k] = q_s[k].numpy()
        if t == end or not any(m == "sim" for m in mode):
            continue
        if engine == "v8":
            V8.law_step(t, mode, M, pos, vel, q_s, p_s, mass, ctx, N)
        else:
            q = torch.zeros(1, N, 2); p = torch.zeros(1, N, 2); act = torch.zeros(1, N)
            yaw = torch.zeros(1, N); L = torch.zeros(1, N)
            for k in range(N):
                if mode[k] == "sim":
                    act[0, k] = 1; q[0, k] = q_s[k]; p[0, k] = p_s[k]; yaw[0, k] = y_s[k]; L[0, k] = L_s[k]
            p, _ = resolve_voxel(q, p, mass, radius, act, attrs[:N, 1], shapes, yaw[0].numpy())
            q, p, yaw, L = model.step_spatial(q, p, yaw, L, mass, radius, e, act, phys, 1.0, create_graph=False)
            for k in range(N):
                if mode[k] == "sim":
                    q_s[k] = q[0, k].detach(); p_s[k] = p[0, k].detach()
                    y_s[k] = yaw[0, k].detach(); L_s[k] = L[0, k].detach()
        for k in range(N):
            if mode[k] == "sim" and float(q_s[k].abs().max()) > TABLE_EXTENT:
                mode[k] = "gone"
    return P


def touched(X, i, j, t0, t1, rad):
    """Did i and j come within the benchmark's collision check in frames t0..t1?"""
    for t in range(max(0, t0), min(X.shape[0] - 1, t1) + 1):
        if np.isfinite(X[t, i]).all() and np.isfinite(X[t, j]).all():
            if np.linalg.norm(X[t, i] - X[t, j]) - CAL_SCALE * (rad[i] + rad[j]) <= CAL_GAP:
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=("v7", "v8"), required=True)
    ap.add_argument("--clips", type=int, default=200)
    ap.add_argument("--lookbacks", default="12,30")
    ap.add_argument("--vel-frames", type=int, default=3,
                    help="start velocity from a line through the last K past positions (K>3)")
    ap.add_argument("--entry-fix", action="store_true",
                    help="correct objects that entered recently with v8/entry_profile.json")
    a = ap.parse_args()
    global VEL_FRAMES, ENTRY_FIX
    VEL_FRAMES = a.vel_frames
    if a.entry_fix:
        ENTRY_FIX = json.load(open(os.path.join(ROOT, "v8", "entry_profile.json")))
    torch.set_num_threads(1)
    model = load_spatial(os.path.join(ROOT, "v6_voxel.pt"), yaw_known=False)
    laws = json.load(open(os.path.join(ROOT, "v8", "laws.json")))
    files = {_vid_num(f): f for f in glob.glob(os.path.join(ROOT, "data", "trajectories_3d_yaw", "*.npz"))}
    val = sorted(k for k in load_split_set(os.path.join(ROOT, "split3d_yaw.json"), "val") if k in files)[:a.clips]
    lookbacks = [int(x) for x in a.lookbacks.split(",")]
    recs, t0 = [], time.time()
    for n_clip, v in enumerate(val):
        z = np.load(files[v], allow_pickle=True)
        pos = z["positions"].astype(np.float32); pres = z["presence"]; attrs = z["attrs"]
        T = pos.shape[0]; N = min(8, pos.shape[1]); rad = attrs[:N, 15]
        real = np.where((pres[:, :N] > 0)[..., None], pos[:, :N], np.nan)
        cols = sorted((int(f), int(i), int(j)) for f, i, j in z["collisions"] if int(i) < N and int(j) < N)
        enter = {k: int(np.argmax(pres[:, k] > 0)) for k in range(N) if (pres[:, k] > 0).any()}
        for f, i, j in cols:
            for L in lookbacks:
                s, end = f - L, f + 5
                if s < 0 or end >= T or pres[s, i] <= 0 and enter.get(i, T) > s or False:
                    pass
                if s < 0 or end >= T:
                    continue
                sim = simulate(a.engine, model, laws, z, s, end)
                ok = touched(sim, i, j, f - 5, f + 5, rad)
                depth = sum(1 for g, x, y in cols if s < g < f and ({x, y} & {i, j}))
                rec = {"video": v, "f": f, "i": i, "j": j, "L": L, "reproduced": bool(ok), "depth": depth,
                       "real_seen": bool(touched(real, i, j, f - 5, f + 5, rad))}
                if not ok:
                    involved = {i, j} | {x for g, x, y in cols if s < g <= f and y in (i, j)} \
                                      | {y for g, x, y in cols if s < g <= f and x in (i, j)}
                    first = None
                    for t in range(s, end + 1):
                        for k in involved:
                            if np.isfinite(sim[t, k]).all() and np.isfinite(real[t, k]).all() \
                                    and np.linalg.norm(sim[t, k] - real[t, k]) > DEV:
                                first = (t, k); break
                        if first:
                            break
                    if first is None:
                        cause = "never diverged"
                    else:
                        t_, k_ = first
                        near = [(g, x, y) for g, x, y in cols if t_ - 8 <= g <= t_ and k_ in (x, y)]
                        if near:
                            g, x, y = near[-1]
                            cause = "wrong bounce" if touched(sim, x, y, g - 5, g + 5, rad) else "missed collision"
                        elif enter.get(k_, -99) >= t_ - 10:
                            cause = "bad entry"
                        elif any(touched(sim, k_, m, t_ - 8, t_, rad) and not touched(real, k_, m, t_ - 8, t_, rad)
                                 for m in range(N) if m != k_):
                            cause = "invented contact"
                        else:
                            cause = "slow drift"
                        rec["diverged_after"] = int(t_ - s)
                    rec["cause"] = cause
                recs.append(rec)
        if (n_clip + 1) % 25 == 0:
            print("  %d/%d clips, %d replays, %.0fs" % (n_clip + 1, len(val), len(recs), time.time() - t0), flush=True)

    out = os.path.join(ROOT, "probe", "shadow_%s%s%s.jsonl" % (a.engine, "" if a.vel_frames == 3 else "_vf%d" % a.vel_frames,
                                                            "_entryfix" if a.entry_fix else ""))
    with open(out, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    print("\n" + "=" * 78)
    print("SHADOW REPLAY, engine %s: %d val clips, real collisions replayed from L frames before" % (a.engine, len(val)))
    print("=" * 78)
    for L in lookbacks:
        sub = [r for r in recs if r["L"] == L]
        if not sub:
            continue
        rep = sum(r["reproduced"] for r in sub)
        seen = sum(r["real_seen"] for r in sub)
        print("\nstart %d frames before: reproduced %.1f%% (%d/%d)   [the collision check sees %.1f%% on the real recording]"
              % (L, 100 * rep / len(sub), rep, len(sub), 100 * seen / len(sub)))
        for name, cond in (("no earlier bounce needed", lambda r: r["depth"] == 0),
                           ("1 earlier bounce needed", lambda r: r["depth"] == 1),
                           ("2+ earlier bounces needed", lambda r: r["depth"] >= 2)):
            s2 = [r for r in sub if cond(r)]
            if s2:
                print("   %-28s %5.1f%% (%d/%d)" % (name, 100 * sum(r["reproduced"] for r in s2) / len(s2),
                                                  sum(r["reproduced"] for r in s2), len(s2)))
        fails = [r for r in sub if not r["reproduced"]]
        c = Counter(r["cause"] for r in fails)
        print("   WHY THE MISSES HAPPENED (first thing that went wrong), %d misses:" % len(fails))
        for cause, n in c.most_common():
            da = [r["diverged_after"] for r in fails if r["cause"] == cause and "diverged_after" in r]
            print("      %-18s %4d  (%4.1f%%)%s" % (cause, n, 100 * n / max(len(fails), 1),
                                                 ("   typically %d frames after the start" % int(np.median(da))) if da else ""))


if __name__ == "__main__":
    main()
