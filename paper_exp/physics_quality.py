"""
paper_exp/physics_quality.py  —  how good is each simulator AS PHYSICS (no questions)
====================================================================================

The x-axis of the audit: for every simulator the benchmark is run with, two
measurements on the same 200 VAL-A videos (detection-only tracks):

  rollout error   every object starts from its recorded state at frames 10, 40 and 70
                  and is simulated alone (later entries start when they appear);
                  mean distance to the recording after 20 and after 40 frames.
  replay          for every annotated collision, start L = 12 and 30 frames before it;
                  reproduced = the pair passes the benchmark's collision check within
                  +-5 frames (probe/shadow_replay.py's test).
Annotations are used only to locate real collisions for the replay.

    python paper_exp/physics_quality.py --sim learned --model v6_voxel.pt --out paper_exp/quality/learned.json
    python paper_exp/physics_quality.py --sim laws --out paper_exp/quality/laws.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for d in ("src2", "v4", "v5", "v6", "v7"):
    sys.path.insert(0, os.path.join(ROOT, d))
from benchmark_eval import load_split_set, _vid_num, load_questions      # noqa: E402
from world import load_spatial, strip_colour                             # noqa: E402
from resolve import resolve_voxel                                        # noqa: E402
from diagnose import shape_name                                          # noqa: E402
import importlib.util                                                    # noqa: E402
_spec = importlib.util.spec_from_file_location("paper_cf_world_q", os.path.join(HERE, "cf_world.py"))
CFW = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CFW)

TABLE_EXTENT = 0.95
CAL_SCALE, CAL_GAP = 1.2, 0.03
LAWS = json.load(open(os.path.join(ROOT, "v8", "laws.json")))
ENTRY = json.load(open(os.path.join(HERE, "entry_profile.json")))


def simulate(sim, model, z, s, end, no_voxel=False):
    pos = z["positions"].astype(np.float32); vel = z["velocities"].astype(np.float32)
    pres = z["presence"]; attrs = z["attrs"].astype(np.float32)
    yaw_d = z["yaw"].astype(np.float32) if "yaw" in z else np.zeros(pres.shape, np.float32)
    T = pos.shape[0]; N = min(8, pos.shape[1]); end = min(end, T - 1)
    shapes = [shape_name(attrs, k) for k in range(N)]
    at = torch.from_numpy(attrs[:N]).unsqueeze(0); phys = strip_colour(at)
    with torch.no_grad():
        mass, radius, e = model.properties(phys)
    ctx = CFW.make_law_context(z, shapes, N, LAWS)
    mode = ["absent"] * N
    q_s = torch.zeros(N, 2); p_s = torch.zeros(N, 2); y_s = torch.zeros(N); L_s = torch.zeros(N)
    M = np.zeros((T, N), np.int8)
    P = np.full((T, N, 2), np.nan, np.float32)

    def start(k, t):
        vs = [vel[u, k] for u in range(max(0, t - 2), t + 1) if pres[u, k] > 0]
        v0 = np.mean(vs, axis=0) if vs else vel[t, k]
        p0 = pos[t, k].astype(np.float64)
        on = np.where(pres[:, k] > 0)[0]
        e0 = int(on[0]) if on.size else 0
        tau = t - e0
        if e0 > 0 and 0 <= tau < len(ENTRY["ratio"]):
            v0 = np.asarray(v0, np.float64) / max(ENTRY["ratio"][tau], 0.3)
            p0 = p0 - ENTRY["lag"][tau] * v0
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
        if sim == "laws":
            CFW.law_step(t, mode, M, pos, vel, q_s, p_s, mass, ctx, N)
        elif sim in ("straight", "straight_friction"):
            CFW.simple_step(mode, q_s, p_s, mass, ctx, N, sim == "straight_friction")
        else:
            q = torch.zeros(1, N, 2); p = torch.zeros(1, N, 2); act = torch.zeros(1, N)
            yaw = torch.zeros(1, N); L = torch.zeros(1, N)
            for k in range(N):
                if mode[k] == "sim":
                    act[0, k] = 1; q[0, k] = q_s[k]; p[0, k] = p_s[k]; yaw[0, k] = y_s[k]; L[0, k] = L_s[k]
            if not no_voxel:
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
    for t in range(max(0, t0), min(X.shape[0] - 1, t1) + 1):
        if np.isfinite(X[t, i]).all() and np.isfinite(X[t, j]).all():
            if np.linalg.norm(X[t, i] - X[t, j]) - CAL_SCALE * (rad[i] + rad[j]) <= CAL_GAP:
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="learned", choices=["learned", "laws", "straight", "straight_friction"])
    ap.add_argument("--model", default="v6_voxel.pt")
    ap.add_argument("--no-voxel", action="store_true")
    ap.add_argument("--no-force", action="store_true")
    ap.add_argument("--drag-scale", type=float, default=1.0)
    ap.add_argument("--clips", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    model = load_spatial(os.path.join(ROOT, a.model), yaw_known=False)
    if a.drag_scale != 1.0:
        cls = type(model); s_ = a.drag_scale
        model.drag_coeff = lambda e_, pr_: (lambda c_: None if c_ is None else c_ * s_)(cls.drag_coeff(model, e_, pr_))
        model.rot_drag_coeff = lambda e_, pr_: (lambda c_: None if c_ is None else c_ * s_)(cls.rot_drag_coeff(model, e_, pr_))
    if a.no_force:
        model.forces_torques = lambda q_, yaw_, *r_, **k_: (torch.zeros_like(q_), torch.zeros_like(yaw_))
    files = {_vid_num(f): f for f in glob.glob(os.path.join(ROOT, "data", "trajectories_3d_det", "*.npz"))}
    qv = set(_vid_num(r["video"]) for r in load_questions(os.path.join(ROOT, "zechennlp/counterfactual/validation-00000-of-00001.json")))
    vids = sorted(v for v in load_split_set(os.path.join(ROOT, "split3d_yaw.json"), "val") if v in files and v in qv)[:a.clips]
    err20, err40, rep = [], [], {12: [], 30: []}
    for v in vids:
        z = np.load(files[v], allow_pickle=True)
        pos, pres = z["positions"], z["presence"] > 0
        N = min(8, pos.shape[1]); rad = z["attrs"][:N, 15]
        for s in (10, 40, 70):
            P = simulate(a.sim, model, z, s, s + 40, a.no_voxel)
            for h, bucket in ((20, err20), (40, err40)):
                t = s + h
                if t >= pos.shape[0]:
                    continue
                for k in range(N):
                    if pres[t, k] and np.isfinite(P[t, k]).all() and pres[s:t + 1, k].all():
                        bucket.append(float(np.linalg.norm(P[t, k] - pos[t, k])))
        for f, i, j in sorted((int(f), int(i), int(j)) for f, i, j in z["collisions"] if int(i) < N and int(j) < N):
            for L in (12, 30):
                s = f - L
                if s < 0:
                    continue
                P = simulate(a.sim, model, z, s, f + 6, a.no_voxel)
                rep[L].append(bool(touched(P, i, j, f - 5, f + 5, rad)))
    out = {"sim": a.sim, "model": a.model, "no_voxel": a.no_voxel, "no_force": a.no_force, "drag_scale": a.drag_scale,
           "videos": len(vids), "rollout_err_20": float(np.mean(err20)), "rollout_err_40": float(np.mean(err40)),
           "n_err40": len(err40), "replay_12": float(np.mean(rep[12])), "replay_30": float(np.mean(rep[30])),
           "n_replay": len(rep[12])}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
