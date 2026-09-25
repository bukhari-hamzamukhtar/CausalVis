"""
probe/physics_laws.py  —  can textbook physics laws, with fitted constants, beat the
                         learned world model on CLEVRER motion and bounces?
====================================================================================

A TEST, not a new version. It answers one question before any version is built:
if the system is given a library of known laws and picks the best one for each
kind of object, how good are its predictions compared with the learned model?

LAW LIBRARY
  motion between collisions, per (shape, material):
    const   constant velocity (no friction)
    exp     speed shrinks by a fixed PERCENTAGE each frame (damping, c)
    coul    speed shrinks by a fixed AMOUNT each frame (sliding friction, a)
  collisions, per material pair:
    C1      textbook bounce along the contact normal, bounciness e, mass ratio
    C2      C1 plus sliding friction at the contact (coefficient mu)
  The constants are fitted on TRAIN videos; the law for each object kind is
  chosen by its TRAIN fit; everything is scored on VAL.

THREE MEASUREMENTS
  A  free motion: from a true state, predict 20 and 40 frames ahead
     (objects that do not collide in that window)
  B  bounce given the contact: pre-collision velocities and the contact normal
     from the recording -> predicted post-collision velocities
  C  fair head-to-head on the same collisions: start both objects 8 frames
     before the recorded collision and let each system run 16 frames itself;
     compare the outgoing direction with the recording
"""

import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("src2", "v4", "v5", "v6"):
    sys.path.insert(0, os.path.join(ROOT, d))
from benchmark_eval import load_split_set, _vid_num                      # noqa: E402
from world import load_spatial, strip_colour                             # noqa: E402
from voxel import shell_local, place, touching, contact_normal           # noqa: E402
from resolve import resolve_voxel                                        # noqa: E402

torch.set_num_threads(4)
WORLD = 6.0
SIZES = json.load(open(os.path.join(ROOT, "v7", "measured_sizes.json")))
FILES = {_vid_num(f): f for f in glob.glob(os.path.join(ROOT, "data", "trajectories_3d_yaw", "*.npz"))}
TRAIN = sorted(k for k in load_split_set(os.path.join(ROOT, "split3d_yaw.json"), "train") if k in FILES)[:1500]
VAL = sorted(k for k in load_split_set(os.path.join(ROOT, "split3d_yaw.json"), "val") if k in FILES)[:700]
TT = np.arange(1, 41)


def shp(a, k):
    return "cube" if a[k, 2] > 0.5 else ("sphere" if a[k, 3] > 0.5 else "cylinder")


def slope(P, t0, t1):
    ts = np.arange(t0, t1 + 1, dtype=float); X = ts - ts.mean(); Y = P[t0:t1 + 1] - P[t0:t1 + 1].mean(0)
    return (X[:, None] * Y).sum(0) / (X ** 2).sum()


def angle(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b) / (na * nb)))))


def load(v):
    z = np.load(FILES[v], allow_pickle=True)
    keys = [str(k) for k in z["obj_keys"]]
    N = min(8, len(keys), z["positions"].shape[1])
    yaw = z["yaw"] if "yaw" in z else np.zeros(z["presence"].shape, np.float32)
    return z, keys, N, z["positions"].astype(np.float64), z["presence"], z["attrs"].astype(np.float32), yaw


# ============================================================ A. free motion
def free_segments(ids, H=40):
    P0, V0, F, G, ATT = [], [], [], [], []
    for v in ids:
        z, keys, N, pos, pres, at, _ = load(v)
        T = pos.shape[0]
        col = {}
        for f, i, j in z["collisions"]:
            for k in (int(i), int(j)):
                col.setdefault(k, []).append(int(f))
        for k in range(N):
            cf = np.array(col.get(k, []))
            for s in range(4, T - H - 1, 8):
                if not (pres[s - 3:s + H + 1, k] > 0).all():
                    continue
                if cf.size and ((cf >= s - 8) & (cf <= s + H + 8)).any():
                    continue
                v0 = slope(pos[:, k], s - 3, s)
                if np.linalg.norm(v0) < 0.0015:
                    continue
                P0.append(pos[s, k]); V0.append(v0); F.append(pos[s + 1:s + H + 1, k])
                G.append(shp(at, k) + "/" + keys[k].split("_")[1]); ATT.append(at[k])
    return np.array(P0), np.array(V0), np.array(F), np.array(G), np.array(ATT, np.float32)


def pred_exp(P0, V0, c):
    fac = TT.astype(float) if c <= 0 else (1 - c) * (1 - (1 - c) ** TT) / c
    return P0[:, None] + V0[:, None] * fac[None, :, None]


def pred_coul(P0, V0, a):
    sp = np.linalg.norm(V0, axis=1); d = V0 / np.maximum(sp, 1e-12)[:, None]
    dist = np.cumsum(np.clip(sp[:, None] - a * TT[None, :], 0, None), axis=1)
    return P0[:, None] + d[:, None, :] * dist[:, :, None]


def err(pred, F, h):
    return np.linalg.norm(pred[:, h - 1] - F[:, h - 1], axis=1)


def part_a(model):
    t0 = time.time()
    tr = free_segments(TRAIN); va = free_segments(VAL)
    C_GRID = np.linspace(0, 0.04, 41); A_GRID = np.linspace(0, 0.0002, 41)
    laws = {}
    for g in sorted(set(tr[3])):
        m = tr[3] == g
        P0, V0, F = tr[0][m], tr[1][m], tr[2][m]
        const = err(pred_exp(P0, V0, 0.0), F, 20).mean()
        ce = [err(pred_exp(P0, V0, c), F, 20).mean() for c in C_GRID]
        ca = [err(pred_coul(P0, V0, a), F, 20).mean() for a in A_GRID]
        c_best, a_best = C_GRID[int(np.argmin(ce))], A_GRID[int(np.argmin(ca))]
        choice = min((("const", const, 0.0), ("exp", min(ce), c_best), ("coul", min(ca), a_best)), key=lambda x: x[1])
        laws[g] = {"exp_c": float(c_best), "coul_a": float(a_best), "chosen": choice[0], "param": float(choice[2]), "n_train": int(m.sum())}
    P0, V0, F, G, ATT = va
    out = {name: np.zeros((len(P0), 2)) for name in ("const", "exp", "coul", "chosen", "learned")}
    for g, L in laws.items():
        m = G == g
        if not m.any():
            continue
        for name, pred in (("const", pred_exp(P0[m], V0[m], 0.0)), ("exp", pred_exp(P0[m], V0[m], L["exp_c"])),
                           ("coul", pred_coul(P0[m], V0[m], L["coul_a"]))):
            out[name][m, 0] = err(pred, F[m], 20); out[name][m, 1] = err(pred, F[m], 40)
        chosen = out[L["chosen"] if L["chosen"] != "const" else "const"]
        out["chosen"][m] = chosen[m]
    preds = []
    for s in range(0, len(P0), 2000):
        at_t = torch.from_numpy(ATT[s:s + 2000]).unsqueeze(1)
        mass, radius, e = model.properties(strip_colour(at_t))
        q = torch.from_numpy(P0[s:s + 2000]).float().unsqueeze(1)
        p = mass.unsqueeze(-1) * torch.from_numpy(V0[s:s + 2000]).float().unsqueeze(1)
        pr = torch.ones(q.shape[0], 1); traj = []
        for _ in range(40):
            q, p, _ = model.step(q, p, mass, radius, e, pr, dt=1.0, F=None, create_graph=False)
            traj.append(q[:, 0].detach().numpy().copy())
        preds.append(np.stack(traj, 1))
    pred = np.concatenate(preds, 0)
    out["learned"][:, 0] = err(pred, F, 20); out["learned"][:, 1] = err(pred, F, 40)
    print("\n=== A. FREE MOTION (no collision in the window)   train segments %d, val segments %d   (%.0fs)"
          % (len(tr[0]), len(P0), time.time() - t0))
    print("   law chosen per object kind (fitted on train):")
    for g, L in laws.items():
        print("      %-18s chosen %-5s  (damping c=%.4f/frame, friction a=%.6f/frame, n=%d)"
              % (g, L["chosen"], L["exp_c"], L["coul_a"], L["n_train"]))
    print("   VAL mean position error      @20 frames   @40 frames")
    for name, label in (("const", "constant velocity"), ("exp", "damping law (fitted)"), ("coul", "friction law (fitted)"),
                        ("chosen", "LAW CHOSEN PER KIND"), ("learned", "learned world model (v6_voxel)")):
        print("      %-30s  %.5f      %.5f" % (label, out[name][:, 0].mean(), out[name][:, 1].mean()))
    return laws


# ============================================================ B/C. collisions
def collisions(ids):
    rows = []
    for v in ids:
        z, keys, N, pos, pres, at, yaw = load(v)
        T = pos.shape[0]
        C = [(int(f), int(i), int(j)) for f, i, j in z["collisions"] if int(i) < N and int(j) < N]
        for f, i, j in C:
            if f - 11 < 0 or f + 10 >= T:
                continue
            if not ((pres[f - 11:f + 11, i] > 0).all() and (pres[f - 11:f + 11, j] > 0).all()):
                continue
            if any((g, a, b) != (f, i, j) and (a in (i, j) or b in (i, j)) and abs(g - f) <= 12 for g, a, b in C):
                continue
            fc = min(range(f - 2, f + 3), key=lambda t: np.linalg.norm(pos[t, i] - pos[t, j]))
            d = pos[fc, i] - pos[fc, j]; nc = d / (np.linalg.norm(d) + 1e-12)
            si, sj = shp(at, i), shp(at, j)
            Ri, Rj = SIZES[si], SIZES[sj]
            a_ = place(shell_local(si, Ri * WORLD, float(yaw[fc, i])), pos[fc, i] * WORLD)
            b_ = place(shell_local(sj, Rj * WORLD, float(yaw[fc, j])), pos[fc, j] * WORLD)
            hit, point = touching(a_, b_, dilate=6)
            nv = nc
            if hit:
                n3 = contact_normal(a_, b_, point); n2 = np.array([n3[0], n3[1]])
                if np.linalg.norm(n2) > 1e-9:
                    nv = n2 / np.linalg.norm(n2)
            rows.append({"v": v, "f": f, "i": i, "j": j,
                         "VI0": slope(pos[:, i], f - 8, f - 3), "VJ0": slope(pos[:, j], f - 8, f - 3),
                         "VI1": slope(pos[:, i], f + 3, f + 8), "VJ1": slope(pos[:, j], f + 3, f + 8),
                         "nc": nc, "nv": nv, "mi": keys[i].split("_")[1], "mj": keys[j].split("_")[1],
                         "gi": si + "/" + keys[i].split("_")[1], "gj": sj + "/" + keys[j].split("_")[1],
                         "Ri": Ri, "Rj": Rj, "si": si, "sj": sj, "ai": at[i], "aj": at[j],
                         "yi": float(yaw[f - 8, i]), "yj": float(yaw[f - 8, j]),
                         "pi_s": pos[f - 8, i].copy(), "pj_s": pos[f - 8, j].copy(),
                         "vi_s": slope(pos[:, i], f - 10, f - 8), "vj_s": slope(pos[:, j], f - 10, f - 8)})
    return rows


def pair_class(mi, mj):
    return "".join(sorted(("m" if mi == "metal" else "r") + ("m" if mj == "metal" else "r")))


def apply_law(VI, VJ, N_, MI, MJ, E, MU):
    vrel = VI - VJ; vn = (vrel * N_).sum(1); inv = 1.0 / MI + 1.0 / MJ
    Jn = np.where(vn < 0, -(1.0 + E) * vn / inv, 0.0)
    VIp = VI + (Jn / MI)[:, None] * N_; VJp = VJ - (Jn / MJ)[:, None] * N_
    if MU > 0:
        vt = vrel - vn[:, None] * N_; vtn = np.linalg.norm(vt, axis=1); Tn = vt / np.maximum(vtn, 1e-12)[:, None]
        Jt = np.minimum(MU * Jn, vtn / inv)
        VIp = VIp - (Jt / MI)[:, None] * Tn; VJp = VJp + (Jt / MJ)[:, None] * Tn
    return VIp, VJp


def arrays(rows, ratio, E, key):
    VI = np.array([r["VI0"] for r in rows]); VJ = np.array([r["VJ0"] for r in rows])
    N_ = np.array([r[key] for r in rows])
    MI = np.array([ratio if r["mi"] == "metal" else 1.0 for r in rows])
    MJ = np.array([ratio if r["mj"] == "metal" else 1.0 for r in rows])
    EE = np.array([E[pair_class(r["mi"], r["mj"])] for r in rows])
    return VI, VJ, N_, MI, MJ, EE


def loss(rows, VIp, VJp):
    VI1 = np.array([r["VI1"] for r in rows]); VJ1 = np.array([r["VJ1"] for r in rows])
    VI0 = np.array([r["VI0"] for r in rows]); VJ0 = np.array([r["VJ0"] for r in rows])
    return (np.linalg.norm(VIp - VI1, axis=1) + np.linalg.norm(VJp - VJ1, axis=1)) / (
        np.linalg.norm(VI0, axis=1) + np.linalg.norm(VJ0, axis=1) + 1e-9)


def dir_errors(rows, VIp, VJp):
    out, speed = [], []
    for r, a, b in zip(rows, VIp, VJp):
        for pred, obs in ((a, r["VI1"]), (b, r["VJ1"])):
            if np.linalg.norm(obs) > 0.001:
                out.append(angle(pred, obs)); speed.append(np.linalg.norm(pred) / np.linalg.norm(obs))
    return np.array(out), np.array(speed)


def fit_collision_law(rows, key):
    best = None
    for ratio in (0.5, 0.67, 1.0, 1.5, 2.0, 3.0):
        E = {}
        total = 0.0
        for cls in ("mm", "mr", "rr"):
            sub = [r for r in rows if pair_class(r["mi"], r["mj"]) == cls]
            if not sub:
                E[cls] = 0.8; continue
            scores = []
            for e in np.arange(0.1, 1.01, 0.1):
                VI, VJ, N_, MI, MJ, _ = arrays(sub, ratio, {cls: e, "mm": e, "mr": e, "rr": e}, key)
                VIp, VJp = apply_law(VI, VJ, N_, MI, MJ, np.full(len(sub), e), 0.0)
                scores.append((loss(sub, VIp, VJp).sum(), e))
            s, e = min(scores)
            E[cls] = float(e); total += s
        if best is None or total < best[0]:
            best = (total, ratio, E)
    _, ratio, E = best
    mus = []
    for mu in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
        VI, VJ, N_, MI, MJ, EE = arrays(rows, ratio, E, key)
        VIp, VJp = apply_law(VI, VJ, N_, MI, MJ, EE, mu)
        mus.append((loss(rows, VIp, VJp).mean(), mu))
    return ratio, E, min(mus)[1]


def part_bc(model, laws):
    t0 = time.time()
    tr, va = collisions(TRAIN), collisions(VAL)
    print("\n=== B. BOUNCE GIVEN THE CONTACT   isolated two-object collisions: train %d, val %d   (%.0fs)"
          % (len(tr), len(va), time.time() - t0))
    fits = {}
    for key, label in (("nc", "normal = line between centres"), ("nv", "normal = voxel contact patch")):
        ratio, E, mu = fit_collision_law(tr, key)
        fits[key] = (ratio, E, mu)
        print("   fitted on train (%s): metal/rubber mass ratio %.2f, bounciness %s, contact friction %.2f"
              % (label, ratio, {k: round(v, 1) for k, v in E.items()}, mu))
    print("   VAL outgoing direction error (degrees)        median   p75    speed ratio (median)")
    VI0 = np.array([r["VI0"] for r in va]); VJ0 = np.array([r["VJ0"] for r in va])
    de, sp = dir_errors(va, VI0, VJ0)
    print("      %-40s %6.1f  %6.1f   %.2f" % ("no collision at all (keep going)", np.median(de), np.percentile(de, 75), np.median(sp)))
    for key, label in (("nc", "centres"), ("nv", "voxel")):
        ratio, E, mu = fits[key]
        for use_mu, name in ((0.0, "C1 bounce"), (mu, "C2 bounce + contact friction")):
            VI, VJ, N_, MI, MJ, EE = arrays(va, ratio, E, key)
            VIp, VJp = apply_law(VI, VJ, N_, MI, MJ, EE, use_mu)
            de, sp = dir_errors(va, VIp, VJp)
            print("      %-40s %6.1f  %6.1f   %.2f" % ("%s, %s normal" % (name, label), np.median(de), np.percentile(de, 75), np.median(sp)))

    # ---------------------------------------------------------- C. fair head-to-head
    ratio, E, mu = fits["nc"]
    law_err, law_hit, mdl_err, mdl_hit = [], 0, [], 0
    t0 = time.time()
    for r in va:
        # law engine: chosen motion law per kind + C2 bounce at the simulated contact
        pi, pj, vi, vj = r["pi_s"].copy(), r["pj_s"].copy(), r["vi_s"].copy(), r["vj_s"].copy()
        hit, ti, tj = False, [], []
        for _ in range(16):
            d = pi - pj; dist = np.linalg.norm(d)
            if not hit and dist - (r["Ri"] + r["Rj"]) <= 0.01 and (vi - vj) @ d < 0:
                n = d / max(dist, 1e-12)
                a, b = apply_law(vi[None], vj[None], n[None], np.array([ratio if r["mi"] == "metal" else 1.0]),
                                 np.array([ratio if r["mj"] == "metal" else 1.0]),
                                 np.array([E[pair_class(r["mi"], r["mj"])]]), mu)
                vi, vj, hit = a[0], b[0], True
            for side in ("i", "j"):
                L = laws.get(r["g" + side], {"chosen": "const", "param": 0.0})
                vel = vi if side == "i" else vj
                if L["chosen"] == "exp":
                    vel = vel * (1 - L["param"])
                elif L["chosen"] == "coul":
                    s = np.linalg.norm(vel); vel = vel * (max(s - L["param"], 0.0) / max(s, 1e-12))
                if side == "i":
                    vi = vel
                else:
                    vj = vel
            pi = pi + vi; pj = pj + vj; ti.append(pi.copy()); tj.append(pj.copy())
        law_hit += hit
        PI, PJ = np.array(ti), np.array(tj)
        for P, obs in ((PI, r["VI1"]), (PJ, r["VJ1"])):
            if np.linalg.norm(obs) > 0.001:
                law_err.append(angle(slope(P, 10, 15), obs))
        # learned world model: the same two objects, the same start, voxel bounces
        at = torch.from_numpy(np.stack([r["ai"], r["aj"]])).unsqueeze(0); phys = strip_colour(at)
        with torch.no_grad():
            mass, radius, e = model.properties(phys)
        q = torch.from_numpy(np.stack([r["pi_s"], r["pj_s"]])).float().unsqueeze(0)
        p = mass.unsqueeze(-1) * torch.from_numpy(np.stack([r["vi_s"], r["vj_s"]])).float().unsqueeze(0)
        yaw = torch.tensor([[r["yi"], r["yj"]]]); L_ = torch.zeros(1, 2); act = torch.ones(1, 2)
        hitm, traj = False, []
        for _ in range(16):
            q, p, yaw, L_ = (x.detach() for x in (q, p, yaw, L_))
            p, ev = resolve_voxel(q, p, mass, radius, act, at[0, :, 1].numpy(), [r["si"], r["sj"]], yaw[0].numpy())
            hitm = hitm or bool(ev)
            q, p, yaw, L_ = model.step_spatial(q, p, yaw, L_, mass, radius, e, act, phys, 1.0, create_graph=False)
            traj.append(q[0].detach().numpy().copy())
        mdl_hit += hitm
        TR = np.array(traj)
        for k, obs in ((0, r["VI1"]), (1, r["VJ1"])):
            if np.linalg.norm(obs) > 0.001:
                mdl_err.append(angle(slope(TR[:, k], 10, 15), obs))
    law_err, mdl_err = np.array(law_err), np.array(mdl_err)
    print("\n=== C. FAIR HEAD-TO-HEAD: start 8 frames before the recorded collision, each system runs 16 frames   (%.0fs)" % (time.time() - t0))
    print("   %-46s contact made   direction error median   p75" % "")
    print("   %-46s %5.1f%%        %6.1f°               %6.1f°" % ("law engine (chosen motion laws + C2 bounce)", 100 * law_hit / len(va), np.median(law_err), np.percentile(law_err, 75)))
    print("   %-46s %5.1f%%        %6.1f°               %6.1f°" % ("learned world model v6_voxel (voxel bounces)", 100 * mdl_hit / len(va), np.median(mdl_err), np.percentile(mdl_err, 75)))


if __name__ == "__main__":
    model = load_spatial(os.path.join(ROOT, "v6_voxel.pt"), yaw_known=False)
    laws = part_a(model)
    part_bc(model, laws)
