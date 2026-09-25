"""
paper_exp/cophy_balls.py  —  CausalVis' engine on CoPhy BallsCF (counterfactual trajectories)
==========================================================================================

CoPhy (Baradel et al., ICLR 2020), BallsCF: 2-6 balls slide in a walled box. Each ball has
HIDDEN confounders (mass, friction, restitution). A video AB is observed; then one ball's
starting position is changed (C) and the task is to predict the new 30-step trajectory D.
Metric (CoPhy): mean squared error of 2D positions, averaged over time steps and balls.

INPUT: object states (positions, velocities) from CoPhy's files -- the state-based
setting of the CoPhy paper's tables. Confounders are NEVER read at test time; they are
inferred from AB, exactly as in our ComPhy and CLEVRER engines.

 1. LAWS fitted on the 4-ball TRAIN split: ball radius, wall position, how two
    restitutions combine, wall restitution (grid search, simulated AB error).
 2. INFER each ball's (mass, restitution) by coordinate descent: the values whose
    simulation of AB best matches the observed AB.
 3. PREDICT D by simulating from C's initial state with the inferred values.

    python paper_exp/cophy_balls.py baselines          # reproduce the paper's copy baselines
    python paper_exp/cophy_balls.py fit                # laws on the 4-ball train split
    python paper_exp/cophy_balls.py eval --n 4         # test split, n balls
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "external", "cophy", "data", "CoPhy_224", "ballsCF")
SPLITS = os.path.join(ROOT, "external", "cophy", "splits")
LAWS_P = os.path.join(HERE, "cophy_balls_laws.json")
DT = 0.2
SUB = 40
MASSES = (1.0, 2.0, 5.0)
RESTS = (0.1, 0.5, 1.0)


def ids(split, n):
    return [l.strip() for l in open(os.path.join(SPLITS, "ballsCF_%s_%d.txt" % (split, n))) if l.strip()]


def load(n, i):
    d = os.path.join(DATA, str(n), i)
    ab = np.load(os.path.join(d, "ab", "states.npy")).astype(np.float64)
    cd = np.load(os.path.join(d, "cd", "states.npy")).astype(np.float64)
    conf = np.load(os.path.join(d, "confounders.npy")).astype(np.float64)
    act = np.abs(ab[0, :, :3]).sum(1) > 0
    return {"ab": ab[:, act], "cd": cd[:, act], "conf": conf[act]}


def mse(pred, gt):
    """CoPhy's metric as reproduced by the copy baselines (see `baselines`)."""
    return float(((pred[:, :, :2] - gt[:, :, :2]) ** 2).sum(-1).mean())


def simulate(p0, v0, m, e, L, steps):
    """Balls in a square box: constant velocity, ball-ball impulses (restitution combined
    as L['combine']), wall bounces (restitution e_i * L['wall_e'])."""
    p, v = p0.copy(), v0.copy()
    n = len(m); h = DT / SUB; R = L["radius"]; W = L["wall"]
    out = np.zeros((steps, n, 2)); out[0] = p
    for t in range(1, steps):
        for _ in range(SUB):
            p += v * h
            for i in range(n):
                for ax in range(2):
                    if p[i, ax] > W - R and v[i, ax] > 0:
                        v[i, ax] = -v[i, ax] * e[i] * L["wall_e"]
                    elif p[i, ax] < -W + R and v[i, ax] < 0:
                        v[i, ax] = -v[i, ax] * e[i] * L["wall_e"]
            for i in range(n):
                for j in range(i + 1, n):
                    d = p[i] - p[j]; dist = np.sqrt(d @ d)
                    if dist < 2 * R and dist > 1e-9:
                        nrm = d / dist; vn = (v[i] - v[j]) @ nrm
                        if vn < 0:
                            ee = e[i] * e[j] if L["combine"] == "product" else min(e[i], e[j]) if L["combine"] == "min" else 0.5 * (e[i] + e[j])
                            J = -(1 + ee) * vn / (1 / m[i] + 1 / m[j])
                            v[i] += J / m[i] * nrm; v[j] -= J / m[j] * nrm
        out[t] = p
    return out


def infer(ab, L, sweeps=2):
    n = ab.shape[1]
    P, V = ab[:, :, :2], ab[:, :, 3:5]
    m = np.full(n, 2.0); e = np.full(n, 0.5)

    def cost(m, e):
        sim = simulate(P[0], V[0], m, e, L, len(P))
        return float(((sim - P) ** 2).sum())
    c0 = cost(m, e)
    for _ in range(sweeps):
        for i in range(n):
            best = (c0, m[i], e[i])
            for mm, ee in itertools.product(MASSES, RESTS):
                m2, e2 = m.copy(), e.copy(); m2[i], e2[i] = mm, ee
                c = cost(m2, e2)
                if c < best[0] - 1e-12:
                    best = (c, mm, ee)
            c0, m[i], e[i] = best
    return m, e


def baselines(n=4, split="test", limit=None):
    """Copy B (D := the observed B) and Copy C (D := C held still)."""
    cb, cc = [], []
    for i in ids(split, n)[:limit]:
        s = load(n, i)
        cb.append(mse(s["ab"], s["cd"]))
        cc.append(mse(np.repeat(s["cd"][:1], len(s["cd"]), 0), s["cd"]))
    print("BallsCF %d balls, %s (%d): copy B %.3f   copy C %.3f   (CoPhy paper 4->4: 2.688 / 6.538)"
          % (n, split, len(cb), np.mean(cb), np.mean(cc)))


def fit(limit=300):
    tr = ids("train", 4)[:limit]
    scenes = [load(4, i) for i in tr]
    best = None
    for R in (0.55, 0.6, 0.65, 0.7):
        for W in (4.5, 4.75, 5.0, 5.25):
            for comb in ("product", "min"):
                for we in (1.0, 0.9):
                    L = {"radius": R, "wall": W, "combine": comb, "wall_e": we}
                    # true confounders are allowed on TRAIN, to fit the laws themselves
                    err = np.mean([mse(simulate(sc["ab"][0, :, :2], sc["ab"][0, :, 3:5], sc["conf"][:, 0], sc["conf"][:, 2], L,
                                                len(sc["ab"])), sc["ab"]) for sc in scenes[:120]])
                    if best is None or err < best[0]:
                        best = (err, L)
                    print(L, "AB error %.4f" % err, flush=True)
    json.dump(best[1], open(LAWS_P, "w"), indent=1)
    print("chosen", best)


def evaluate(n, split="test", limit=None, oracle=False):
    L = json.load(open(LAWS_P))
    errs, cb = [], []
    for i in ids(split, n)[:limit]:
        s = load(n, i)
        if oracle:
            m, e = s["conf"][:, 0], s["conf"][:, 2]
        else:
            m, e = infer(s["ab"], L)
        cd = s["cd"]
        pred = simulate(cd[0, :, :2], cd[0, :, 3:5], m, e, L, len(cd))
        errs.append(mse(pred, cd))
        cb.append(mse(s["ab"], cd))
    res = {"n": n, "split": split, "scenes": len(errs), "mse": float(np.mean(errs)), "copy_B": float(np.mean(cb)), "oracle": oracle}
    print(json.dumps(res), flush=True)
    return res


def contacts(P, R, gap=0.05):
    """Contact onsets (step, i, j) in a trajectory [T, n, 2]."""
    T, n = P.shape[:2]
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(P[:, i] - P[:, j], axis=-1)
            hit = d < 2 * R + gap
            on = np.where(hit & ~np.concatenate([[False], hit[:-1]]))[0]
            out += [(int(t), i, j) for t in on]
    return out


def audit(n, limit=300):
    """Right for the right reason in the counterfactual world itself: CoPhy gives the true
    counterfactual trajectory D, so every collision our predicted D contains can be matched
    with a collision in the true D (same pair, within one step)."""
    L = json.load(open(LAWS_P))
    tp = fp = fn = 0
    for i in ids("test", n)[:limit]:
        s_ = load(n, i)
        m, e = infer(s_["ab"], L)
        cd = s_["cd"]
        pred = simulate(cd[0, :, :2], cd[0, :, 3:5], m, e, L, len(cd))
        ours, real = contacts(pred, L["radius"]), contacts(cd[:, :, :2], L["radius"])
        used = set()
        for t, a, b in ours:
            k = [q for q, (u, x, y) in enumerate(real) if (x, y) == (a, b) and abs(u - t) <= 1 and q not in used]
            if k:
                used.add(k[0]); tp += 1
            else:
                fp += 1
        fn += len(real) - len(used)
    res = {"n": n, "scenes": min(limit, len(ids("test", n))), "event_precision": tp / max(1, tp + fp),
           "event_recall": tp / max(1, tp + fn), "tp": tp, "fp": fp, "fn": fn}
    print(json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["baselines", "fit", "eval", "audit"])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--oracle", action="store_true")
    a = ap.parse_args()
    if a.mode == "audit":
        audit(a.n, a.limit or 300)
    elif a.mode == "baselines":
        baselines(a.n, a.split, a.limit)
    elif a.mode == "fit":
        fit()
    else:
        evaluate(a.n, a.split, a.limit, a.oracle)
