"""
v5/analyse_failures.py  —  what exactly is the 26% failing at?
===============================================================

Two questions, one script.

**1. What kind of wrong?** A false POSITIVE (model invents a collision) and a
false NEGATIVE (model misses one) have opposite fixes: the first wants a
tighter contact threshold or stiffer repulsion, the second wants a looser one
or better trajectories. Reporting only "74.1%" hides which.

**2. Could a better READER fix any of it?** `evaluate.py` answers from a yes/no
lookup on the simulation, which is already a perfect reading of that binary. So
a language model reading the same store can only beat it by using what the
binary throws away: HOW CLOSE the pair came. This records, for every wrong
answer, the closest approach the simulation actually produced.

  - wrong answers where the sim came within a hair of the right call
    -> the information IS in the store, a smarter reader can recover it,
       the VLM step is worth having
  - wrong answers where the sim was nowhere near
    -> the store is simply wrong, no reader can fix it, and the VLM is
       decoration

That distinction decides whether the language-model step is a real component
or a fig leaf, so it is worth measuring before building more of it.
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
sys.path.insert(0, HERE)
from benchmark_eval import (load_model, load_questions, normalise_choices,   # noqa: E402
                            answers_from_conversations, parse_descriptors,
                            program_has, resolve_all, load_split_set,
                            lightcone_rollout, _vid_num, STRIP)
from evaluate import closest_approach, WINDOW, TABLE_EXTENT       # noqa: E402
from world import load_spatial                                    # noqa: E402


def load_any(path):
    """A SpatialWorld checkpoint carries rot_drag_head.*; a plain one does not.
    Detect from the file so the caller never has to know which it has."""
    sd = torch.load(path, map_location="cpu")
    if any(k.startswith("rot_drag_head.") for k in sd):
        return load_spatial(path, yaw_known=False)
    return load_model(path)


def simulate_pair_detail(model, z, removed, i, j, s, mass, radius, e,
                         thresh, release, end=None):
    """Like evaluate.simulate_pair but also returns the MINIMUM gap reached.

    The gap is what a richer reader would see: 'they passed within 0.004 of
    touching' is very different evidence from 'they never came closer than
    0.3', even though the yes/no verdict is 'no collision' for both.
    """
    pos = z["positions"].astype(np.float32); vel = z["velocities"].astype(np.float32)
    pres = z["presence"]; T = pos.shape[0]; N = mass.shape[1]
    q = torch.zeros(1, N, 2); p = torch.zeros(1, N, 2); active = torch.zeros(1, N)
    for k in range(N):
        if k in removed or pres[s, k] <= 0:
            continue
        active[0, k] = 1.0
        q[0, k] = torch.from_numpy(pos[s, k]); p[0, k] = mass[0, k] * torch.from_numpy(vel[s, k])
    if active[0, i] <= 0 or active[0, j] <= 0:
        return None, None
    contact = float(z["attrs"][i, 15] + z["attrs"][j, 15])
    prev = float((q[0, i] - q[0, j]).norm()); in_c = (prev - contact) <= thresh
    min_gap = prev - contact
    stop = min(T, (end if end is not None else s + 2 * WINDOW) + 1)
    for t in range(s + 1, stop):
        for k in range(N):
            if k not in removed and active[0, k] <= 0 and pres[t, k] > 0 and (pres[:t, k] <= 0).all():
                active[0, k] = 1.0; q[0, k] = torch.from_numpy(pos[t, k])
                p[0, k] = mass[0, k] * torch.from_numpy(vel[t, k])
        q, p = q.detach(), p.detach()
        q, p, _ = model.step(q, p, mass, radius, e, active, dt=1.0, F=None, create_graph=False)
        for k in range(N):
            if active[0, k] > 0 and float(q[0, k].abs().max()) > TABLE_EXTENT:
                active[0, k] = 0.0
        if active[0, i] <= 0 or active[0, j] <= 0:
            return None, min_gap
        d = float((q[0, i] - q[0, j]).norm()); gap = d - contact
        min_gap = min(min_gap, gap)
        if gap <= thresh and d < prev and not in_c:
            return t, min_gap
        if gap > thresh + release:
            in_c = False
        prev = d
    return None, min_gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--data", default="data/trajectories_3d")
    ap.add_argument("--split", default="split3d.json")
    ap.add_argument("--which", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--thresh", type=float, default=0.02)
    ap.add_argument("--release", type=float, default=0.02)
    ap.add_argument("--out", default="v5/failures.json")
    a = ap.parse_args()

    model = load_any(a.model)
    files = sorted(glob.glob(os.path.join(a.data, "*.npz")))
    by_num = {_vid_num(f): f for f in files if _vid_num(f) is not None}
    allowed = load_split_set(a.split, a.which)
    rows = load_questions(a.questions)

    recs = []
    cone_cache = {}
    n_q = 0
    for row in rows:
        v = _vid_num(str(row.get("video_filename", row.get("video", ""))))
        if v is None or v not in by_num or v not in allowed:
            continue
        if a.limit and n_q >= a.limit:
            break
        z = np.load(by_num[v], allow_pickle=True)
        keys = [str(k) for k in z["obj_keys"]]; N = min(8, z["positions"].shape[1])
        qd = parse_descriptors(row.get("program", []))
        if not qd:
            continue
        removed = set(k for k in resolve_all(qd[0], keys) if k < N)
        if not removed:
            continue
        negate = program_has(row.get("program", []), "negate")
        ck = (v, frozenset(removed))
        if ck not in cone_cache:
            _, _, taint, _ = lightcone_rollout(model, z, removed, a.thresh, a.release, 0.06)
            cone_cache[ck] = taint
        taint = cone_cache[ck]
        at = torch.from_numpy(z["attrs"][:N].astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            mass, radius, e = model.properties(STRIP(at))
        pos, pres = z["positions"], z["presence"]
        choices = normalise_choices(row.get("choices"))
        if choices and all(c["answer"] is None for c in choices):
            answers_from_conversations(row, choices)
        scored = False
        for c in choices:
            truth = c["answer"]
            if truth not in ("correct", "wrong"):
                continue
            cd = parse_descriptors(c["program"])
            if len(cd) < 2:
                continue
            h1 = [k for k in resolve_all(cd[0], keys) if k < N]
            h2 = [k for k in resolve_all(cd[1], keys) if k < N]
            if len(h1) != 1 or len(h2) != 1 or h1[0] == h2[0]:
                continue
            i, j = h1[0], h2[0]
            if i in removed or j in removed:
                happens, why, mg = False, "participant_removed", None
            else:
                tainted = [taint[k] for k in (i, j) if k in taint]
                if tainted:
                    s, end, why = max(0, min(tainted)), pos.shape[0], "in_cone"
                else:
                    ca = closest_approach(pos, pres, i, j)
                    if ca is None:
                        continue
                    s, end, why = max(0, ca - WINDOW), ca + WINDOW, "outside_cone"
                fr, mg = simulate_pair_detail(model, z, removed, i, j, s, mass, radius,
                                              e, a.thresh, a.release, end=end)
                happens = fr is not None
            label = "correct" if (happens != negate) else "wrong"
            # ground truth about the PHYSICAL event, with the negation undone
            truth_collides = (truth == "correct") != negate
            recs.append({"video": v, "pair": [int(i), int(j)], "path": why,
                         "negate": bool(negate),
                         "model_collides": bool(happens),
                         "truth_collides": bool(truth_collides),
                         "ok": label == truth,
                         "min_gap": (None if mg is None else round(float(mg), 5)),
                         "contact": round(float(z["attrs"][i, 15] + z["attrs"][j, 15]), 5)})
            scored = True
        if scored:
            n_q += 1

    json.dump(recs, open(a.out, "w"))
    tot = len(recs); right = sum(r["ok"] for r in recs)
    print("=" * 70)
    print("model:", a.model, "  choices:", tot, "  correct: %.1f%%" % (100.0 * right / max(tot, 1)))
    print("=" * 70)
    print("\nWHAT KIND OF WRONG")
    fp = [r for r in recs if not r["ok"] and r["model_collides"] and not r["truth_collides"]]
    fn = [r for r in recs if not r["ok"] and not r["model_collides"] and r["truth_collides"]]
    print("   invented a collision (false positive) : %5d  (%.1f%% of all choices)"
          % (len(fp), 100.0 * len(fp) / max(tot, 1)))
    print("   missed a collision  (false negative)  : %5d  (%.1f%%)"
          % (len(fn), 100.0 * len(fn) / max(tot, 1)))
    print("\n   by path:")
    for path in ("in_cone", "outside_cone", "participant_removed"):
        sub = [r for r in recs if r["path"] == path]
        if not sub:
            continue
        w = [r for r in sub if not r["ok"]]
        print("      %-20s %5d choices, %5d wrong (%.1f%%)  FP %d / FN %d"
              % (path, len(sub), len(w), 100.0 * len(w) / len(sub),
                 sum(1 for r in w if r["model_collides"]),
                 sum(1 for r in w if not r["model_collides"])))

    print("\nCOULD A BETTER READER FIX IT? (how close the simulation actually came)")
    print("   For each wrong answer, min_gap is the closest the pair got.")
    print("   gap <= 0 means they overlapped; contact distance is ~0.117.\n")
    for name, sub in (("false negatives (missed)", fn), ("false positives (invented)", fp)):
        gaps = np.array([r["min_gap"] for r in sub if r["min_gap"] is not None])
        if not gaps.size:
            continue
        near = int((np.abs(gaps) < 0.02).sum())
        print("   %-28s n=%d   median gap %+.4f" % (name, gaps.size, np.median(gaps)))
        print("      within 0.02 of the right call (a threshold tweak away): %d (%.0f%%)"
              % (near, 100.0 * near / gaps.size))
        print("      nowhere near (|gap| > 0.05)                          : %d (%.0f%%)"
              % (int((np.abs(gaps) > 0.05).sum()),
                 100.0 * int((np.abs(gaps) > 0.05).sum()) / gaps.size))
    print("\nwrote per-choice records to", a.out)


if __name__ == "__main__":
    main()
