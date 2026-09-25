"""
v6/diagnose.py  —  every benchmark choice taken apart: where does it fail?
=========================================================================

For each of the 3328 test choices this records, side by side:

  MODEL      the benchmark's own verdict, reproduced exactly (same rollout,
             same voxel impulses, same centre-distance contact rule), so the
             totals must equal v6/cmp/v6_voxel_trained.log

  SENSOR     the voxel-sensor reading of the SAME simulated motion: the lattice
             distance between the two objects' surface voxels at every frame
             (0 = they press the same voxel, 1 = neighbouring voxels)

  RECORDING  the same two readings on the RECORDED motion from the video, no
             model involved -- what the sensors see when the trajectory is
             right by construction

  VIDEO      the annotated collisions between the pair, whether either object
             was hit by the removed object, and whether both objects were on
             the table when the simulation started

With those, every wrong answer can be put in exactly one bucket: the
simulation refused to run, the cone mis-labelled the pair, the reconstruction
never shows contact even in the real recording, the sensor saw contact but the
rule did not count it, the simulation came close, or it never came close.

Sharded so four processes can run it in parallel on this CPU.

    python v6/diagnose.py --model v6_voxel.pt --shard 0 --nshards 4
    python v6/diagnose_report.py
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
for d in ("src2", "v4", "v5", "v6"):
    sys.path.insert(0, os.path.join(ROOT, d))

from scipy.spatial import cKDTree                                        # noqa: E402
from benchmark_eval import (load_questions, normalise_choices,           # noqa: E402
                            answers_from_conversations, parse_descriptors,
                            program_has, resolve_all, load_split_set,
                            lightcone_rollout, _vid_num)
from evaluate import closest_approach, WINDOW, TABLE_EXTENT, _wrap_q     # noqa: E402
from world import load_spatial, strip_colour                             # noqa: E402
from voxel import shell_local, VOXEL                                     # noqa: E402
from resolve import resolve_voxel                                        # noqa: E402

WORLD = 6.0
PREFILTER = 0.20      # normalised; beyond this the surfaces are > ~25 voxels apart
FAR = 99.0
NEAR_KEEP = 12.0      # store per-frame voxel gaps up to this many voxels
THRESH = 0.02
RELEASE = 0.02

_TREES = {}


def shape_name(attrs, k):
    return "cube" if attrs[k, 2] > 0.5 else ("sphere" if attrs[k, 3] > 0.5 else "cylinder")


def _tree(shape, r, yaw):
    loc = shell_local(shape, r * WORLD, yaw)          # cached array, stable identity
    key = id(loc)
    if key not in _TREES:
        _TREES[key] = (loc, cKDTree(loc.astype(np.float64)))
    return _TREES[key]


def voxel_gap(sh_i, r_i, y_i, c_i, sh_j, r_j, y_j, c_j):
    """Lattice distance between the surface voxels of i and j. Centres in
    normalised units. 0 = a shared voxel. FAR when the centres are too far apart
    for the surfaces to be within ~25 voxels."""
    if float(np.hypot(c_i[0] - c_j[0], c_i[1] - c_j[1])) > PREFILTER:
        return FAR
    loc_i, _ = _tree(sh_i, r_i, y_i)
    _, tr_j = _tree(sh_j, r_j, y_j)
    off_i = np.round(np.array([c_i[0], c_i[1], 0.0]) * WORLD / VOXEL).astype(np.int64)
    off_j = np.round(np.array([c_j[0], c_j[1], 0.0]) * WORLD / VOXEL).astype(np.int64)
    d, _ = tr_j.query((loc_i + off_i - off_j).astype(np.float64), k=1)
    return float(d.min())


def rule_verdict(dists, contact):
    """evaluate.py's contact rule on a distance sequence; dists[0] = start frame.
    Returns the index of the first counted contact, or None."""
    prev = dists[0]
    in_c = (prev - contact) <= THRESH
    for k in range(1, len(dists)):
        d = dists[k]
        gap = d - contact
        if gap <= THRESH and d < prev and not in_c:
            return k
        if gap > THRESH + RELEASE:
            in_c = False
        prev = d
    return None


def simulate_full(model, z, removed, i, j, s, end):
    """simulate_pair_spatial with voxel impulses, run over the whole window
    instead of stopping at the first contact, recording i and j every frame.
    Frames stop where evaluate.py would return (i or j left the table)."""
    pos = z["positions"].astype(np.float32); vel = z["velocities"].astype(np.float32)
    pres = z["presence"]; attrs = z["attrs"].astype(np.float32)
    yaw_d = z["yaw"].astype(np.float32) if "yaw" in z else np.zeros(pres.shape, np.float32)
    T = pos.shape[0]; N = min(8, pos.shape[1])
    at = torch.from_numpy(attrs[:N]).unsqueeze(0); phys = strip_colour(at)
    with torch.no_grad():
        mass, radius, e = model.properties(phys)
    I = model.inertia(phys, mass, radius)
    shapes = [shape_name(attrs, k) for k in range(N)]
    q = torch.zeros(1, N, 2); p = torch.zeros(1, N, 2); active = torch.zeros(1, N)
    yaw = torch.zeros(1, N); L = torch.zeros(1, N)

    def inject(k, t):
        active[0, k] = 1.0; q[0, k] = torch.from_numpy(pos[t, k])
        p[0, k] = mass[0, k] * torch.from_numpy(vel[t, k]); yaw[0, k] = float(yaw_d[t, k])
        w = _wrap_q(float(yaw_d[min(t + 1, T - 1), k] - yaw_d[t, k])) if t + 1 < T else 0.0
        L[0, k] = float(I[0, k]) * w

    for k in range(N):
        if k not in removed and pres[s, k] > 0:
            inject(k, s)
    out = {"present_i_at_s": bool(pres[s, i] > 0), "present_j_at_s": bool(pres[s, j] > 0),
           "refused": None, "left_table": None, "frames": [], "qi": [], "qj": [],
           "yi": [], "yj": []}
    if active[0, i] <= 0 or active[0, j] <= 0:
        out["refused"] = "absent_at_start"
        return out

    def record(t):
        out["frames"].append(t)
        out["qi"].append(q[0, i].detach().numpy().copy()); out["qj"].append(q[0, j].detach().numpy().copy())
        out["yi"].append(float(yaw[0, i])); out["yj"].append(float(yaw[0, j]))

    record(s)
    stop = min(T, end + 1)
    for t in range(s + 1, stop):
        for k in range(N):
            if k not in removed and active[0, k] <= 0 and pres[t, k] > 0 and (pres[:t, k] <= 0).all():
                inject(k, t)
        q, p, yaw, L = (x.detach() for x in (q, p, yaw, L))
        p, _ = resolve_voxel(q, p, mass, radius, active, attrs[:N, 1], shapes,
                             yaw[0].detach().cpu().numpy())
        q, p, yaw, L = model.step_spatial(q, p, yaw, L, mass, radius, e, active, phys, 1.0,
                                          create_graph=False)
        for k in range(N):
            if active[0, k] > 0 and float(q[0, k].abs().max()) > TABLE_EXTENT:
                active[0, k] = 0.0
        if active[0, i] <= 0 or active[0, j] <= 0:
            out["left_table"] = int(t)
            break
        record(t)
    return out


def near_list(frames, gaps):
    return [[int(f), round(g, 2)] for f, g in zip(frames, gaps) if g <= NEAR_KEEP]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v6_voxel.pt")
    ap.add_argument("--data", default="data/trajectories_3d_yaw")
    ap.add_argument("--split", default="split3d_yaw.json")
    ap.add_argument("--which", default="test")
    ap.add_argument("--questions",
                    default="zechennlp/counterfactual/validation-00000-of-00001.json")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="v6/cmp/diag")
    a = ap.parse_args()
    torch.set_num_threads(1)
    os.makedirs(a.out, exist_ok=True)
    fout = open(os.path.join(a.out, "shard%d.jsonl" % a.shard), "w", encoding="utf-8")

    model = load_spatial(a.model, yaw_known=False)
    files = sorted(glob.glob(os.path.join(a.data, "*.npz")))
    by_num = {_vid_num(f): f for f in files if _vid_num(f) is not None}
    allowed = load_split_set(a.split, a.which)
    rows = load_questions(a.questions)

    qn = -1
    n_done = 0
    cone_cache = {}
    for row in rows:
        v = _vid_num(str(row.get("video_filename", row.get("video", ""))))
        if v is None or v not in by_num or v not in allowed:
            continue
        qn += 1
        if qn % a.nshards != a.shard:
            continue
        if a.limit and n_done >= a.limit:
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
            cone_out = lightcone_rollout(model, z, removed, THRESH, RELEASE, 0.06)
            cone_cache[ck] = cone_out
        cone_out = cone_cache[ck]
        taint = cone_out[2]
        cause = cone_out[3] if len(cone_out) > 3 and isinstance(cone_out[3], dict) else {}
        pos, pres = z["positions"], z["presence"]
        attrs = z["attrs"].astype(np.float32)
        yaw_d = z["yaw"] if "yaw" in z else np.zeros(pres.shape, np.float32)
        T = pos.shape[0]
        cols = [(int(f), int(x), int(y)) for f, x, y in z["collisions"]]
        choices = normalise_choices(row.get("choices"))
        if choices and all(c["answer"] is None for c in choices):
            answers_from_conversations(row, choices)

        for ci, c in enumerate(choices):
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
            rec = {"video": v, "qid": str(row.get("question_id")), "question": row.get("question", ""),
                   "choice_index": ci, "choice": c["choice"], "truth": truth, "negate": bool(negate),
                   "truth_collides": (truth == "correct") != bool(negate),
                   "n_objects": int(N), "removed": sorted(int(k) for k in removed),
                   "removed_shape": [shape_name(attrs, k) for k in sorted(removed)],
                   "i": int(i), "j": int(j), "names": [keys[i], keys[j]],
                   "shapes": sorted([shape_name(attrs, i), shape_name(attrs, j)]),
                   "obs_frames": [f for f, x, y in cols if {x, y} == {i, j}],
                   "i_hit_removed_obs": any((x == i and y in removed) or (y == i and x in removed) for f, x, y in cols),
                   "j_hit_removed_obs": any((x == j and y in removed) or (y == j and x in removed) for f, x, y in cols),
                   "enter_i": int(np.argmax(pres[:, i] > 0)) if (pres[:, i] > 0).any() else None,
                   "enter_j": int(np.argmax(pres[:, j] > 0)) if (pres[:, j] > 0).any() else None}
            if i in removed or j in removed:
                rec.update({"path": "participant_removed", "model_happens": False})
                rec["model_label"] = "correct" if (False != bool(negate)) else "wrong"
                rec["ok"] = rec["model_label"] == truth
                fout.write(json.dumps(rec) + "\n")
                continue
            tainted = [taint[k] for k in (i, j) if k in taint]
            if tainted:
                s, end, path = max(0, min(tainted)), T, "in_cone"
            else:
                ca = closest_approach(pos, pres, i, j)
                if ca is None:
                    continue
                s, end, path = max(0, ca - WINDOW), ca + WINDOW, "outside_cone"
            contact = float(attrs[i, 15] + attrs[j, 15])
            rec.update({"path": path, "s": int(s), "end": int(min(end, T - 1)),
                        "horizon": int(min(end, T - 1) - s),
                        "taint_i": taint.get(i), "taint_j": taint.get(j),
                        "cause_i": (cause.get(i) or {}).get("kind") if isinstance(cause.get(i), dict) else None,
                        "cause_j": (cause.get(j) or {}).get("kind") if isinstance(cause.get(j), dict) else None,
                        "contact": round(contact, 4)})

            # ---- the model: exact benchmark verdict + sensor reading --------
            sim = simulate_full(model, z, removed, i, j, s, end)
            rec["present_i_at_s"] = sim["present_i_at_s"]; rec["present_j_at_s"] = sim["present_j_at_s"]
            rec["refused"] = sim["refused"]; rec["left_table"] = sim["left_table"]
            sh_i, sh_j = shape_name(attrs, i), shape_name(attrs, j)
            ri, rj = float(attrs[i, 15]), float(attrs[j, 15])
            if sim["refused"] is None:
                dists = [float(np.linalg.norm(a_ - b_)) for a_, b_ in zip(sim["qi"], sim["qj"])]
                k_rule = rule_verdict(dists, contact)
                gaps = [voxel_gap(sh_i, ri, yi_, qi_, sh_j, rj, yj_, qj_)
                        for qi_, qj_, yi_, yj_ in zip(sim["qi"], sim["qj"], sim["yi"], sim["yj"])]
                rec["model_happens"] = k_rule is not None
                rec["model_frame"] = sim["frames"][k_rule] if k_rule is not None else None
                rec["sim_min_centre_gap"] = round(min(d - contact for d in dists), 4)
                rec["sim_vgap_start"] = round(gaps[0], 2)
                rec["sim_vgap_min"] = round(min(gaps[1:]) if len(gaps) > 1 else FAR, 2)
                rec["sim_near"] = near_list(sim["frames"], gaps)
                rec["sim_frames"] = len(sim["frames"])
            else:
                rec["model_happens"] = False
                rec["model_frame"] = None
            rec["model_label"] = "correct" if (rec["model_happens"] != bool(negate)) else "wrong"
            rec["ok"] = rec["model_label"] == truth

            # ---- the recording: same readings, no model ---------------------
            fr = [t for t in range(s, min(T, end + 1)) if pres[t, i] > 0 and pres[t, j] > 0]
            if fr:
                d_rec = [float(np.linalg.norm(pos[t, i] - pos[t, j])) for t in fr]
                g_rec = [voxel_gap(sh_i, ri, float(yaw_d[t, i]), pos[t, i], sh_j, rj,
                                   float(yaw_d[t, j]), pos[t, j]) for t in fr]
                rec["rec_rule"] = rule_verdict(d_rec, contact) is not None
                rec["rec_min_centre_gap"] = round(min(d - contact for d in d_rec), 4)
                rec["rec_vgap_min"] = round(min(g_rec), 2)
                rec["rec_near"] = near_list(fr, g_rec)
            else:
                rec["rec_rule"] = None
            both = [t for t in range(T) if pres[t, i] > 0 and pres[t, j] > 0]
            if both:
                rec["rec_vgap_min_clip"] = round(min(
                    voxel_gap(sh_i, ri, float(yaw_d[t, i]), pos[t, i], sh_j, rj,
                              float(yaw_d[t, j]), pos[t, j]) for t in both), 2)
            fout.write(json.dumps(rec) + "\n")
        fout.flush()
        n_done += 1
        print("shard %d  question %d  video %d  done" % (a.shard, n_done, v), flush=True)
    fout.close()


if __name__ == "__main__":
    main()
