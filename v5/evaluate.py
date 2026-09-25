"""
v5/evaluate.py  —  the counterfactual benchmark with NO COPIED ANSWERS
=======================================================================

THE RULE
--------
Ground truth is used for INITIAL CONDITIONS only. It is never used as an
answer.

v3's light-cone evaluator did two things with one mechanism. Objects the
intervention could not reach were (a) started from their observed state -- a
legitimate initial condition -- and (b) had their COLLISIONS copied from the
observed record and reported as answers. (b) is the crutch: it answered 831
of 3332 test choices at 100% without consulting the model.

Here every choice the question asks about is answered by SIMULATION:

  for a choice "do objects i and j collide?"
     if i or j was removed              -> no   (logic, not physics; not scored
                                                 as a model decision)
     otherwise:
        s = the frame the simulation starts, chosen as
              min( first frame i or j enters the causal cone,
                   the observed closest approach of i and j  -  WINDOW )
        the WHOLE scene is simulated from its observed state at frame s for
        HORIZON frames, and the answer is whether i and j touch in that
        simulation.

So the model decides every answer. The taint machinery survives only to
choose WHEN each simulation starts, which is the part you asked to keep: a
few frames before the event, from a true state, not 128 frames blind.

WHAT GROUND TRUTH STILL INFORMS, STATED PLAINLY
-----------------------------------------------
1. the initial state at frame s (any simulator needs one)
2. the choice of s for pairs outside the cone: their observed closest
   approach. That leaks "these two get near each other around frame t" but
   not whether they touch, and the question itself already names the pair.
Nothing else. If i and j never get within reach in the observed record, s is
taken from their closest approach anyway and the model is free to say no.

RUN
    python v5/evaluate.py --model v3_6_3d.pt --data data/trajectories_3d \
        --split split3d.json --which test \
        --questions zechennlp/counterfactual/validation-00000-of-00001.json
"""

import argparse
import glob
import os
import sys

import math
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src2"))
from benchmark_eval import (load_model, load_questions, normalise_choices,   # noqa: E402
                            answers_from_conversations, parse_descriptors,
                            program_has, resolve_all, load_split_set,
                            lightcone_rollout, _vid_num, STRIP)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "v6"))
from world import load_spatial, strip_colour, CUBE_MEAN_SUPPORT      # noqa: E402
from fit3d import exact_contact                                      # noqa: E402

CONTACT_SCALE = 1.0   # set from --contact-scale
VOXEL_MODE = False
VOXEL_SHAPES = None
VOXEL_YAW = None
WINDOW = 15        # frames before the event the simulation starts (the design)
HORIZON = 30       # frames simulated from s
TABLE_EXTENT = 0.95


def closest_approach(pos, pres, i, j):
    """Frame of minimum observed centre distance between i and j."""
    d = np.linalg.norm(pos[:, i] - pos[:, j], axis=-1)
    d = np.where((pres[:, i] > 0) & (pres[:, j] > 0), d, np.inf)
    return int(np.argmin(d)) if np.isfinite(d).any() else None


def simulate_pair(model, z, removed, i, j, s, mass, radius, e, thresh, release, end=None):
    """Roll the whole scene from observed state at frame s; return whether
    i and j come into contact (approach-gated, hysteresis, same rule as v3)."""
    pos = z["positions"].astype(np.float32); vel = z["velocities"].astype(np.float32)
    pres = z["presence"]; T = pos.shape[0]; N = mass.shape[1]
    q = torch.zeros(1, N, 2); p = torch.zeros(1, N, 2); active = torch.zeros(1, N)
    for k in range(N):
        if k in removed or pres[s, k] <= 0:
            continue
        active[0, k] = 1.0
        q[0, k] = torch.from_numpy(pos[s, k]); p[0, k] = mass[0, k] * torch.from_numpy(vel[s, k])
    if active[0, i] <= 0 or active[0, j] <= 0:
        return None
    contact = float(z["attrs"][i, 15] + z["attrs"][j, 15]) * CONTACT_SCALE
    prev = float((q[0, i] - q[0, j]).norm()); in_c = (prev - contact) <= thresh
    # Run to the pair's closest approach plus a margin. A fixed horizon from a
    # cone entry cannot reach an event that happens later than it -- object j
    # may not even have entered the scene yet -- and silently biases toward
    # 'no collision'. Each object still starts from its LAST TRUE state.
    stop = min(T, (end if end is not None else s + 2 * WINDOW) + 1)
    for t in range(s + 1, stop):
        for k in range(N):                       # late entries are boundary conditions
            if k not in removed and active[0, k] <= 0 and pres[t, k] > 0 and (pres[:t, k] <= 0).all():
                active[0, k] = 1.0; q[0, k] = torch.from_numpy(pos[t, k])
                p[0, k] = mass[0, k] * torch.from_numpy(vel[t, k])
        q, p = q.detach(), p.detach()
        if VOXEL_MODE:
            from resolve import resolve_voxel
            p, _ev = resolve_voxel(q, p, mass, radius, active,
                                   z["attrs"][:N, 1], VOXEL_SHAPES, VOXEL_YAW)
        q, p, _ = model.step(q, p, mass, radius, e, active, dt=1.0, F=None, create_graph=False)
        for k in range(N):
            if active[0, k] > 0 and float(q[0, k].abs().max()) > TABLE_EXTENT:
                active[0, k] = 0.0
        if active[0, i] <= 0 or active[0, j] <= 0:
            return None
        d = float((q[0, i] - q[0, j]).norm()); gap = d - contact
        if gap <= thresh and d < prev and not in_c:
            return t                                   # the contact frame
        if gap > thresh + release:
            in_c = False
        prev = d
    return None


def _wrap_q(a):
    q = math.pi / 2
    return (a + q / 2) % q - q / 2


def simulate_pair_spatial(model, z, removed, i, j, s, thresh, release, end=None):
    """Same contract as simulate_pair, but the rollout goes through
    SpatialWorld.step_spatial with yaw and angular momentum as state, and the
    contact test uses the true support-function distance when the model is
    orientation-aware. With yaw_known=False this reduces to simulate_pair's
    physics exactly (torque 0, r_i + r_j), so the two models are compared on
    the same code path."""
    pos = z["positions"].astype(np.float32); vel = z["velocities"].astype(np.float32)
    pres = z["presence"]; attrs = z["attrs"].astype(np.float32)
    yaw_d = z["yaw"].astype(np.float32) if "yaw" in z else np.zeros(pres.shape, np.float32)
    T = pos.shape[0]; N = min(8, pos.shape[1])
    at = torch.from_numpy(attrs[:N]).unsqueeze(0); phys = strip_colour(at)
    with torch.no_grad():
        mass, radius, e = model.properties(phys)
    I = model.inertia(phys, mass, radius)
    shapes = ["cube" if attrs[k, 2] > 0.5 else ("sphere" if attrs[k, 3] > 0.5 else "cylinder")
              for k in range(N)]

    def size_of(k):                                   # exact_contact wants half-side for cubes
        r = float(attrs[k, 15]); return r / CUBE_MEAN_SUPPORT if shapes[k] == "cube" else r

    def contact_ij(yi, yj, dx, dy):
        if not model.yaw_known:
            return float(attrs[i, 15] + attrs[j, 15])
        return exact_contact(shapes[i], size_of(i), yi, shapes[j], size_of(j), yj, dx, dy)

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
    if active[0, i] <= 0 or active[0, j] <= 0:
        return None
    d0 = q[0, j] - q[0, i]
    prev = float(d0.norm()); in_c = (prev - contact_ij(float(yaw[0, i]), float(yaw[0, j]), float(d0[0]), float(d0[1]))) <= thresh
    stop = min(T, (end if end is not None else s + 2 * WINDOW) + 1)
    for t in range(s + 1, stop):
        for k in range(N):
            if k not in removed and active[0, k] <= 0 and pres[t, k] > 0 and (pres[:t, k] <= 0).all():
                inject(k, t)
        q, p, yaw, L = (x.detach() for x in (q, p, yaw, L))
        if VOXEL_MODE:
            # contact resolved from voxel occupancy, BEFORE the step, so the
            # normal matches the configuration that produced the contact
            from resolve import resolve_voxel
            p, _ev = resolve_voxel(q, p, mass, radius, active, attrs[:N, 1],
                                   [("cube" if attrs[k,2] > .5 else
                                     ("sphere" if attrs[k,3] > .5 else "cylinder"))
                                    for k in range(N)],
                                   yaw[0].detach().cpu().numpy())
        q, p, yaw, L = model.step_spatial(q, p, yaw, L, mass, radius, e, active, phys, 1.0,
                                          create_graph=False)
        for k in range(N):
            if active[0, k] > 0 and float(q[0, k].abs().max()) > TABLE_EXTENT:
                active[0, k] = 0.0
        if active[0, i] <= 0 or active[0, j] <= 0:
            return None
        dv = q[0, j] - q[0, i]; d = float(dv.norm())
        gap = d - contact_ij(float(yaw[0, i]), float(yaw[0, j]), float(dv[0]), float(dv[1]))
        if gap <= thresh and d < prev and not in_c:
            return t
        if gap > thresh + release:
            in_c = False
        prev = d
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True, nargs="+")
    ap.add_argument("--data", default="data/trajectories_3d")
    ap.add_argument("--split", default=None)
    ap.add_argument("--which", default="test")
    ap.add_argument("--limit", type=int, default=None, help="max questions")
    ap.add_argument("--thresh", type=float, default=0.02)
    ap.add_argument("--release", type=float, default=0.02)
    ap.add_argument("--spatial", action="store_true",
                    help="load as SpatialWorld and simulate through step_spatial")
    ap.add_argument("--yaw-known", action="store_true",
                    help="with --spatial: orientation-aware contact and torque ON")
    ap.add_argument("--contact-scale", type=float, default=1.0,
                    help="multiply r_i + r_j. Radii come from sqrt(silhouette "
                         "area/pi) and detectors under-segment boundaries, so "
                         "they run ~20%% small: swept on VAL, x1.20 with "
                         "thresh 0.03 detects 98.3%% of annotated collisions "
                         "against 79.3%% at x1.00/0.02, for 3.7pp more false "
                         "positives. This is CALIBRATION of the measurement, "
                         "not a change to the physics.")
    ap.add_argument("--voxel", action="store_true",
                    help="resolve contacts from VOXEL OCCUPANCY: the contact "
                         "point is where the two occupied sets meet and the "
                         "normal comes from the pressed geometry. Exact on a "
                         "flat face struck off-centre, where line-of-centres "
                         "is wrong by up to 34 degrees.")
    ap.add_argument("--impulse", action="store_true",
                    help="resolve contacts with the textbook impulse at eval time "
                         "(works on a model trained without it -- the potential "
                         "still pushes, so this is a lower bound on the benefit)")
    ap.add_argument("--max-clip", type=int, default=None,
                    help="only clips numbered below this (match the training set)")
    a = ap.parse_args()

    global CONTACT_SCALE, VOXEL_MODE
    CONTACT_SCALE = a.contact_scale
    VOXEL_MODE = a.voxel
    if a.voxel:
        print("contact: VOXEL OCCUPANCY (contact point + normal from pressed voxels)")
    if a.contact_scale != 1.0:
        print("contact scale: x%.2f  threshold %.3f" % (a.contact_scale, a.thresh))
    if a.spatial:
        model = load_spatial(a.model, yaw_known=a.yaw_known)
        print("spatial model, yaw_known =", a.yaw_known)
    else:
        model = load_model(a.model)
    if a.impulse:
        model.impulse = True
        print("contact: IMPULSE along the normal")
    files = sorted(glob.glob(os.path.join(a.data, "*.npz")))
    by_num = {_vid_num(f): f for f in files if _vid_num(f) is not None}
    if a.max_clip is not None:
        by_num = {k: v for k, v in by_num.items() if k < a.max_clip}
    allowed = load_split_set(a.split, a.which) if a.split else None
    rows = []
    for qf in a.questions:
        rows.extend(load_questions(qf))

    print("model :", a.model, "  data :", a.data)
    print("rule  : ground truth = initial conditions ONLY; every answer is simulated")
    print("=" * 66)
    n_q = n_q_right = n_c = n_c_right = unscore = 0
    paths = {"participant_removed": [0, 0], "simulated_in_cone": [0, 0],
             "simulated_outside_cone": [0, 0]}
    cone_cache = {}

    for row in rows:
        v = _vid_num(str(row.get("video_filename", row.get("video", ""))))
        if v is None or v not in by_num or (allowed is not None and v not in allowed):
            continue
        if a.limit and n_q >= a.limit:
            break
        z = np.load(by_num[v], allow_pickle=True)
        keys = [str(k) for k in z["obj_keys"]]; N = min(8, z["positions"].shape[1])
        qd = parse_descriptors(row.get("program", []))
        if not qd: unscore += 1; continue
        removed = set(k for k in resolve_all(qd[0], keys) if k < N)
        if not removed: unscore += 1; continue
        negate = program_has(row.get("program", []), "negate")

        ck = (v, frozenset(removed))
        if ck not in cone_cache:
            _, _, taint, _ = lightcone_rollout(model, z, removed, a.thresh, a.release, 0.06)
            cone_cache[ck] = taint
        taint = cone_cache[ck]
        at = torch.from_numpy(z["attrs"][:N].astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            mass, radius, e = model.properties(STRIP(at))
        if getattr(model, "impulse", False):
            model._restitution = at[..., 1]        # metal 0.8 / rubber 0.5
        if VOXEL_MODE:
            global VOXEL_SHAPES, VOXEL_YAW
            from voxel import shape_of as _sh
            VOXEL_SHAPES = [_sh(z["attrs"], k) for k in range(N)]
            VOXEL_YAW = (z["yaw"][0, :N] if "yaw" in z else np.zeros(N))
        pos, pres = z["positions"], z["presence"]

        choices = normalise_choices(row.get("choices"))
        if choices and all(c["answer"] is None for c in choices):
            answers_from_conversations(row, choices)
        scored, allr = False, True
        for c in choices:
            truth = c["answer"]
            if truth not in ("correct", "wrong"): unscore += 1; continue
            cd = parse_descriptors(c["program"])
            if len(cd) < 2: unscore += 1; continue
            h1 = [k for k in resolve_all(cd[0], keys) if k < N]
            h2 = [k for k in resolve_all(cd[1], keys) if k < N]
            if len(h1) != 1 or len(h2) != 1 or h1[0] == h2[0]: unscore += 1; continue
            i, j = h1[0], h2[0]
            if i in removed or j in removed:
                happens, why = False, "participant_removed"
            else:
                tainted = [taint[k] for k in (i, j) if k in taint]
                if tainted:
                    # counterfactual pair: the event time is unknowable from the
                    # recording (the observed closest approach lies on a path
                    # the intervention erased), so simulate from the last true
                    # state -- the cone entry -- to the END of the clip
                    s, end = max(0, min(tainted)), pos.shape[0]
                else:
                    # factual pair: observed closest approach is the event
                    ca = closest_approach(pos, pres, i, j)
                    if ca is None: unscore += 1; continue
                    s, end = max(0, ca - WINDOW), ca + WINDOW
                if a.spatial:
                    happens = simulate_pair_spatial(model, z, removed, i, j, s,
                                                    a.thresh, a.release, end=end) is not None
                else:
                    happens = simulate_pair(model, z, removed, i, j, s, mass, radius, e,
                                            a.thresh, a.release, end=end) is not None
                why = "simulated_in_cone" if (i in taint or j in taint) else "simulated_outside_cone"
            label = "correct" if (happens != negate) else "wrong"
            ok = label == truth
            n_c += 1; n_c_right += int(ok); paths[why][1] += 1; paths[why][0] += int(ok)
            scored = True; allr = allr and ok
        if scored:
            n_q += 1; n_q_right += int(allr)

    print("questions scored :", n_q, "   choices scored :", n_c, "   unscoreable :", unscore)
    if n_c: print("PER-CHOICE   : %.1f%%  (%d/%d)" % (100.0 * n_c_right / n_c, n_c_right, n_c))
    if n_q: print("PER-QUESTION : %.1f%%  (%d/%d)" % (100.0 * n_q_right / n_q, n_q_right, n_q))
    print("-" * 66)
    mr = mt = 0
    for k, (r, t) in paths.items():
        if t: print("   %-24s %6.1f%%  (%d/%d)" % (k, 100.0 * r / t, r, t))
        if k != "participant_removed": mr += r; mt += t
    if mt:
        print("   %-24s %6.1f%%  (%d/%d)" % ("MODEL-DECIDED", 100.0 * mr / mt, mr, mt))
        print("   model-decided share of all choices : %.0f%%   (v3 light-cone: 18%%)" % (100.0 * mt / max(n_c, 1)))
    print("=" * 66)


if __name__ == "__main__":
    main()
