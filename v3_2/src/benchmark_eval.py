"""
benchmark_eval.py  (src2)  —  CLEVRER counterfactual benchmark, LIGHT-CONE design
=================================================================================

One timeline per question, but only the part of the world the intervention can
actually change is simulated.  Removing an object only alters the future of
objects it would have touched (directly or through a chain).  Everything
outside that causal cone is IDENTICAL in the factual and counterfactual worlds
-- and the factual world is not a prediction problem: it is the observed video
stored in the .npz.

Per question:

  1. the video number is read from the question ("video_00013.mp4" -> sim_00013.npz)
  2. the factual world is read straight from that .npz (positions, velocities,
     presence, attrs, obj_keys, recorded collisions)
  3. the question program names the object to delete; if the description
     matches several objects ("the cylinder" with two cylinders), ALL matches
     are removed -- that is the do(.) intervention
  4. ONE pass over the full clip builds the edited-world event log:
        - every object starts UNTAINTED: its counterfactual trajectory equals
          its observed trajectory (read from the npz, exact, zero drift)
        - an object becomes TAINTED the first frame the observed video brings
          it within contact range of a removed object (that encounter now
          cannot happen), or the simulation brings it near an already-tainted
          object (chains).  From that frame it is handed to the physics
          engine, initialised at its exact factual state -- the same accurate
          short-horizon regime that reproduces 97.5% of collisions.
        - collisions between two untainted objects are copied from the
          recorded ground truth (exact).  Collisions involving anything
          tainted are logged from the simulation (per-pair measured contact
          r_i + r_j, approach-gated, with hysteresis).
  5. each choice is answered by LOOKUP in that single event log.

The model does all of the COUNTERFACTUAL work: everything inside the cone,
which is the only part of the world that is actually counterfactual.  Nothing
outside the cone was ever a prediction problem.

Decision rule for "does collision (i,j) happen in the edited world?":
      participant deleted                  -> NO  (trivially cannot happen)
      pair in the edited-world event log   -> YES (exact copy or simulated hit)
      recorded factually, but a participant
        was tainted before it and the sim
        lost it                            -> NO  (the causal vanish signal)
      otherwise                            -> NO  (never happens anywhere)

RUN
    python src2/benchmark_eval.py --model v3_2_dynamics.pt --split split.json \
           --which test --questions zechennlp/counterfactual/train-00000-of-00001.json \
           --limit 400
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import torch

import importlib.util

from dynamics import HamiltonianDynamics, strip_colour

STRIP = strip_colour        # set per checkpoint in load_model (src vs src2)


def load_model(path, legacy_dynamics=None):
    """Load a checkpoint from either generation of the project.

    src2 checkpoints (v3_2_*) match the local dynamics.py.  src checkpoints
    (v3_1_*) were trained with the OLD class (radius_head, 7-dim physical
    attrs); they are detected by their state-dict keys and loaded through
    ../src/dynamics.py (or --legacy-dynamics <path>)."""
    global STRIP
    sd = torch.load(path, map_location="cpu")
    legacy = any(k.startswith("radius_head.") for k in sd)
    if not legacy:
        model = HamiltonianDynamics()
        STRIP = strip_colour
    else:
        cand = legacy_dynamics
        if cand is None:
            here = os.path.dirname(os.path.abspath(__file__))
            cand = os.path.join(os.path.dirname(here), "src", "dynamics.py")
        if not os.path.isfile(cand):
            raise SystemExit(
                f"{path} is a src/-generation checkpoint (radius_head keys) "
                f"but {cand} was not found; pass --legacy-dynamics <path to "
                f"src/dynamics.py>")
        spec = importlib.util.spec_from_file_location("dynamics_legacy", cand)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.HamiltonianDynamics()
        STRIP = mod.strip_colour
        print(f"[legacy checkpoint] using dynamics class from {cand}")
    model.load_state_dict(sd)
    model.eval()
    return model

COLORS = {"gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"}
SHAPES = {"cube", "sphere", "cylinder"}
MATERIALS = {"metal", "rubber"}


# ---------------------------------------------------------------------------
# question JSON loading (tolerant: array, {"rows":...}, or JSON-lines)
# ---------------------------------------------------------------------------

def load_questions(path):
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read().strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("rows", "data", "questions", "train"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
        if isinstance(data, dict):
            data = [data]
        rows = []
        for r in data:
            rows.append(r["row"] if isinstance(r, dict) and "row" in r else r)
        return rows
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows


def normalise_choices(raw):
    """Return a list of {'choice':str,'program':list,'answer':str|None} whether
    the source is a list of dicts or a columnar dict of lists (HF export)."""
    if raw is None:
        return []
    if isinstance(raw, list):
        out = []
        for c in raw:
            out.append({
                "choice": c.get("choice", ""),
                "program": c.get("program", []),
                "answer": c.get("answer", c.get("answers", c.get("label"))),
            })
        return out
    if isinstance(raw, dict):
        progs = raw.get("program", [])
        texts = raw.get("choice", [])
        n = max(len(progs), len(texts))
        ans = raw.get("answer", raw.get("answers", raw.get("label", [None] * n)))
        out = []
        for k in range(n):
            out.append({
                "choice": texts[k] if k < len(texts) else "",
                "program": progs[k] if k < len(progs) else [],
                "answer": ans[k] if isinstance(ans, list) and k < len(ans) else None,
            })
        return out
    return []


def answers_from_conversations(row, choices):
    """Fallback when no per-choice 'answer' labels exist: read the gpt reply
    ('A' / 'A, C' ...) out of the conversations field and mark those letters
    correct, the rest wrong."""
    conv = row.get("conversations")
    if conv is None:
        return False
    values, roles = [], []
    if isinstance(conv, dict):
        roles = conv.get("from", [])
        values = conv.get("value", [])
    elif isinstance(conv, list):
        for m in conv:
            roles.append(m.get("from", ""))
            values.append(m.get("value", ""))
    reply = None
    for role, val in zip(roles, values):
        if role != "human":
            reply = str(val)
    if reply is None:
        return False
    letters = set(re.findall(r"\b([A-H])\b", reply.upper()))
    if not letters:
        return False
    for k, c in enumerate(choices):
        c["answer"] = "correct" if chr(ord("A") + k) in letters else "wrong"
    return True


# ---------------------------------------------------------------------------
# symbolic program parsing  (value token comes BEFORE its filter op:
#  ["objects","cyan","filter_color","cube","filter_shape","unique",...])
# ---------------------------------------------------------------------------

def _tok(step):
    if isinstance(step, str):
        return step
    if isinstance(step, dict):
        return str(step.get("function", step.get("op", step.get("type", ""))))
    return str(step)


def parse_descriptors(program):
    """Split a program into object descriptors.  A descriptor opens at each
    'objects' token; each 'filter_color/shape/material' consumes the token
    just before it as the value.  Returns list of dicts with color/shape/
    material sets (empty set = unconstrained)."""
    descs, cur, prev = [], None, None
    for step in program:
        t = _tok(step)
        if t == "objects":
            cur = {"color": set(), "shape": set(), "material": set()}
            descs.append(cur)
        elif t in ("filter_color", "filter_shape", "filter_material") and cur is not None:
            slot = t.split("_")[1]
            if prev in COLORS | SHAPES | MATERIALS:
                cur[slot].add(prev)
        prev = t
    return descs


def program_has(program, token):
    return any(_tok(s) == token for s in program)


def resolve_all(desc, obj_keys):
    """ALL object indices matching a descriptor.  obj_keys are
    'color_material_shape' strings."""
    hits = []
    for idx, key in enumerate(obj_keys):
        parts = str(key).split("_")
        if len(parts) < 3:
            continue
        colour, material, shape = parts[0], parts[1], parts[2]
        if desc["color"] and colour not in desc["color"]:
            continue
        if desc["material"] and material not in desc["material"]:
            continue
        if desc["shape"] and shape not in desc["shape"]:
            continue
        hits.append(idx)
    return hits


# ---------------------------------------------------------------------------
# split handling
# ---------------------------------------------------------------------------

def _vid_num(name):
    m = re.search(r"(\d+)", os.path.basename(str(name)))
    return int(m.group(1)) if m else None


def load_split_set(path, which):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    entries = data[which] if isinstance(data, dict) else data
    out = set()
    for e in entries:
        n = e if isinstance(e, int) else _vid_num(e)
        if n is not None:
            out.add(int(n))
    return out


# ---------------------------------------------------------------------------
# the light-cone rollout: observed data outside the intervention's causal
# cone, physics simulation inside it
# ---------------------------------------------------------------------------

def lightcone_rollout(model, z, removed, thresh, release, margin,
                      lookback=12, vel_smooth=3, contact_override=None,
                      max_objects=8, record_traj=False):
    """Build the edited-world event log for one clip.

    Returns (events, pair_set, taint_frame):
        events      : list of dicts {frame, i, j, source: 'observed'|'simulated'}
        pair_set    : set of frozenset({i, j}) that collide in the edited world
        taint_frame : dict k -> frame the object entered the causal cone
    """
    pos = z["positions"].astype(np.float32)
    vel = z["velocities"].astype(np.float32)
    pres = z["presence"]
    attrs = z["attrs"].astype(np.float32)
    T = pos.shape[0]
    N = min(max_objects, pos.shape[1])
    removed = set(k for k in removed if k < N)

    at = torch.from_numpy(attrs[:N])
    phys = STRIP(at.unsqueeze(0))
    with torch.no_grad():
        mass, radius_m, e = model.properties(phys)
    mass, radius_m, e = mass.detach(), radius_m.detach(), e.detach()

    # measured contact radius (src2: attrs column 16 = index 15).  For npz
    # files that predate the measured column (src/trajectories_out) use the
    # --contact scalar if given (e.g. the measured 0.119), else the model's
    # own radius head (often too small -- prefer --contact for src data).
    if attrs.shape[1] >= 16:
        r = attrs[:N, 15]
        contact = r[:, None] + r[None, :]
    elif contact_override is not None:
        contact = np.full((N, N), float(contact_override), np.float32)
    else:
        r = radius_m[0].numpy()
        contact = r[:, None] + r[None, :]

    taint_frame = {}                       # k -> frame it entered the cone
    taint_cause = {}                       # k -> {"kind", "other", ...}
    sim_q = torch.zeros(N, 2)              # state, meaningful only for tainted
    sim_p = torch.zeros(N, 2)

    events, pair_set = [], set()
    in_contact, prev_dist = {}, {}
    traj = np.full((T, N, 2), np.nan, np.float32) if record_traj else None
    traj_mode = np.zeros((T, N), np.int8) if record_traj else None
    # traj_mode: 0 = absent, 1 = observed (outside cone), 2 = simulated (in cone)

    # ---- GT-anchored taint schedule ----------------------------------------
    # An object's counterfactual path diverges from its observed path only
    # when a RECORDED contact with a removed object fails to happen.  Taint it
    # `lookback` frames before that recorded collision (the accurate-start
    # recipe of the 97.5% collision check), not on mere proximity: a near
    # pass with no recorded collision changes nothing, and eager tainting
    # only buys extra free-running drift.
    gt_taint_at = {}                       # k -> scheduled taint frame
    gt_taint_why = {}                      # k -> (removed partner, collision frame)
    for row in z["collisions"]:
        f, i, j = int(row[0]), int(row[1]), int(row[2])
        if i >= N or j >= N:
            continue
        pair_rm = None
        if i in removed and j not in removed:
            pair_rm = j
        elif j in removed and i not in removed:
            pair_rm = i
        if pair_rm is None:
            continue
        start = None
        for u in range(max(0, f - lookback), f + 1):
            if pres[u, pair_rm] > 0:
                start = u
                break
        if start is not None:
            if start <= gt_taint_at.get(pair_rm, start):
                gt_taint_at[pair_rm] = start
                gt_taint_why[pair_rm] = (i if pair_rm == j else j, f)

    def taint(k, t, cause):
        if k in taint_frame or k in removed:
            return
        taint_frame[k] = t
        taint_cause[k] = cause
        sim_q[k] = torch.from_numpy(pos[t, k])
        # detector velocity is noisy frame to frame; launch with the mean over
        # the last few frames the object was on screen (all pre-divergence)
        vs = [vel[u, k] for u in range(max(0, t - vel_smooth + 1), t + 1)
              if pres[u, k] > 0]
        v0 = np.mean(vs, axis=0) if vs else vel[t, k]
        sim_p[k] = mass[0, k] * torch.from_numpy(np.asarray(v0, np.float32))

    def coord(k, t):
        """Edited-world position of k at frame t (None if absent there)."""
        if k in removed:
            return None
        if k in taint_frame:
            return sim_q[k]
        if pres[t, k] > 0:
            return torch.from_numpy(pos[t, k])
        return None

    for t in range(T):
        # ---- taint propagation (before logging, so a contact is simulated
        #      by the engine rather than copied from a world it never had) ---
        for j in range(N):
            if j in removed or j in taint_frame or pres[t, j] <= 0:
                continue
            cause = None
            # 1) a RECORDED collision with a removed object is coming up:
            #    that contact cannot happen in the edited world (GT-anchored)
            if gt_taint_at.get(j) == t:
                rm_p, col_f = gt_taint_why[j]
                cause = {"kind": "lost_contact", "other": int(rm_p),
                         "col_frame": int(col_f)}
            # 2) safety net: the video shows j essentially TOUCHING a removed
            #    object without a recorded collision (annotation gaps)
            if cause is None:
                for rr in removed:
                    if pres[t, rr] > 0:
                        d = float(np.linalg.norm(pos[t, j] - pos[t, rr]))
                        if d - contact[j, rr] <= thresh:
                            cause = {"kind": "touch_removed", "other": int(rr)}
                            break
            # 3) a simulated (tainted) object comes near j: chain taint
            #    (no ground truth exists for simulated paths, so proximity
            #    with the wider margin is the only available signal)
            if cause is None:
                for k in taint_frame:
                    d = float((torch.from_numpy(pos[t, j]) - sim_q[k]).norm().item())
                    if d - contact[j, k] <= margin:
                        cause = {"kind": "chain", "other": int(k)}
                        break
            if cause is not None:
                taint(j, t, cause)

        if record_traj:
            for k in range(N):
                ck_ = coord(k, t)
                if ck_ is not None:
                    traj[t, k] = ck_.numpy() if hasattr(ck_, "numpy") else ck_
                    traj_mode[t, k] = 2 if k in taint_frame else 1

        # ---- log simulated collisions (any pair with >= 1 tainted member) --
        for i in range(N):
            ci = coord(i, t)
            if ci is None:
                continue
            for j in range(i + 1, N):
                if i not in taint_frame and j not in taint_frame:
                    continue                     # untainted pairs come from GT
                cj = coord(j, t)
                if cj is None:
                    continue
                d = float((ci - cj).norm().item())
                key = (i, j)
                gap = d - float(contact[i, j])
                if key not in prev_dist:
                    # First frame this pair is comparable, which is the taint
                    # frame.  There is no approach history yet, so the gate
                    # below would pass unconditionally and a pair that is
                    # ALREADY touching at its observed position would be
                    # logged as a brand new counterfactual collision.  It is
                    # not new: it is the observed configuration.  Seed the
                    # state instead of logging.
                    in_contact[key] = gap <= thresh
                    prev_dist[key] = d
                    continue
                approaching = d < prev_dist[key]
                if gap <= thresh and approaching and not in_contact.get(key, False):
                    in_contact[key] = True
                    events.append({"frame": t, "i": i, "j": j, "source": "simulated"})
                    pair_set.add(frozenset(key))
                elif gap > thresh + release:
                    in_contact[key] = False
                prev_dist[key] = d

        # ---- advance the simulation one frame ------------------------------
        # untainted objects are teacher-forced to their observed state (their
        # counterfactual path IS the observed path); tainted objects free-run
        if t < T - 1 and taint_frame:
            q = torch.zeros(1, N, 2)
            p = torch.zeros(1, N, 2)
            active = torch.zeros(1, N)
            for k in range(N):
                if k in removed:
                    continue
                if k in taint_frame:
                    active[0, k] = 1.0
                    q[0, k] = sim_q[k]
                    p[0, k] = sim_p[k]
                elif pres[t, k] > 0:
                    active[0, k] = 1.0
                    q[0, k] = torch.from_numpy(pos[t, k])
                    p[0, k] = mass[0, k] * torch.from_numpy(vel[t, k])
            if active.sum().item() > 0:
                q, p = q.detach(), p.detach()
                q, p, _ = model.step(q, p, mass, radius_m, e, active,
                                     dt=1.0, F=None, create_graph=False)
                for k in taint_frame:
                    sim_q[k] = q[0, k].detach()
                    sim_p[k] = p[0, k].detach()

    # ---- copy the untouched part of the world from the observed record -----
    # a recorded collision is exact in the edited world iff neither participant
    # was removed and neither was inside the cone when it happened
    for row in z["collisions"]:
        f, i, j = int(row[0]), int(row[1]), int(row[2])
        if i >= N or j >= N or i in removed or j in removed:
            continue
        ti = taint_frame.get(i, T + 1)
        tj = taint_frame.get(j, T + 1)
        if f < ti and f < tj:
            events.append({"frame": f, "i": i, "j": j, "source": "observed"})
            pair_set.add(frozenset((i, j)))

    events.sort(key=lambda ev: ev["frame"])
    if record_traj:
        return events, pair_set, taint_frame, taint_cause, traj, traj_mode
    return events, pair_set, taint_frame, taint_cause


# ---------------------------------------------------------------------------
# the decision rule (documented at the top of the file)
# ---------------------------------------------------------------------------

def pair_happens_cf(pair, removed, cf_pairs, gt_pairs, obs_pairs, paths):
    """-> (happens_in_edited_world, decision_path)"""
    i, j = tuple(pair)
    if i in removed or j in removed:
        paths["participant_removed"] += 1
        return False, "participant_removed"
    if pair in cf_pairs:
        if pair in obs_pairs:
            paths["observed_exact"] += 1
            return True, "observed_exact"
        paths["sim_hit"] += 1
        return True, "sim_hit"
    if pair in gt_pairs:
        paths["vanished"] += 1        # in the cone and the sim lost it
        return False, "vanished"
    paths["never"] += 1
    return False, "never"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True, nargs="+",
                    help="one or more question JSON files (e.g. the train and "
                         "validation counterfactual files together)")
    ap.add_argument("--data", default="data/trajectories_v2")
    ap.add_argument("--split", default=None)
    ap.add_argument("--which", default="test")
    ap.add_argument("--limit", type=int, default=None,
                    help="max question rows to scan from the JSON")
    ap.add_argument("--thresh", type=float, default=0.02,
                    help="contact slack: hit if gap <= thresh")
    ap.add_argument("--release", type=float, default=0.02,
                    help="hysteresis: a pair must separate past thresh+release "
                         "before a new event can be logged")
    ap.add_argument("--contact", type=float, default=None,
                    help="scalar contact distance for npz files WITHOUT the "
                         "measured-radius column (src/trajectories_out); "
                         "e.g. the measured 0.119.  Ignored when measured "
                         "radii are present.")
    ap.add_argument("--legacy-dynamics", default=None,
                    help="path to src/dynamics.py for v3_1-generation "
                         "checkpoints (auto-detected at ../src/dynamics.py)")
    ap.add_argument("--lookback", type=int, default=12,
                    help="frames before a recorded removed-object collision "
                         "at which the partner enters the cone (the 97.5%% "
                         "start-window recipe)")
    ap.add_argument("--vel-smooth", type=int, default=3,
                    help="frames of observed velocity averaged when an object "
                         "is injected into the simulation")
    ap.add_argument("--taint-margin", type=float, default=0.06,
                    help="an object enters the causal cone when it comes within "
                         "contact+margin of a removed or tainted object (must "
                         "be > thresh so contacts are simulated, not copied)")
    ap.add_argument("--trace-out", default=None,
                    help="write one JSON record per scored question (the "
                         "grounded explanation trace consumed by explain.py)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    model = load_model(a.model, a.legacy_dynamics)

    rows = []
    for qf in a.questions:
        rows.extend(load_questions(qf))
    split_set = load_split_set(a.split, a.which) if a.split else None

    # index the available .npz files by video number once
    npz_by_num = {}
    for f in glob.glob(os.path.join(a.data, "*.npz")):
        n = _vid_num(f)
        if n is not None:
            npz_by_num[n] = f

    n_questions = n_questions_right = 0
    n_choices = n_choices_right = 0
    unscoreable = 0
    skipped_video = 0
    paths = {"participant_removed": 0, "observed_exact": 0, "sim_hit": 0,
             "vanished": 0, "never": 0}
    path_acc = {k: [0, 0] for k in paths}    # path -> [right, scored]
    cone_sizes = []      # tainted objects per question (cone footprint)
    in_cone_total = in_cone_kept = 0   # recorded collisions the cone had to re-derive
    in_cone_seen = set()               # interventions already counted in that diagnostic
    confusion = {}       # (polarity, truth, model) -> count
    rollout_cache = {}   # (video, frozenset(removed)) -> rollout result

    trace_fh = open(a.trace_out, "w", encoding="utf-8") if a.trace_out else None

    scanned = 0
    for row in rows:
        if a.limit is not None and scanned >= a.limit:
            break
        scanned += 1

        if str(row.get("question_type", "counterfactual")) != "counterfactual":
            continue
        vnum = _vid_num(row.get("video", ""))
        if vnum is None:
            continue
        if split_set is not None and vnum not in split_set:
            continue
        if vnum not in npz_by_num:
            skipped_video += 1
            continue

        z = np.load(npz_by_num[vnum], allow_pickle=True)
        obj_keys = [str(k) for k in z["obj_keys"]]
        N = min(8, z["positions"].shape[1])

        # ---- the do(.) target: remove ALL objects matching the description
        qprog = row.get("program", [])
        qdescs = parse_descriptors(qprog)
        if not qdescs:
            unscoreable += 1
            continue
        removed = set(k for k in resolve_all(qdescs[0], obj_keys) if k < N)
        if not removed:
            unscoreable += 1
            continue
        negate = program_has(qprog, "negate")

        # ---- ground-truth factual pairs from the npz
        gt_pairs = set()
        for r_ in z["collisions"]:
            i, j = int(r_[1]), int(r_[2])
            if i < N and j < N:
                gt_pairs.add(frozenset((i, j)))

        # ---- ONE light-cone pass per question (cached per intervention) ----
        ck = (vnum, frozenset(removed))
        if ck not in rollout_cache:
            rollout_cache[ck] = lightcone_rollout(
                model, z, removed, a.thresh, a.release, a.taint_margin,
                lookback=a.lookback, vel_smooth=a.vel_smooth,
                contact_override=a.contact)
        cf_events, cf_pairs, taint_frame, taint_cause = rollout_cache[ck]
        obs_pairs = set(frozenset((ev["i"], ev["j"])) for ev in cf_events
                        if ev["source"] == "observed")
        cone_sizes.append(len(taint_frame))

        # in-cone diagnostic: recorded collisions the simulation had to
        # re-derive (a participant tainted before the collision frame) --
        # how many did the model keep vs lose under the intervention?
        T_clip = z["positions"].shape[0]
        count_cone = ck not in in_cone_seen
        in_cone_seen.add(ck)
        for row_ in z["collisions"]:
            if not count_cone:
                break
            f_, i_, j_ = int(row_[0]), int(row_[1]), int(row_[2])
            if i_ >= N or j_ >= N or i_ in removed or j_ in removed:
                continue
            if min(taint_frame.get(i_, T_clip + 1),
                   taint_frame.get(j_, T_clip + 1)) <= f_:
                in_cone_total += 1
                if frozenset((i_, j_)) in cf_pairs:
                    in_cone_kept += 1

        # ---- score every choice by lookup in the logs
        choices = normalise_choices(row.get("choices"))
        if choices and all(c["answer"] is None for c in choices):
            answers_from_conversations(row, choices)

        q_scored = q_all_right = True
        any_scored = False
        trace_choices = []
        for c in choices:
            truth = c["answer"]
            if truth not in ("correct", "wrong"):
                unscoreable += 1
                continue
            cdescs = parse_descriptors(c["program"])
            if len(cdescs) < 2:
                unscoreable += 1
                continue
            hits1 = [k for k in resolve_all(cdescs[0], obj_keys) if k < N]
            hits2 = [k for k in resolve_all(cdescs[1], obj_keys) if k < N]
            if len(hits1) != 1 or len(hits2) != 1 or hits1[0] == hits2[0]:
                unscoreable += 1
                continue
            pair = frozenset((hits1[0], hits2[0]))

            happens, why = pair_happens_cf(pair, removed, cf_pairs, gt_pairs,
                                           obs_pairs, paths)
            model_label = "correct" if (happens != negate) else "wrong"

            any_scored = True
            n_choices += 1
            ok = model_label == truth
            n_choices_right += int(ok)
            path_acc[why][1] += 1
            path_acc[why][0] += int(ok)
            if trace_fh is not None:
                p1, p2 = sorted(pair)
                ev_frames = [ev["frame"] for ev in cf_events
                             if {ev["i"], ev["j"]} == set(pair)]
                gt_frames = [int(r_[0]) for r_ in z["collisions"]
                             if {int(r_[1]), int(r_[2])} == set(pair)]
                trace_choices.append({
                    "text": c["choice"], "pair": [int(p1), int(p2)],
                    "happens": bool(happens), "path": why,
                    "model": model_label, "truth": truth, "ok": bool(ok),
                    "cf_frames": ev_frames, "gt_frames": gt_frames})
            if not ok:
                q_all_right = False
            pol = "NOT " if negate else "WILL"
            confusion[(pol, truth, model_label)] = \
                confusion.get((pol, truth, model_label), 0) + 1

            if a.verbose:
                nm = lambda k: obj_keys[k].replace("_", " ")
                p1, p2 = tuple(pair)
                print(f"  vid {vnum:05d} [{pol}] {nm(p1)} <-> {nm(p2)}: "
                      f"truth={truth} model={model_label} "
                      f"{'ok' if ok else 'MISS'}")

        if any_scored:
            n_questions += 1
            n_questions_right += int(q_all_right)
            if trace_fh is not None:
                trace_fh.write(json.dumps({
                    "video": int(vnum),
                    "npz": npz_by_num[vnum],
                    "question": row.get("question", ""),
                    "negate": bool(negate),
                    "obj_keys": obj_keys[:N],
                    "removed": sorted(int(k) for k in removed),
                    "match_mode": "description",
                    "taints": [{"idx": int(k), "frame": int(taint_frame[k]),
                                "cause": taint_cause.get(k, {})}
                               for k in sorted(taint_frame)],
                    "gt_collisions": [
                        {"frame": int(r_[0]), "i": int(r_[1]), "j": int(r_[2])}
                        for r_ in z["collisions"]
                        if int(r_[1]) < N and int(r_[2]) < N],
                    "cf_events": cf_events,
                    "choices": trace_choices}) + "\n")

    if trace_fh is not None:
        trace_fh.close()
        print(f"[trace] wrote {a.trace_out}")

    # ---- report -----------------------------------------------------------
    print("=" * 64)
    print(f"model      : {a.model}")
    print(f"questions  : {', '.join(a.questions)}")
    print(f"split      : {a.which if a.split else '(no split filter)'}")
    print("=" * 64)
    print(f"counterfactual questions scored : {n_questions}")
    print(f"choices scored                  : {n_choices}")
    if n_choices:
        print(f"PER-CHOICE accuracy             : "
              f"{100.0 * n_choices_right / n_choices:.1f}%  "
              f"({n_choices_right}/{n_choices})")
    if n_questions:
        print(f"PER-QUESTION accuracy (all right): "
              f"{100.0 * n_questions_right / n_questions:.1f}%  "
              f"({n_questions_right}/{n_questions})")
    print(f"unscoreable choices (ambiguous) : {unscoreable}")
    if skipped_video:
        print(f"questions skipped (no .npz)     : {skipped_video}")
    print("-" * 64)
    print("WHERE THE MISSES ARE  (truth -> model, by question type):")
    for (pol, truth, mdl), cnt in sorted(confusion.items()):
        mark = "ok  " if truth == mdl else "MISS"
        print(f"   [{pol}] truth={truth:<7s} model={mdl:<7s} {mark} {cnt:3d}")
    print("-" * 64)
    print("HOW EACH ANSWER WAS DECIDED (light-cone: observed outside, simulated inside):")
    print(f"   participant deleted -> no             : {paths['participant_removed']}")
    print(f"   outside the cone, observed -> yes     : {paths['observed_exact']}   (exact, zero drift)")
    print(f"   inside the cone, simulated hit -> yes : {paths['sim_hit']}")
    print(f"   inside the cone, sim lost it -> no    : {paths['vanished']}   (the causal signal)")
    print(f"   never happens anywhere -> no          : {paths['never']}")
    print("-" * 64)
    print("ACCURACY BY DECISION PATH (right/scored):")
    for key, label in (("observed_exact", "outside cone (observed, no model)"),
                       ("never", "never happens (no model)"),
                       ("participant_removed", "participant deleted (no model)"),
                       ("sim_hit", "simulated hit"),
                       ("vanished", "simulated vanish")):
        r, tot = path_acc[key]
        if tot:
            print(f"   {label:<36s}: {100.0 * r / tot:5.1f}%  ({r}/{tot})")
    md_r = path_acc["sim_hit"][0] + path_acc["vanished"][0]
    md_t = path_acc["sim_hit"][1] + path_acc["vanished"][1]
    if md_t:
        print(f"   {'MODEL-DECIDED subset (the comparison metric)':<36s}: "
              f"{100.0 * md_r / md_t:5.1f}%  ({md_r}/{md_t})")
    if cone_sizes:
        print("-" * 64)
        print(f"causal cone footprint: {sum(cone_sizes) / len(cone_sizes):.1f} "
              f"objects tainted per question on average")
    if in_cone_total:
        print(f"in-cone re-derivation: of {in_cone_total} recorded collisions the "
              f"intervention forced the model to re-simulate, it kept "
              f"{in_cone_kept} and vanished {in_cone_total - in_cone_kept}")
    print("=" * 64)
    print("Note: per-choice accuracy is the standard CLEVRER counterfactual metric.")
    print("Random-guess baseline is 50% per choice (each is a yes/no).")


if __name__ == "__main__":
    main()