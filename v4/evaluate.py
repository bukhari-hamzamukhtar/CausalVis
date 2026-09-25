"""
v4/evaluate.py  —  the CLEVRER counterfactual benchmark with NO light cone
==========================================================================

WHAT CHANGED FROM v3
--------------------
v3 computed a causal light cone: which objects the intervention could possibly
reach. Everything outside it was frozen to its OBSERVED trajectory and only the
rest was simulated. The benchmark then reported

    outside cone (observed, no model) : 100.0%  (831/831)

831 choices answered perfectly because the cone handed them the answer. Only
595 of 3332 choices (18%) ever consulted the model. That is the crutch.

Here there is no cone. The model is given the scene at its first fully-observed
frame and simulates EVERY object to the end of the clip on its own. If an
object should be unaffected by the intervention, the model has to keep it on
its trajectory by getting the physics right -- nobody tells it which those are.
That is the whole point, and it is how learned simulators are normally built:
GNS (arXiv 2002.09405) rebuilds its interaction graph by nearest neighbour
every timestep rather than pruning in advance.

TWO BOUNDARY CONDITIONS, STATED PLAINLY
---------------------------------------
These are inputs to the scene, not predictions, and are marked source=0 in the
store so the accounting stays honest:

  ENTRIES. CLEVRER objects walk in from off-screen at annotated frames. No
  physics can predict an object that is not yet in the world, so an entering
  object is injected at its observed entry state. It is then simulated.

  EXITS. An object that leaves the table is NOT read from the annotation. The
  model decides, by predicting a position outside the table extent.

Everything else -- every position, every collision, every vanish -- is the
model's.

RUN
    python v4/evaluate.py --model v3_5_drag.pt --data data/trajectories_3d \\
        --split split3d.json --which test \\
        --questions zechennlp/counterfactual/validation-00000-of-00001.json
"""

import argparse
import glob
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src2"))

from store import RolloutStore                                   # noqa: E402
from benchmark_eval import (load_model, load_questions, normalise_choices,   # noqa: E402
                            answers_from_conversations, parse_descriptors,
                            program_has, resolve_all, load_split_set,
                            _vid_num, STRIP)
import benchmark_eval as BE                                      # noqa: E402


TABLE_EXTENT = 0.95      # normalised units; beyond this an object has left


def full_scene_rollout(model, z, removed, world_scale=6.0, max_objects=8):
    """Simulate the whole scene with no cone and no teacher forcing.

    Returns a RolloutStore holding the entire predicted future.
    """
    pos = z["positions"].astype(np.float32)
    vel = z["velocities"].astype(np.float32)
    pres = z["presence"]
    attrs = z["attrs"].astype(np.float32)
    T = pos.shape[0]
    N = min(max_objects, pos.shape[1])
    obj_keys = [str(k) for k in z["obj_keys"]][:N]

    store = RolloutStore(_vid_num(str(z["video_name"])) or 0, removed,
                         obj_keys, attrs[:N], T, N, world_scale)

    at = torch.from_numpy(attrs[:N]).unsqueeze(0)
    with torch.no_grad():
        mass, radius, e = model.properties(STRIP(at))
    mass, radius, e = mass.detach(), radius.detach(), e.detach()

    # first frame where at least two surviving objects are on screen
    keep = [k for k in range(N) if k not in removed]
    t0 = None
    for t in range(T):
        if sum(1 for k in keep if pres[t, k] > 0) >= 2:
            t0 = t
            break
    if t0 is None:
        return store

    q = torch.zeros(1, N, 2)
    p = torch.zeros(1, N, 2)
    active = torch.zeros(1, N)
    for k in keep:
        if pres[t0, k] > 0:
            active[0, k] = 1.0
            q[0, k] = torch.from_numpy(pos[t0, k])
            p[0, k] = mass[0, k] * torch.from_numpy(vel[t0, k])
    store.write(t0, q[0].numpy(), (p[0] / mass[0].unsqueeze(-1)).numpy(),
                active[0].numpy(), simulated=False)

    p_prev = float(p.sum(dim=1).norm())
    for t in range(t0 + 1, T):
        # ENTRIES: an object appearing for the first time is a boundary
        # condition, not a prediction. Inject its observed entry state.
        newly = [k for k in keep
                 if pres[t, k] > 0 and active[0, k] == 0
                 and (pres[:t, k] <= 0).all()]
        for k in newly:
            active[0, k] = 1.0
            q[0, k] = torch.from_numpy(pos[t, k])
            p[0, k] = mass[0, k] * torch.from_numpy(vel[t, k])

        if active.sum().item() >= 1:
            q, p = q.detach(), p.detach()
            q, p, _ = model.step(q, p, mass, radius, e, active,
                                 dt=1.0, F=None, create_graph=False)
            now = float(p.sum(dim=1).norm())
            # the pairwise interaction cannot change total momentum, so any
            # change is drag handing momentum to the table, plus float32 noise
            store.ledger["momentum_to_table"] += abs(p_prev - now)
            store.ledger["steps"] += 1
            p_prev = now

        # EXITS: decided by the model, not read from the annotation
        for k in range(N):
            if active[0, k] > 0 and float(q[0, k].abs().max()) > TABLE_EXTENT:
                active[0, k] = 0.0

        store.write(t, q[0].detach().numpy(),
                    (p[0] / mass[0].unsqueeze(-1)).detach().numpy(),
                    active[0].numpy(), simulated=True)
    return store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True, nargs="+")
    ap.add_argument("--data", default="data/trajectories_3d")
    ap.add_argument("--split", default=None)
    ap.add_argument("--which", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--thresh", type=float, default=0.02)
    ap.add_argument("--release", type=float, default=0.02)
    ap.add_argument("--dump-store", default=None,
                    help="write one store as JSON+text, for inspection")
    a = ap.parse_args()

    model = load_model(a.model)
    files = sorted(glob.glob(os.path.join(a.data, "*.npz")))
    npz_by_num = {_vid_num(f): f for f in files if _vid_num(f) is not None}
    allowed = load_split_set(a.split, a.which) if a.split else None

    rows = []
    for qf in a.questions:
        rows.extend(load_questions(qf))
    print("model     :", a.model)
    print("data      :", a.data)
    print("questions :", len(rows), "rows")
    print("cone      : NONE -- the model simulates every object itself")
    print("=" * 64)

    cache = {}
    n_q = n_q_right = n_c = n_c_right = unscoreable = 0
    paths = {"participant_removed": [0, 0], "model_says_hit": [0, 0],
             "model_says_no": [0, 0]}
    dumped = False

    for row in rows:
        vnum = _vid_num(str(row.get("video_filename", row.get("video", ""))))
        if vnum is None or vnum not in npz_by_num:
            continue
        if allowed is not None and vnum not in allowed:
            continue
        if a.limit and n_q >= a.limit:
            break

        z = np.load(npz_by_num[vnum], allow_pickle=True)
        obj_keys = [str(k) for k in z["obj_keys"]]
        N = min(8, z["positions"].shape[1])
        qdescs = parse_descriptors(row.get("program", []))
        if not qdescs:
            unscoreable += 1
            continue
        removed = set(k for k in resolve_all(qdescs[0], obj_keys) if k < N)
        if not removed:
            unscoreable += 1
            continue
        negate = program_has(row.get("program", []), "negate")

        ck = (vnum, frozenset(removed))
        if ck not in cache:
            st = full_scene_rollout(model, z, removed)
            st.contacts(thresh=a.thresh, release=a.release)
            cache[ck] = st
        store = cache[ck]
        cf_pairs = set(frozenset((e["i"], e["j"])) for e in store.to_dict()["collisions"])

        if a.dump_store and not dumped:
            store.save(a.dump_store)
            with open(a.dump_store.replace(".json", ".txt"), "w",
                      encoding="utf-8") as fh:
                fh.write(store.to_text())
            print("[dump] wrote", a.dump_store, "and .txt")
            dumped = True

        choices = normalise_choices(row.get("choices"))
        if choices and all(c["answer"] is None for c in choices):
            answers_from_conversations(row, choices)

        scored_any, all_right = False, True
        for c in choices:
            truth = c["answer"]
            if truth not in ("correct", "wrong"):
                unscoreable += 1
                continue
            cd = parse_descriptors(c["program"])
            if len(cd) < 2:
                unscoreable += 1
                continue
            h1 = [k for k in resolve_all(cd[0], obj_keys) if k < N]
            h2 = [k for k in resolve_all(cd[1], obj_keys) if k < N]
            if len(h1) != 1 or len(h2) != 1 or h1[0] == h2[0]:
                unscoreable += 1
                continue
            pair = frozenset((h1[0], h2[0]))

            if h1[0] in removed or h2[0] in removed:
                happens, why = False, "participant_removed"
            elif pair in cf_pairs:
                happens, why = True, "model_says_hit"
            else:
                happens, why = False, "model_says_no"

            label = "correct" if (happens != negate) else "wrong"
            ok = label == truth
            n_c += 1
            n_c_right += int(ok)
            paths[why][1] += 1
            paths[why][0] += int(ok)
            scored_any = True
            all_right = all_right and ok

        if scored_any:
            n_q += 1
            n_q_right += int(all_right)

    print("counterfactual questions scored :", n_q)
    print("choices scored                  :", n_c)
    if n_c:
        print("PER-CHOICE accuracy             : "
              "%.1f%%  (%d/%d)" % (100.0 * n_c_right / n_c, n_c_right, n_c))
    if n_q:
        print("PER-QUESTION accuracy           : "
              "%.1f%%  (%d/%d)" % (100.0 * n_q_right / n_q, n_q_right, n_q))
    print("unscoreable choices             :", unscoreable)
    print("-" * 64)
    print("HOW EACH ANSWER WAS DECIDED (no cone -- almost all are the model)")
    model_right = model_tot = 0
    for k, (r, t) in paths.items():
        if t:
            print("   %-22s %6.1f%%  (%d/%d)" % (k, 100.0 * r / t, r, t))
        if k != "participant_removed":
            model_right += r
            model_tot += t
    if model_tot:
        print("   %-22s %6.1f%%  (%d/%d)" %
              ("MODEL-DECIDED subset", 100.0 * model_right / model_tot,
               model_right, model_tot))
        print("   model-decided share of all choices : %.0f%%"
              % (100.0 * model_tot / max(n_c, 1)))
    print("=" * 64)


if __name__ == "__main__":
    main()
