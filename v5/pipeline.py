"""
v5/pipeline.py  —  one counterfactual question, end to end, nothing hardcoded
=============================================================================

    question text ─► VLM: which video, which object is removed
                  ─► world model: taint rollout of the edited world
                  ─► RolloutStore: the WHOLE predicted future, as JSON + text
                  ─► VLM: reads that text, answers every choice
                  ─► GIF: observed world (left) vs predicted edited world (right)

WHAT IS AND IS NOT THE MODEL'S
------------------------------
The taint design is kept on purpose (it is how the benchmark was built and it
is Pearl-correct): an object's observed trajectory IS its counterfactual
trajectory until the intervention's effects reach it, then it is simulated
from its last true state. The store records which frames were observed and
which were simulated, so the accounting is visible in the output.

NO TEMPLATES ON THE ANSWER PATH. The question is parsed by the language model
from its natural-language text, and every choice is answered by the language
model from the stored future. If no API credential is present the pipeline
STOPS at that step and says so -- it does not fall back to a lookup, because
a lookup is exactly what this replaces.

The language model never sees pixels. It sees the same numbers a renderer
would draw, written as text: which objects exist, which frames were simulated,
every collision the simulator produced, and the momentum the table absorbed.

RUN
    set ANTHROPIC_API_KEY=...            (console.anthropic.com; Pro is not API)
    python v5/pipeline.py --model v3_6_3d.pt --data data/trajectories_3d \
        --questions zechennlp/counterfactual/validation-00000-of-00001.json \
        --index 0 --out v5/demo
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# v4/ also has an evaluate.py; insert HERE last so v5's modules win the lookup
sys.path.insert(0, os.path.join(ROOT, "v4"))
sys.path.insert(0, os.path.join(ROOT, "src2"))
sys.path.insert(0, HERE)

import torch                                                            # noqa: E402
from store import RolloutStore                                          # noqa: E402
from benchmark_eval import (load_model, load_questions, normalise_choices,   # noqa: E402
                            answers_from_conversations, lightcone_rollout,
                            _vid_num, STRIP)
from evaluate import closest_approach, simulate_pair, WINDOW, TABLE_EXTENT   # noqa: E402

MODEL_ID = "claude-opus-5"


# ---------------------------------------------------------------------------
#                       the language model, twice
# ---------------------------------------------------------------------------
from lm import pick_provider, parse_intervention, answer_choices  # noqa: E402


# ---------------------------------------------------------------------------
#                     store + gif from the taint rollout
# ---------------------------------------------------------------------------
def start_frame(model, z, removed, thresh=0.02, release=0.02):
    """When does the intervention start to matter? The earliest frame any
    object enters the causal cone; failing that, the earliest observed close
    approach between two surviving objects, minus WINDOW. Ground truth chooses
    WHEN; it never chooses an answer."""
    _, _, taint, _ = lightcone_rollout(model, z, removed, thresh, release, 0.06)
    if taint:
        return max(0, min(taint.values()) - 1), taint
    pos, pres, at = z["positions"], z["presence"], z["attrs"]
    N = min(8, pos.shape[1]); best = None
    for i in range(N):
        for j in range(i + 1, N):
            if i in removed or j in removed:
                continue
            t = closest_approach(pos, pres, i, j)
            if t is None:
                continue
            d = float(np.linalg.norm(pos[t, i] - pos[t, j]))
            if d < 2.0 * float(at[i, 15] + at[j, 15]):
                best = t if best is None else min(best, t)
    return (max(0, best - WINDOW) if best is not None else 0), taint


def honest_rollout(model, z, removed, s, world_scale):
    """From frame s the model owns EVERY object. Frames before s are the
    observed record and are marked as such in the store. Late entries are
    injected as boundary conditions; exits are the model's call."""
    pos = z["positions"].astype(np.float32); vel = z["velocities"].astype(np.float32)
    pres = z["presence"]; attrs = z["attrs"].astype(np.float32)
    T = pos.shape[0]; N = min(8, pos.shape[1])
    keys = [str(k) for k in z["obj_keys"]][:N]
    st = RolloutStore(_vid_num(str(z["video_name"])) or 0, removed, keys, attrs[:N], T, N, world_scale)
    yaw = z["yaw"] if "yaw" in z else np.zeros((T, N), np.float32)
    keep = [k for k in range(N) if k not in removed]
    for t in range(s):                                    # observed prefix
        pr = np.array([1.0 if (k in keep and pres[t, k] > 0) else 0.0 for k in range(N)], np.float32)
        st.write(t, pos[t, :N], vel[t, :N], pr, yaw=yaw[t, :N], simulated=False)

    at = torch.from_numpy(attrs[:N]).unsqueeze(0)
    with torch.no_grad():
        mass, radius, e = model.properties(STRIP(at))
    q = torch.zeros(1, N, 2); p = torch.zeros(1, N, 2); active = torch.zeros(1, N)
    for k in keep:
        if pres[s, k] > 0:
            active[0, k] = 1.0; q[0, k] = torch.from_numpy(pos[s, k])
            p[0, k] = mass[0, k] * torch.from_numpy(vel[s, k])
    st.write(s, q[0].numpy(), (p[0] / mass[0].unsqueeze(-1)).numpy(), active[0].numpy(),
             yaw=yaw[s, :N], simulated=False)
    p_prev = float(p.sum(dim=1).norm())
    for t in range(s + 1, T):
        for k in keep:                                    # boundary condition: entries
            if pres[t, k] > 0 and active[0, k] == 0 and (pres[:t, k] <= 0).all():
                active[0, k] = 1.0; q[0, k] = torch.from_numpy(pos[t, k])
                p[0, k] = mass[0, k] * torch.from_numpy(vel[t, k])
        if active.sum().item() >= 1:
            q, p = q.detach(), p.detach()
            q, p, _ = model.step(q, p, mass, radius, e, active, dt=1.0, F=None, create_graph=False)
            now = float(p.sum(dim=1).norm())
            st.ledger["momentum_to_table"] += abs(p_prev - now); st.ledger["steps"] += 1; p_prev = now
        for k in range(N):                                # exits: the model's call
            if active[0, k] > 0 and float(q[0, k].abs().max()) > TABLE_EXTENT:
                active[0, k] = 0.0
        st.write(t, q[0].detach().numpy(), (p[0] / mass[0].unsqueeze(-1)).detach().numpy(),
                 active[0].numpy(), yaw=yaw[s, :N], simulated=True)
    return st


def render_gif(z, store, out_path, world_scale, stride=2, frames=64, orbit=True):
    """Left: the observed world. Right: the model's predicted world.

    The camera ORBITS both panels together. A fixed camera (CLEVRER's own
    viewpoint) makes overlapping objects impossible to judge -- you cannot see
    whether two things actually touch or just line up from that one angle.
    Circling the scene is the whole point of having real 3D: if the geometry
    were wrong, objects would swim or interpenetrate as the view changes.
    """
    try:
        from render3d import render_scene, save_gif, make_camera
        from lift3d import Camera
    except ImportError as ex:
        print("   gif skipped:", ex); return None
    cam_blob = json.load(open(os.path.join(ROOT, "camera_fit.json")))
    cam_fixed = Camera(cam_blob["camera"])
    W, H = cam_blob["camera"]["width"], cam_blob["camera"]["height"]
    keys = [str(k) for k in z["obj_keys"]]
    attrs = z["attrs"]

    def objs_from(pos_t, pres_t, yaw_t, skip=()):
        out = []
        for k in range(store.N):
            if k in skip or pres_t[k] <= 0 or not np.isfinite(pos_t[k]).all():
                continue
            col, mat, shape = keys[k].split("_")[:3]
            r = float(attrs[k, 15]) * world_scale
            size = r / (4.0 / np.pi) if shape == "cube" else r   # half-side for a cube
            out.append({"shape": shape, "size": size,
                        "x": float(pos_t[k, 0]) * world_scale,
                        "y": float(pos_t[k, 1]) * world_scale,
                        "yaw": float(yaw_t[k]), "color": col, "material": mat})
        return out

    pos_obs, pres_obs = z["positions"], z["presence"]
    yaw_obs = z["yaw"] if "yaw" in z else np.zeros(pres_obs.shape, np.float32)
    ts = list(range(0, store.T, stride))[:frames]
    imgs = []
    for n, t in enumerate(ts):
        if orbit:
            ang = 2 * np.pi * n / max(len(ts) - 1, 1)      # one full turn per clip
            cam = make_camera((9.0 * np.cos(ang), 9.0 * np.sin(ang), 5.2),
                              (0, 0, 0.4), 34.2, W, H)
        else:
            cam = cam_fixed
        left = render_scene(cam, objs_from(pos_obs[t], pres_obs[t], yaw_obs[t]), W, H)
        right = render_scene(cam, objs_from(store.positions[t], store.present[t],
                                            store.yaw[t], skip=set(store.removed)), W, H)
        imgs.append(np.concatenate([left, right], axis=1))
    save_gif(imgs, out_path, fps=12)
    return out_path


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/trajectories_3d")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--index", type=int, default=0, help="which question row")
    ap.add_argument("--video", type=int, default=None, help="or: first question on this video")
    ap.add_argument("--out", default="v5/demo")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--fixed-camera", action="store_true",
                    help="do not orbit (CLEVRER's original viewpoint)")
    ap.add_argument("--provider", default=None,
                    help="anthropic | groq | openrouter | together | cerebras "
                         "(default: whichever key is set)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = load_questions(a.questions)
    if a.video is not None:
        rows = [r for r in rows if _vid_num(str(r.get("video_filename", r.get("video", "")))) == a.video]
    row = rows[a.index]
    vnum = _vid_num(str(row.get("video_filename", row.get("video", ""))))
    if vnum is None:
        raise SystemExit(f"could not read a video number from row keys {sorted(row.keys())}")
    npz = os.path.join(a.data, f"sim_{vnum:05d}.npz")
    z = np.load(npz, allow_pickle=True)
    world_scale = float(z["world_scale"]) if "world_scale" in z else 1.0
    obj_names = [str(k).replace("_", " ") for k in z["obj_keys"]]
    question = row.get("question", "")
    choices = normalise_choices(row.get("choices"))
    if choices and all(c["answer"] is None for c in choices):
        answers_from_conversations(row, choices)

    print("=" * 70)
    print(f"VIDEO     : {vnum}   ({npz})")
    print(f"QUESTION  : {question}")
    print("OBJECTS   : " + "; ".join(f"{i}:{n}" for i, n in enumerate(obj_names)))
    print("=" * 70)

    prov, cfg, why = pick_provider(a.provider)
    if cfg:
        print(f"LM        : {prov}  model={cfg['model']}")

    # ---- 1. VLM reads the question ---------------------------------------
    if cfg is None:
        print(f"\n[1] VLM parse       : BLOCKED -- {why}")
        print("    (the rollout, store and gif below still run; the ANSWER does not)")
        removed = None
    else:
        parsed = parse_intervention(cfg, prov, question, obj_names)
        removed = {int(parsed["removed_index"])}
        print(f"\n[1] VLM parse       : remove object {parsed['removed_index']} "
              f"({obj_names[parsed['removed_index']]}) -- {parsed.get('reason','')}")

    if removed is None:
        # Nothing to simulate without an intervention. Use the ground-truth
        # program ONLY to keep the demo moving and say so loudly.
        from benchmark_eval import parse_descriptors, resolve_all
        qd = parse_descriptors(row.get("program", []))
        removed = set(resolve_all(qd[0], [str(k) for k in z["obj_keys"]])) if qd else set()
        print(f"    intervention taken from the question PROGRAM for the demo only: "
              f"remove {sorted(removed)}")

    # ---- 2. world model rolls out the edited world -----------------------
    model = load_model(a.model)
    s0, taint = start_frame(model, z, removed)
    st = honest_rollout(model, z, removed, s0, world_scale)
    n_sim = int((st.source == 1).sum()); n_obs = int(((st.source == 0) & (st.present > 0)).sum())
    print(f"\n[2] world model     : simulates EVERY object from frame {s0} "
          f"(cone entries {dict(sorted(taint.items())) or 'none'}); "
          f"{n_sim} object-frames simulated, {n_obs} observed prefix. Nothing copied after frame {s0}.")

    # ---- 2b. focused re-simulation of EVERY surviving pair ------------------
    # No ground-truth pair selection: all pairs, each from a true state a few
    # frames before its own closest approach (or its cone entry). The long
    # rollout above drifts for late events; these short windows do not.
    at = torch.from_numpy(z["attrs"][:st.N].astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        mass, radius, e = model.properties(STRIP(at))
    st.focused = []
    keep = [k for k in range(st.N) if k not in removed]
    for ii in range(len(keep)):
        for jj in range(ii + 1, len(keep)):
            i, j = keep[ii], keep[jj]
            tainted = [taint[k] for k in (i, j) if k in taint]
            if tainted:                       # counterfactual pair: cone entry -> end
                s_ij, end = max(0, min(tainted)), st.T
            else:                             # factual pair: around closest approach
                ca = closest_approach(z["positions"], z["presence"], i, j)
                if ca is None:
                    continue
                s_ij, end = max(0, ca - WINDOW), ca + WINDOW
            fr = simulate_pair(model, z, removed, i, j, s_ij, mass, radius, e, 0.02, 0.02,
                               end=end)
            st.focused.append({"i": i, "j": j, "start": int(s_ij),
                               "frame": (int(fr) if fr is not None else None)})
    print(f"    focused re-simulations: {len(st.focused)} pairs, "
          f"{sum(1 for f in st.focused if f['frame'] is not None)} predicted to collide")

    # ---- 3. store the predicted future ------------------------------------
    st.contacts()
    sp = os.path.join(a.out, f"q{a.index}_video{vnum}_store.json")
    st.save(sp)
    txt = st.to_text()
    open(sp.replace(".json", ".txt"), "w", encoding="utf-8").write(txt)
    print(f"\n[3] stored future   : {sp}  ({len(st.to_dict()['collisions'])} predicted collisions)")
    print("    " + txt.replace("\n", "\n    "))

    # ---- 4. VLM answers from the store -----------------------------------
    print("\n[4] answers")
    print(f"    {'#':>2} {'ground truth':>13} {'model':>13}   statement")
    if cfg is None:
        for k, c in enumerate(choices):
            print(f"    {k:>2} {str(c['answer']):>13} {'BLOCKED':>13}   {c['choice']}")
        print("    -> set one of the provider keys above to get the model's answers")
    else:
        ans = answer_choices(cfg, prov, txt, question, [c["choice"] for c in choices])
        by = {int(x["index"]): x for x in ans.get("answers", [])}
        right = 0
        for k, c in enumerate(choices):
            got = by.get(k, {}).get("label", "?")
            ok = got == c["answer"]; right += int(ok)
            print(f"    {k:>2} {str(c['answer']):>13} {got:>13}   {c['choice']}  {'ok' if ok else 'MISS'}")
            if by.get(k, {}).get("evidence"):
                print(f"       evidence: {by[k]['evidence']}")
        print(f"    -> {right}/{len(choices)} correct")

    # ---- 5. gif of observed vs predicted ----------------------------------
    if not a.no_gif:
        gp = os.path.join(a.out, f"q{a.index}_video{vnum}_observed_vs_predicted.gif")
        g = render_gif(z, st, gp, world_scale, orbit=not a.fixed_camera)
        if g:
            print(f"\n[5] gif             : {g}   (left = observed, right = predicted edited world)")


if __name__ == "__main__":
    main()
