"""
v6/demo.py  —  one counterfactual, simulated with voxel contacts, rendered high
==============================================================================

Runs the whole thing on a single question and produces:

    <clip>_voxel_store.json   summary a language model can read
    <clip>_voxel_store.npz    the FULL predicted future, per frame:
                              positions, velocities, yaw, presence, source,
                              plus every voxel contact event with its 3D point
                              and normal
    <clip>_voxel.gif          the predicted world, camera orbiting, contact
                              points marked where the voxels were pressed

WHY THE STORE CARRIES THE VOXEL EVENTS
--------------------------------------
The contact point is the one quantity every earlier version threw away. It is
what makes the bounce direction correct, and it is the thing that can be drawn
-- a marker at the exact 3D point where two objects pressed the same voxels.
Storing it means the render, the physics and the record all refer to the same
event, which has not been true before in this project.

RENDER QUALITY
--------------
Rendered at 2x the frame size the fitted camera uses, because the geometry
supports it: the poses come from silhouette fitting at 0.95 median IoU, so the
limit on the picture is the rasteriser, not the reconstruction. Objects are
drawn as smooth meshes rather than as their voxels -- the voxel grid is the
physics representation, and drawing it would show 0.02-unit stair-steps that
are an artefact of the contact test, not of the recovered scene.

RUN
    python v6/demo.py --model v5b_noyaw.pt --index 40 --out v6/demo
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for d in ("src2", "v4", "v5", "v6"):
    sys.path.insert(0, os.path.join(ROOT, d))

from benchmark_eval import (load_questions, normalise_choices,          # noqa: E402
                            answers_from_conversations, parse_descriptors,
                            resolve_all, lightcone_rollout, _vid_num, STRIP)
from world import load_spatial                                         # noqa: E402
from store import RolloutStore                                         # noqa: E402
from voxel import shape_of, VOXEL                                      # noqa: E402
from resolve import resolve_voxel                                      # noqa: E402
from evaluate import closest_approach, WINDOW, TABLE_EXTENT            # noqa: E402
from render3d import render_scene, save_gif, make_camera, fill_poly, project_pts  # noqa: E402
from lift3d import Camera                                              # noqa: E402

WORLD = 6.0


def voxel_rollout(model, z, removed, s0):
    """Simulate every surviving object from frame s0, resolving contacts from
    voxel occupancy. Returns a store plus the contact events."""
    pos = z["positions"].astype(np.float32); vel = z["velocities"].astype(np.float32)
    pres = z["presence"]; attrs = z["attrs"].astype(np.float32)
    yaw_d = z["yaw"] if "yaw" in z else np.zeros(pres.shape, np.float32)
    T = pos.shape[0]; N = min(8, pos.shape[1])
    keys = [str(k) for k in z["obj_keys"]][:N]
    shapes = [shape_of(attrs, k) for k in range(N)]
    st = RolloutStore(_vid_num(str(z["video_name"])) or 0, removed, keys,
                      attrs[:N], T, N, WORLD)

    at = torch.from_numpy(attrs[:N]).unsqueeze(0)
    with torch.no_grad():
        mass, radius, e = model.properties(STRIP(at))
    keep = [k for k in range(N) if k not in removed]
    for t in range(s0):                                   # observed prefix
        pr = np.array([1.0 if (k in keep and pres[t, k] > 0) else 0.0
                       for k in range(N)], np.float32)
        st.write(t, pos[t, :N], vel[t, :N], pr, yaw=yaw_d[t, :N], simulated=False)

    q = torch.zeros(1, N, 2); p = torch.zeros(1, N, 2); active = torch.zeros(1, N)
    for k in keep:
        if pres[s0, k] > 0:
            active[0, k] = 1.0
            q[0, k] = torch.from_numpy(pos[s0, k])
            p[0, k] = mass[0, k] * torch.from_numpy(vel[s0, k])
    st.write(s0, q[0].numpy(), (p[0] / mass[0].unsqueeze(-1)).numpy(),
             active[0].numpy(), yaw=yaw_d[s0, :N], simulated=False)

    yaw_now = yaw_d[s0, :N].astype(np.float64).copy()
    events = []
    for t in range(s0 + 1, T):
        for k in keep:                                    # entries: boundary condition
            if pres[t, k] > 0 and active[0, k] == 0 and (pres[:t, k] <= 0).all():
                active[0, k] = 1.0
                q[0, k] = torch.from_numpy(pos[t, k])
                p[0, k] = mass[0, k] * torch.from_numpy(vel[t, k])
        if active.sum().item() >= 1:
            q, p = q.detach(), p.detach()
            p, ev = resolve_voxel(q, p, mass, radius, active, attrs[:N, 1],
                                  shapes, yaw_now)
            for x in ev:
                x["frame"] = int(t)
                events.append(x)
            q, p, _ = model.step(q, p, mass, radius, e, active, dt=1.0,
                                 F=None, create_graph=False)
        for k in range(N):
            if active[0, k] > 0 and float(q[0, k].abs().max()) > TABLE_EXTENT:
                active[0, k] = 0.0
        st.write(t, q[0].detach().numpy(),
                 (p[0] / mass[0].unsqueeze(-1)).detach().numpy(),
                 active[0].numpy(), yaw=yaw_now, simulated=True)
    return st, events, shapes


def render(z, st, events, out_path, scale=2, stride=2, frames=64):
    """Orbiting render of the PREDICTED world with contact points marked."""
    blob = json.load(open(os.path.join(ROOT, "camera_fit.json")))
    W = int(blob["camera"]["width"] * scale)
    H = int(blob["camera"]["height"] * scale)
    keys = [str(k) for k in z["obj_keys"]]
    attrs = z["attrs"]
    by_frame = {}
    for e in events:
        by_frame.setdefault(e["frame"], []).append(e)

    def objs(t):
        out = []
        for k in range(st.N):
            if k in st.removed or st.present[t, k] <= 0:
                continue
            if not np.isfinite(st.positions[t, k]).all():
                continue
            col, mat, shp = keys[k].split("_")[:3]
            r = float(attrs[k, 15]) * WORLD
            out.append({"shape": shp, "size": r / (4.0 / np.pi) if shp == "cube" else r,
                        "x": float(st.positions[t, k, 0]) * WORLD,
                        "y": float(st.positions[t, k, 1]) * WORLD,
                        "yaw": float(st.yaw[t, k]), "color": col, "material": mat})
        return out

    ts = list(range(0, st.T, stride))[:frames]
    imgs = []
    recent = []
    for n, t in enumerate(ts):
        ang = 2 * math.pi * n / max(len(ts) - 1, 1)
        cam = make_camera((9.0 * math.cos(ang), 9.0 * math.sin(ang), 5.2),
                          (0, 0, 0.4), 34.2, W, H)
        img = render_scene(cam, objs(t), W, H)
        # contact markers: every voxel contact in the last few frames, drawn at
        # its true 3D point so the picture and the physics agree
        for tt in range(max(0, t - 2 * stride), t + 1):
            for ev in by_frame.get(tt, []):
                recent.append((tt, np.asarray(ev["point_world"], float)))
        recent = [(tt, pnt) for (tt, pnt) in recent if t - tt <= 8]
        for tt, pnt in recent:
            uv, d, front = project_pts(cam, pnt.reshape(1, 3))
            if not front[0]:
                continue
            u, v = uv[0]
            rad = max(3, int(7 * scale * (1.0 - (t - tt) / 12.0)))
            poly = np.array([[u + rad * math.cos(a), v + rad * math.sin(a)]
                             for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)])
            fill_poly(img, np.full((H, W), np.inf), poly, 0.0, (245, 70, 60))
        imgs.append(img)
    save_gif(imgs, out_path, fps=12)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v5b_noyaw.pt")
    ap.add_argument("--data", default="data/trajectories_3d_yaw")
    ap.add_argument("--questions",
                    default="zechennlp/counterfactual/validation-00000-of-00001.json")
    ap.add_argument("--index", type=int, default=40)
    ap.add_argument("--out", default="v6/demo")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--provider", default=None,
                    help="groq | cerebras | openrouter | together | anthropic "
                         "(default: whichever key is set)")
    ap.add_argument("--no-gif", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from lm import pick_provider, parse_intervention, answer_choices

    rows = load_questions(a.questions)
    row = rows[a.index]
    v = _vid_num(str(row.get("video_filename", row.get("video", ""))))
    npz = os.path.join(a.data, "sim_%05d.npz" % v)
    if not os.path.exists(npz):
        raise SystemExit("no yaw-fitted clip for video %d" % v)
    z = np.load(npz, allow_pickle=True)
    keys = [str(k) for k in z["obj_keys"]]
    names = [k.replace("_", " ") for k in keys]
    N = min(8, z["positions"].shape[1])
    question = row.get("question", "")
    choices = normalise_choices(row.get("choices"))
    if choices and all(c["answer"] is None for c in choices):
        answers_from_conversations(row, choices)

    print("=" * 70)
    print("VIDEO    :", v)
    print("QUESTION :", question)
    print("OBJECTS  :", "; ".join("%d:%s" % (k, n) for k, n in enumerate(names)))
    print("=" * 70)

    # ---- 1. the language model reads the question -------------------------
    prov, cfg, why = pick_provider(a.provider)
    if cfg is not None:
        parsed = parse_intervention(cfg, prov, question, names)
        removed = {int(parsed["removed_index"])}
        print("\n[1] LM (%s) reads the question: remove %d (%s) -- %s"
              % (prov, parsed["removed_index"], names[parsed["removed_index"]],
                 parsed.get("reason", "")))
    else:
        qd = parse_descriptors(row.get("program", []))
        removed = set(k for k in resolve_all(qd[0], keys) if k < N)
        print("\n[1] no LM key (%s) -- intervention from the question PROGRAM: remove %s"
              % (why, sorted(removed)))

    # ---- 2. world model simulates the edited world, voxel contacts ---------
    model = load_spatial(a.model, yaw_known=False)
    _, _, taint, _ = lightcone_rollout(model, z, removed, 0.02, 0.02, 0.06)
    s0 = max(0, min(taint.values()) - 1) if taint else 0
    print("\n[2] world model %s simulates every object from frame %d, contacts from VOXEL OCCUPANCY"
          % (a.model, s0))
    st, events, shapes = voxel_rollout(model, z, removed, s0)
    st.contacts()

    # ---- 3. store the predicted future --------------------------------------
    base = os.path.join(a.out, "q%d_video%d_voxel_store" % (a.index, v))
    st.save(base + ".json")
    d = json.load(open(base + ".json")); d["voxel_contacts"] = events
    d["voxel_size_world"] = VOXEL
    json.dump(d, open(base + ".json", "w"))
    np.savez_compressed(base + "_contacts.npz",
                        frames=np.array([e["frame"] for e in events], np.int64),
                        pair=np.array([[e["i"], e["j"]] for e in events], np.int64).reshape(-1, 2),
                        point_world=np.array([e["point_world"] for e in events], np.float32).reshape(-1, 3),
                        normal=np.array([e["normal"] for e in events], np.float32).reshape(-1, 2),
                        impulse=np.array([e["impulse"] for e in events], np.float32))
    txt = st.to_text()
    if events:
        txt += ("\n\nVOXEL CONTACT EVENTS in the predicted future (frame: object hits object "
                "at 3D contact point in world units, contact normal, impulse):\n")
        for e in events[:40]:
            txt += ("  frame %d: %s hits %s at (%.2f, %.2f, %.2f), normal (%+.2f, %+.2f), J=%.4f\n"
                    % (e["frame"], st.name(e["i"]), st.name(e["j"]),
                       *e["point_world"], *e["normal"], e["impulse"]))
    else:
        txt += "\n\nVOXEL CONTACT EVENTS: none -- no two objects pressed the same voxels.\n"
    open(base + ".txt", "w", encoding="utf-8").write(txt)
    print("\n[3] stored future: %s.json / .txt / .npz / _contacts.npz  (%d voxel contacts)"
          % (base, len(events)))
    print("    " + txt.replace("\n", "\n    "))

    # ---- 4. the language model answers FROM THE STORE ONLY -----------------
    print("\n[4] answers  (the LM sees only the stored predicted future, never ground truth)")
    print("    %2s %13s %13s   statement" % ("#", "ground truth", "LM answer"))
    if cfg is None:
        for k, c in enumerate(choices):
            print("    %2d %13s %13s   %s" % (k, c["answer"], "BLOCKED", c["choice"]))
    else:
        ans = answer_choices(cfg, prov, txt, question, [c["choice"] for c in choices])
        by = {int(x["index"]): x for x in ans.get("answers", [])}
        right = 0
        for k, c in enumerate(choices):
            got = by.get(k, {}).get("label", "?")
            ok = got == c["answer"]; right += int(ok)
            print("    %2d %13s %13s   %s  %s" % (k, c["answer"], got, c["choice"],
                                                  "ok" if ok else "MISS"))
            if by.get(k, {}).get("evidence"):
                print("       evidence: %s" % by[k]["evidence"])
        print("    -> %d/%d correct" % (right, len(choices)))
        from lm import _resolve_model
        json.dump({"question": question, "provider": prov,
                   "model": _resolve_model(cfg) if prov != "anthropic" else cfg["model"],
                   "answers": ans, "truth": [c["answer"] for c in choices]},
                  open(base.replace("_store", "_answers") + ".json", "w"), indent=1)

    # ---- 5. orbiting gif of the predicted world -----------------------------
    if not a.no_gif:
        gif = render(z, st, events, base.replace("_store", "") + ".gif", scale=a.scale)
        print("\n[5] gif: %s  (%dx%d, camera orbiting, voxel contact points marked)"
              % (gif, 480 * a.scale, 320 * a.scale))


if __name__ == "__main__":
    main()
