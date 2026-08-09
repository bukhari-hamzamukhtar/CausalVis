"""
viewer2.py  (src2)  —  CausalVis V3 merged viewer: worlds, conservation, explanation
====================================================================================

One page, three tabs, one engine.  The engine (light-cone rollout, taint rules,
event logging, CCD maths) is byte-for-byte the benchmark's; this file only
draws it.

  WORLDS        the observed video on the left (data, not prediction) and the
                edited world on the right, from the SAME light-cone pass the
                benchmark uses.  Each panel has its own frame slider.  Trails
                are drawn only up to the slider frame, never ahead of it, with
                the oldest quarter faded.  Hovering an object shows its
                material, mass, kinetic energy, potential-energy share and
                speed at that frame, and in the edited world whether it is
                currently observed or simulated.  A totals line under each
                panel tracks total KE, total PE and total energy.

  CONSERVATION  the CCD diagnostic for the selected scene, three physics modes
                at inference (symplectic = ours, euler = energy unprotected,
                broken = momentum unprotected).  The explanation under the
                charts is written from the curves themselves, in plain words.

  EXPLANATION   the full grounded narration for the chosen intervention.

RUN
    pip install fastapi uvicorn            (once)
    python src2/viewer2.py --data data/trajectories_v2 --model v3_2_dynamics.pt
    then open  http://127.0.0.1:8000
"""

import argparse
import glob
import os
import re

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from benchmark_eval import lightcone_rollout, load_model
from explain import narrate

STATE = {}

COLOR_HEX = {"gray": "#8a8a8a", "red": "#d62728", "blue": "#1f77b4",
             "green": "#2ca02c", "brown": "#8c564b", "purple": "#9467bd",
             "cyan": "#17becf", "yellow": "#e8b800"}
SHAPE_SYM = {"cube": "square", "sphere": "circle", "cylinder": "diamond"}


class WorldReq(BaseModel):
    scene: str
    remove: list = []
    thresh: float = 0.02
    release: float = 0.02
    taint_margin: float = 0.06
    lookback: int = 12
    contact: float = None


class CCDReq(BaseModel):
    scene: str
    remove: list = []       # if given, sweep THESE objects cumulatively
    steps: int = 40
    max_remove: int = 3
    repeats: int = 5        # random draws averaged per deletion count


def _load_npz(name):
    return np.load(os.path.join(STATE["data"], name), allow_pickle=True)


def _vid_num(name):
    m = re.search(r"(\d+)", os.path.basename(str(name)))
    return int(m.group(1)) if m else 0


def _objects(z, N):
    out = []
    for k in range(N):
        parts = str(z["obj_keys"][k]).split("_")
        colour = parts[0] if parts else "gray"
        material = parts[1] if len(parts) > 1 else "rubber"
        shape = parts[2] if len(parts) > 2 else "sphere"
        out.append({"name": str(z["obj_keys"][k]).replace("_", " "),
                    "hex": COLOR_HEX.get(colour, "#555555"),
                    "symbol": SHAPE_SYM.get(shape, "circle"),
                    "metal": material == "metal"})
    return out


@torch.no_grad()
def _panel_report(model, pos_np, vel_np, present_np, attrs_np):
    """UI data for one panel, matching the original viewer's panel_report:
    per-frame KE, PE-share, speed, totals, and per-object 'moved'.
    pos_np may contain NaN where an object is absent."""
    T, N, _ = pos_np.shape
    pos_c = np.nan_to_num(pos_np, nan=0.0)
    positions = torch.from_numpy(pos_c.astype(np.float32))
    velocities = torch.from_numpy(np.nan_to_num(vel_np, nan=0.0).astype(np.float32))
    present = torch.from_numpy(present_np.astype(np.float32))
    from benchmark_eval import STRIP
    a1 = torch.from_numpy(attrs_np[:N].astype(np.float32)).unsqueeze(0)
    mass1, radius1, e1 = model.properties(STRIP(a1))
    mass = mass1.expand(T, -1)
    radius = radius1.expand(T, -1)
    e = e1.expand(T, -1, -1)
    p = mass.unsqueeze(-1) * velocities * present.unsqueeze(-1)
    ke = (p ** 2).sum(-1) / (2 * mass)
    speed = velocities.norm(dim=-1) * present
    if hasattr(model, "pair_potentials"):
        Vmat = model.pair_potentials(positions, mass, radius, e, present)
        pe = 0.5 * (Vmat.sum(2) + Vmat.sum(1))
    else:                                   # legacy checkpoints: no PE share
        pe = torch.zeros(T, N)
    moved = []
    for i in range(N):
        seen = present_np[:, i] > 0
        if seen.sum() < 2:
            moved.append(False)
        else:
            span = pos_c[seen, i, :]
            moved.append(bool((span.max(0) - span.min(0)).max() > 0.02))
    r = lambda t: [[round(float(x), 5) for x in row] for row in t.numpy().tolist()]
    return {"pos": [[[round(float(x), 4) for x in xy] for xy in fr]
                    for fr in pos_c.tolist()],
            "present": present_np.astype(int).tolist(),
            "ke": r(ke), "pe": r(pe), "speed": r(speed),
            "tke": [round(float(x), 5) for x in ke.sum(-1).tolist()],
            "tpe": [round(float(x), 5) for x in pe.sum(-1).tolist()],
            "moved": moved, "W": T}


app = FastAPI()


@app.get("/api/scenes")
def scenes():
    files = sorted(os.path.basename(f) for f in
                   glob.glob(os.path.join(STATE["data"], "*.npz")))
    return {"scenes": files[:400], "dir": STATE["data"]}


@app.get("/api/scene/{name}")
def scene_info(name: str):
    z = _load_npz(name)
    N = min(8, z["positions"].shape[1])
    cols = [{"frame": int(r[0]), "i": int(r[1]), "j": int(r[2])}
            for r in z["collisions"] if int(r[1]) < N and int(r[2]) < N]
    return {"objects": _objects(z, N), "collisions": cols,
            "frames": int(z["positions"].shape[0])}


@app.post("/api/world")
def api_world(req: WorldReq):
    z = _load_npz(req.scene)
    pos = z["positions"].astype(np.float32)
    vel = z["velocities"].astype(np.float32)
    pres = z["presence"]
    T = pos.shape[0]
    N = min(8, pos.shape[1])
    removed = sorted(int(k) for k in req.remove if 0 <= int(k) < N)

    events, pair_set, taint_frame, taint_cause, traj, traj_mode = \
        lightcone_rollout(STATE["model"], z, set(removed), req.thresh,
                          req.release, req.taint_margin, lookback=req.lookback,
                          contact_override=req.contact, record_traj=True)

    gt_cols = [{"frame": int(r[0]), "i": int(r[1]), "j": int(r[2])}
               for r in z["collisions"] if int(r[1]) < N and int(r[2]) < N]
    rec = {"video": _vid_num(req.scene), "question": "", "negate": False,
           "obj_keys": [str(k) for k in z["obj_keys"]][:N],
           "removed": removed, "match_mode": "explicit",
           "taints": [{"idx": int(k), "frame": int(taint_frame[k]),
                       "cause": taint_cause.get(k, {})}
                      for k in sorted(taint_frame)],
           "gt_collisions": gt_cols, "cf_events": events, "choices": []}
    if removed:
        lines = narrate(rec)
    else:
        lines = ["No intervention selected. Both panels show the observed "
                 "world. Tick one or more objects and run again to build an "
                 "edited world."]

    # ---- panel data in the original viewer's format ------------------------
    fact_pos = np.where(pres[:, :N, None] > 0, pos[:, :N], np.nan)
    fact_present = (pres[:, :N] > 0).astype(np.float32)
    panel_f = _panel_report(STATE["model"], fact_pos, vel[:, :N],
                            fact_present, z["attrs"])

    edit_present = np.isfinite(traj[:, :, 0]).astype(np.float32)
    edit_vel = np.zeros_like(vel[:, :N])
    for t in range(T):
        for k in range(N):
            if traj_mode[t, k] == 1:                     # observed: real velocity
                edit_vel[t, k] = vel[t, k]
            elif traj_mode[t, k] == 2:                   # simulated: back-diff
                if t > 0 and np.isfinite(traj[t - 1, k, 0]):
                    edit_vel[t, k] = traj[t, k] - traj[t - 1, k]
                else:
                    edit_vel[t, k] = vel[t, k]
    panel_c = _panel_report(STATE["model"], traj, edit_vel, edit_present,
                            z["attrs"])
    panel_c["mode"] = traj_mode.tolist()

    # ---- the short blurbs under the two world panels -----------------------
    names = [o["name"] for o in _objects(z, N)]
    n_gt = len(gt_cols)
    first_f = min((c["frame"] for c in gt_cols), default=None)
    fact_blurb = (f"The observed video, as recorded: {N} tracked objects and "
                  f"{n_gt} recorded collision(s)"
                  + (f", the first at frame {first_f}" if first_f is not None else "")
                  + ". This panel is data, not prediction.")

    kept = sum(1 for ev in events if ev["source"] == "observed")
    new = sum(1 for ev in events
              if ev["source"] == "simulated"
              and frozenset((ev["i"], ev["j"])) not in
              set(frozenset((c["i"], c["j"])) for c in gt_cols))
    rederived = sum(1 for ev in events if ev["source"] == "simulated") - new
    vanish = sum(1 for c in gt_cols
                 if frozenset((c["i"], c["j"])) not in pair_set)
    rm_names = " and the ".join(names[k] for k in removed)
    if not removed:
        edit_blurb = ("No object removed, so this panel is the observed "
                      "world unchanged. Tick objects on the top bar and run "
                      "again to see the edited world.")
    elif taint_frame:
        edit_blurb = (f"Without the {rm_names}: {len(taint_frame)} object(s) "
                      f"are re-simulated by the physics engine, the rest keep "
                      f"their observed paths. {kept} collision(s) carry over "
                      f"exactly, {rederived} re-derived in the cone, "
                      f"{vanish} vanish, {new} new one(s) appear.")
    else:
        edit_blurb = (f"Without the {rm_names}: nothing else ever interacts "
                      f"with it, so every trajectory here is identical to the "
                      f"observed video. {vanish} collision(s) vanish only "
                      f"because a participant was removed.")

    meta = _objects(z, N)
    with torch.no_grad():
        from benchmark_eval import STRIP
        a1 = torch.from_numpy(z["attrs"][:N].astype(np.float32)).unsqueeze(0)
        massv, _, _ = STATE["model"].properties(STRIP(a1))
    for k, m in enumerate(meta):
        m["mass"] = round(float(massv[0, k]), 3)

    return {"objects": meta, "frames": T,
            "real": panel_f, "edited": panel_c,
            "events": events, "gt_collisions": gt_cols,
            "taints": rec["taints"], "removed": removed,
            "narration": lines,
            "blurbs": {"factual": fact_blurb, "edited": edit_blurb}}


# ---------------------------------------------------------------------------
# the CCD tab: three physics modes at inference, exactly as in ccd_eval.py
# ---------------------------------------------------------------------------

def _ccd_roll(model, q0, v0, attrs, present, steps, mode, dt=1.0,
              break_idx=None):
    from benchmark_eval import STRIP
    phys = STRIP(attrs.unsqueeze(0))
    mass, radius, e = model.properties(phys)
    mass, radius, e = mass.detach(), radius.detach(), e.detach()
    pres = present.unsqueeze(0)

    # the broken ablation must act on an object that is actually PRESENT.
    # Hardcoding index 0 silently turns the ablation off whenever object 0 is
    # the one deleted, which makes the broken curve collapse onto the
    # symplectic one at that deletion count.
    if break_idx is None:
        q_probe = q0.unsqueeze(0) * pres.unsqueeze(-1)
        F0 = model.forces(q_probe, mass, radius, e, pres,
                          create_graph=False).detach()
        mag = F0[0].norm(dim=-1) * present            # zero out absent objects
        break_idx = int(torch.argmax(mag)) if float(mag.max()) > 0 else 0

    def forces(qcur, broken):
        F = model.forces(qcur, mass, radius, e, pres, create_graph=False)
        if broken:
            F = F.clone()
            F[:, break_idx, :] = F[:, break_idx, :] * 1.5
        return F.detach()

    q = q0.unsqueeze(0) * pres.unsqueeze(-1)
    p = mass.unsqueeze(-1) * v0.unsqueeze(0) * pres.unsqueeze(-1)
    H0 = float(((p ** 2).sum(-1).div(2 * mass).sum(-1)
                + model.V(q, mass, radius, e, pres)).item())
    P0 = p.sum(1)[0].detach().clone()
    Hs, Ps = [H0], [P0]
    for _ in range(steps):
        q, p = q.detach(), p.detach()
        if mode == "symplectic":
            F = forces(q, False); p = p + 0.5 * dt * F
            q = q + dt * p / mass.unsqueeze(-1)
            F2 = forces(q, False); p = p + 0.5 * dt * F2
        elif mode == "euler":
            F = forces(q, False)
            q = q + dt * p / mass.unsqueeze(-1)
            p = p + dt * F
        else:  # broken
            F = forces(q, True); p = p + 0.5 * dt * F
            q = q + dt * p / mass.unsqueeze(-1)
            F2 = forces(q, True); p = p + 0.5 * dt * F2
        H = (p ** 2).sum(-1).div(2 * mass).sum(-1) + model.V(q, mass, radius, e, pres)
        Hs.append(H.item())
        Ps.append(p.sum(1)[0].detach())
    Hs = np.array(Hs)
    mom = max((Pt - P0).norm().item() for Pt in Ps)
    eng = float(np.abs(Hs - H0).max() / (np.abs(Hs).mean() + 1e-9))
    return eng, mom


def _times_words(ratio):
    """Turn a ratio into everyday words, keeping it honest."""
    if ratio < 10:
        return "a few times"
    p = int(np.floor(np.log10(ratio)))
    names = {1: "about ten times", 2: "about a hundred times",
             3: "about a thousand times", 4: "about ten thousand times",
             5: "about a hundred thousand times", 6: "about a million times",
             7: "about ten million times", 8: "about a hundred million times",
             9: "about a billion times"}
    return names.get(p, f"about 10^{p} times")


def _trend(vals):
    """'flat' or 'climbs' for one curve across the deletions."""
    lo, hi = min(vals), max(vals)
    if hi <= 1e-12 or hi <= 5 * max(lo, 1e-30):
        return "flat"
    return "climbs"


def _ccd_blurb(removes, eng, mom, sweep="random", swept=None):
    """Plain-words, pattern-driven reading of the two charts."""
    out = []
    if sweep == "selection" and swept:
        order = ", then the ".join(swept)
        out.append(f"This sweep follows your own intervention: first nothing "
                   f"is deleted, then the {order}. Each point on the x axis "
                   f"is one more of your ticked objects taken away.")
    else:
        out.append("No objects are ticked, so this sweep deletes 0, 1, 2 and "
                   "3 objects at random and averages several draws at each "
                   "count.")
    out.append("What the charts ask: after we delete objects and replay the "
               "scene, does the physics quietly leak? Lower is better, and a "
               "flat line means deleting objects changes nothing about how "
               "honest the physics stays.")

    sm, bm = max(mom["symplectic"]), max(mom["broken"])
    t_sym, t_brk = _trend(mom["symplectic"]), _trend(mom["broken"])
    if bm > 10 * max(sm, 1e-30):
        out.append(f"Momentum chart: our model (symplectic, blue) is the "
                   f"{'flat ' if t_sym == 'flat' else ''}line at the bottom. "
                   f"Its pushes always come in equal and opposite pairs, so "
                   f"no momentum can appear or disappear, no matter how many "
                   f"objects we delete. The broken line (green) "
                   f"{'climbs with every deletion' if t_brk == 'climbs' else 'sits far above it'}: "
                   f"once forces stop cancelling, the scene gains or loses "
                   f"momentum out of nowhere, ending up "
                   f"{_times_words(bm / max(sm, 1e-30))} worse than ours.")
    else:
        out.append("Momentum chart: the three lines sit close together on "
                   "this scene, so it does not separate the modes much. Try "
                   "another scene for a clearer gap.")

    se, ee = max(eng["symplectic"]), max(eng["euler"])
    if ee > 5 * max(se, 1e-30):
        out.append(f"Energy chart: our leapfrog stepper (blue) makes energy "
                   f"wobble in place instead of draining away, so its line "
                   f"stays low. The plain euler stepper (orange) leaks a "
                   f"little energy every single frame, which adds up to "
                   f"{_times_words(ee / max(se, 1e-30))} more drift.")
    else:
        out.append("Energy chart: both steppers hold energy about equally "
                   "well on this scene.")

    out.append("The takeaway in one line: flat and low means the model's "
               "physics survives being edited, which is exactly what a "
               "counterfactual needs.")
    return " ".join(out)


@app.post("/api/ccd")
def api_ccd(req: CCDReq):
    import random
    random.seed(0)
    model = STATE["model"]
    z = _load_npz(req.scene)
    pos, vel, pres, attrs = (z["positions"], z["velocities"], z["presence"],
                             z["attrs"])
    T, N = pos.shape[0], pos.shape[1]
    cand = [t for t in range(T) if (pres[t] > 0).sum() >= 3]
    if not cand:
        return JSONResponse({"error": "fewer than 3 objects ever on screen"},
                            status_code=400)
    s = cand[len(cand) // 2]
    on = list(np.where(pres[s] > 0)[0])
    q0 = torch.from_numpy(pos[s].astype(np.float32))
    v0 = torch.from_numpy(vel[s].astype(np.float32))
    at = torch.from_numpy(attrs.astype(np.float32))

    modes = ["symplectic", "euler", "broken"]

    # If the user ticked objects, sweep THEIR intervention: delete the first
    # one, then the first two, and so on.  Deterministic, so no averaging is
    # needed and the curve is exactly about the chosen do(.) sequence.
    sel = [int(k) for k in req.remove
           if 0 <= int(k) < N and int(k) in on]
    sel = list(dict.fromkeys(sel))                   # keep order, drop repeats
    sel = sel[:max(0, len(on) - 2)]                  # leave at least 2 objects
    if sel:
        removes = list(range(len(sel) + 1))
        sweep = "selection"
    else:
        removes = [r for r in range(req.max_remove + 1) if r <= len(on) - 2]
        sweep = "random"
    eng = {m: [] for m in modes}
    mom = {m: [] for m in modes}
    eng_lo = {m: [] for m in modes}
    eng_hi = {m: [] for m in modes}
    mom_lo = {m: [] for m in modes}
    mom_hi = {m: [] for m in modes}

    for r in removes:
        # r = 0 has only one possible draw; a user selection is deterministic;
        # otherwise average several so one unlucky subset cannot define the curve
        n_draw = 1 if (r == 0 or sweep == "selection") else max(1, int(req.repeats))
        acc_e = {m: [] for m in modes}
        acc_m = {m: [] for m in modes}
        for _ in range(n_draw):
            drop = set(sel[:r]) if sweep == "selection" else set(random.sample(on, r))
            keep = [idx for idx in on if idx not in drop]
            present = torch.zeros(N)
            for idx in keep:
                present[idx] = 1.0
            for m in modes:
                ed, md = _ccd_roll(model, q0, v0, at, present, req.steps, m)
                acc_e[m].append(ed)
                acc_m[m].append(md)
        for m in modes:
            eng[m].append(float(np.mean(acc_e[m])))
            mom[m].append(float(np.mean(acc_m[m])))
            eng_lo[m].append(float(np.min(acc_e[m])))
            eng_hi[m].append(float(np.max(acc_e[m])))
            mom_lo[m].append(float(np.min(acc_m[m])))
            mom_hi[m].append(float(np.max(acc_m[m])))

    return {"removes": removes, "energy": eng, "momentum": mom,
            "energy_lo": eng_lo, "energy_hi": eng_hi,
            "momentum_lo": mom_lo, "momentum_hi": mom_hi,
            "repeats": 1 if sweep == "selection" else int(req.repeats),
            "sweep": sweep,
            "swept": [str(z["obj_keys"][k]).replace("_", " ") for k in sel],
            "start_frame": int(s), "steps": req.steps,
            "blurb": _ccd_blurb(removes, eng, mom, sweep,
                                [str(z["obj_keys"][k]).replace("_", " ")
                                 for k in sel])}


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------

HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>CausalVis V3</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
 body{font-family:Arial,Helvetica,sans-serif;margin:0;background:#f6f4f0;color:#1a1a1a}
 header{background:#73000a;color:#fff;padding:14px 22px;font-size:20px;font-weight:bold}
 .wrap{padding:16px 22px;max-width:1280px;margin:0 auto}
 .row{display:flex;gap:18px;flex-wrap:wrap}
 .panel{background:#fff;border:1px solid #d8d4cc;border-radius:4px;padding:12px;flex:1;min-width:300px}
 h3{margin:0 0 10px;color:#73000a;font-size:15px}
 select,button{font-size:14px;padding:6px 8px}
 button{background:#73000a;color:#fff;border:0;border-radius:3px;cursor:pointer}
 button:disabled{background:#999}
 label{display:block;margin:3px 0;font-size:14px}
 .muted{color:#666;font-size:13px}
 .tot{font-size:12.5px;color:#444;margin-top:4px}
 input[type=range]{width:70%;vertical-align:middle}
 .blurb{font-size:13.5px;line-height:1.45;background:#faf8f4;border:1px solid #d8d4cc;
        border-radius:3px;padding:8px 10px;margin-top:8px}
 pre{white-space:pre-wrap;font-size:13px;line-height:1.5;background:#fbfaf8;
     border:1px solid #d8d4cc;padding:12px;border-radius:3px}
 .tabs{display:flex;gap:0;margin-top:14px;border-bottom:2px solid #73000a}
 .tab{padding:9px 18px;font-size:14px;font-weight:bold;cursor:pointer;
      background:#e8e2d8;border:1px solid #d8d4cc;border-bottom:0;border-radius:4px 4px 0 0;margin-right:4px}
 .tab.on{background:#73000a;color:#fff}
 .page{display:none;margin-top:14px}
 .page.on{display:block}
</style></head><body>
<header>CausalVis V3 &mdash; worlds, conservation, explanation</header>
<div id="err" style="display:none;background:#a2140f;color:#fff;padding:10px 22px;
     font-size:14px;white-space:pre-wrap"></div>
<div class="wrap">
  <div class="panel">
    <h3>Scene and intervention</h3>
    <select id="scene"></select>
    <button id="run" style="margin-left:12px">Run</button>
    <span id="status" class="muted"></span>
    <div class="muted" style="margin-top:8px">Tick the objects to delete
      (none ticked = just show the observed world), then press Run.</div>
    <div id="objs" style="margin-top:6px"></div>
  </div>

  <div class="tabs">
    <div class="tab on" data-p="pw">Worlds</div>
    <div class="tab" data-p="pcc">Conservation (CCD)</div>
    <div class="tab" data-p="pe">Explanation</div>
  </div>

  <div class="page on" id="pw">
    <div class="row">
      <div class="panel"><h3>Real world (observed)</h3>
        <div id="pf" style="height:400px"></div>
        <div id="frwrapR" style="display:none;margin-top:6px">frame
          <input type="range" id="frameR" min="0" max="0" value="0">
          <span id="flabelR" class="muted"></span></div>
        <div id="ftot" class="tot"></div>
        <div class="blurb" id="blF">run to see the worlds</div></div>
      <div class="panel"><h3>Edited world (light-cone)</h3>
        <div id="pc2" style="height:400px"></div>
        <div id="frwrapC" style="display:none;margin-top:6px">frame
          <input type="range" id="frameC" min="0" max="0" value="0">
          <span id="flabelC" class="muted"></span></div>
        <div id="ctot" class="tot"></div>
        <div class="blurb" id="blC">&nbsp;</div></div>
    </div>
  </div>

  <div class="page" id="pcc">
    <div class="panel"><h3>Counterfactual Conservation Drift for this scene</h3>
      <button id="runccd">Compute CCD (three physics modes)</button>
      <span id="ccdstat" class="muted"></span>
      <div class="muted" style="margin-top:6px">With objects ticked above, this
        sweeps your own intervention: nothing deleted, then the first ticked
        object, then the first two, and so on. With nothing ticked it falls
        back to deleting 0, 1, 2 and 3 objects at random, averaged over
        several draws.</div>
      <div class="row" style="margin-top:10px">
        <div style="flex:1;min-width:320px"><div id="figE" style="height:340px"></div></div>
        <div style="flex:1;min-width:320px"><div id="figP" style="height:340px"></div></div>
      </div>
      <div class="blurb" id="blCCD">press the button to run the diagnostic on the selected scene</div>
    </div>
  </div>

  <div class="page" id="pe">
    <div class="panel"><h3>Grounded explanation of the intervention</h3>
      <pre id="narr">run an intervention first</pre></div>
  </div>
</div>
<script>
let D=null;
function fail(msg){const e=document.getElementById('err');
  e.style.display='block'; e.textContent=msg;}
window.onerror=(m,src,line)=>{fail('Page error: '+m+' (line '+line+')');};
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('on'));
  document.querySelectorAll('.page').forEach(x=>x.classList.remove('on'));
  t.classList.add('on'); document.getElementById(t.dataset.p).classList.add('on');
});
function lighten(hex,f){const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);
  const L=x=>Math.round(x+(255-x)*f);return `rgb(${L(r)},${L(g)},${L(b)})`;}
async function loadScenes(){
  let d;
  try{ d=await (await fetch('/api/scenes')).json(); }
  catch(e){ fail('Could not reach the server for the scene list: '+e); return; }
  if(!d.scenes || !d.scenes.length){
    fail('The server found no .npz files in\\n  '+(d.dir||'(unknown folder)')+
         '\\nStart it from the project root (G:\\\\CausalVis) or pass '+
         '--data with the correct path, then reload this page.');
    return;
  }
  const s=document.getElementById('scene');
  s.innerHTML=d.scenes.map(n=>`<option>${n}</option>`).join('');
  s.onchange=loadObjs; await loadObjs();
}
async function loadObjs(){
  const name=document.getElementById('scene').value;
  let d;
  try{ d=await (await fetch('/api/scene/'+name)).json(); }
  catch(e){ fail('Scene '+name+' failed to load: '+e); return; }
  if(!d.objects){ fail('Scene '+name+': server error: '+JSON.stringify(d)); return; }
  document.getElementById('objs').innerHTML=d.objects.map((o,i)=>
    `<label style="display:inline-block;margin-right:14px">
       <input type="checkbox" class="rm" value="${i}">
       <span style="color:${o.hex};font-size:16px">${o.symbol==='square'?'&#9632;':(o.symbol==='diamond'?'&#9670;':'&#9679;')}</span>
       [${i}] ${o.name}</label>`).join('');
}

// ---- the original viewer's panel drawing: trails only up to the slider ----
function panelTraces(P, upto, isEdited){
  const out=[];
  D.objects.forEach((m,i)=>{
    let lastSeen=-1;
    for(let t=0;t<=upto;t++){ if(P.present[t] && P.present[t][i]) lastSeen=t; }
    if(lastSeen<0) return;
    const light=lighten(m.hex,0.45), lighter=lighten(m.hex,0.7);
    const xs=[],ys=[];
    for(let t=0;t<=upto;t++){ if(P.present[t] && P.present[t][i]){
      xs.push(P.pos[t][i][0]); ys.push(P.pos[t][i][1]); } }
    if(P.moved[i] && xs.length>1){
      const k=Math.max(1,Math.floor(xs.length*0.25));
      out.push({x:xs.slice(0,k+1),y:ys.slice(0,k+1),mode:'lines',
        line:{color:lighter,width:3},opacity:0.35,hoverinfo:'skip',showlegend:false});
      out.push({x:xs.slice(k),y:ys.slice(k),mode:'lines',
        line:{color:light,width:3},opacity:0.9,hoverinfo:'skip',showlegend:false});
    }
    const hx=P.pos[lastSeen][i][0], hy=P.pos[lastSeen][i][1];
    let tag='';
    if(isEdited && P.mode) tag = P.mode[lastSeen][i]===2 ? ' [simulated]'
                             : (P.mode[lastSeen][i]===1 ? ' [observed]' : '');
    const info=`${m.name} (${m.metal?'metal':'rubber'})${tag}<br>mass ${m.mass}`
      +`<br>KE ${P.ke[lastSeen][i]}<br>PE-share ${P.pe[lastSeen][i]}`
      +`<br>speed ${P.speed[lastSeen][i]}/frame`;
    out.push({x:[hx],y:[hy],mode:'markers',
      marker:{color:m.hex,symbol:m.symbol,size:P.moved[i]?15:13,
              line:{color:'#111',width:m.metal?2.5:0}},
      text:[info],hoverinfo:'text',showlegend:false});
  });
  return out;
}
const LAY={margin:{l:28,r:8,t:8,b:26},xaxis:{range:[0,1],zeroline:false,fixedrange:true},
  yaxis:{range:[1,0],zeroline:false,scaleanchor:'x',fixedrange:true},plot_bgcolor:'#fbfaf8'};

function drawPanel(div,P,totId,labId,t,isEdited){
  if(typeof Plotly==='undefined'){
    document.getElementById(div).innerHTML='<div style="padding:20px;color:#a2140f">'+
      'Plotly did not load from cdn.plot.ly. Check the internet connection '+
      'or unblock that domain, then reload.</div>'; return;}
  Plotly.react(div,panelTraces(P,t,isEdited),LAY,{displayModeBar:false});
  document.getElementById(totId).innerHTML=
    `total KE ${P.tke[t]} &nbsp; total PE ${P.tpe[t]} &nbsp; total energy ${(P.tke[t]+P.tpe[t]).toFixed(5)}`;
  document.getElementById(labId).textContent=`frame ${t}/${P.W-1}`;
}

document.getElementById('run').onclick=async()=>{
  const btn=document.getElementById('run'); btn.disabled=true;
  document.getElementById('status').textContent=' running the light-cone pass...';
  const rm=[...document.querySelectorAll('.rm:checked')].map(e=>+e.value);
  let r,d;
  try{
    r=await fetch('/api/world',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({scene:document.getElementById('scene').value,remove:rm})});
    d=await r.json();
  }catch(e){ fail('The intervention request failed: '+e);
    btn.disabled=false; document.getElementById('status').textContent=''; return; }
  btn.disabled=false; document.getElementById('status').textContent='';
  if(!r.ok){document.getElementById('blF').textContent='error: '+(d.error||r.status);return;}
  D=d;
  const R=document.getElementById('frameR'), C=document.getElementById('frameC');
  R.max=D.real.W-1;   R.value=0;
  C.max=D.edited.W-1; C.value=0;
  document.getElementById('frwrapR').style.display='block';
  document.getElementById('frwrapC').style.display='block';
  R.oninput=()=>drawPanel('pf',D.real,'ftot','flabelR',+R.value,false);
  C.oninput=()=>drawPanel('pc2',D.edited,'ctot','flabelC',+C.value,true);
  drawPanel('pf',D.real,'ftot','flabelR',0,false);
  drawPanel('pc2',D.edited,'ctot','flabelC',0,true);
  document.getElementById('blF').textContent=D.blurbs.factual;
  document.getElementById('blC').textContent=D.blurbs.edited;
  document.getElementById('narr').textContent=D.narration.join('\\n');
};
document.getElementById('runccd').onclick=async()=>{
  const b=document.getElementById('runccd'); b.disabled=true;
  document.getElementById('ccdstat').textContent=' simulating three modes x four interventions...';
  let r,d;
  try{
    r=await fetch('/api/ccd',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({scene:document.getElementById('scene').value,
        remove:[...document.querySelectorAll('.rm:checked')].map(e=>+e.value)})});
    d=await r.json();
  }catch(e){ fail('The CCD request failed: '+e);
    b.disabled=false; document.getElementById('ccdstat').textContent=''; return; }
  b.disabled=false; document.getElementById('ccdstat').textContent='';
  if(!r.ok){document.getElementById('blCCD').textContent='error: '+(d.error||r.status);return;}
  const cols={symplectic:'#73000a',euler:'#8a8a8a',broken:'#333333'};
  const rgba=(hex,a)=>{const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),
    b=parseInt(hex.slice(5,7),16);return `rgba(${r},${g},${b},${a})`;};
  const mk=(store,lo,hi,title,div)=>{
    const traces=[];
    Object.keys(store).forEach(m=>{                    // spread band first
      if(lo&&hi&&d.repeats>1){
        traces.push({x:d.removes.concat([...d.removes].reverse()),
          y:lo[m].concat([...hi[m]].reverse()),
          fill:'toself',fillcolor:rgba(cols[m],0.13),line:{width:0},
          hoverinfo:'skip',showlegend:false});
      }
    });
    Object.keys(store).forEach(m=>{
      traces.push({x:d.removes,y:store[m],name:m,mode:'lines+markers',
        line:{color:cols[m],width:2.5},marker:{size:8}});
    });
    Plotly.react(div,traces,{title:{text:title,font:{size:13}},
      xaxis:{title:{text:'objects deleted',standoff:6},dtick:1,zeroline:false},
      yaxis:{type:'log',title:'conservation drift',zeroline:false,
             exponentformat:'power',showexponent:'all'},
      margin:{l:66,r:10,t:26,b:52},
      legend:{orientation:'h',y:1.14,x:0},
      plot_bgcolor:'#fbfaf8'},{displayModeBar:false});
  };
  mk(d.energy,d.energy_lo,d.energy_hi,'Energy','figE');
  mk(d.momentum,d.momentum_lo,d.momentum_hi,'Momentum','figP');
  document.getElementById('blCCD').textContent=d.blurb;
};
loadScenes();
</script></body></html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/trajectories_v2")
    ap.add_argument("--model", default="v3_2_dynamics.pt")
    ap.add_argument("--legacy-dynamics", default=None)
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    STATE["data"] = os.path.abspath(a.data)
    n_npz = len(glob.glob(os.path.join(STATE["data"], "*.npz")))
    print(f"data folder : {STATE['data']}   ({n_npz} npz files found)")
    if n_npz == 0:
        print("WARNING: no .npz files there. Start the server from the "
              "project root (G:\\\\CausalVis) or pass --data with the right "
              "path. The page will say the same.")
    STATE["model"] = load_model(a.model, a.legacy_dynamics)
    print(f"open http://127.0.0.1:{a.port}")
    uvicorn.run(app, host="127.0.0.1", port=a.port)