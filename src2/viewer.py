"""
viewer.py  (src2)  —  rich re-simulation viewer (Plotly)
========================================================

Left  = REAL world: the ground-truth trajectory from the data (whole clip, all
        frames, real collisions show as bends). No model, no drift.
Right = EDITED world: the model's prediction when you delete objects (this world
        never existed, so it must be simulated).

Per object: cubes are squares, spheres/cylinders are circles, real colours.
Static objects (never move) are a single marker. Moving objects are a solid
light-coloured trail whose oldest quarter is faded, with a solid head showing
where it ended. Drag the frame slider to see the world up to any frame, with
each object's kinetic energy, potential-energy share and speed at that frame.

The CCD panel compares the structured engine against two ablations (euler,
broken) on THIS scene, energy and momentum, integral form.

INSTALL   pip install fastapi uvicorn
RUN       python viewer.py --data data/trajectories_v2 --model v3_2_dynamics.pt
          open http://127.0.0.1:8000
"""

import argparse, glob, os, random
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from dynamics import HamiltonianDynamics, strip_colour
from intervene import do_remove
from readout import build_scene, simulate, pair_collides, window_start

STATE = {}
app = FastAPI()

CSS = {"gray": "#8a8a8a", "red": "#d62728", "blue": "#1f77b4", "green": "#2ca02c",
       "brown": "#8c564b", "purple": "#9467bd", "cyan": "#17becf", "yellow": "#e8b800"}


class SimReq(BaseModel):
    scene: str
    remove: list = []
    lookback: int = 12
    extra: int = 25
    thresh: float = 0.02


def load(name):
    return np.load(os.path.join(STATE["data"], name), allow_pickle=True)


def style_of(key):
    parts = key.split("_")                       # colour_material_shape
    colour = CSS.get(parts[0], "#555")
    material = parts[1] if len(parts) >= 3 else "rubber"
    shape = parts[-1] if len(parts) >= 3 else "sphere"
    return colour, ("square" if shape == "cube" else "circle"), shape, material


@torch.no_grad()
def panel_report(model, positions, velocities, present, attrs):
    """positions,velocities: [W,N,2] tensors; present: [W,N]; attrs: [N,16].
    Returns per-frame KE, PE-share, speed, and per-object 'moved'."""
    W, N, _ = positions.shape
    a1 = attrs.unsqueeze(0)
    mass1, radius1, e1 = model.properties(strip_colour(a1))       # [1,N],[1,N],[1,N,emb]
    mass = mass1.expand(W, -1)
    radius = radius1.expand(W, -1)
    e = e1.expand(W, -1, -1)
    pres = present
    p = mass.unsqueeze(-1) * velocities * pres.unsqueeze(-1)
    ke = (p ** 2).sum(-1) / (2 * mass)                            # [W,N]
    speed = velocities.norm(dim=-1) * pres                        # [W,N]
    Vmat = model.pair_potentials(positions, mass, radius, e, pres)  # [W,N,N]
    pe = 0.5 * (Vmat.sum(2) + Vmat.sum(1))                        # [W,N] raw pairwise share
    # moved: position range over frames where present
    moved = []
    P = positions.cpu().numpy()
    M = present.cpu().numpy()
    for i in range(N):
        seen = M[:, i] > 0
        if seen.sum() < 2:
            moved.append(False)
        else:
            span = P[seen, i, :]
            moved.append(bool((span.max(0) - span.min(0)).max() > 0.02))
    r = lambda t: [[round(float(x), 5) for x in row] for row in t.cpu().numpy().tolist()]
    return {
        "pos": [[[round(float(x), 4) for x in xy] for xy in fr]
                for fr in positions.cpu().numpy().tolist()],
        "present": present.cpu().numpy().astype(int).tolist(),
        "ke": r(ke), "pe": r(pe), "speed": r(speed),
        "tke": [round(float(x), 5) for x in ke.sum(-1).tolist()],
        "tpe": [round(float(x), 5) for x in pe.sum(-1).tolist()],
        "moved": moved,
    }


def roll_mode(model, scene, steps, mode, dt=1.0):
    """One rollout under a physics mode; returns (energy_ccd, momentum_ccd),
    both as the integral of the absolute step-to-step change."""
    q0, v0, attrs, present = scene[0], scene[1], scene[2], scene[3]
    phys = strip_colour(attrs.unsqueeze(0))
    mass, radius, e = model.properties(phys)
    mass, radius, e = mass.detach(), radius.detach(), e.detach()
    pres = present.unsqueeze(0)

    def forces(qc, broken):
        F = model.forces(qc, mass, radius, e, pres, create_graph=False)
        if broken:
            F = F.clone(); F[:, 0, :] = F[:, 0, :] * 1.5
        return F.detach()

    q = q0.unsqueeze(0) * pres.unsqueeze(-1)
    p = mass.unsqueeze(-1) * v0.unsqueeze(0) * pres.unsqueeze(-1)
    Hs, Ps = [], []
    for _ in range(steps):
        q, p = q.detach(), p.detach()
        if mode == "symplectic":
            F = forces(q, False); p = p + 0.5 * dt * F
            q = q + dt * p / mass.unsqueeze(-1)
            F2 = forces(q, False); p = p + 0.5 * dt * F2
        elif mode == "euler":
            F = forces(q, False); q = q + dt * p / mass.unsqueeze(-1); p = p + dt * F
        else:
            F = forces(q, True); p = p + 0.5 * dt * F
            q = q + dt * p / mass.unsqueeze(-1)
            F2 = forces(q, True); p = p + 0.5 * dt * F2
        with torch.no_grad():
            H = (p ** 2).sum(-1).div(2 * mass).sum(-1) + model.V(q, mass, radius, e, pres)
        Hs.append(H.item()); Ps.append(p.sum(1)[0].detach())
    Hs = np.array(Hs)
    eng = float(np.abs(np.diff(Hs)).sum() / (np.abs(Hs).mean() + 1e-9))
    mom = float(sum((Ps[t] - Ps[t - 1]).norm().item() for t in range(1, len(Ps))))
    return eng, mom


def ccd_compare(model, scene, steps):
    steps = min(steps, 35)
    on = [i for i in range(scene[3].shape[0]) if float(scene[3][i]) > 0]
    removes = list(range(0, min(3, len(on) - 2) + 1))
    random.seed(0)
    out = {"removes": removes,
           "energy": {m: [] for m in ("symplectic", "euler", "broken")},
           "momentum": {m: [] for m in ("symplectic", "euler", "broken")}}
    for r in removes:
        sc = [scene[0].clone(), scene[1].clone(), scene[2].clone(), scene[3].clone()]
        for k in random.sample(on, r):
            sc = do_remove(sc, k)
        for m in ("symplectic", "euler", "broken"):
            eng, mom = roll_mode(model, sc, steps, m)
            out["energy"][m].append(eng)
            out["momentum"][m].append(mom)
    return out


@app.get("/api/scenes")
def scenes():
    fs = sorted(os.path.basename(f) for f in glob.glob(os.path.join(STATE["data"], "*.npz")))
    return {"scenes": fs[:400]}


@app.get("/api/scene/{name}")
def scene_info(name):
    z = load(name)
    keys = [str(x) for x in z["obj_keys"]]
    cols = [{"frame": int(r[0]), "i": int(r[1]), "j": int(r[2])} for r in z["collisions"]]
    return {"objects": [k.replace("_", " ") for k in keys],
            "styles": [style_of(k) for k in keys], "collisions": cols}


@app.post("/api/simulate")
def api_sim(req: SimReq):
    model = STATE["model"]
    z = load(req.scene)
    pos, vel, pres = z["positions"], z["velocities"], z["presence"]
    attrs = z["attrs"]
    keys = [str(x) for x in z["obj_keys"]]
    cols = z["collisions"]
    T, N, _ = pos.shape
    if len(cols) == 0:
        return JSONResponse({"error": "clip has no recorded collisions"}, status_code=400)

    first_col = min(int(r[0]) for r in cols)
    last_col = max(int(r[0]) for r in cols)
    at = torch.from_numpy(attrs.astype(np.float32))

    # LEFT (real world): the FULL video from ground truth -> the whole animation.
    gt_pos = torch.from_numpy(pos.astype(np.float32))
    gt_vel = torch.from_numpy(vel.astype(np.float32))
    gt_pres = torch.from_numpy((pres > 0).astype(np.float32))
    real = panel_report(model, gt_pos, gt_vel, gt_pres, at)
    real["start"] = 0; real["W"] = T

    # RIGHT (edited world): the model MUST start where objects are already moving,
    # i.e. just before the first collision -- otherwise it begins from a dead frame
    # (zero velocity, nothing nearby) and correctly stays frozen. Start 15 frames
    # before the first collision and run through the last collision + margin.
    start_m = max(0, first_col - 15)
    end_m = int(min(T - 1, last_col + 20))
    W_m = end_m - start_m + 1
    scene = build_scene(z, start_m)
    cf = scene
    for k in req.remove:
        if 0 <= k < N:
            cf = do_remove(cf, k)
    qs, vs = simulate(model, cf, W_m)
    cf_pres = cf[3].unsqueeze(0).expand(W_m, -1).clone()
    edited = panel_report(model, qs, vs, cf_pres, cf[2])
    edited["start"] = start_m; edited["W"] = W_m

    # per-collision verdicts (short accurate windows, per-pair measured contact)
    with torch.no_grad():
        massv, radii, _ = model.properties(strip_colour(scene[2].unsqueeze(0)))
    massv = massv[0]; radii = radii[0]
    events = []
    for r in cols:
        f, i, j = int(r[0]), int(r[1]), int(r[2])
        ws = window_start(pres, f, i, j, req.lookback)
        if ws is None:
            continue
        wsteps = (f - ws) + req.extra
        sc = build_scene(z, ws)
        qf, _ = simulate(model, sc, wsteps)
        contact = float(radii[i] + radii[j])
        hit_f, _ = pair_collides(qf, i, j, contact, req.thresh)
        if not hit_f:
            continue
        removed_here = [k for k in req.remove if 0 <= k < N and float(sc[3][k]) > 0]
        part = (i in removed_here) or (j in removed_here)
        if part:
            hit_c = False
        else:
            cf2 = sc
            for k in removed_here:
                cf2 = do_remove(cf2, k)
            qc, _ = simulate(model, cf2, wsteps)
            hit_c, _ = pair_collides(qc, i, j, contact, req.thresh)
        events.append({"frame": f, "i": i, "j": j, "survives": bool(hit_c),
                       "tag": ("" if hit_c else ("direct" if part else "caused"))})

    ccd = ccd_compare(model, scene, W_m)

    meta = []
    for i, k in enumerate(keys):
        c, sym, shp, mat = style_of(k)
        meta.append({"i": i, "colour": c, "symbol": sym, "shape": shp,
                     "metal": mat == "metal", "mass": round(float(massv[i]), 3),
                     "radius": round(float(radii[i]), 4)})

    return {"names": [k.replace("_", " ") for k in keys], "meta": meta,
            "real": real, "edited": edited, "events": events, "ccd": ccd}


HTML = r"""
<!doctype html><html><head><meta charset="utf-8"><title>CausalVis V3</title>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<style>
 body{font-family:Arial,Helvetica,sans-serif;margin:0;background:#f6f4f0;color:#1a1a1a}
 header{background:#73000a;color:#fff;padding:13px 20px;font-size:19px;font-weight:bold}
 .wrap{padding:16px 20px;max-width:1280px;margin:0 auto}
 .panel{background:#fff;border:1px solid #d8d4cc;border-radius:4px;padding:12px 14px;margin-bottom:14px}
 h3{margin:0 0 8px;color:#73000a;font-size:15px}
 select,button{font-size:14px;padding:6px 8px}
 button{background:#73000a;color:#fff;border:0;border-radius:3px;cursor:pointer}
 button:disabled{background:#999}
 label{display:inline-block;margin:3px 12px 3px 0;font-size:14px}
 .row{display:flex;gap:14px;flex-wrap:wrap}
 .col{flex:1;min-width:380px}
 .ev{font-size:13.5px;padding:4px 6px;border-bottom:1px solid #eee}
 .van{color:#a2140f;font-weight:bold}.sur{color:#137a2b}
 .tot{font-size:13px;color:#333;margin-top:6px}
 .muted{color:#666;font-size:13px}
 #framewrap{margin:6px 0 2px}
 input[type=range]{width:70%;vertical-align:middle}
</style></head><body>
<header>CausalVis V3 &mdash; re-simulation viewer</header>
<div class="wrap">
 <div class="panel">
   <h3>1. Scene</h3>
   <select id="scene"></select>
   <button id="run">Re-simulate</button>
   <span id="status" class="muted"></span>
 </div>
 <div class="row">
   <div class="panel col" style="flex:0 0 250px;min-width:230px">
     <h3>2. Delete objects</h3><div id="objs"></div>
   </div>
   <div class="panel col"><h3>3. What changes</h3><div id="events" class="muted">run a simulation</div></div>
 </div>
 <div class="row">
   <div class="panel col"><h3>Real world (data)</h3><div id="pf" style="height:400px"></div>
     <div id="frwrapR" style="display:none;margin-top:6px">frame <input type="range" id="frameR" min="0" max="0"> <span id="flabelR" class="muted"></span></div>
     <div id="ftot" class="tot"></div></div>
   <div class="panel col"><h3>Edited world (model)</h3><div id="pc" style="height:400px"></div>
     <div id="frwrapC" style="display:none;margin-top:6px">frame <input type="range" id="frameC" min="0" max="0"> <span id="flabelC" class="muted"></span></div>
     <div id="ctot" class="tot"></div></div>
 </div>
 <div class="panel">
   <h3>CCD &mdash; structured engine vs ablations (this scene, integral form)</h3>
   <div class="row">
     <div class="col"><div id="ccdE" style="height:300px"></div></div>
     <div class="col"><div id="ccdM" style="height:300px"></div></div>
   </div>
   <div class="muted">Blue (structured) stays low and flat however many objects are deleted; the ablations drift far above it.</div>
 </div>
</div>
<script>
let SCENE=null, D=null;
function lighten(hex,f){const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);
  const L=x=>Math.round(x+(255-x)*f);return `rgb(${L(r)},${L(g)},${L(b)})`;}

async function loadScenes(){
  const d=await (await fetch('/api/scenes')).json();
  const s=document.getElementById('scene');
  s.innerHTML=d.scenes.map(n=>`<option>${n}</option>`).join('');
  s.onchange=loadObjs; await loadObjs();
}
async function loadObjs(){
  SCENE=await (await fetch('/api/scene/'+document.getElementById('scene').value)).json();
  document.getElementById('objs').innerHTML=SCENE.objects.map((o,i)=>{
    const c=SCENE.styles[i][0], sq=SCENE.styles[i][1]==='square';
    return `<label><input type="checkbox" class="rm" value="${i}"> `
      +`<span style="color:${c};font-size:16px">${sq?'&#9632;':'&#9679;'}</span> [${i}] ${o}</label><br>`;
  }).join('');
  document.getElementById('events').innerHTML='<div class="muted">recorded collisions:<br>'
    +SCENE.collisions.map(c=>`frame ${c.frame}: ${SCENE.objects[c.i]} &harr; ${SCENE.objects[c.j]}`).join('<br>')+'</div>';
  document.getElementById('framewrap').style.display='none';
}

function panelTraces(P, upto){
  const out=[];
  D.meta.forEach(m=>{
    const i=m.i;
    // is this object present anywhere up to 'upto'?
    let lastSeen=-1;
    for(let t=0;t<=upto;t++){ if(P.present[t] && P.present[t][i]) lastSeen=t; }
    if(lastSeen<0) return;
    const light=lighten(m.colour,0.45), lighter=lighten(m.colour,0.7);
    // collect the visible path (present frames only) up to 'upto'
    const xs=[],ys=[];
    for(let t=0;t<=upto;t++){ if(P.present[t] && P.present[t][i]){ xs.push(P.pos[t][i][0]); ys.push(P.pos[t][i][1]); } }
    if(P.moved[i] && xs.length>1){
      const k=Math.max(1,Math.floor(xs.length*0.25));
      out.push({x:xs.slice(0,k+1),y:ys.slice(0,k+1),mode:'lines',
        line:{color:lighter,width:3},opacity:0.35,hoverinfo:'skip',showlegend:false});     // faded tail
      out.push({x:xs.slice(k),y:ys.slice(k),mode:'lines',
        line:{color:light,width:3},opacity:0.9,hoverinfo:'skip',showlegend:false});         // solid body
    }
    // head marker at the last seen frame
    const hx=P.pos[lastSeen][i][0], hy=P.pos[lastSeen][i][1];
    const info=`${D.names[i]} (${m.metal?'metal':'rubber'})<br>mass ${m.mass}`
      +`<br>KE ${P.ke[lastSeen][i]}<br>PE-share ${P.pe[lastSeen][i]}<br>speed ${P.speed[lastSeen][i]}/frame`;
    out.push({x:[hx],y:[hy],mode:'markers',
      marker:{color:m.colour,symbol:m.symbol,size:P.moved[i]?15:13,
              line:{color:'#111',width:m.metal?2.5:0}},   // metal = solid outline, rubber = none
      text:[info],hoverinfo:'text',showlegend:false});
  });
  return out;
}
const LAY={margin:{l:28,r:8,t:8,b:26},xaxis:{range:[0,1],zeroline:false,fixedrange:true},
  yaxis:{range:[1,0],zeroline:false,scaleanchor:'x',fixedrange:true},plot_bgcolor:'#fbfaf8'};

function drawPanel(div,P,totId,labId,t){
  Plotly.react(div,panelTraces(P,t),LAY,{displayModeBar:false});
  document.getElementById(totId).innerHTML=`total KE ${P.tke[t]} &nbsp; total PE ${P.tpe[t]} &nbsp; total energy ${(P.tke[t]+P.tpe[t]).toFixed(5)}`;
  document.getElementById(labId).textContent=`frame ${t+1}/${P.W} (video frame ${P.start+t})`;
}
function drawCCD(){
  const modes=[['symplectic','#1f77b4'],['euler','#ff7f0e'],['broken','#2ca02c']];
  const mk=(field)=>modes.map(([m,c])=>({x:D.ccd.removes,y:D.ccd[field][m],mode:'lines+markers',
      name:m,line:{color:c},marker:{size:7}}));
  const lay=t=>({margin:{l:60,r:8,t:26,b:38},title:{text:t,font:{size:13}},
    xaxis:{title:'objects deleted',dtick:1},yaxis:{title:'CCD (integral)',type:'log'}});
  Plotly.react('ccdE',mk('energy'),lay('Energy'),{displayModeBar:false});
  Plotly.react('ccdM',mk('momentum'),lay('Momentum'),{displayModeBar:false});
}

document.getElementById('run').onclick=async()=>{
  const btn=document.getElementById('run');btn.disabled=true;
  document.getElementById('status').textContent=' simulating...';
  const rm=[...document.querySelectorAll('.rm:checked')].map(e=>+e.value);
  const r=await fetch('/api/simulate',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({scene:document.getElementById('scene').value,remove:rm})});
  btn.disabled=false;document.getElementById('status').textContent='';
  if(!r.ok){document.getElementById('events').textContent='error: '+(await r.text());return;}
  D=await r.json();
  const R=document.getElementById('frameR'), C=document.getElementById('frameC');
  R.max=D.real.W-1;   R.value=D.real.W-1;
  C.max=D.edited.W-1; C.value=D.edited.W-1;
  document.getElementById('frwrapR').style.display='block';
  document.getElementById('frwrapC').style.display='block';
  R.oninput=()=>drawPanel('pf',D.real,'ftot','flabelR',+R.value);
  C.oninput=()=>drawPanel('pc',D.edited,'ctot','flabelC',+C.value);
  drawPanel('pf',D.real,'ftot','flabelR',D.real.W-1);
  drawPanel('pc',D.edited,'ctot','flabelC',D.edited.W-1);
  drawCCD();
  let html='';
  if(!D.events.length) html='<div class="muted">no collisions reproduced in this window</div>';
  D.events.forEach(e=>{const nm=i=>D.names[i];
    html+=`<div class="ev">frame ${e.frame}: ${nm(e.i)} &harr; ${nm(e.j)} `
      +(e.survives?`<span class="sur">SURVIVES</span>`:`<span class="van">VANISHES${e.tag?' ['+e.tag+']':''}</span>`)+`</div>`;});
  const van=D.events.filter(e=>!e.survives).length;
  html+=`<div style="margin-top:6px"><b>${van} of ${D.events.length}</b> reproduced collisions would not happen.</div>`;
  document.getElementById('events').innerHTML=html;
};
loadScenes();
</script></body></html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/trajectories_v2")
    ap.add_argument("--model", default="v3_2_dynamics.pt")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    m = HamiltonianDynamics()
    m.load_state_dict(torch.load(a.model, map_location="cpu"))
    m.eval()
    STATE["model"] = m
    STATE["data"] = a.data
    print(f"open http://127.0.0.1:{a.port}")
    uvicorn.run(app, host="127.0.0.1", port=a.port)