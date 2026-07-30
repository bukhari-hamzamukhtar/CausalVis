"""
viewer.py  —  CausalVis V3: the re-simulation viewer
====================================================

A browser interface over the SAME engine used in readout.py. You pick a scene,
tick the objects to delete, and the server re-simulates both worlds and draws
them side by side: the real world on the left, the edited world on the right,
with the collisions that vanish listed underneath.

Nothing new is trained or approximated here -- it calls the identical
simulate() / pair_collides() used to produce the command-line results.

INSTALL (once)
    pip install fastapi uvicorn

RUN
    python viewer.py --data data/trajectories_out --model v3_1_dynamics.pt
    then open  http://127.0.0.1:8000
"""

import argparse
import glob
import os
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


class SimReq(BaseModel):
    scene: str
    remove: list = []
    contact: float = 0.119
    lookback: int = 12
    extra: int = 15
    thresh: float = 0.02


def load_npz(name):
    path = os.path.join(STATE["data"], name)
    return np.load(path, allow_pickle=True)


@app.get("/api/scenes")
def scenes():
    files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(STATE["data"], "*.npz")))
    return {"scenes": files[:400]}


@app.get("/api/scene/{name}")
def scene_info(name: str):
    z = load_npz(name)
    names = [str(x).replace("_", " ") for x in z["obj_keys"]]
    cols = [{"frame": int(r[0]), "i": int(r[1]), "j": int(r[2])} for r in z["collisions"]]
    return {"objects": names, "collisions": cols, "frames": int(z["positions"].shape[0])}


@app.post("/api/simulate")
def api_simulate(req: SimReq):
    model = STATE["model"]
    z = load_npz(req.scene)
    pos, pres = z["positions"], z["presence"]
    names = [str(x).replace("_", " ") for x in z["obj_keys"]]
    cols = z["collisions"]
    N = pos.shape[1]

    if len(cols) == 0:
        return JSONResponse({"error": "this clip has no recorded collisions"}, status_code=400)

    # one window covering the collisions we can test, anchored like readout.py
    first = min(int(r[0]) for r in cols)
    s = max(0, first - req.lookback)
    last = max(int(r[0]) for r in cols)
    steps = int(min(pos.shape[0] - s, (last - s) + req.extra))

    scene = build_scene(z, s)
    qs_f, _ = simulate(model, scene, steps)

    cf = scene
    for k in req.remove:
        if 0 <= k < N:
            cf = do_remove(cf, k)
    qs_c, _ = simulate(model, cf, steps)

    results = []
    for r in cols:
        f, i, j = int(r[0]), int(r[1]), int(r[2])
        ws = window_start(pres, f, i, j, req.lookback)
        if ws is None:
            continue
        wsteps = (f - ws) + req.extra
        sc = build_scene(z, ws)
        qf, _ = simulate(model, sc, wsteps)
        hit_f, _ = pair_collides(qf, i, j, req.contact, req.thresh)
        if not hit_f:
            continue
        removed_here = [k for k in req.remove if 0 <= k < N and sc[3][k].item() > 0]
        cf2 = sc
        for k in removed_here:
            cf2 = do_remove(cf2, k)

        dmin_f = min((qf[t, i] - qf[t, j]).norm().item() for t in range(wsteps))

        # A deleted object is parked at the origin, so distances involving it are
        # meaningless. If either participant is deleted the collision cannot
        # happen -- decide that directly instead of measuring a ghost.
        participant_removed = (i in removed_here) or (j in removed_here)
        if participant_removed:
            hit_c, dmin_c = False, None
        else:
            qc, _ = simulate(model, cf2, wsteps)
            hit_c, _ = pair_collides(qc, i, j, req.contact, req.thresh)
            dmin_c = round(min((qc[t, i] - qc[t, j]).norm().item() for t in range(wsteps)), 4)

        tag = ("direct" if participant_removed else "caused") if not hit_c else ""
        results.append({"frame": f, "i": i, "j": j, "survives": bool(hit_c),
                        "tag": tag, "dmin_factual": round(dmin_f, 4),
                        "dmin_cf": dmin_c})

    present = [bool(v > 0) for v in scene[3].tolist()]
    return {
        "objects": names,
        "present": present,
        "start": s,
        "steps": steps,
        "contact": req.contact,
        "factual": qs_f.tolist(),
        "counterfactual": qs_c.tolist(),
        "events": results,
    }


HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>CausalVis V3</title>
<style>
 body{font-family:Arial,Helvetica,sans-serif;margin:0;background:#f6f4f0;color:#1a1a1a}
 header{background:#73000a;color:#fff;padding:14px 22px;font-size:20px;font-weight:bold}
 .wrap{padding:18px 22px;max-width:1180px;margin:0 auto}
 .row{display:flex;gap:20px;flex-wrap:wrap}
 .panel{background:#fff;border:1px solid #d8d4cc;border-radius:4px;padding:14px;flex:1;min-width:300px}
 h3{margin:0 0 10px;color:#73000a;font-size:15px}
 canvas{border:1px solid #d8d4cc;background:#fbfaf8;width:100%;height:auto}
 select,button{font-size:14px;padding:6px 8px}
 button{background:#73000a;color:#fff;border:0;border-radius:3px;cursor:pointer}
 button:disabled{background:#999}
 label{display:block;margin:4px 0;font-size:14px}
 .ev{font-size:14px;padding:6px 8px;border-bottom:1px solid #eee}
 .van{color:#a2140f;font-weight:bold}
 .sur{color:#137a2b}
 .muted{color:#666;font-size:13px}
</style></head><body>
<header>CausalVis V3 &mdash; re-simulation viewer</header>
<div class="wrap">
  <div class="panel">
    <h3>1. Pick a scene</h3>
    <select id="scene"></select>
    <button id="run">Re-simulate</button>
    <span id="status" class="muted"></span>
  </div>
  <div class="row" style="margin-top:16px">
    <div class="panel">
      <h3>2. Delete objects</h3>
      <div id="objs"></div>
    </div>
    <div class="panel" style="flex:2">
      <h3>3. What changes</h3>
      <div id="events" class="muted">run a simulation to see events</div>
    </div>
  </div>
  <div class="row" style="margin-top:16px">
    <div class="panel"><h3>Real world</h3><canvas id="cf" width="480" height="320"></canvas></div>
    <div class="panel"><h3>Edited world</h3><canvas id="cc" width="480" height="320"></canvas></div>
  </div>
</div>
<script>
const PAL=["#1f77b4","#d62728","#2ca02c","#ff7f0e","#9467bd","#8c564b","#17becf","#7f7f7f"];
let SCENE=null, LAST=null;

async function loadScenes(){
  const r=await fetch('/api/scenes'); const d=await r.json();
  const s=document.getElementById('scene');
  s.innerHTML=d.scenes.map(n=>`<option>${n}</option>`).join('');
  s.onchange=loadObjs; await loadObjs();
}
async function loadObjs(){
  const name=document.getElementById('scene').value;
  const r=await fetch('/api/scene/'+name); SCENE=await r.json();
  document.getElementById('objs').innerHTML=SCENE.objects.map((o,i)=>
    `<label><input type="checkbox" class="rm" value="${i}"> <span style="color:${PAL[i%8]}">&#9632;</span> [${i}] ${o}</label>`).join('');
  document.getElementById('events').innerHTML=
    '<div class="muted">recorded collisions: '+SCENE.collisions.map(c=>
      `frame ${c.frame}: ${SCENE.objects[c.i]} &harr; ${SCENE.objects[c.j]}`).join('<br>')+'</div>';
  clear('cf'); clear('cc');
}
function clear(id){const c=document.getElementById(id).getContext('2d');c.clearRect(0,0,480,320);}
function draw(id, traj, present){
  const cv=document.getElementById(id), g=cv.getContext('2d');
  g.clearRect(0,0,480,320);
  const T=traj.length, N=traj[0].length;
  for(let n=0;n<N;n++){
    if(present && !present[n]) continue;
    if(traj[0][n][0]===0 && traj[0][n][1]===0) continue;
    g.strokeStyle=PAL[n%8]; g.lineWidth=2; g.beginPath();
    for(let t=0;t<T;t++){const x=traj[t][n][0]*480, y=traj[t][n][1]*320;
      t?g.lineTo(x,y):g.moveTo(x,y);}
    g.stroke();
    const lx=traj[T-1][n][0]*480, ly=traj[T-1][n][1]*320;
    g.fillStyle=PAL[n%8]; g.beginPath(); g.arc(lx,ly,5,0,6.29); g.fill();
  }
}
document.getElementById('run').onclick=async()=>{
  const btn=document.getElementById('run'); btn.disabled=true;
  document.getElementById('status').textContent=' simulating...';
  const rm=[...document.querySelectorAll('.rm:checked')].map(e=>+e.value);
  const body={scene:document.getElementById('scene').value, remove:rm, contact:0.119};
  const r=await fetch('/api/simulate',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)});
  btn.disabled=false; document.getElementById('status').textContent='';
  if(!r.ok){document.getElementById('events').textContent='error: '+(await r.text());return;}
  const d=await r.json(); LAST=d;
  const cfPresent=d.present.map((p,i)=>p && !rm.includes(i));
  draw('cf', d.factual, d.present);
  draw('cc', d.counterfactual, cfPresent);
  let html='';
  if(!d.events.length) html='<div class="muted">no collisions reproduced in this window</div>';
  d.events.forEach(e=>{
    const nm=i=>d.objects[i];
    html += `<div class="ev">frame ${e.frame}: ${nm(e.i)} &harr; ${nm(e.j)} &nbsp;`
      + (e.survives ? `<span class="sur">SURVIVES</span>`
                    : `<span class="van">VANISHES${e.tag?' ['+e.tag+']':''}</span>`)
      + `<br><span class="muted">closest: real ${e.dmin_factual} &nbsp;|&nbsp; edited ${e.dmin_cf===null?'&mdash; (object deleted)':e.dmin_cf}`
      + ` &nbsp;|&nbsp; contact ${d.contact}</span></div>`;
  });
  const van=d.events.filter(e=>!e.survives).length;
  html += `<div style="margin-top:8px"><b>${van} of ${d.events.length}</b> reproduced collisions
           would not happen.</div>`;
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
    ap.add_argument("--data", default="data/trajectories_out")
    ap.add_argument("--model", default="v3_1_dynamics.pt")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    m = HamiltonianDynamics()
    m.load_state_dict(torch.load(a.model, map_location="cpu"))
    m.eval()
    STATE["model"] = m
    STATE["data"] = a.data
    print(f"open http://127.0.0.1:{a.port}")
    uvicorn.run(app, host="127.0.0.1", port=a.port)