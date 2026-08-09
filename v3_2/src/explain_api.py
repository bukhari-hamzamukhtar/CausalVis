"""
explain_api.py  (src2)  —  live grounded explanations for CausalVis V3
======================================================================

The benchmark's light-cone engine, served interactively: pick a scene, tick
the objects to delete, and the server runs ONE edited-world pass (observed
trajectories outside the intervention's causal cone, physics simulation
inside it) and returns the same grounded narration that explain.py produces
for the benchmark -- intervention, who is re-simulated and why, the full
edited-world event log with sources, and which recorded collisions vanish.

Every line is rendered from the engine's own outputs.  No text is generated
that the pipeline did not compute.

RUN STANDALONE (its own page, safe next to the Plotly viewer)
    pip install fastapi uvicorn            (once)
    python src2/explain_api.py --data data/trajectories_v2 \\
           --model v3_2_dynamics.pt --port 8001
    then open  http://127.0.0.1:8001

MOUNT INTO AN EXISTING VIEWER (two lines in viewer.py, after STATE is set)
    import explain_api
    explain_api.attach(app, data_dir=a.data, model_path=a.model)
    # ->  POST /api/explain  {"scene": "sim_00042.npz", "remove": [3]}
    #     returns {"lines": [...], "events": [...], "taints": [...]}
"""

import argparse
import glob
import os
import re

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from benchmark_eval import lightcone_rollout, load_model
from explain import narrate

STATE = {}


class ExplainReq(BaseModel):
    scene: str
    remove: list = []
    thresh: float = 0.02
    release: float = 0.02
    taint_margin: float = 0.06
    lookback: int = 12
    contact: float = None          # scalar fallback for npz without measured radii


def _load_npz(name):
    return np.load(os.path.join(STATE["data"], name), allow_pickle=True)


def _vid_num(name):
    m = re.search(r"(\d+)", os.path.basename(str(name)))
    return int(m.group(1)) if m else 0


def build_record(z, name, removed, req):
    """Run one light-cone pass and package it as an explain.py trace record."""
    events, pair_set, taint_frame, taint_cause = lightcone_rollout(
        STATE["model"], z, set(removed), req.thresh, req.release,
        req.taint_margin, lookback=req.lookback,
        contact_override=req.contact)
    N = min(8, z["positions"].shape[1])
    return {
        "video": _vid_num(name),
        "question": "",                       # live mode: no benchmark question
        "negate": False,
        "obj_keys": [str(k) for k in z["obj_keys"]][:N],
        "removed": sorted(int(k) for k in removed),
        "taints": [{"idx": int(k), "frame": int(taint_frame[k]),
                    "cause": taint_cause.get(k, {})}
                   for k in sorted(taint_frame)],
        "gt_collisions": [{"frame": int(r[0]), "i": int(r[1]), "j": int(r[2])}
                          for r in z["collisions"]
                          if int(r[1]) < N and int(r[2]) < N],
        "cf_events": events,
        "choices": [],
    }


def register_routes(app):
    @app.post("/api/explain")
    def api_explain(req: ExplainReq):
        z = _load_npz(req.scene)
        N = min(8, z["positions"].shape[1])
        removed = [int(k) for k in req.remove if 0 <= int(k) < N]
        if not removed:
            return JSONResponse({"error": "tick at least one object to remove"},
                                status_code=400)
        rec = build_record(z, req.scene, removed, req)
        return {"lines": narrate(rec), "events": rec["cf_events"],
                "taints": rec["taints"], "removed": rec["removed"]}

    @app.get("/api/explain_scenes")
    def explain_scenes():
        files = sorted(os.path.basename(f) for f in
                       glob.glob(os.path.join(STATE["data"], "*.npz")))
        return {"scenes": files[:400]}

    @app.get("/api/explain_scene/{name}")
    def explain_scene(name: str):
        z = _load_npz(name)
        N = min(8, z["positions"].shape[1])
        return {"objects": [str(x).replace("_", " ")
                            for x in z["obj_keys"]][:N]}


def attach(app, data_dir, model_path, legacy_dynamics=None):
    """Mount /api/explain onto an existing FastAPI viewer app."""
    STATE["data"] = data_dir
    STATE["model"] = load_model(model_path, legacy_dynamics)
    register_routes(app)


# ---------------------------------------------------------------------------
# standalone page
# ---------------------------------------------------------------------------

HTML = """
<!doctype html><html><head><meta charset="utf-8"><title>CausalVis V3 - explainer</title>
<style>
 body{font-family:Arial,Helvetica,sans-serif;margin:0;background:#f6f4f0;color:#1a1a1a}
 header{background:#73000a;color:#fff;padding:14px 22px;font-size:20px;font-weight:bold}
 .wrap{padding:18px 22px;max-width:1180px;margin:0 auto}
 .row{display:flex;gap:20px;flex-wrap:wrap}
 .panel{background:#fff;border:1px solid #d8d4cc;border-radius:4px;padding:14px;flex:1;min-width:300px}
 h3{margin:0 0 10px;color:#73000a;font-size:15px}
 select,button{font-size:14px;padding:6px 8px}
 button{background:#73000a;color:#fff;border:0;border-radius:3px;cursor:pointer}
 button:disabled{background:#999}
 label{display:block;margin:4px 0;font-size:14px}
 pre{white-space:pre-wrap;font-size:13px;line-height:1.5;background:#fbfaf8;
     border:1px solid #d8d4cc;padding:12px;border-radius:3px}
 .muted{color:#666;font-size:13px}
</style></head><body>
<header>CausalVis V3 &mdash; grounded explainer</header>
<div class="wrap">
  <div class="panel">
    <h3>1. Pick a scene</h3>
    <select id="scene"></select>
    <button id="run">Explain the intervention</button>
    <span id="status" class="muted"></span>
  </div>
  <div class="row" style="margin-top:16px">
    <div class="panel">
      <h3>2. Delete objects</h3>
      <div id="objs" class="muted">pick a scene first</div>
    </div>
    <div class="panel" style="flex:2.5">
      <h3>3. The grounded explanation</h3>
      <pre id="out">tick one or more objects and press the button</pre>
    </div>
  </div>
</div>
<script>
async function loadScenes(){
  const d=await (await fetch('/api/explain_scenes')).json();
  const s=document.getElementById('scene');
  s.innerHTML=d.scenes.map(n=>`<option>${n}</option>`).join('');
  s.onchange=loadObjs; await loadObjs();
}
async function loadObjs(){
  const name=document.getElementById('scene').value;
  const d=await (await fetch('/api/explain_scene/'+name)).json();
  document.getElementById('objs').innerHTML=d.objects.map((o,i)=>
    `<label><input type="checkbox" class="rm" value="${i}"> [${i}] ${o}</label>`).join('');
  document.getElementById('out').textContent='tick one or more objects and press the button';
}
document.getElementById('run').onclick=async()=>{
  const btn=document.getElementById('run'); btn.disabled=true;
  document.getElementById('status').textContent=' simulating the edited world...';
  const rm=[...document.querySelectorAll('.rm:checked')].map(e=>+e.value);
  const r=await fetch('/api/explain',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({scene:document.getElementById('scene').value, remove:rm})});
  btn.disabled=false; document.getElementById('status').textContent='';
  const d=await r.json();
  document.getElementById('out').textContent =
      r.ok ? d.lines.join('\\n') : ('error: ' + (d.error || r.status));
};
loadScenes();
</script></body></html>
"""


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/trajectories_v2")
    ap.add_argument("--model", default="v3_2_dynamics.pt")
    ap.add_argument("--legacy-dynamics", default=None)
    ap.add_argument("--port", type=int, default=8001)
    a = ap.parse_args()

    app = FastAPI()
    attach(app, a.data, a.model, a.legacy_dynamics)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML

    print(f"open http://127.0.0.1:{a.port}")
    uvicorn.run(app, host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()