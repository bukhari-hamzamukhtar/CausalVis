# -*- coding: utf-8 -*-
"""Build the hosted version of the app from app/static/index.html.

    python app/capture_static.py site/data 10002,10020
    python app/build_static.py

The app normally talks to a Python server that runs the world model per request, which no
static host can do. The frontend funnels every call through one api() function, so that one
function is replaced with a router reading precomputed worlds from data/. Everything else,
the renderer, the camera, the timeline, the two panels, is the original file untouched.
"""
import io
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "app", "static", "index.html")
DST = os.path.join(ROOT, "site", "index.html")

s = io.open(SRC, encoding="utf-8", newline="").read().replace("\r\n", "\n")
n = 0


def swap(old, new):
    global s, n
    assert old in s, "NOT FOUND: %r" % old[:90]
    assert s.count(old) == 1, "found %d times: %r" % (s.count(old), old[:90])
    s = s.replace(old, new, 1)
    n += 1


# ------------------------------------------------------------------ the one seam
swap('''async function api(path, body){
  const opt = body === undefined ? {cache:"no-store"} : {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)};
  const r = await fetch(path, opt); const j = await r.json().catch(() => ({error:"The server did not answer."}));
  if(!r.ok) throw new Error(j.error || ("HTTP " + r.status));
  return j;
}''', '''// Hosted build. Every world below was produced by the real engine and saved as JSON,
// because the engine runs a learned physics model per request and this page has no server.
const STATIC_JOBS = new Map();
let STATIC_INDEX = null;
async function getIndex(){
  if(!STATIC_INDEX){
    const r = await fetch("data/index.json", {cache:"force-cache"});
    if(!r.ok) throw new Error("Could not load the list of scenes.");
    STATIC_INDEX = await r.json();
  }
  return STATIC_INDEX;
}
async function getJSON(name){
  const r = await fetch("data/" + name, {cache:"force-cache"});
  if(!r.ok) throw new Error("That world did not load.");
  return await r.json();
}
const pad5 = (v) => String(v).padStart(5, "0");
function planKey(plan){
  const ev = (plan && plan.events) || [];
  if(ev.length !== 1) return null;
  const e = ev[0], m = e.motion || {};
  if(e.do === "remove") return "remove-" + e.object;
  if(e.do === "pin") return "pin-" + e.object;
  if(e.do === "move" && m.speed_factor === 2) return "fast-" + e.object;
  if(e.do === "move" && m.stop) return "stop-" + e.object;
  return null;
}
async function api(path, body){
  const idx = await getIndex();
  const ids = idx.videos.map(x => x.video);

  if(path.startsWith("/api/status"))
    return {llm:false, llm_model:null, llm_why:"hosted demo", detector:idx.detector,
            extend:idx.extend, videos:idx.corpus, test_videos:idx.test_videos};

  if(path.startsWith("/api/reset")) return {ok:true};

  if(path.startsWith("/api/random"))
    return {video: ids[Math.floor(Math.random()*ids.length)]};

  if(path.startsWith("/api/job/")){
    const r = STATIC_JOBS.get(path.split("/").pop());
    if(!r) throw new Error("Unknown job.");
    return {status:"done", stage:"done", progress:1, elapsed:r.seconds, result:r.result};
  }

  if(path.startsWith("/api/load")){
    const v = parseInt(body && body.video, 10);
    if(!ids.includes(v))
      throw new Error("This page carries " + ids.length + " scenes, computed ahead of time. " +
                      "Use the buttons above, or run the engine locally for any of CLEVRER's 20,000.");
    return await getJSON("v" + pad5(v) + ".json");
  }

  if(path.startsWith("/api/intervene")){
    const v = parseInt(body && body.video, 10);
    const entry = idx.videos.find(x => x.video === v);
    const key = planKey(body && body.plan);
    if(!key)
      throw new Error("A what-if written in words is read by a language model and simulated on the " +
                      "spot, so it needs the engine running. The four buttons beside each object " +
                      "are computed ahead of time here.");
    const w = entry && entry.worlds.find(x => x.key === key);
    if(!w)
      throw new Error("That one is not stored on this page. Try another object or another scene.");
    const id = "j" + Math.random().toString(36).slice(2);
    STATIC_JOBS.set(id, {result: await getJSON(w.file), seconds: w.seconds || 0});
    return {job:id};
  }

  if(path.startsWith("/api/gif"))
    throw new Error("A GIF is drawn frame by frame on the server, and this page has none.");

  throw new Error("Not available on the hosted page.");
}''')

# ------------------------------------------------------------------ no server session to end
swap('window.addEventListener("pagehide", () => { navigator.sendBeacon("/api/reset"); });',
     '// no server session to reset on the hosted page')

# ------------------------------------------------------------------ what the status line says
swap('$("#llmTag").textContent = st.llm ? `language model: ${st.llm_model} (Groq)` : '
     '"language model: off, use the buttons next to objects";',
     '$("#llmTag").textContent = `detector: ${st.detector} \\u00b7 '
     '${st.extend} frames past the video end`;')

# ------------------------------------------------------------------ "options" is the right unit
swap(u'<span class="tag">what-if physics \u00b7 learned world model \u00b7 '
     u'90.4% on held-out removal questions</span>',
     u'<span class="tag">counterfactual physics \u00b7 learned world model \u00b7 '
     u'90.4% of held-out removal options</span>')

# ------------------------------------------------------------------ pick a scene, do not type a number
swap(u'''    <div class="label">1 \u00b7 Pick a video</div>
    <div class="row">
      <input id="vid" type="number" min="0" max="19999" placeholder="e.g. 10002" aria-label="Video number">
      <button id="loadBtn">Load</button>
    </div>
    <div class="row" style="margin-top:8px">
      <button id="randTest" title="A video from the held-out test set, never used for training or tuning">Random test video</button>
      <button id="randAny">Any video</button>
    </div>''',
     u'''    <div class="label">1 \u00b7 Pick a scene</div>
    <div class="examples" id="scenes"><span class="note">loading\u2026</span></div>
    <div class="row" style="margin-top:8px">
      <button id="randTest" title="One of the scenes on this page, all from the held-out test set">Surprise me</button>
      <button id="randAny" hidden></button>
      <button id="loadBtn" hidden></button>
    </div>
    <input id="vid" type="hidden" value="">''')

# ------------------------------------------------------------------ typing needs the local engine
swap(u'''    <div class="label">3 \u00b7 Ask what if</div>
    <textarea id="req" placeholder="Anything that changes the scene: remove, add, move, speed up, turn, come from another side, change material, shape, size, mass, bounciness or friction, glue to the table\u2026 and combinations."></textarea>
    <div class="row" style="margin-top:8px">
      <button id="markBtn" title="Click a point on the table, then say 'here' in your question">Mark a spot</button>
      <button id="clearMark" class="mini" hidden>Clear spot</button>
    </div>
    <p class="note" id="markNote"></p>
    <div class="row" style="margin-top:8px"><button class="primary" id="runBtn" disabled>Run the world model</button></div>''',
     u'''    <div class="label">3 \u00b7 Ask what if, in words</div>
    <textarea id="req" disabled placeholder="Running locally, this box takes anything that changes the scene: remove, add, move, speed up, turn, come in from another side, change material, shape, size, mass, bounciness or friction, glue it down, tilt the table\u2026 and combinations."></textarea>
    <p class="note">A sentence is read by a language model, turned into a plan of events and
    simulated on the spot, so it needs the engine running. Here, use the four buttons beside each
    object: those worlds are already computed.</p>
    <div class="row">
      <button id="markBtn" hidden></button>
      <button id="clearMark" class="mini" hidden>Clear spot</button>
      <button class="primary" id="runBtn" hidden disabled></button>
    </div>
    <p class="note" id="markNote"></p>''')

# ------------------------------------------------------------------ only offer worlds we have
swap('''    [["remove","remove"],["fast","2× speed"],["stop","stop"],["pin","glue down"]].forEach(([k, lbl]) => {
      const q = quickPlan(k, o), btn = el("button", "mini", lbl); btn.title = q.understood; btn.onclick = () => run(q); acts.append(btn);
    });''',
     '''    [["remove","remove"],["fast","2× speed"],["stop","stop"],["pin","glue down"]].forEach(([k, lbl]) => {
      const q = quickPlan(k, o);
      if(!hasWorld(info.video, planKey(q))) return;
      const btn = el("button", "mini", lbl); btn.title = q.understood; btn.onclick = () => run(q); acts.append(btn);
    });
    if(!acts.children.length) acts.append(el("small", "note", "nothing stored for this one"));''')

swap('''function renderInfo(info){''',
     '''function hasWorld(video, key){
  const e = STATIC_INDEX && STATIC_INDEX.videos.find(x => x.video === video);
  return !!(e && key && e.worlds.some(w => w.key === key));
}
function renderInfo(info){''')

# ------------------------------------------------------------------ the object panel, reworded
swap(u'''    <div class="label">2 · Objects in this video</div>
    <div class="objs" id="objs"><span class="note">Load a video to see its objects.</span></div>
    <p class="note">Click a name to put it into your question. The small buttons run that change straight away.</p>''',
     u'''    <div class="label">2 · Objects in this scene</div>
    <div class="objs" id="objs"><span class="note">Pick a scene to see its objects.</span></div>
    <p class="note">Each small button is one what-if, already computed. Press it and the right
    panel becomes that world.</p>''')

# ------------------------------------------------------------------ a name cannot go into a disabled box
swap('''    const nm = el("button", "oname", o.name); nm.title = "Put this object into your question"; nm.onclick = () => insertText(`the ${o.name}`);''',
     '''    const nm = el("span", "oname", o.name);''')

# ------------------------------------------------------------------ the dataset questions run their removal
swap('''    const btn = el("button", null, `the ${q.removed_name}`); btn.title = q.question;
    btn.onclick = () => { $("#req").value = q.question; };''',
     '''    const btn = el("button", null, `the ${q.removed_name}`); btn.title = q.question;
    btn.onclick = () => { $("#req").value = q.question;
      run(quickPlan("remove", {index:q.removed, name:q.removed_name})); };''')

swap(u'''    <div class="label">The dataset asks about removing…</div>
    <div id="qs"></div>''',
     u'''    <div class="label">The dataset asks about removing…</div>
    <div id="qs"></div>
    <p class="note">These are the questions CausalVis is scored on. Pressing one runs that
    removal and shows the question above.</p>''')

# ------------------------------------------------------------------ examples become a list, not buttons
swap('''    <div class="label">Try one of these</div>
    <div class="examples" id="examples"><span class="note">Examples appear when a video is loaded.</span></div>''',
     '''    <div class="label">What the full version answers</div>
    <div class="examples" id="examples"></div>''')

swap('''  const opp = {left:"right", right:"left", near:"back", far:"front"};''',
     '''  const opp = {left:"right", right:"left", near:"back", far:"front"};
  box.append(el("p", "note", "These run on the local engine, not here. Each sentence becomes a plan "
    + "of composable events, and the world model simulates the scene from the first frame the change "
    + "can reach an object."));''')

swap('''  ].forEach(([q, spot]) => {
    const btn = el("button", null, q);
    btn.onclick = () => { $("#req").value = q; if(spot && !S.marked){ S.marking = true; updateMark(); } $("#req").focus(); };
    box.append(btn);
  });''',
     '''  ].forEach(([q]) => { box.append(el("span", "exq", q)); });''')

# ------------------------------------------------------------------ the missing explanation
swap('''"No language model is connected. Start the server with GROQ_API_KEY set to get explanations."''',
     '''"The explanation in words is written by a language model, which runs alongside the engine. Everything else on this page, the two panels, the collisions, the comparison with the dataset, comes from the world model itself."''')

# ------------------------------------------------------------------ no server to draw GIFs
swap('''      <button class="mini" id="gif_rec" disabled>Save real video as GIF (this view)</button>
      <button class="mini" id="gif_cf" disabled>Save what-if as GIF (this view)</button>''',
     '''      <button class="mini" id="gif_rec" hidden disabled></button>
      <button class="mini" id="gif_cf" hidden disabled></button>''')

swap('<p class="note">Everything this page creates is deleted when you load another video, '
     'refresh, or close the page.</p>',
     '<p class="note">Nothing is uploaded and nothing is stored. The page fetches static JSON '
     'and draws every frame in your browser.</p>')

# ------------------------------------------------------------------ the scene buttons go busy too
swap('document.querySelectorAll("#objs .mini, #examples button, #qs button")',
     'document.querySelectorAll("#objs .mini, #scenes button, #qs button")')

# ------------------------------------------------------------------ boot: list the scenes first
swap('''(async () => {
  try{
    await api("/api/reset", {});
    const st = await api("/api/status");''',
     '''async function renderScenes(){
  const idx = await getIndex(), box = $("#scenes");
  box.replaceChildren();
  idx.videos.forEach(v => {
    const b = el("button", null, "scene " + v.video);
    b.title = v.objects.map(o => o.name).join(", ");
    b.onclick = () => loadVideo(v.video);
    box.append(b);
  });
}

(async () => {
  try{
    await renderScenes();
    const st = await api("/api/status");''')

# ------------------------------------------------------------------ a banner, so nobody mistakes it
BANNER = u'''<div class="hosted">
  <b>Hosted demo.</b> Pick a scene, then press a button beside any object: remove it, double its
  speed, stop it, or glue it to the table. The left panel is the recorded video and the right panel
  is the counterfactual world the engine computed. Drag either view to turn it, scroll to zoom,
  press space to play. The worlds here were computed ahead of time.
  <a href="/paper">What this measures</a> \u00b7
  <a href="https://github.com/bukhari-hamzamukhtar/CausalVis">Run the engine yourself</a>
</div>
'''
i = s.index("</header>") + len("</header>")
s = s[:i] + "\n" + BANNER + s[i:]
n += 1

# ------------------------------------------------------------------ styles for the bits added above
swap("</style>", '''.oname{cursor:default}
.oname:hover{text-decoration:none}
.hosted{padding:12px 24px;border-bottom:1px solid var(--line);background:var(--tint);
  font-size:.95rem;line-height:1.5;color:var(--ink2)}
.hosted b{color:var(--ink);font-family:var(--display);font-stretch:90%}
.hosted a{color:var(--sim);font-weight:500}
#scenes{max-height:none;display:flex;flex-wrap:wrap;gap:5px}
#scenes button{display:inline-block;width:auto;margin:0;font-size:.78rem;padding:5px 9px;
  font-family:var(--mono)}
#examples{max-height:none}
.exq{display:block;border-left:2px solid var(--line);padding:2px 0 2px 9px;margin:5px 0;
  font-size:.9rem;color:var(--ink2)}
</style>''')

swap("<title>CausalVis Lab</title>",
     '<title>CausalVis Lab</title>\n'
     '<meta name="description" content="Counterfactual physics on CLEVRER: remove an object and '
     'see the world a learned model computes, beside the real recording.">')

io.open(DST, "w", encoding="utf-8", newline="\n").write(s)
print("%d edits -> %s (%.0f KB)" % (n, DST, os.path.getsize(DST) / 1024.0))
