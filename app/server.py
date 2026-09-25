"""
app/server.py  —  CausalVis Lab: pick a video, ask any "what if", turn the 3D view yourself
=========================================================================================

A local web app around v7.1: v7's learned world model with the entry-state
correction (88.4% of options on the held-out test questions, which are all
removals). One what-if world per request, detector "cal", 30 frames past the end
of the video, world model v6_voxel.pt.

    GROQ_API_KEY=...  python app/server.py        then open http://127.0.0.1:8000

Requests it can simulate (app/intervene.py): any combination of general events,
each at any frame, on any object including ones the request adds --
  remove / disappear, add (standing or rolling in), move (speed, stop, turn, head
  somewhere, push, come in from a side), place (start or jump somewhere),
  shift_time (come in earlier or later), set (material, shape, size, mass,
  bounciness, friction), pin (glued to the table) -- plus world settings
  (friction everywhere, a tilted table, how far past the video to predict).
Only removals have dataset answers to check against; other changes are compared
with the real video, and the page says so.

What happens when you ask:
  1. the language model reads your sentence and writes a PLAN (fixed JSON shape)
  2. the plan becomes start frames, positions and velocities (camera + tracks)
  3. the world model builds ONE what-if world (v7_1/cf_world.py)
  4. collisions are read off that world with the benchmark's own detector
  5. the page draws both worlds in 3D from the tracks; you turn the camera
  6. the language model explains what changed, using only the stored world

STORAGE. Everything a session produces lives in app/session_files/. It is
deleted when you load another video, refresh or close the page (the page calls
/api/reset), and every time the server starts or stops.
"""

import json
import math
import os
import random
import shutil
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import render_worker  # noqa: E402  (light: numpy + renderer only; puts src2 on the path)

STATIC = os.path.join(HERE, "static")
FILES = os.path.join(HERE, "session_files")
DATA = os.path.join(ROOT, "data", "trajectories_3d_yaw")
MODEL_PATH = os.path.join(ROOT, "v6_voxel.pt")
QUESTION_FILES = [os.path.join(ROOT, "zechennlp", "counterfactual", "train-00000-of-00001.json"),
                  os.path.join(ROOT, "zechennlp", "counterfactual", "validation-00000-of-00001.json")]

DETECTOR, EXTEND, WORLD = "cal + path change", 30, 6.0     # the v7.3 test configuration
LOOKBACK = "3"                                            # v7.3: hand off 3 frames before a lost contact


def collision_frames(w, i, j, ext):
    """v7.3 detector: the calibrated distance rule OR a sharp path change with the
    surfaces close (kick2). Hits closer than 6 frames are one collision."""
    a, b = min(i, j), max(i, j)
    lim = w["T"] + ext
    fr = sorted(set(int(f) for det in ("cal", "kick2") for f in w["events"][det].get((a, b), []) if f < lim))
    out = []
    for f in fr:
        if not out or f - out[-1] > 6:
            out.append(f)
    return out
STRIDE, FPS, W, H = 2, 15, 480, 320           # GIF export
AMBER, RED = (236, 160, 30), (222, 58, 46)    # collision that also happened / new collision

B = {}                                        # backend, filled by load_backend()
JOBS = {}
SESSION = {"video": None, "z": None, "rec": None, "cf": None, "tag": uuid.uuid4().hex[:8]}
WORK = threading.Lock()
POOL = None

EXPLAIN_SYSTEM = (
    "You explain a physics world model's 'what if' prediction to a student whose English is "
    "intermediate. Use short sentences and everyday words; if you must use a technical word, explain "
    "it in brackets. The change may remove objects, change how an object moves, or add new objects. "
    "Use ONLY the facts given: the dataset's recorded collisions for the real video, the change as it "
    "was simulated, the stored what-if world, the computed comparison, and the dataset's answers when "
    "given. Never invent a collision, a frame or a cause. The real video has frames 0-127; frames from "
    "128 on are the world model's prediction after the video ends, so say that when it matters. Explain "
    "WHY things changed using the 'how each object was produced' lines (which object was simulated from "
    "which frame, and because of what). summary: 2-3 sentences. story: 4-8 items in time order. "
    "changes: one item per collision that disappeared or appeared, with its reason. unchanged: one "
    "sentence. ground_truth_check: compare with the real video's recorded collisions; if dataset "
    "answers are given, say how many options match (each dataset line is one OPTION of a question, not "
    "a separate question); if none are given, say plainly that the dataset has no answers for this "
    "change, so the prediction can only be compared with the real video. caution: one honest sentence "
    "about what might be wrong; the world model was only ever scored on removals.")

CAUSE = {"lost_contact": "it was about to touch the %s in the video, which no longer moves the same way",
         "new_contact": "the simulated %s came near it",
         "video_ended": "the video ended, so the world model predicts the rest"}


# ----------------------------------------------------------------------------- backend
def load_backend():
    for d in ("src2", "v4", "v5", "v6", "v7"):
        sys.path.insert(0, os.path.join(ROOT, d))
    import importlib.util
    import torch
    torch.set_num_threads(4)
    import benchmark_eval as be
    import lm
    from world import load_spatial
    from render3d import save_gif

    def by_path(name, *rel):
        spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, *rel))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    # src2, v6, v7 and v7_1 share module names (intervene, lm_eval, cf_world) and
    # sit in front of app/ on sys.path: load the ones meant, by path
    intervene = by_path("app_intervene", "app", "intervene.py")
    os.environ["CF_LOOKBACK"] = LOOKBACK            # read by v7_3/cf_world.py at import
    cfw = by_path("v7_3_cf_world", "v7_3", "cf_world.py")
    v7_lm_eval = by_path("v7_lm_eval", "v7", "lm_eval.py")
    with open(os.path.join(ROOT, "v7_1", "entry_profile.json"), encoding="utf-8") as fh:
        entry = json.load(fh)
    B.update(be=be, lm=lm, intervene=intervene, build_world=cfw.build_world, pair_answer=cfw.pair_answer,
             event_frames=v7_lm_eval.event_frames, save_gif=save_gif, entry=entry)
    B["model"] = load_spatial(MODEL_PATH, yaw_known=False)
    prov, cfg, why = lm.pick_provider("groq" if os.environ.get("GROQ_API_KEY") else None)
    B["llm"] = (prov, cfg) if cfg else None
    B["llm_why"] = why
    B["llm_model"] = None
    if cfg:
        B["llm_model"] = lm._resolve_model(cfg) if prov != "anthropic" else cfg["model"]
    ids = sorted(v for v in (be._vid_num(f) for f in os.listdir(DATA) if f.endswith(".npz")) if v is not None)
    split = {}
    for w_ in ("train", "val", "test"):
        for v in be.load_split_set(os.path.join(ROOT, "split3d_yaw.json"), w_):
            split[v] = w_
    questions = {}
    for qf in QUESTION_FILES:
        if os.path.exists(qf):
            for row in be.load_questions(qf):
                v = be._vid_num(str(row.get("video_filename", row.get("video", ""))))
                if v is not None:
                    questions.setdefault(v, []).append(row)
    B.update(ids=ids, idset=set(ids), split=split, questions=questions)


def get_pool():
    global POOL
    if POOL is None:
        POOL = ProcessPoolExecutor(max_workers=3)
    return POOL


def wipe_files():
    shutil.rmtree(FILES, ignore_errors=True)
    os.makedirs(FILES, exist_ok=True)
    SESSION["tag"] = uuid.uuid4().hex[:8]


def file_url(name):
    return "/files/%s?t=%d" % (name, int(time.time() * 1000))


def with_retry(fn, *args):
    """Groq's free tier answers 429 when calls come too fast: wait and try again."""
    for k in range(4):
        try:
            return fn(*args)
        except Exception as exc:                                    # noqa: BLE001
            msg = str(exc)
            if k == 3 or not ("429" in msg or "rate" in msg.lower()):
                raise
            time.sleep(8 * (k + 1))


# ----------------------------------------------------------------------------- jobs
def set_stage(jid, stage=None, progress=None):
    if stage is not None:
        JOBS[jid]["stage"] = stage
    if progress is not None:
        JOBS[jid]["progress"] = round(float(progress), 3)


def start_job(fn, *args):
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"status": "running", "stage": "waiting", "progress": 0.0, "started": time.time()}

    def run():
        with WORK:
            try:
                JOBS[jid]["result"] = fn(jid, *args)
                JOBS[jid]["status"] = "done"
            except Exception as exc:                            # noqa: BLE001
                traceback.print_exc()
                JOBS[jid]["status"] = "error"
                JOBS[jid]["error"] = str(exc)
        JOBS[jid]["seconds"] = round(time.time() - JOBS[jid]["started"], 1)

    threading.Thread(target=run, daemon=True).start()
    return jid


# ----------------------------------------------------------------------------- scenes for the page
def names_of(z):
    keys = [str(k) for k in z["obj_keys"]][:8]
    return keys, [k.replace("_", " ") for k in keys]


def truth_list(z, N):
    return sorted((int(f), int(i), int(j)) for f, i, j in z["collisions"] if int(i) < N and int(j) < N)


def marks_of(events, P, attrs, names, kind_of):
    """One mark per collision, at the contact point, in render units."""
    out = []
    for f, i, j in events:
        if f >= P.shape[0] or not (np.isfinite(P[f, i]).all() and np.isfinite(P[f, j]).all()):
            continue
        ri, rj = float(attrs[i, 15]), float(attrs[j, 15])
        c = P[f, i] + (P[f, j] - P[f, i]) * (ri / max(ri + rj, 1e-9))
        out.append({"f": int(f), "x": round(float(c[0]) * WORLD, 4), "y": round(float(c[1]) * WORLD, 4),
                    "kind": kind_of(i, j), "a": names[i], "b": names[j]})
    return out


def scene_payload(sc):
    """Tracks for the page's 3D player: per object, per frame x, y (render units), yaw, mode."""
    P, M, Yw, attrs = sc["P"], sc["M"], sc["Yw"], sc["attrs"]
    TT = P.shape[0]
    objs = []
    for k, key in enumerate(sc["keys"]):
        col, mat, shp = key.split("_")[:3]
        r = float(attrs[k, 15]) * WORLD
        fin = np.isfinite(P[:, k]).all(axis=-1)

        def arr(a):
            return [round(float(a[t]), 4) if fin[t] else None for t in range(TT)]
        objs.append({"index": k, "key": key, "name": sc["names"][k], "color": col, "material": mat, "shape": shp,
                     "size": round(r / (4.0 / math.pi) if shp == "cube" else r, 4), "status": sc["status"][k],
                     "x": arr(P[:, k, 0] * WORLD), "y": arr(P[:, k, 1] * WORLD), "yaw": arr(Yw[:, k]),
                     "mode": [int(m) for m in M[:, k]]})
    return {"frames": int(TT), "objects": objs, "marks": sc["marks"]}


def recorded_scene(z):
    keys, names = names_of(z)
    N = len(keys)
    pres = z["presence"][:, :N] > 0
    P = np.where(pres[..., None], z["positions"][:, :N].astype(np.float32), np.nan)
    Yw = (z["yaw"] if "yaw" in z else np.zeros(pres.shape, np.float32))[:, :N]
    attrs = z["attrs"][:N].astype(np.float32)
    marks = marks_of(truth_list(z, N), P, attrs, names, lambda i, j: "truth")
    return {"keys": keys, "names": names, "attrs": attrs, "P": P, "M": pres.astype(np.int8), "Yw": Yw,
            "marks": marks, "status": ["video"] * N}


def dataset_questions(v, keys, N):
    be = B["be"]
    out = []
    for row in B["questions"].get(v, []):
        qd = be.parse_descriptors(row.get("program", []))
        if not qd:
            continue
        removed = sorted(k for k in be.resolve_all(qd[0], keys) if k < N)
        negate = be.program_has(row.get("program", []), "negate")
        choices = be.normalise_choices(row.get("choices"))
        if choices and all(c["answer"] is None for c in choices):
            be.answers_from_conversations(row, choices)
        ch = []
        for c in choices:
            cd = be.parse_descriptors(c["program"])
            if c["answer"] not in ("correct", "wrong") or len(cd) < 2:
                continue
            h1 = [k for k in be.resolve_all(cd[0], keys) if k < N]
            h2 = [k for k in be.resolve_all(cd[1], keys) if k < N]
            if len(h1) != 1 or len(h2) != 1 or h1[0] == h2[0]:
                continue
            ch.append({"text": c["choice"], "truth": c["answer"], "i": h1[0], "j": h2[0]})
        if removed:
            out.append({"question": row.get("question", ""), "removed": removed, "negate": bool(negate), "choices": ch})
    return out


def video_info(v, z):
    keys, names = names_of(z)
    sc = B["intervene"].Scene(z)
    N = len(keys)
    objects = []
    for k in range(N):
        col, mat, shp = keys[k].split("_")[:3]
        on = sc.on(k)
        side = None
        if on.size and on[0] > 0:
            side = B["intervene"].entry_side(sc.pos[on[0], k], sc.height(k))
        objects.append({"index": k, "name": names[k], "color": col, "material": mat, "shape": shp,
                        "enters": int(on[0]) if on.size else None, "leaves": int(on[-1]) if on.size else None,
                        "moves": sc.first_moving(k) is not None, "side": side})
    truth = [{"frame": f, "a": names[i], "b": names[j]} for f, i, j in truth_list(z, N)]
    qs = dataset_questions(v, keys, N)
    return {"video": v, "split": B["split"].get(v, "none"), "frames": int(z["positions"].shape[0]),
            "objects": objects, "truth": truth,
            "questions": [{"question": q["question"], "removed": q["removed"][0],
                           "removed_name": names[q["removed"][0]], "options": len(q["choices"])} for q in qs]}


# ----------------------------------------------------------------------------- the what-if job
def job_intervene(jid, v, text, plan, marked):
    z = SESSION["z"]
    if SESSION["video"] != v or z is None:
        raise RuntimeError("Load the video first.")
    iv = B["intervene"]
    sc = iv.Scene(z)
    N, T = sc.N, sc.T

    set_stage(jid, "reading your request", 0.02)
    if plan is None:
        if not text.strip():
            raise RuntimeError("Type a what-if question first.")
        if not B["llm"]:
            raise RuntimeError("No GROQ_API_KEY is set, so typed requests can't be read. "
                               "Use the buttons next to each object instead.")
        prov, cfg = B["llm"]
        plan = with_retry(iv.ask_plan, B["lm"], cfg, prov, text, sc, marked)
    set_stage(jid, "reading your request: placing the change in the scene", 0.1)
    R = iv.resolve(plan, sc, B["model"], marked, B["entry"])
    removed, names, ext = set(R["removed"]), R["names"], R["extend"]

    set_stage(jid, "building the what-if world", 0.2)
    with iv.friction(B["model"], R["drag_scale"]):
        w = B["build_world"](B["model"], R["z"], removed, extend=ext, edits=R["edits"], entry_fix=B["entry"])
    NA = w["N"]

    set_stage(jid, "finding collisions", 0.72)
    cf = sorted((int(f), i, j) for i in range(NA) for j in range(i + 1, NA)
                for f in collision_frames(w, i, j, ext))
    truth = truth_list(z, N)
    tp, cp = {}, {}
    for f, i, j in truth:
        tp.setdefault((min(i, j), max(i, j)), []).append(f)
    for f, i, j in cf:
        cp.setdefault((i, j), []).append(f)

    def item(p, frames):
        return {"a": names[p[0]], "b": names[p[1]], "frames": frames, "after_end": any(f >= T for f in frames)}

    kept = [dict(item(p, cp[p]), truth_frames=tp[p]) for p in sorted(cp) if p in tp]
    new = [item(p, cp[p]) for p in sorted(cp) if p not in tp]
    gone = [dict(item(p, tp[p]), with_removed=bool(set(p) & removed)) for p in sorted(tp) if p not in cp]

    status = ["removed" if k in removed else "added" if k >= N else "changed" if k in R["changed"] else "video"
              for k in range(NA)]
    affected = []
    for k in range(NA):
        tk = w["taint"].get(k)
        eds = R["edits"]["change"].get(k, [])
        replaced = any(ed.get("replace_track") for ed in eds)
        on = sc.on(k) if k < N else np.zeros(0, int)
        enters, leaves = (int(on[0]), int(on[-1])) if on.size else (None, None)
        if status[k] == "added" or replaced:
            enters, leaves = (tk[0] if tk else None), None
        affected.append({"name": names[k], "status": status[k], "replaced": replaced,
                         "enters": enters, "leaves": leaves, "change_frame": min(int(ed["frame"]) for ed in eds) if eds else None,
                         "sim_from": tk[0] if tk else None, "cause": tk[1] if tk else None,
                         "because_of": names[tk[2]] if tk and tk[2] is not None else None})

    checks = []
    if R["pure_removal"]:
        for q in dataset_questions(v, sc.keys, N):
            if set(q["removed"]) != removed:
                continue
            for c in q["choices"]:
                happens = (c["i"] not in removed and c["j"] not in removed
                           and bool(collision_frames(w, c["i"], c["j"], ext)))
                ours = "correct" if happens != q["negate"] else "wrong"
                checks.append({"question": q["question"], "option": c["text"], "truth": c["truth"],
                               "ours": ours, "match": ours == c["truth"]})

    attrs = R["render_attrs"]
    cf_marks = marks_of(cf, w["positions"], attrs, names, lambda i, j: "kept" if (i, j) in tp else "new")
    SESSION["cf"] = {"keys": R["keys"], "names": names, "attrs": attrs, "P": w["positions"], "M": w["modes"],
                     "Yw": w["yaw"], "marks": cf_marks, "status": status}
    text_store = world_text(w, names, R, T, cf)
    base = "%s_whatif_%d_%d" % (SESSION["tag"], v, int(time.time()))
    with open(os.path.join(FILES, base + ".json"), "w", encoding="utf-8") as fh:
        json.dump({"video": v, "request": text, "plan": plan, "steps": R["steps"], "notes": R["notes"],
                   "detector": DETECTOR, "extend": ext, "kept": kept, "new": new, "gone": gone,
                   "affected": affected, "checks": checks, "store_text": text_store}, fh, indent=1, default=str)
    np.savez_compressed(os.path.join(FILES, base + ".npz"), positions=w["positions"], modes=w["modes"], yaw=w["yaw"])

    explanation, llm_error = None, None
    if B["llm"]:
        set_stage(jid, "writing the explanation", 0.8)
        prov, cfg = B["llm"]
        try:
            explanation = with_retry(explain, prov, cfg, v, names, truth, text_store, kept, new, gone, checks)
        except Exception as exc:                                    # noqa: BLE001
            llm_error = str(exc)
    set_stage(jid, "done", 1.0)
    return {"understood": R["understood"], "steps": R["steps"], "notes": R["notes"],
            "pure_removal": R["pure_removal"], "scene": scene_payload(SESSION["cf"]),
            "kept": kept, "new": new, "gone": gone, "affected": affected, "checks": checks,
            "explanation": explanation, "llm_error": llm_error, "video_frames": T, "total_frames": T + ext}


def world_text(w, names, R, T, cf):
    removed = set(R["removed"])
    lines = ["THE REQUESTED CHANGE, as simulated:"] + ["  - " + s for s in R["steps"]]
    lines += ["  note: " + s for s in R["notes"]]
    lines += ["The recorded video has frames 0-%d; the world model continued %d frames past its end." % (T - 1, R["extend"]),
              "", "HOW EACH OBJECT WAS PRODUCED:"]
    for k in range(w["N"]):
        tk = w["taint"].get(k)
        eds = R["edits"]["change"].get(k, [])
        at = (" The change to it is applied at frame%s %s." % ("s" if len(eds) > 1 else "",
              ", ".join(str(int(e_["frame"])) for e_ in eds))) if eds else ""
        if k in removed:
            lines.append("  %s: removed." % names[k])
        elif k >= w["n_original"]:
            lines.append("  %s: added by the change; simulated by the world model from frame %s."
                         % (names[k], tk[0] if tk else "?"))
        elif any(e_.get("replace_track") for e_ in eds):
            lines.append("  %s: its real path was replaced by the change; simulated from frame %s."
                         % (names[k], tk[0] if tk else "?"))
        elif tk is None:
            lines.append("  %s: follows its recorded motion (the change never reaches it)." % names[k])
        elif tk[1] == "vanished":
            lines.append("  %s: follows its recorded motion until frame %d, where it disappears (the change)."
                         % (names[k], tk[0]))
        elif tk[1] == "edited":
            lines.append("  %s: recorded motion until frame %d, where the change is applied; simulated after.%s"
                         % (names[k], tk[0], at if len(eds) > 1 else ""))
        else:
            cause = CAUSE.get(tk[1], tk[1])
            if "%s" in cause:
                cause = cause % (names[tk[2]] if tk[2] is not None else "an object")
            lines.append("  %s: recorded motion until frame %d, simulated after (%s).%s" % (names[k], tk[0], cause, at))
    lines += ["", "COLLISIONS IN THIS WORLD (frame: object and object):"]
    lines += ["  frame %d%s: %s and %s collide" % (f, " (after the video ends)" if f >= T else "", names[i], names[j])
              for f, i, j in cf] or ["  none"]
    lines.append("Any pair not listed above does NOT collide in this world.")
    return "\n".join(lines)


def explain(prov, cfg, v, names, truth, text_store, kept, new, gone, checks):
    lines = ["VIDEO %d. Objects: %s." % (v, "; ".join(names)),
             "", "GROUND TRUTH: collisions recorded in the REAL video (from the dataset):"]
    lines += ["  frame %d: %s and %s" % (f, names[i], names[j]) for f, i, j in truth] or ["  none"]
    lines += ["", "THE WORLD MODEL'S STORED WHAT-IF WORLD:", text_store,
              "", "COMPARISON (computed from the two lists, not guessed):",
              "  still happen: " + ("; ".join("%s + %s (frames %s)" % (x["a"], x["b"], x["frames"]) for x in kept) or "none"),
              "  no longer happen: " + ("; ".join("%s + %s%s" % (x["a"], x["b"], " (a removed object was one of them)"
                                                                 if x["with_removed"] else "") for x in gone) or "none"),
              "  new: " + ("; ".join("%s + %s (frames %s%s)" % (x["a"], x["b"], x["frames"],
                                                               ", after the video ends" if x["after_end"] else "")
                                     for x in new) or "none")]
    if checks:
        lines += ["", "THE DATASET'S OWN ANSWERS FOR THIS EXACT REMOVAL:"]
        lines += ["  question '%s', option '%s': dataset says %s, world model says %s"
                  % (c["question"], c["option"], c["truth"], c["ours"]) for c in checks]
    else:
        lines += ["", "THE DATASET HAS NO ANSWERS FOR THIS CHANGE."]
    user = "\n".join(lines) + ('\n\nReturn JSON: {"summary": "...", "story": ["..."], "changes": ["..."], '
                               '"unchanged": "...", "ground_truth_check": "...", "caution": "..."}')
    return B["lm"].ask_json(cfg, prov, EXPLAIN_SYSTEM, user)


# ----------------------------------------------------------------------------- GIF export from the user's view
def scene_objects(keys, attrs, P, present, yaw):
    out = []
    for k in range(len(keys)):
        if not present[k] or not np.isfinite(P[k]).all():
            continue
        col, mat, shp = keys[k].split("_")[:3]
        r = float(attrs[k, 15]) * WORLD
        out.append({"shape": shp, "size": r / (4.0 / math.pi) if shp == "cube" else r,
                    "x": float(P[k, 0]) * WORLD, "y": float(P[k, 1]) * WORLD,
                    "yaw": float(yaw[k]), "color": col, "material": mat})
    return out


def job_gif(jid, which, view):
    sc = SESSION.get(which)
    if sc is None:
        raise RuntimeError("There is nothing to save yet.")
    set_stage(jid, "drawing the GIF", 0.02)
    P = sc["P"]
    frames = []
    for t in range(0, P.shape[0], STRIDE):
        present = np.isfinite(P[t]).all(axis=-1)
        objs = scene_objects(sc["keys"], sc["attrs"], P[t], present, sc["Yw"][t])
        marks = [(m["x"], m["y"], 0.35, RED if m["kind"] == "new" else AMBER, max(4, 10 - (t - m["f"])))
                 for m in sc["marks"] if 0 <= t - m["f"] <= 8]
        frames.append((t, objs, marks))
    pool = get_pool()
    futs = [pool.submit(render_worker.render_one, (t, view, objs, marks, W, H)) for t, objs, marks in frames]
    imgs = {}
    for n, fut in enumerate(as_completed(futs)):
        t, img = fut.result()
        imgs[t] = img
        set_stage(jid, None, 0.05 + 0.9 * (n + 1) / len(futs))
    name = "%s_%s_%d.gif" % (SESSION["tag"], which, int(time.time()))
    B["save_gif"]([imgs[t] for t, _, _ in frames], os.path.join(FILES, name), fps=FPS)
    return {"gif": file_url(name), "download": "video%s_%s.gif" % (SESSION["video"], "whatif" if which == "cf" else "recorded")}


# ----------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def json_out(self, obj, code=200):
        self._send(json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8", code)

    def json_in(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        except ValueError:
            return {}

    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        if p in ("/", "/index.html"):
            with open(os.path.join(STATIC, "index.html"), "rb") as fh:
                return self._send(fh.read(), "text/html; charset=utf-8")
        if p.startswith("/files/"):
            name = os.path.basename(p[len("/files/"):])
            full = os.path.join(FILES, name)
            if name and os.path.isfile(full):
                with open(full, "rb") as fh:
                    return self._send(fh.read(), "image/gif" if name.endswith(".gif") else "application/octet-stream")
            return self.json_out({"error": "That file was discarded. Make it again."}, 404)
        if p == "/api/status":
            return self.json_out({
                "llm": bool(B.get("llm")), "llm_model": B.get("llm_model"),
                "llm_why": None if B.get("llm") else str(B.get("llm_why")),
                "detector": DETECTOR, "extend": EXTEND, "videos": len(B["ids"]),
                "test_videos": sum(1 for v in B["ids"] if B["split"].get(v) == "test" and v in B["questions"])})
        if p == "/api/random":
            pool = parse_qs(u.query).get("pool", ["test"])[0]
            ids = B["ids"]
            if pool == "test":
                ids = [v for v in ids if B["split"].get(v) == "test" and v in B["questions"]]
            elif pool == "questions":
                ids = [v for v in ids if v in B["questions"]]
            return self.json_out({"video": random.choice(ids)})
        if p.startswith("/api/job/"):
            j = JOBS.get(p.rsplit("/", 1)[-1])
            if not j:
                return self.json_out({"status": "error", "error": "Unknown job."}, 404)
            out = {k: j[k] for k in ("status", "stage", "progress", "error", "result", "seconds") if k in j}
            out["elapsed"] = round(time.time() - j["started"], 1)
            return self.json_out(out)
        return self.json_out({"error": "Not found."}, 404)

    def do_POST(self):
        p = urlparse(self.path).path
        data = self.json_in()
        if p == "/api/reset":
            if WORK.acquire(blocking=False):
                try:
                    wipe_files()
                    SESSION.update(video=None, z=None, rec=None, cf=None)
                finally:
                    WORK.release()
            return self.json_out({"ok": True})
        if p == "/api/load":
            try:
                v = int(data.get("video"))
            except (TypeError, ValueError):
                return self.json_out({"error": "Type a video number."}, 400)
            if v not in B["idset"]:
                return self.json_out({"error": "Video %d has no 3D tracks. Pick 0 to 19999." % v}, 404)
            with WORK:
                wipe_files()                  # a new video discards everything from the last one
                z = np.load(os.path.join(DATA, "sim_%05d.npz" % v), allow_pickle=True)
                SESSION.update(video=v, z=z, rec=recorded_scene(z), cf=None)
                info = video_info(v, z)
                info["scene"] = scene_payload(SESSION["rec"])
            return self.json_out(info)
        if p == "/api/intervene":
            try:
                v = int(data.get("video"))
            except (TypeError, ValueError):
                return self.json_out({"error": "Load a video first."}, 400)
            plan = data.get("plan") if isinstance(data.get("plan"), dict) else None
            marked = None
            m = data.get("marked")
            if isinstance(m, (list, tuple)) and len(m) == 2:
                try:
                    marked = [float(m[0]) / WORLD, float(m[1]) / WORLD]
                except (TypeError, ValueError):
                    marked = None
            return self.json_out({"job": start_job(job_intervene, v, str(data.get("text") or ""), plan, marked)})
        if p == "/api/gif":
            which = data.get("which")
            if which not in ("rec", "cf"):
                return self.json_out({"error": "Say which view to save."}, 400)
            vw = data.get("view") or {}
            try:
                view = {"az": float(vw.get("az", -69.17)), "el": min(max(float(vw.get("el", 34.9)), 2.0), 89.0),
                        "dist": min(max(float(vw.get("dist", 8.29)), 3.0), 30.0)}
            except (TypeError, ValueError):
                return self.json_out({"error": "The view angles are not numbers."}, 400)
            return self.json_out({"job": start_job(job_gif, which, view)})
        return self.json_out({"error": "Not found."}, 404)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    wipe_files()
    print("loading the world model and the video catalogue ...", flush=True)
    load_backend()
    get_pool()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print("CausalVis Lab (v7.1) ready on http://127.0.0.1:%d   language model: %s"
          % (a.port, B["llm_model"] or "off (%s)" % B["llm_why"]), flush=True)
    try:
        srv.serve_forever()
    finally:
        wipe_files()


if __name__ == "__main__":
    main()
