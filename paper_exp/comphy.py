"""
paper_exp/comphy.py  —  CausalVis' counterfactual engine on ComPhy (hidden mass and charge)
==========================================================================================

ComPhy (Chen et al., ICLR 2022): CLEVRER-like scenes where objects have HIDDEN mass
(1 or 5) or charge (-1, 0, +1; like charges repel, opposite attract). Questions:
"If the gray object were heavier / lighter / uncharged / oppositely charged, which of
the following would (not) happen?" and "What will happen next?".

INPUT (stated plainly in the paper): object tracks and visible attributes are read from
ComPhy's annotation (ORACLE PERCEPTION, only while the object is in view and only the
first 125 frames = the 5 s the video shows). Masses and charges are NEVER read: they are
inferred from motion.

 1. LAWS, fitted on ComPhy TRAIN annotations (where properties are known):
      friction  deceleration a_s + b_s * speed per shape
      charge    a_i = k q_i q_j (p_i - p_j) / |p_i - p_j|^3 / m_i
      contact   distance r_i + r_j per shape pair; bounce restitution e
 2. INFER each object's (mass, charge) by coordinate descent: the assignment whose
    short simulated windows best reproduce the observed motion (target video only).
 3. ANSWER with CausalVis' rule: objects follow the recording until the intervention
    can reach them (a changed mass acts at its next contact; a changed charge acts on
    every charged object at once), then they are simulated; +E frames past the end.
    Predictive: every object simulated from the last observed frame.

    python paper_exp/comphy.py fit                         # laws from TRAIN annotations
    python paper_exp/comphy.py eval --split train --limit 300 --out ...   # dev
    python paper_exp/comphy.py eval --split val --out ...
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CP = os.path.join(ROOT, "external", "comphy")
ANN = os.path.join(CP, "ann", "target_annotation")
LAWS_P = os.path.join(HERE, "comphy_laws.json")
DT = 0.04          # one video frame
SUB = 10           # physics sub-steps per frame (ComPhy's engine runs at 250 Hz)
T_OBS = 125        # frames the video shows (5 s)
LOOKBACK = 3
USE_REFS = os.environ.get("COMPHY_REFS", "1") == "1"


def ann_path(idx):
    lo = idx // 1000 * 1000
    return os.path.join(ANN, "annotation_%05d_%05d" % (lo, lo + 1000), "%05d.json" % idx)


def load(idx):
    a = json.load(open(ann_path(idx)))
    props = a["object_property"]
    mt = a["motion_trajectory"]
    P = np.array([[o["location"][:2] for o in fr["objects"]] for fr in mt], np.float64)
    V = np.array([[o["velocity"][:2] for o in fr["objects"]] for fr in mt], np.float64)
    ins = np.array([[bool(o["inside_scene"]) for o in fr["objects"]] for fr in mt])
    attrs = [{"color": p["color"], "material": p["material"], "shape": p["shape"]} for p in props]
    hidden = [{"mass": p["mass"], "charge": p["charge"]} for p in props]      # diagnosis / fitting only
    cols = [(int(c["step"]) // 10, tuple(sorted(c["object_idxs"]))) for c in a["collision"]]
    return {"P": P, "V": V, "ins": ins, "attrs": attrs, "hidden": hidden, "cols": cols, "T": len(mt)}


REF = os.path.join(CP, "ann", "reference_annotation")


def load_refs(idx, attrs):
    """The 4 two-second reference videos of a scene (same objects, other starts). Objects
    are matched to the target's by their visible attributes; unmatched ones are dropped."""
    lo = idx // 1000 * 1000
    folder = os.path.join(REF, "annotation_%05d_%05d" % (lo, lo + 1000), "%05d" % idx)
    key = {(a["color"], a["material"], a["shape"]): k for k, a in enumerate(attrs)}
    out = []
    for f in sorted(glob.glob(os.path.join(folder, "*.json"))):
        a = json.load(open(f))
        props = a["object_property"]
        idxs = [key.get((p["color"], p["material"], p["shape"])) for p in props]
        keep = [i for i, k in enumerate(idxs) if k is not None]
        if len(keep) < 2:
            continue
        mt = a["motion_trajectory"]
        P = np.array([[fr["objects"][i]["location"][:2] for i in keep] for fr in mt], np.float64)
        V = np.array([[fr["objects"][i]["velocity"][:2] for i in keep] for fr in mt], np.float64)
        ins = np.array([[bool(fr["objects"][i]["inside_scene"]) for i in keep] for fr in mt])
        out.append({"P": P, "V": V, "ins": ins, "map": [idxs[i] for i in keep]})
    return out


# --------------------------------------------------------------------------- laws
def fit():
    files = sorted(glob.glob(os.path.join(ANN, "*", "*.json")))
    train = [f for f in files if not (4000 <= int(os.path.basename(f)[:5]) < 6000)]
    fr = {s: ([], []) for s in ("sphere", "cube", "cylinder")}
    kx, cont, rest = [], defaultdict(list), []
    train = train[::max(1, len(train) // 1500)][:1500]
    for f in train:
        try:
            d = load(int(os.path.basename(f)[:5]))
        except (ValueError, KeyError):              # a file cut off by an unfinished download
            continue
        P, V, n = d["P"], d["V"], len(d["attrs"])
        q = np.array([h["charge"] for h in d["hidden"]], float); m = np.array([h["mass"] for h in d["hidden"]], float)
        near = set()
        for t, (i, j) in d["cols"]:
            near |= {(u, i) for u in range(t - 3, t + 4)} | {(u, j) for u in range(t - 3, t + 4)}
            if 0 < t < d["T"] - 1:
                cont[tuple(sorted((d["attrs"][i]["shape"], d["attrs"][j]["shape"])))].append(float(np.linalg.norm(P[t, i] - P[t, j])))
        for t in range(1, d["T"] - 1):
            A = (V[t + 1] - V[t - 1]) / (2 * DT)
            for i in range(n):
                if (t, i) in near:
                    continue
                g = np.zeros(2)
                for j in range(n):
                    if j != i and q[i] * q[j] != 0:
                        dd = P[t, i] - P[t, j]; g += q[i] * q[j] * dd / np.linalg.norm(dd) ** 3
                g /= m[i]
                sp = np.linalg.norm(V[t, i])
                if q[i] != 0 and np.linalg.norm(g) > 0.2:
                    kx.append((float(A[i] @ g), float(g @ g)))
                elif q[i] == 0 and sp > 0.05:
                    fr[d["attrs"][i]["shape"]][0].append(sp)
                    fr[d["attrs"][i]["shape"]][1].append(-float(A[i] @ V[t, i]) / sp)
        for t, (i, j) in d["cols"]:
            if 3 <= t < d["T"] - 4:
                nrm = P[t, i] - P[t, j]; nrm /= max(np.linalg.norm(nrm), 1e-9)
                v0 = (V[t - 3, i] - V[t - 3, j]) @ nrm; v1 = (V[t + 3, i] - V[t + 3, j]) @ nrm
                if v0 < -0.3:
                    rest.append(-v1 / v0)
    friction = {}
    for s, (x, y) in fr.items():
        X = np.stack([np.ones(len(x)), np.array(x)], 1)
        a, b = np.linalg.lstsq(X, np.array(y), rcond=None)[0]
        friction[s] = [float(max(a, 0.0)), float(max(b, 0.0))]
    kx = np.array(kx)
    laws = {"friction": friction, "k": float(kx[:, 0].sum() / kx[:, 1].sum()),
            "contact": {"|".join(k): float(np.median(v)) for k, v in cont.items()},
            "restitution": float(np.median(rest)), "videos": len(train)}
    json.dump(laws, open(LAWS_P, "w"), indent=1)
    print(json.dumps(laws, indent=1))


# --------------------------------------------------------------------------- simulator
class Sim:
    def __init__(self, laws, attrs):
        self.L = laws
        self.shape = [a["shape"] for a in attrs]
        n = len(attrs)
        self.cd = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                key = "|".join(sorted((self.shape[i], self.shape[j])))
                self.cd[i, j] = laws["contact"].get(key, 0.4)
        self.fa = np.array([laws["friction"][s][0] for s in self.shape])
        self.fb = np.array([laws["friction"][s][1] for s in self.shape])

    def run(self, p, v, m, q, active, frames, record_from=0, entries=None):
        """Advance `frames` frames. active[k]: object k is simulated. entries: {frame: [(k, p, v)]}
        objects that appear later. Returns positions per frame and contact events (frame, i, j)."""
        p, v, active = p.copy(), v.copy(), active.copy()
        n = len(m); h = DT / SUB; e = self.L["restitution"]; k = self.L["k"]
        out = np.full((frames + 1, n, 2), np.nan); events = []
        incont = np.zeros((n, n), bool)
        for f in range(frames + 1):
            if entries and f in entries:
                for kk, pp, vv in entries[f]:
                    p[kk], v[kk], active[kk] = pp, vv, True
            out[f, active] = p[active]
            if f == frames:
                break
            for _ in range(SUB):
                idx = np.where(active)[0]
                acc = np.zeros_like(v)
                for a_ in idx:
                    if q[a_] == 0:
                        continue
                    for b_ in idx:
                        if b_ != a_ and q[b_] != 0:
                            dd = p[a_] - p[b_]; r = max(np.linalg.norm(dd), 0.2)
                            acc[a_] += k * q[a_] * q[b_] * dd / r ** 3 / m[a_]
                v[active] += acc[active] * h
                sp = np.linalg.norm(v, axis=1)
                dec = (self.fa + self.fb * sp) * h
                scale = np.where(sp > 1e-9, np.maximum(sp - dec, 0) / np.maximum(sp, 1e-9), 0)
                v[active] *= scale[active, None]
                p[active] += v[active] * h
                for ai in range(len(idx)):
                    for bi in range(ai + 1, len(idx)):
                        i, j = idx[ai], idx[bi]
                        dd = p[i] - p[j]; dist = np.linalg.norm(dd)
                        touching = dist < self.cd[i, j]
                        if touching and not incont[i, j]:
                            events.append((record_from + f, i, j))
                        incont[i, j] = incont[j, i] = touching
                        if touching and dist > 1e-9:
                            nrm = dd / dist; vn = (v[i] - v[j]) @ nrm
                            if vn < 0:
                                J = -(1 + e) * vn / (1 / m[i] + 1 / m[j])
                                v[i] += J / m[i] * nrm; v[j] -= J / m[j] * nrm
        return out, events


# --------------------------------------------------------------------------- inference
def observed(d):
    ins = d["ins"][:T_OBS]
    first = {k: int(np.argmax(ins[:, k])) for k in range(ins.shape[1]) if ins[:, k].any()}
    return ins, first


def infer(d, sim, window=10, sweeps=3, mode="both", refs=None, laws=None):
    """Masses and charges that best explain the observed motion: the target video's 125
    frames plus, when given, its reference videos (only while objects are in view)."""
    P, V = d["P"][:T_OBS], d["V"][:T_OBS]
    ins, first = observed(d)
    n = len(d["attrs"])
    clips = [(P, V, ins, list(range(n)), sim)]
    for r in refs or []:
        sub_attrs = [d["attrs"][k] for k in r["map"]]
        clips.append((r["P"], r["V"], r["ins"], r["map"], Sim(laws, sub_attrs)))

    def cost(m, q):
        c = 0.0
        for Pc, Vc, insc, mp, simc in clips:
            mm, qq = m[mp], q[mp]
            for s in range(0, len(Pc) - window, window):
                act = insc[s].copy()
                if act.sum() == 0:
                    continue
                out, _ = simc.run(Pc[s], Vc[s], mm, qq, act, window)
                ok = act & insc[s + window]
                if ok.any():
                    c += float(((out[window, ok] - Pc[s + window, ok]) ** 2).sum())
        return c

    best = None
    families = ([("mass", [(1, 0), (5, 0)])] if mode in ("both", "mass") else []) + \
               ([("charge", [(1, -1), (1, 0), (1, 1)])] if mode in ("both", "charge") else [])
    for fam, opts in families:
        m = np.ones(n); q = np.zeros(n)
        c0 = cost(m, q)
        if fam == "charge" and n >= 2:
            # one charge has nothing to act on, so single-object moves cannot leave
            # "all uncharged": start from the best charged PAIR (the overall sign of all
            # charges cannot be seen, so +1 is fixed for the first object of the pair)
            best_pair = None
            for i in range(n):
                for j in range(i + 1, n):
                    for sj in (1, -1):
                        q2 = np.zeros(n); q2[i], q2[j] = 1, sj
                        c = cost(m, q2)
                        if best_pair is None or c < best_pair[0]:
                            best_pair = (c, q2)
            if best_pair[0] < c0:
                c0, q = best_pair[0], best_pair[1].copy()
        for _ in range(sweeps):
            changed = False
            for i in range(n):
                cur = (m[i], q[i]); bc, bo = c0, cur
                for o in opts:
                    if o == cur:
                        continue
                    m[i], q[i] = o
                    c = cost(m, q)
                    if c < bc - 1e-9:
                        bc, bo = c, o
                m[i], q[i] = bo
                if bo != cur:
                    changed, c0 = True, bc
            if not changed:
                break
        if best is None or c0 < best[0]:
            best = (c0, fam, m.copy(), q.copy())
    return best[1], best[2], best[3]


# --------------------------------------------------------------------------- worlds
def recorded_contacts(d, sim):
    P, ins = d["P"][:T_OBS], d["ins"][:T_OBS]
    n = len(d["attrs"])
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            both = ins[:, i] & ins[:, j]
            dist = np.linalg.norm(P[:, i] - P[:, j], axis=1)
            hit = both & (dist < sim.cd[i, j] * 1.05)
            on = np.where(hit & ~np.concatenate([[False], hit[:-1]]))[0]
            out += [(int(t), i, j) for t in on]
    return sorted(out)


def run_world(d, sim, m, q, sim_from, ext):
    """CausalVis' rule on ComPhy. Objects replay the recording until they must be
    simulated: at sim_from[k]; when an object they touched in the recording is already
    simulated (lost contact, LOOKBACK frames before that touch); when a simulated object
    touches them (new contact); at the end of the video. Returns contact onsets (frame, i, j)."""
    P, V, ins = d["P"][:T_OBS], d["V"][:T_OBS], d["ins"][:T_OBS]
    n = len(m)
    rec = recorded_contacts(d, sim)
    start = min(sim_from.values()) if sim_from else T_OBS - 1
    mode = ["replay"] * n
    p = np.zeros((n, 2)); v = np.zeros((n, 2))
    h = DT / SUB; e = sim.L["restitution"]; k = sim.L["k"]
    incont = np.zeros((n, n), bool); events = []
    lost = defaultdict(list)
    for t, i, j in rec:
        lost[max(0, t - LOOKBACK)].append((i, j))
    for g in range(start, T_OBS - 1 + ext):
        for kk, f0 in sim_from.items():
            if f0 == g and mode[kk] == "replay":
                if g < T_OBS and ins[g, kk]:
                    p[kk], v[kk] = P[g, kk], V[g, kk]
                    mode[kk] = "sim"
                else:
                    mode[kk] = "sim_pending"          # starts when it appears
        for i, j in lost.get(g, []):
            for a_, b_ in ((i, j), (j, i)):
                if mode[a_] == "sim" and mode[b_] == "replay" and g < T_OBS and ins[g, b_]:
                    p[b_], v[b_] = P[g, b_], V[g, b_]; mode[b_] = "sim"
        for kk in range(n):
            if mode[kk] == "sim_pending" and g < T_OBS and ins[g, kk]:
                p[kk], v[kk] = P[g, kk], V[g, kk]; mode[kk] = "sim"
            if mode[kk] == "replay":
                if g >= T_OBS - 1:
                    if ins[T_OBS - 1, kk]:
                        p[kk], v[kk] = P[T_OBS - 1, kk], V[T_OBS - 1, kk]; mode[kk] = "sim"
                    else:
                        mode[kk] = "gone"
                elif ins[g, kk]:
                    p[kk], v[kk] = P[g, kk], V[g, kk]
        act = np.array([mode[kk] == "sim" or (mode[kk] == "replay" and g < T_OBS and ins[g, kk]) for kk in range(n)])
        idx = np.where(act)[0]
        for _ in range(SUB):
            acc = np.zeros_like(v)
            for a_ in idx:
                if q[a_] == 0:
                    continue
                for b_ in idx:
                    if b_ != a_ and q[b_] != 0:
                        dd = p[a_] - p[b_]; r = max(np.linalg.norm(dd), 0.2)
                        acc[a_] += k * q[a_] * q[b_] * dd / r ** 3 / m[a_]
            v[act] += acc[act] * h
            sp = np.linalg.norm(v, axis=1)
            dec = (sim.fa + sim.fb * sp) * h
            scale = np.where(sp > 1e-9, np.maximum(sp - dec, 0) / np.maximum(sp, 1e-9), 0)
            v[act] *= scale[act, None]
            p[act] += v[act] * h
            for ai in range(len(idx)):
                for bi in range(ai + 1, len(idx)):
                    i, j = idx[ai], idx[bi]
                    dd = p[i] - p[j]; dist = np.linalg.norm(dd)
                    touching = dist < sim.cd[i, j]
                    if touching and not incont[i, j]:
                        events.append((g, i, j))
                        for x_, y_ in ((i, j), (j, i)):         # new contact
                            if mode[x_] == "sim" and mode[y_] == "replay":
                                mode[y_] = "sim"
                    incont[i, j] = incont[j, i] = touching
                    if touching and dist > 1e-9:
                        nrm = dd / dist; vn = (v[i] - v[j]) @ nrm
                        if vn < 0:
                            J = -(1 + e) * vn / (1 / m[i] + 1 / m[j])
                            v[i] += J / m[i] * nrm; v[j] -= J / m[j] * nrm
    pre = [(t, i, j) for t, i, j in rec if t < start]
    return pre + events


def world(d, sim, m, q, change=None, ext=50):
    """change = (object, 'heavier'|'lighter'|'uncharged'|'opposite') or None (the factual
    future, for predictive questions)."""
    ins, first = observed(d)
    n = len(m)
    m2, q2 = m.copy(), q.copy()
    sim_from = {}
    if change is not None:
        kk, kind = change
        if kind == "heavier":
            m2[kk] = 5
        elif kind == "lighter":
            m2[kk] = 1
        elif kind == "uncharged":
            q2[kk] = 0
        elif kind == "opposite":
            q2[kk] = -q[kk]
        if kind in ("uncharged", "opposite") and q[kk] != 0:
            for j in range(n):                            # the field changes everywhere at once
                if q[j] != 0 and j in first:
                    sim_from[j] = first[j]
        elif kind in ("heavier", "lighter") and m2[kk] != m[kk]:
            rec = [t for t, i, j in recorded_contacts(d, sim) if kk in (i, j)]
            if rec:
                sim_from[kk] = max(0, rec[0] - LOOKBACK)   # its path changes at its first touch
    return run_world(d, sim, m2, q2, sim_from, ext)


def pairs_in(events, lo, hi):
    return {frozenset((i, j)) for t, i, j in events if lo <= t < hi}


# --------------------------------------------------------------------------- questions
COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'yellow', 'cyan', 'purple']
MATS = ['metal', 'rubber']
SHAPES = ['sphere', 'cylinder', 'cube']
CF_OPS = {"counterfact_heavier": "heavier", "counterfact_lighter": "lighter",
          "counterfact_uncharged": "uncharged", "counterfact_opposite": "opposite"}


def resolve(tokens, attrs):
    cand = list(range(len(attrs)))
    for t in tokens:
        if t in COLORS:
            cand = [k for k in cand if attrs[k]["color"] == t]
        elif t in MATS:
            cand = [k for k in cand if attrs[k]["material"] == t]
        elif t in SHAPES:
            cand = [k for k in cand if attrs[k]["shape"] == t]
    return cand


def segments(prog):
    """Split a program at 'objects' into attribute groups."""
    groups, cur = [], None
    for t in prog:
        if t == "objects":
            if cur is not None:
                groups.append(cur)
            cur = []
        elif cur is not None and t in COLORS + MATS + SHAPES:
            cur.append(t)
    if cur is not None:
        groups.append(cur)
    return groups


def evaluate(split, limit, out, ext_grid=(0, 25, 50), mode="both", shard=0, nshards=1):
    laws = json.load(open(LAWS_P))
    qs = json.load(open(os.path.join(CP, {"train": "train.json", "val": "val.json"}[split])))
    if split == "train":                             # dev: an even sample of the training scenes
        qs = [it for it in qs if not (4000 <= it["scene_index"] < 6000) and os.path.exists(ann_path(it["scene_index"]))]
        qs = qs[::max(1, len(qs) // (limit or len(qs)))]
    qs = qs[shard::nshards]
    recs = []
    done = 0
    # resumable: every finished scene is appended to <out>.jsonl and skipped on a rerun
    part = out + ".jsonl"
    finished = set()
    if os.path.exists(part):
        for line in open(part, encoding="utf-8"):
            r = json.loads(line)
            recs.append(r); finished.add(r["scene"])
    fpart = open(part, "a", encoding="utf-8")
    for item in qs:
        idx = item["scene_index"]
        if not os.path.exists(ann_path(idx)):
            continue
        if idx in finished:
            done += 1
            continue
        if split == "train" and 4000 <= idx < 6000:
            continue
        if limit and split != "train" and done >= limit:
            break
        done += 1
        d = load(idx)
        sim = Sim(laws, d["attrs"])
        refs = load_refs(idx, d["attrs"]) if USE_REFS else None
        fam, m, q = infer(d, sim, mode=mode, refs=refs, laws=laws)
        truth_h = d["hidden"]
        worlds = {}
        for qu in item["questions"]:
            qt = qu.get("question_type", "factual")
            if qt == "factual":
                continue
            prog = qu["program"]
            if qt.startswith("counterfactual"):
                op = next((o for o in prog if o in CF_OPS), None)
                g = segments(prog)
                tgt = resolve(g[0], d["attrs"]) if g else []
                key = (tgt[0], CF_OPS[op]) if len(tgt) == 1 and op else None
            else:
                key = "future"
            if key not in worlds:
                if key == "future":
                    worlds[key] = world(d, sim, m, q, None, ext=max(ext_grid))
                elif key is None:
                    worlds[key] = []
                else:
                    worlds[key] = world(d, sim, m, q, key, ext=max(ext_grid))
            ev = worlds[key]
            neg = "not" in prog or "negate" in prog
            for c in qu["choices"]:
                g = segments(c["program"])
                a = resolve(g[0], d["attrs"]) if len(g) > 0 else []
                b = resolve(g[1], d["attrs"]) if len(g) > 1 else []
                pair = frozenset((a[0], b[0])) if len(a) == 1 and len(b) == 1 and a[0] != b[0] else None
                hap = {}
                for e_ in ext_grid:
                    lo = T_OBS if key == "future" else 0
                    hap[e_] = bool(pair is not None and pair in pairs_in(ev, lo, T_OBS + e_))
                new_rec = {"scene": idx, "qid": qu["question_id"], "type": qt, "negate": neg, "answer": c["answer"],
                             "happens": hap, "family": fam, "resolved": pair is not None and key is not None,
                             "props_ok": bool(any(all((mm == h["mass"]) and (sg * qq == h["charge"]) for mm, qq, h in zip(m, q, truth_h))
                                                  for sg in (1, -1)))}
                recs.append(new_rec)
                fpart.write(json.dumps(new_rec) + chr(10))
        fpart.flush()
        if done % 25 == 0:
            print("scenes %d" % done, flush=True)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(recs, open(out, "w"))
    report(recs, ext_grid)


def report(recs, ext_grid=(0, 25, 50)):
    for qt in ("counterfactual_multiple_choice", "predictive_multiple_choice"):
        sub = [r for r in recs if r["type"] == qt]
        if not sub:
            continue
        for e_ in ext_grid:
            ok = defaultdict(lambda: True); n_ok = 0
            for r in sub:
                pred = "correct" if (r["happens"][str(e_)] if str(e_) in r["happens"] else r["happens"][e_]) != r["negate"] else "wrong"
                g = pred == r["answer"]; n_ok += g; ok[(r["scene"], r["qid"])] &= g
            print("%-32s +%2d frames: per option %.1f%% (%d/%d)  per question %.1f%% (%d/%d)"
                  % (qt, e_, 100.0 * n_ok / len(sub), n_ok, len(sub), 100.0 * sum(ok.values()) / len(ok), sum(ok.values()), len(ok)))
    scenes = {}
    for r in recs:
        scenes[r["scene"]] = r["props_ok"]
    print("hidden properties fully right in %.1f%% of scenes (%d)" % (100.0 * np.mean(list(scenes.values())), len(scenes)))


# --------------------------------------------------------------------------- audit
def audit_predictive(limit=400, shard=0, nshards=1, out=None):
    """Right for the right reason on predictive questions: ComPhy's annotation contains the
    future (frames 125-174), so every simulated collision that decides an answer can be
    checked against a real future collision of the same pair."""
    laws = json.load(open(LAWS_P))
    qs = json.load(open(os.path.join(CP, "val.json")))
    qs = [it for it in qs if any(q.get("question_type", "").startswith("predictive") for q in it["questions"])]
    qs = qs[::max(1, len(qs) // limit)][:limit][shard::nshards]
    recs = []
    for item in qs:
        idx = item["scene_index"]
        d = load(idx)
        sim = Sim(laws, d["attrs"])
        fam, m, q = infer(d, sim, refs=load_refs(idx, d["attrs"]) if USE_REFS else None, laws=laws)
        ev = world(d, sim, m, q, None, ext=50)
        gt = [(t, tuple(sorted(p))) for t, p in d["cols"] if T_OBS <= t < T_OBS + 50]
        for qu in item["questions"]:
            if not qu.get("question_type", "").startswith("predictive"):
                continue
            neg = "not" in qu["program"] or "negate" in qu["program"]
            for c in qu["choices"]:
                g = segments(c["program"])
                a = resolve(g[0], d["attrs"]) if g else []
                b = resolve(g[1], d["attrs"]) if len(g) > 1 else []
                if not (len(a) == 1 and len(b) == 1 and a[0] != b[0]):
                    continue
                pair = tuple(sorted((a[0], b[0])))
                ours = sorted(t for t, i, j in ev if tuple(sorted((i, j))) == pair and T_OBS <= t < T_OBS + 50)
                real = sorted(t for t, p in gt if p == pair)
                recs.append({"scene": idx, "answer": c["answer"], "negate": neg, "ours": ours, "real": real})
    json.dump(recs, open(out, "w"))
    print("audit records", len(recs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fit", "eval", "report", "audit"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "comphy_runs", "dev.json"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.mode == "audit":
        audit_predictive(a.limit or 400, a.shard, a.nshards, a.out)
    elif a.mode == "fit":
        fit()
    elif a.mode == "eval":
        evaluate(a.split, a.limit, a.out, shard=a.shard, nshards=a.nshards)
    else:
        recs = []
        for f in a.out.split(","):
            recs += json.load(open(f))
        report(recs)
