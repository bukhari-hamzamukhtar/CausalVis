"""
paper_exp/humans.py  —  CLEVRER-Humans (human causal judgments) with CausalVis
=============================================================================

CLEVRER-Humans (Mao et al., NeurIPS 2022 D&B) asks "Which of the following is
responsible for <event B>?" with options <event A> written freely by people
("the green cube came from the left"), labelled by human causal judgments.

Pipeline, per video:
 1. GROUND: a language model (Groq, temperature 0, cached) maps every event sentence to
    {kind: collision | enter | motion | stop | other, objects: [our object ids]} using
    our detected object list. It never sees answers.
 2. LOCATE: the event in OUR scene (detection-only tracks, the benchmark's union
    detector): a collision -> that pair's first detected collision; enter -> the
    object's first frame; motion / stop -> the object's first collision (what changed
    its motion), else its entry.
 3. DECIDE "A responsible for B", three rules, chosen on the CLEVRER-Humans TRAIN split:
      trace   CLEVRER's own rule: A is an ancestor of B in the objects' event chains
      butfor  CausalVis counterfactual: remove the object that A adds beyond B (or A's
              object if A and B share all objects) and simulate; A is responsible if B
              no longer happens within the video (+30 frames)
      yes     always "responsible" (label balance baseline)
Reported per option and per question (both options right), as in the paper.

    GROQ_API_KEY=... python paper_exp/humans.py ground --split valid
    python paper_exp/humans.py score --split train
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for d in ("src2", "v4", "v5", "v6", "v7"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.environ.setdefault("CF_LOOKBACK", "3")
import importlib.util                                                    # noqa: E402
_spec = importlib.util.spec_from_file_location("paper_cf_world_h", os.path.join(HERE, "cf_world.py"))
CFW = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CFW)
from world import load_spatial                                           # noqa: E402
sys.path.insert(0, HERE)
import detonly                                                           # noqa: E402

DATA = os.path.join(ROOT, "external", "clevrer_humans")
TRACKS = os.path.join(ROOT, "data", "trajectories_3d_det_humans")
QFILE = {"train": os.path.join(HERE, "humans_train_dev_questions.json"),     # the grounded dev subset
         "valid": os.path.join(DATA, "q_folder", "valid_question.json")}
CACHE = os.path.join(HERE, "humans_ground_%s.json")
NICK = {"sphere": "sphere / ball", "cube": "cube / block / box", "cylinder": "cylinder"}

GROUND_SYSTEM = (
    "You map short English descriptions of events in a video of colored objects sliding on a table "
    "to a fixed schema. Objects are listed with ids; people may call a metal gray object 'silver', "
    "a sphere 'ball', a cube 'block' or 'box'. For EACH numbered sentence return "
    '{"kind": one of "collision","enter","motion","stop","other", "objects": [ids involved, the '
    'main actor first]}. "collision" = two objects touch/hit/bump/push; "enter" = an object comes '
    'into the scene or comes from a side; "motion" = an object moves/turns/changes direction; '
    '"stop" = an object stops or slows. If an object cannot be identified use []. '
    'Reply with JSON only: {"events": [ ... one per sentence, same order ... ]}.')


def track_path(v):
    return os.path.join(TRACKS, "sim_%05d.npz" % v)


def ensure_track(v, cam, sizes):
    p = track_path(v)
    if not os.path.exists(p):
        os.makedirs(TRACKS, exist_ok=True)
        rec = detonly.build_det(os.path.join(detonly.PROPOSALS, "sim_%05d.json" % v), cam, sizes)
        if rec is None:
            return None
        np.savez_compressed(p, **rec)
    return p


def names(keys):
    out = []
    for i, k in enumerate(keys):
        c, m, s = str(k).split("_")[:3]
        out.append("%d: %s %s %s (%s)" % (i, c, m, s, NICK[s]))
    return out


def ground(split, limit=None):
    from lm import pick_provider, ask_json
    prov, cfg, err = pick_provider("groq")
    if err:
        raise SystemExit(err)
    blob = json.load(open(os.path.join(ROOT, "camera_fit.json")))
    cam, sizes = detonly.lift3d.Camera(blob["camera"]), blob["sizes"]
    qs = json.load(open(QFILE[split]))
    cache_p = CACHE % split
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    items = list(qs.items())[:limit]
    for n, (vid, item) in enumerate(items):
        if vid in cache:
            continue
        p = ensure_track(int(vid), cam, sizes)
        if p is None:
            cache[vid] = {"error": "no track"}
            continue
        keys = [str(k) for k in np.load(p, allow_pickle=True)["obj_keys"]]
        sents = []
        for q in item["questions"]:
            eff = q["question"].split("responsible for", 1)[-1].strip(" ?")
            sents.append(eff)
            sents += [c["choice"] for c in q["choices"]]
        user = "Objects:\n" + "\n".join(names(keys)) + "\n\nSentences:\n" + "\n".join(
            "%d. %s" % (i, s) for i, s in enumerate(sents))
        ev = None
        for attempt in range(4):
            try:
                ev = ask_json(cfg, prov, GROUND_SYSTEM, user).get("events", [])
                break
            except Exception as ex:                   # rate limit or bad JSON: wait and retry
                print("  ground error", vid, str(ex)[:160], flush=True)
                import time
                time.sleep(20 * (attempt + 1))
        if ev is None:
            continue                                  # not cached, so a rerun retries it
        cache[vid] = {"keys": keys, "sentences": sents, "events": ev}
        json.dump(cache, open(cache_p, "w"))
        print("grounded %s (%d/%d)" % (vid, n + 1, len(items)), flush=True)


class SceneEvents:
    def __init__(self, z, model, det="union"):
        self.z, self.model = z, model
        self.T = int(z["presence"].shape[0])
        self.entry = {}
        pres = z["presence"] > 0
        for k in range(pres.shape[1]):
            on = np.where(pres[:, k])[0]
            if on.size:
                self.entry[k] = int(on[0])
        w = CFW.build_world(model, z, set(), extend=0, entry_fix=json.load(open(os.path.join(HERE, "entry_profile.json"))))
        self.cols = {}
        for (i, j), fr in w["events"]["cal"].items():
            self.cols.setdefault((i, j), []).extend(fr)
        if det == "union":
            for (i, j), fr in w["events"]["kick2"].items():
                self.cols.setdefault((i, j), []).extend(fr)
        self.cols = {p: sorted(set(f for f in fr if f < self.T)) for p, fr in self.cols.items()}
        self.cols = {p: fr for p, fr in self.cols.items() if fr}
        self._cf = {}

    def locate(self, e):
        """-> (kind, objects, frame) of the matching scene event, or None."""
        if not isinstance(e, dict):
            return None
        kind, objs = e.get("kind"), [o for o in e.get("objects", []) if isinstance(o, int)]
        if kind == "collision" and len(objs) >= 2:
            p = tuple(sorted(objs[:2]))
            fr = self.cols.get(p)
            return ("collision", p, fr[0]) if fr else None
        if not objs:
            return None
        k = objs[0]
        if kind == "enter":
            return ("enter", (k,), self.entry.get(k, 0)) if k in self.entry else None
        own = sorted((f, p) for p, fr in self.cols.items() if k in p for f in fr)
        if own:
            return ("collision", own[0][1], own[0][0])
        return ("enter", (k,), self.entry[k]) if k in self.entry else None

    def happens(self, ev, removed):
        """Does event ev still happen when `removed` objects are gone?"""
        if set(ev[1]) & removed:
            return False
        if ev[0] == "enter":
            return True
        key = frozenset(removed)
        if key not in self._cf:
            self._cf[key] = CFW.build_world(self.model, self.z, set(removed), extend=30,
                                            entry_fix=json.load(open(os.path.join(HERE, "entry_profile.json"))))
        w = self._cf[key]
        i, j = ev[1]
        fr = w["events"]["cal"].get((i, j), []) + w["events"]["kick2"].get((i, j), [])
        return any(f < w["T"] + 30 for f in fr)

    def trace_ancestor(self, a, b):
        """CLEVRER's rule on our events: a precedes b in some shared object's event chain."""
        evs = [("enter", (k,), f) for k, f in self.entry.items() if f > 0] + \
              [("collision", p, f) for p, fr in self.cols.items() for f in fr]
        chains = {}
        for e in evs:
            for o in e[1]:
                chains.setdefault(o, []).append(e)
        chains = {o: sorted(c, key=lambda e: e[2]) for o, c in chains.items()}
        causes, stack, seen = set(), [b], set()
        while stack:
            t = stack.pop()
            for c in chains.values():
                if t in c:
                    i = c.index(t)
                    if i > 0 and c[i - 1] not in seen:
                        seen.add(c[i - 1]); causes.add(c[i - 1]); stack.append(c[i - 1])
        return a in causes


def decide(rule, sc, a, b):
    if rule == "yes":
        return True
    if a is None or b is None:
        return False
    if a == b:
        return False
    if rule == "trace":
        return sc.trace_ancestor(a, b)
    extra = set(a[1]) - set(b[1])
    removed = extra if extra else {a[1][0]}
    if set(b[1]) <= removed:
        return True
    return not sc.happens(b, removed)


def score(split, rules=("trace", "butfor", "yes"), limit=None):
    cache = json.load(open(CACHE % split))
    qs = dict(list(json.load(open(QFILE[split])).items())[:limit])
    model = load_spatial(os.path.join(ROOT, "v6_voxel.pt"), yaw_known=False)
    torch.set_num_threads(1)
    res = {r: [0, 0, 0, 0] for r in rules}          # options ok, options, questions ok, questions
    for vid, item in qs.items():
        g = cache.get(vid, {})
        ev = g.get("events", [])
        sc = None
        if "keys" in g and os.path.exists(track_path(int(vid))):
            sc = SceneEvents(np.load(track_path(int(vid)), allow_pickle=True), model)
        k = 0
        for q in item["questions"]:
            b = sc.locate(ev[k]) if sc and k < len(ev) else None
            k += 1
            okq = {r: True for r in rules}
            for c in q["choices"]:
                a = sc.locate(ev[k]) if sc and k < len(ev) else None
                k += 1
                for r in rules:
                    pred = decide(r, sc, a, b) if (sc or r == "yes") else False
                    good = ("correct" if pred else "wrong") == c["answer"]
                    res[r][0] += good; res[r][1] += 1; okq[r] &= good
            for r in rules:
                res[r][2] += okq[r]; res[r][3] += 1
    for r, (a, b, c, d) in res.items():
        print("%-7s per option %.1f%% (%d/%d)   per question %.1f%% (%d/%d)" % (r, 100.0 * a / b, a, b, 100.0 * c / d, c, d))
    json.dump(res, open(os.path.join(HERE, "humans_score_%s.json" % split), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ground", "score"])
    ap.add_argument("--split", default="train", choices=["train", "valid"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    ground(a.split, a.limit) if a.mode == "ground" else score(a.split, limit=a.limit)
