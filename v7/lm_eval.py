"""
v7/lm_eval.py  —  the language model reads the question, the world model builds the
                  counterfactual world once, the language model answers from the store
=====================================================================================

The division of labour, and why it is this one:

  LANGUAGE MODEL  reads the question -> which object is removed
                  reads the stored world -> answers every choice, citing lines
  WORLD MODEL     removes the object, simulates the whole scene once (every object,
                  every frame, past the end of the video), stores every event

The language model does NOT choose when to start simulating. A question like
"what happens if the cube is removed?" carries no timing at all; the timing is
physics. The world model decides it on the go: each object follows its recorded
motion only until the removal can reach it, then it is simulated. That is what
the store records, so the language model can see and cite it.

Every choice of one question comes from the SAME stored world, so sub-question
answers can never contradict each other.

Scores the LM and the literal reading of the same store side by side (so any gap
is the reader), with a paired McNemar test. Resumable.

    GROQ_API_KEY=... python v7/lm_eval.py --detector rule --extend 20 --limit 60
"""

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for d in ("src2", "v4", "v5", "v6", "v7"):
    sys.path.insert(0, os.path.join(ROOT, d))

from benchmark_eval import (load_questions, normalise_choices,           # noqa: E402
                            answers_from_conversations, parse_descriptors,
                            program_has, resolve_all, load_split_set, _vid_num)
from world import load_spatial                                           # noqa: E402
from cf_world import build_world                                         # noqa: E402
from lm import pick_provider, parse_intervention, answer_choices         # noqa: E402

CAUSE = {"lost_contact": "it was about to touch %s in the video, which no longer happens",
         "new_contact": "the simulated %s came near it",
         "video_ended": "the video ended, so the world model predicts the rest",
         "entry": "it entered the scene"}


def event_frames(world, i, j, detector, extend, tol):
    a, b = min(i, j), max(i, j)
    limit = world["T"] + extend
    if detector in ("rule", "cal"):
        return [f for f in world["events"][detector].get((a, b), []) if f < limit]
    start = world["first_both"].get((a, b))
    out, last = [], -99
    for f, g in world["near"].get((a, b), []):
        if f < limit and f != start and g <= tol:
            if f - last > 3:                        # one event per contact episode
                out.append(f)
            last = f
    return out


def store_text(world, names, removed, detector, extend, tol):
    T = world["T"]
    lines = ["Counterfactual world after removing: %s." % ", ".join(names[k] for k in sorted(removed)),
             "The recorded video has %d frames; the world model continued %d frames past its end." % (T, extend),
             "", "HOW EACH OBJECT WAS PRODUCED:"]
    for k in range(world["N"]):
        if k in removed:
            lines.append("  %s: removed." % names[k])
            continue
        tk = world["taint"].get(k)
        if tk is None:
            lines.append("  %s: follows its recorded motion (the removal never reaches it)." % names[k])
        else:
            cause = CAUSE.get(tk[1], tk[1])
            if "%s" in cause:
                cause = cause % (names[tk[2]] if tk[2] is not None else "an object")
            lines.append("  %s: recorded motion until frame %d, simulated after (%s)." % (names[k], tk[0], cause))
    ev = []
    for i in range(world["N"]):
        for j in range(i + 1, world["N"]):
            if i in removed or j in removed:
                continue
            for f in event_frames(world, i, j, detector, extend, tol):
                ev.append((f, i, j))
    ev.sort()
    lines += ["", "COLLISIONS IN THIS WORLD (frame: object and object):"]
    if not ev:
        lines.append("  none")
    for f, i, j in ev:
        lines.append("  frame %d%s: %s and %s collide" % (f, " (after the video ends)" if f >= T else "",
                                                         names[i], names[j]))
    lines.append("Any pair not listed above does NOT collide in this world.")
    return "\n".join(lines)


def with_retry(fn, *args, tries=6):
    for k in range(tries):
        try:
            return fn(*args)
        except Exception as exc:                          # noqa: BLE001
            if k == tries - 1:
                raise
            msg = str(exc)
            wait = 20 * (k + 1) if ("429" in msg or "rate" in msg.lower()) else 5
            print("      [lm] %s -- retrying in %ds" % (msg[:120], wait), flush=True)
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v6_voxel.pt")
    ap.add_argument("--data", default="data/trajectories_3d_yaw")
    ap.add_argument("--split", default="split3d_yaw.json")
    ap.add_argument("--which", default="test")
    ap.add_argument("--questions", default="zechennlp/counterfactual/validation-00000-of-00001.json")
    ap.add_argument("--detector", default="rule")
    ap.add_argument("--tol", type=float, default=6.0)
    ap.add_argument("--extend", type=int, default=20)
    ap.add_argument("--no-lost-affected", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--provider", default=None)
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--out", default="v7/runs/lm_eval.jsonl")
    a = ap.parse_args()

    prov, cfg, why = pick_provider(a.provider)
    if cfg is None:
        raise SystemExit("no LM key: " + str(why))
    model = load_spatial(a.model, yaw_known=False)
    files = sorted(glob.glob(os.path.join(a.data, "*.npz")))
    by_num = {_vid_num(f): f for f in files if _vid_num(f) is not None}
    allowed = load_split_set(a.split, a.which)
    rows = load_questions(a.questions)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out, encoding="utf-8"):
            r = json.loads(line); done.add((r["video"], r["question"]))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fout = open(a.out, "a", encoding="utf-8")
    print("world model %s  detector %s  extend +%d  LM %s  split %s" % (a.model, a.detector, a.extend, prov, a.which))

    n_q = len(done)
    worlds = {}
    for row in rows:
        if n_q >= a.limit:
            break
        v = _vid_num(str(row.get("video_filename", row.get("video", ""))))
        if v is None or v not in by_num or v not in allowed:
            continue
        question = row.get("question", "")
        if (v, question) in done:
            continue
        z = np.load(by_num[v], allow_pickle=True)
        keys = [str(k) for k in z["obj_keys"]]; names = [k.replace("_", " ") for k in keys]
        N = min(8, z["positions"].shape[1])
        choices = normalise_choices(row.get("choices"))
        if choices and all(c["answer"] is None for c in choices):
            answers_from_conversations(row, choices)
        scorable = []
        for c in choices:
            if c["answer"] not in ("correct", "wrong"):
                continue
            cd = parse_descriptors(c["program"])
            if len(cd) < 2:
                continue
            h1 = [k for k in resolve_all(cd[0], keys) if k < N]
            h2 = [k for k in resolve_all(cd[1], keys) if k < N]
            if len(h1) != 1 or len(h2) != 1 or h1[0] == h2[0]:
                continue
            scorable.append((c, h1[0], h2[0]))
        if not scorable:
            continue
        negate = program_has(row.get("program", []), "negate")
        qd = parse_descriptors(row.get("program", []))
        prog_removed = sorted(k for k in resolve_all(qd[0], keys) if k < N) if qd else []

        parsed = with_retry(parse_intervention, cfg, prov, question, names)   # 1. LM reads the question
        time.sleep(a.sleep)
        try:
            removed = {int(parsed["removed_index"])}
        except (KeyError, TypeError, ValueError):
            removed = set()

        wk = (v, frozenset(removed))                                         # 2. world model, once
        if wk not in worlds:
            worlds.clear()
            worlds[wk] = build_world(model, z, removed, extend=max(a.extend, 0),
                                     lost_contact_affected=not a.no_lost_affected, strict=a.strict)
        w = worlds[wk]
        txt = store_text(w, names, removed, a.detector, a.extend, a.tol)      # 3. store

        ans = with_retry(answer_choices, cfg, prov, txt, question,           # 4. LM answers from the store
                         [c["choice"] for c, _, _ in scorable])
        time.sleep(a.sleep)
        by = {}
        for x in (ans.get("answers", []) if isinstance(ans, dict) else []):
            try:
                by[int(x["index"])] = x
            except (KeyError, TypeError, ValueError):
                pass
        recs = []
        for k, (c, i, j) in enumerate(scorable):
            happens = bool(i not in removed and j not in removed and event_frames(w, i, j, a.detector, a.extend, a.tol))
            recs.append({"choice": c["choice"], "truth": c["answer"],
                         "lm": by.get(k, {}).get("label", "?"), "lm_happens": by.get(k, {}).get("happens"),
                         "prog": "correct" if happens != bool(negate) else "wrong",
                         "evidence": by.get(k, {}).get("evidence", "")})
        rec = {"video": v, "question": question, "negate": bool(negate), "lm_removed": sorted(removed),
               "program_removed": prog_removed, "store": txt, "choices": recs}
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        n_q += 1
        print("q%-3d video %d  parse %s  LM %d/%d  literal %d/%d"
              % (n_q, v, "ok" if sorted(removed) == prog_removed else "DIFFERS",
                 sum(r["lm"] == r["truth"] for r in recs), len(recs),
                 sum(r["prog"] == r["truth"] for r in recs), len(recs)), flush=True)
    fout.close()
    summarise(a.out)


def summarise(path):
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    ch = [c for r in recs for c in r["choices"]]
    n = max(len(ch), 1)
    lm = sum(c["lm"] == c["truth"] for c in ch); pg = sum(c["prog"] == c["truth"] for c in ch)
    b = sum(c["lm"] == c["truth"] and c["prog"] != c["truth"] for c in ch)
    cc = sum(c["lm"] != c["truth"] and c["prog"] == c["truth"] for c in ch)
    print("=" * 70)
    print("questions %d  choices %d   LM parse matches program %d/%d"
          % (len(recs), len(ch), sum(r["lm_removed"] == r["program_removed"] for r in recs), len(recs)))
    print("per-choice   LM %.1f%% (%d)   literal reading %.1f%% (%d)" % (100 * lm / n, lm, 100 * pg / n, pg))
    print("per-question LM %.1f%%   literal %.1f%%"
          % (100 * sum(all(c["lm"] == c["truth"] for c in r["choices"]) for r in recs) / max(len(recs), 1),
             100 * sum(all(c["prog"] == c["truth"] for c in r["choices"]) for r in recs) / max(len(recs), 1)))
    print("paired: LM right & literal wrong %d, literal right & LM wrong %d, McNemar z=%.2f"
          % (b, cc, (b - cc) / math.sqrt(b + cc) if b + cc else 0.0))
    print("=" * 70)


if __name__ == "__main__":
    main()
