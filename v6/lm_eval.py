"""
v6/lm_eval.py  —  the language model answers counterfactual questions from the
                  stored predicted future, scored on the test split
=============================================================================

For every question:

  1. the LM reads the question and names the removed object (no program used)
  2. the world model simulates the edited world with VOXEL contacts:
       - one continuous rollout of every object from the cone entry
       - a focused re-simulation of EVERY surviving pair (same windows as
         v5/evaluate.py: factual pairs around closest approach, counterfactual
         pairs from cone entry to the end of the clip)
  3. all of it is written to the store as text, voxel contact events included
  4. the LM answers every choice from that text alone -- no ground truth, no
     template, no if/else extraction

Two readings of the SAME store are scored side by side:
  lm    the language model's answer
  prog  the literal reading evaluate.py uses (collides xor negated)
So any gap between them is the reader, not the simulation.

Resumable: finished questions are kept in --out (jsonl) and skipped on restart,
because free-tier rate limits can stop a long run part-way.

    GROQ_API_KEY=... python v6/lm_eval.py --model v6_voxel.pt --limit 60
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for d in ("src2", "v4", "v5", "v6"):
    sys.path.insert(0, os.path.join(ROOT, d))

from benchmark_eval import (load_questions, normalise_choices,           # noqa: E402
                            answers_from_conversations, parse_descriptors,
                            program_has, resolve_all, load_split_set,
                            lightcone_rollout, _vid_num)
import evaluate as ev                                                     # noqa: E402
from evaluate import closest_approach, WINDOW, simulate_pair_spatial     # noqa: E402
from world import load_spatial                                           # noqa: E402
from demo import voxel_rollout                                           # noqa: E402
from lm import pick_provider, parse_intervention, answer_choices         # noqa: E402


def with_retry(fn, *args, tries=6):
    """Free tiers answer 429 when the per-minute budget is spent. Back off."""
    for k in range(tries):
        try:
            return fn(*args)
        except Exception as exc:                          # noqa: BLE001
            msg = str(exc)
            if k == tries - 1:
                raise
            wait = 20 * (k + 1) if ("429" in msg or "rate" in msg.lower()) else 5
            print("      [lm] %s -- retrying in %ds" % (msg[:120], wait), flush=True)
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/trajectories_3d_yaw")
    ap.add_argument("--split", default="split3d_yaw.json")
    ap.add_argument("--which", default="test")
    ap.add_argument("--questions",
                    default="zechennlp/counterfactual/validation-00000-of-00001.json")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--provider", default=None)
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between LM calls")
    ap.add_argument("--out", default="v6/cmp/lm_eval.jsonl")
    a = ap.parse_args()

    prov, cfg, why = pick_provider(a.provider)
    if cfg is None:
        raise SystemExit("no LM key: " + str(why))
    ev.VOXEL_MODE = True
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

    print("model %s   LM %s   voxel contacts ON   test split" % (a.model, prov))
    n_q = len(done)
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
        keys = [str(k) for k in z["obj_keys"]]
        names = [k.replace("_", " ") for k in keys]
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
        prog_removed = set(k for k in resolve_all(qd[0], keys) if k < N) if qd else set()

        # 1. LM reads the question
        parsed = with_retry(parse_intervention, cfg, prov, question, names)
        time.sleep(a.sleep)
        try:
            removed = {int(parsed["removed_index"])}
        except (KeyError, TypeError, ValueError):
            removed = set()

        # 2. simulate: continuous rollout + focused pair re-simulations
        _, _, taint, _ = lightcone_rollout(model, z, removed, 0.02, 0.02, 0.06)
        s0 = max(0, min(taint.values()) - 1) if taint else 0
        st, events, _ = voxel_rollout(model, z, removed, s0)
        st.contacts()
        pos, pres = z["positions"], z["presence"]
        keep = [k for k in range(N) if k not in removed and (pres[:, k] > 0).any()]
        focused = {}
        for ii in range(len(keep)):
            for jj in range(ii + 1, len(keep)):
                i, j = keep[ii], keep[jj]
                tainted = [taint[k] for k in (i, j) if k in taint]
                if tainted:
                    s, end = max(0, min(tainted)), pos.shape[0]
                else:
                    ca = closest_approach(pos, pres, i, j)
                    if ca is None:
                        continue
                    s, end = max(0, ca - WINDOW), ca + WINDOW
                fr = simulate_pair_spatial(model, z, removed, i, j, s, 0.02, 0.02, end=end)
                focused[(i, j)] = (s, fr)

        # 3. the store as text
        txt = st.to_text()
        txt += "\n\nFOCUSED RE-SIMULATIONS (each pair from a true state shortly before it could meet):\n"
        for (i, j), (s, fr) in sorted(focused.items()):
            txt += "  %s and %s: %s\n" % (names[i], names[j],
                                          ("COLLIDE at frame %d" % fr) if fr is not None
                                          else "do NOT collide (simulated from frame %d)" % s)
        if events:
            txt += "\nVOXEL CONTACT EVENTS (frame: object hits object at 3D point, normal):\n"
            for e in events[:40]:
                txt += "  frame %d: %s hits %s at (%.2f, %.2f, %.2f), normal (%+.2f, %+.2f)\n" % (
                    e["frame"], names[e["i"]], names[e["j"]], *e["point_world"], *e["normal"])

        # 4. LM answers from the store alone
        ans = with_retry(answer_choices, cfg, prov, txt, question,
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
            if i in removed or j in removed:
                happens = False
            else:
                key = (min(i, j), max(i, j))
                happens = key in focused and focused[key][1] is not None
            prog = "correct" if (happens != negate) else "wrong"
            lm_label = by.get(k, {}).get("label", "?")
            recs.append({"choice": c["choice"], "truth": c["answer"], "lm": lm_label,
                         "lm_happens": by.get(k, {}).get("happens"),
                         "prog": prog, "evidence": by.get(k, {}).get("evidence", "")})
        rec = {"video": v, "question": question, "negate": bool(negate),
               "lm_removed": sorted(removed), "program_removed": sorted(prog_removed),
               "voxel_contacts": len(events), "choices": recs}
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        n_q += 1
        lm_ok = sum(r["lm"] == r["truth"] for r in recs)
        pg_ok = sum(r["prog"] == r["truth"] for r in recs)
        print("q%-3d video %d  parse %s  LM %d/%d  prog %d/%d  voxel contacts %d"
              % (n_q, v, "ok" if removed == prog_removed else "DIFFERS",
                 lm_ok, len(recs), pg_ok, len(recs), len(events)), flush=True)

    fout.close()
    summarise(a.out)


def summarise(path):
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    ch = [c for r in recs for c in r["choices"]]
    n = max(len(ch), 1)
    lm = sum(c["lm"] == c["truth"] for c in ch)
    pg = sum(c["prog"] == c["truth"] for c in ch)
    agree = sum(c["lm"] == c["prog"] for c in ch)
    lm_q = sum(all(c["lm"] == c["truth"] for c in r["choices"]) for r in recs)
    pg_q = sum(all(c["prog"] == c["truth"] for c in r["choices"]) for r in recs)
    parse_ok = sum(r["lm_removed"] == r["program_removed"] for r in recs)
    print("=" * 66)
    print("questions %d   choices %d" % (len(recs), len(ch)))
    print("LM parse of the intervention matches the program: %d/%d" % (parse_ok, len(recs)))
    print("per-choice   LM %.1f%% (%d)   literal reading %.1f%% (%d)" % (100 * lm / n, lm, 100 * pg / n, pg))
    print("per-question LM %.1f%%       literal reading %.1f%%"
          % (100 * lm_q / max(len(recs), 1), 100 * pg_q / max(len(recs), 1)))
    print("LM agrees with the literal reading on %.1f%% of choices" % (100 * agree / n))
    print("=" * 66)


if __name__ == "__main__":
    main()
