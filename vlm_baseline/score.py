"""
vlm_baseline/score.py  —  score the vision-language model on the same options as the engine
==========================================================================================

Reads the answers the GPU job wrote and turns them into the two numbers the paper uses,
per option and per question, on exactly the keys paper_exp/score.py uses
(video, question id, choice text, occurrence of that text).

    python vlm_baseline/score.py one   questions_testA.json answers.jsonl video
    python vlm_baseline/score.py table questions_testA.json answers.jsonl
    python vlm_baseline/score.py pair  questions_testA.json answers.jsonl video A_learned cal 30

The model is asked "does this event happen?" and the question's polarity is applied here,
so the score is not a test of how well a language model handles the word "not".
Confidence intervals resample VIDEOS, 2,000 draws, fixed seed, as in paper_exp/score.py.
"""

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from job import build_prompt, frame_keys, parse_answers             # noqa: E402
from vlm import AnswerCache, key_of                                 # noqa: E402

CONDITIONS = ("video", "blind")


def load_answers(questions, answers, condition, model="Qwen2.5-VL-7B-Instruct", seed=0,
                 frames=16):
    """One yes/no per option, keyed like the engine's records. Also counts how many replies
    could not be read and how many questions were never asked."""
    cache = AnswerCache([a for a in answers if a])
    qs = json.load(open(questions, encoding="utf-8"))
    said, missing, unreadable = {}, 0, 0
    for q in qs:
        opts = q["options"]
        prompt = build_prompt(q.get("intervention") or q["question"], opts, condition == "blind")
        paths = [] if condition == "blind" else frame_keys(q["video"], frames)
        reply = cache.get(key_of(model, seed, prompt, paths))
        if reply is None:
            missing += 1
            continue
        vals, ok = parse_answers(reply, len(opts))
        unreadable += 0 if ok else 1
        occ = defaultdict(int)
        for t, v in zip(opts, vals):
            k = (q["video"], str(q["qid"]), t, occ[t])
            occ[t] += 1
            said[k] = {"happens": v == "yes", "negate": q["negate"]}
    return said, {"questions": len(qs), "missing": missing, "unreadable": unreadable}


def load_truth(questions):
    truth = {}
    for row in json.load(open(questions + ".truth.json", encoding="utf-8")):
        occ = defaultdict(int)
        for t, ans in zip(row["options"], row["answers"]):
            truth[(row["video"], str(row["qid"]), t, occ[t])] = ans
            occ[t] += 1
    return truth


def tally(said, truth, keys=None):
    keys = [k for k in (said if keys is None else keys) if k in said and k in truth]
    ok = {}
    for k in keys:
        label = "correct" if said[k]["happens"] != said[k]["negate"] else "wrong"
        ok[k] = label == truth[k]
    q = defaultdict(lambda: True)
    for k in keys:
        q[(k[0], k[1])] &= ok[k]
    return ok, q


def summary(said, truth, keys=None):
    ok, q = tally(said, truth, keys)
    return {"options": sum(ok.values()), "n_options": len(ok),
            "questions": sum(q.values()), "n_questions": len(q)}


def boot(said, truth, n=2000, seed=0):
    ok, q = tally(said, truth)
    byv_o, byv_q = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for k, g in ok.items():
        byv_o[k[0]][0] += g
        byv_o[k[0]][1] += 1
    for k, g in q.items():
        byv_q[k[0]][0] += g
        byv_q[k[0]][1] += 1
    vids = sorted(byv_o)
    rng = random.Random(seed)
    so, sq = [], []
    for _ in range(n):
        pick = [rng.choice(vids) for _ in vids]
        so.append(100.0 * sum(byv_o[v][0] for v in pick) / max(1, sum(byv_o[v][1] for v in pick)))
        sq.append(100.0 * sum(byv_q[v][0] for v in pick) / max(1, sum(byv_q[v][1] for v in pick)))
    so.sort()
    sq.sort()
    lo, hi = int(0.025 * n), int(0.975 * n) - 1
    return (so[lo], so[hi]), (sq[lo], sq[hi])


def fmt(s):
    return "options %5.1f%% (%d/%d)  questions %5.1f%% (%d/%d)" % (
        100.0 * s["options"] / max(1, s["n_options"]), s["options"], s["n_options"],
        100.0 * s["questions"] / max(1, s["n_questions"]), s["questions"], s["n_questions"])


def engine(run, det, ext):
    """The engine's own yes/no on the same keys, read from a paper_exp run."""
    sys.path.insert(0, os.path.join(ROOT, "paper_exp"))
    import score as pscore
    recs = pscore.load(run)
    return {k: {"happens": pscore.happens(r, det, ext), "negate": r["negate"]}
            for k, r in recs.items()}


def mcnemar(a, b, truth):
    keys = [k for k in a if k in b and k in truth]
    oa, qa = tally(a, truth, keys)
    ob, qb = tally(b, truth, keys)
    x = sum(oa[k] and not ob[k] for k in keys)
    y = sum(ob[k] and not oa[k] for k in keys)
    qx = sum(qa[k] and not qb[k] for k in qa)
    qy = sum(qb[k] and not qa[k] for k in qa)
    return {"matched": len(keys), "a_only": x, "b_only": y,
            "z": (x - y) / math.sqrt(x + y) if x + y else 0.0,
            "q_a_only": qx, "q_b_only": qy,
            "z_q": (qx - qy) / math.sqrt(qx + qy) if qx + qy else 0.0,
            "a": summary(a, truth, keys), "b": summary(b, truth, keys)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["one", "table", "pair"])
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--answers", default="", help="extra answer files, comma separated")
    ap.add_argument("--model", default="Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=16)
    a = ap.parse_args()

    qfile = a.rest[0] if a.rest else os.path.join(HERE, "questions_testA.json")
    qfile = qfile if os.path.exists(qfile) else os.path.join(HERE, qfile)
    files = [a.rest[1] if len(a.rest) > 1 else os.path.join(HERE, "answers.jsonl")]
    files += [f for f in a.answers.split(",") if f]
    truth = load_truth(qfile)

    def get(cond):
        return load_answers(qfile, files, cond, a.model, a.seed, a.frames)

    if a.cmd == "table":
        for cond in CONDITIONS:
            said, info = get(cond)
            if not said:
                print("%-6s no answers yet" % cond)
                continue
            (ol, oh), (ql, qh) = boot(said, truth)
            print("%-6s %s  95%% CI %.1f-%.1f / %.1f-%.1f  | unanswered %d, unreadable %d"
                  % (cond, fmt(summary(said, truth)), ol, oh, ql, qh,
                     info["missing"], info["unreadable"]))
    elif a.cmd == "one":
        cond = a.rest[2] if len(a.rest) > 2 else "video"
        said, info = get(cond)
        (ol, oh), (ql, qh) = boot(said, truth)
        print("%s  %s" % (cond, fmt(summary(said, truth))))
        print("95%% CI options %.1f-%.1f  questions %.1f-%.1f" % (ol, oh, ql, qh))
        print("questions in the set %d | never answered %d | replies not readable %d"
              % (info["questions"], info["missing"], info["unreadable"]))
    else:
        cond = a.rest[2] if len(a.rest) > 2 else "video"
        run = a.rest[3] if len(a.rest) > 3 else "A_learned"
        det = a.rest[4] if len(a.rest) > 4 else "cal"
        ext = int(a.rest[5]) if len(a.rest) > 5 else 30
        said, _ = get(cond)
        p = mcnemar(said, engine(run, det, ext), truth)
        print("matched %d" % p["matched"])
        print("VLM    %s" % fmt(p["a"]))
        print("engine %s  (%s %s +%d)" % (fmt(p["b"]), run, det, ext))
        print("options: VLM only %d, engine only %d, z = %+.2f" % (p["a_only"], p["b_only"], p["z"]))
        print("questions: VLM only %d, engine only %d, z = %+.2f"
              % (p["q_a_only"], p["q_b_only"], p["z_q"]))


if __name__ == "__main__":
    main()
