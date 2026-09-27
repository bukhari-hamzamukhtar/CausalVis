"""
vlm_baseline/selftest.py  —  check the scorer without a GPU
==========================================================

Writes answer files a model did not produce and scores them, so the plumbing (prompt,
cache key, parser, polarity, keys shared with the engine) is checked against answers whose
score is known in advance:

  oracle      the true answer everywhere              -> must be 100%
  always-no   every event answered "no"               -> must equal "never predict a collision"
  always-yes  every event answered "yes"              -> must equal "always predict a collision"

    python vlm_baseline/selftest.py                      # test set
    python vlm_baseline/selftest.py questions_valA.json
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from job import build_prompt, frame_keys                          # noqa: E402
from vlm import key_of                                            # noqa: E402
import score as S                                                 # noqa: E402

MODES = ("oracle", "always-no", "always-yes")


def write(qfile, mode, out, model="Qwen2.5-VL-7B-Instruct", seed=0, frames=16):
    qs = json.load(open(qfile, encoding="utf-8"))
    truth = {(r["video"], str(r["qid"])): r
             for r in json.load(open(qfile + ".truth.json", encoding="utf-8"))}
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for q in qs:
            t = truth[(q["video"], str(q["qid"]))]
            for cond in S.CONDITIONS:
                prompt = build_prompt(q.get("intervention") or q["question"], q["options"],
                                      cond == "blind")
                names = [] if cond == "blind" else frame_keys(q["video"], frames)
                if mode == "always-no":
                    vals = ["no"] * len(q["options"])
                elif mode == "always-yes":
                    vals = ["yes"] * len(q["options"])
                else:                    # the event happens when the key says correct, polarity aside
                    vals = ["yes" if ((ans == "correct") != q["negate"]) else "no"
                            for ans in t["answers"]]
                fh.write(json.dumps({"key": key_of(model, seed, prompt, names), "model": model,
                                     "seed": seed, "paths": names, "prompt": prompt,
                                     "response": json.dumps({"answers": vals})}) + "\n")
                n += 1
    return n


def main():
    qfile = sys.argv[1] if len(sys.argv) > 1 else "questions_testA.json"
    qfile = qfile if os.path.exists(qfile) else os.path.join(HERE, qfile)
    truth = S.load_truth(qfile)
    tmp = os.path.join(HERE, "_selftest.jsonl")
    scores = {}
    for mode in MODES:
        write(qfile, mode, tmp)
        for cond in S.CONDITIONS:
            said, info = S.load_answers(qfile, [tmp], cond)
            s = S.summary(said, truth)
            scores[(mode, cond)] = s
            print("%-10s %-5s %s | unreadable %d, unanswered %d"
                  % (mode, cond, S.fmt(s), info["unreadable"], info["missing"]))
    os.remove(tmp)
    o = scores[("oracle", "video")]
    assert o["options"] == o["n_options"], "the true answers must score 100%"
    assert o["questions"] == o["n_questions"]
    no, yes = scores[("always-no", "video")], scores[("always-yes", "video")]
    assert no["options"] + yes["options"] == no["n_options"], "yes and no must partition the options"
    print("ok: the true answers score 100%% and the two trivial policies split the %d options "
          "%d / %d" % (no["n_options"], no["options"], yes["options"]))


if __name__ == "__main__":
    main()
