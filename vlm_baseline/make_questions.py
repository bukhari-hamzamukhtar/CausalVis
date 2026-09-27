"""
vlm_baseline/make_questions.py  —  the question set the VLM baseline answers
===========================================================================

Exports exactly the counterfactual options our engine is scored on, so the two can be
compared option by option (same keys as paper_exp/score.py: video, question id, choice text,
occurrence of that text inside the question).

    python vlm_baseline/make_questions.py --which test --out vlm_baseline/questions_testA.json

Writes two files:
  <out>            what the GPU job needs: video id, question id, question text, options
  <out>.truth.json the answers, kept on the laptop for scoring
"""

import argparse
import glob
import re
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for d in ("src2", "v5", "v7"):
    sys.path.insert(0, os.path.join(ROOT, d))
from benchmark_eval import load_questions, load_split_set, _vid_num   # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "paper_exp"))
import qparser                                                        # noqa: E402

QF = os.path.join(ROOT, "zechennlp", "counterfactual", "validation-00000-of-00001.json")


def intervention_of(question):
    """The change the question asks about, as a plain clause.

    CLEVRER phrases it three ways:
      "Which of the following will not happen if the yellow cube is removed?"
      "If the red cylinder is removed, which event will not happen?"
      "Without the sphere, which of the following will happen?"
    Anything unexpected falls back to the question itself, and make_questions reports how
    many of those there are.
    """
    q = " ".join(question.strip().split())
    m = re.match(r"(?:if|suppose)\s+(.+?)\s*,", q, re.I)          # clause first
    if m:
        return m.group(1).strip()
    m = re.match(r"without\s+(.+?)\s*,", q, re.I)
    if m:
        return "%s is removed" % m.group(1).strip()
    m = re.search(r"\bwithout\s+(.+?)\s*\?*$", q, re.I)          # "... without the cube?"
    if m:
        return "%s is removed" % m.group(1).strip().rstrip("?.")
    m = re.search(r"\bif\s+(.+?)\s*\?*$", q, re.I)               # clause last
    if m:
        return m.group(1).strip().rstrip("?.")
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="test")
    ap.add_argument("--questions", default=QF)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "trajectories_3d_det"))
    ap.add_argument("--split", default=os.path.join(ROOT, "split3d_yaw.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "questions_testA.json"))
    a = ap.parse_args()

    have = {_vid_num(f) for f in glob.glob(os.path.join(a.data, "*.npz"))}
    allowed = load_split_set(a.split, a.which)
    qs, truth = [], []
    seen = {}
    for row in load_questions(a.questions):
        v = _vid_num(str(row.get("video_filename", row.get("video", ""))))
        if v is None or v not in allowed or v not in have:
            continue
        ch = row["choices"]
        texts, answers = list(ch["choice"]), list(ch["answer"])
        occ = []
        for t in texts:
            k = (v, str(row["question_id"]), t)
            seen[k] = seen.get(k, -1) + 1
            occ.append(seen[k])
        q_text = row.get("question", "")
        prog = qparser.parse(q_text) or []
        negate = "negate" in prog
        qs.append({"video": v, "qid": str(row["question_id"]), "question": q_text,
                   "intervention": intervention_of(q_text), "negate": negate,
                   "options": texts})
        truth.append({"video": v, "qid": str(row["question_id"]), "options": texts,
                      "occ": occ, "answers": answers, "negate": negate})
    json.dump(qs, open(a.out, "w", encoding="utf-8"), indent=1)
    json.dump(truth, open(a.out + ".truth.json", "w", encoding="utf-8"), indent=1)
    bad = [q for q in qs if re.search(r"\b(which|what|following|happen)\b", q["intervention"], re.I)]
    if bad:
        print("interventions that did not parse cleanly: %d" % len(bad))
        for q in bad[:5]:
            print("   ", q["question"], "->", q["intervention"])
    n_opt = sum(len(q["options"]) for q in qs)
    print("videos %d | questions %d | options %d" % (len({q["video"] for q in qs}), len(qs), n_opt))
    print("wrote", a.out, "and", a.out + ".truth.json")


if __name__ == "__main__":
    main()
