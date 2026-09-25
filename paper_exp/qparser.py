"""
paper_exp/qparser.py  —  question text -> program, learned from TRAIN questions only
=================================================================================

CLEVRER questions are generated from templates, so a question with its attribute
words (colours, materials, shapes) replaced by type-free slots identifies its program
skeleton; each (value, filter_x) pair in a program becomes (slot, FILTER), and the
filter is re-derived from the value's type when parsing. Which text slot feeds which
program position is decided by votes over all TRAIN examples of the template (a
value that appears once in the text pins its slot). Parsing: slot the text, look the
template up, fill the values in. Unknown template -> no program (the question is
then answered as an error, i.e. counted wrong).
Validation exact-match: counterfactual 100%, predictive 100%, explanatory 99.995%,
descriptive 99.80% (88 of 54,990 texts have no TRAIN template).

    python paper_exp/qparser.py fit                   # writes paper_exp/parser_templates.json
    python paper_exp/qparser.py check --qset val      # exact-match vs the dataset programs
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src2"))
from benchmark_eval import load_questions                                # noqa: E402

COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'yellow', 'cyan', 'purple']
MATERIALS = ['metal', 'rubber']
SHAPES = ['sphere', 'cylinder', 'cube']
PLURAL = {'spheres': 'sphere', 'cylinders': 'cylinder', 'cubes': 'cube'}
TYPE = {**{c: 'C' for c in COLORS}, **{m: 'M' for m in MATERIALS}, **{s: 'S' for s in SHAPES}}
FILES = {"counterfactual": "zechennlp/counterfactual/%s-00000-of-00001.json",
         "descriptive": "zechennlp/descriptive/%s-00000-of-000012.json",
         "explanatory": "zechennlp/explanatory/%s-00000-of-00001 (2).json",
         "predictive": "zechennlp/predictive/%s-00000-of-00001 (3).json"}
OUT = os.path.join(HERE, "parser_templates.json")


FILTER = {"C": "filter_color", "M": "filter_material", "S": "filter_shape"}


def slot_text(text):
    """-> (text with every attribute word replaced by a type-free slot, [(type, value)])."""
    words = re.findall(r"[A-Za-z]+|[^A-Za-z\s]", text.strip().lower())
    slots, out = [], []
    for w in words:
        base = PLURAL.get(w, w)
        if base in TYPE:
            slots.append((TYPE[base], base))
            out.append("<Ap>" if w in PLURAL else "<A>")
        else:
            out.append(w)
    return " ".join(out), slots


def skeleton(prog):
    """Program with each (value, filter_x) pair made generic -> (skeleton, values)."""
    sk, vals, i = [], [], 0
    while i < len(prog):
        tok = prog[i]
        if tok in TYPE and i + 1 < len(prog) and prog[i + 1] == FILTER[TYPE[tok]]:
            sk += ["@", "FILTER"]; vals.append(tok); i += 2
        elif tok in TYPE:
            sk.append("@"); vals.append(tok); i += 1
        else:
            sk.append(tok); i += 1
    return sk, vals


def pairs(rows):
    for r in rows:
        yield r["question"], r["program"]
        ch = r.get("choices")
        if isinstance(ch, dict) and ch.get("program"):
            for t, p in zip(ch["choice"], ch["program"]):
                yield t, p


def fit():
    skel = defaultdict(Counter)                  # text template -> skeleton counts
    votes = defaultdict(lambda: defaultdict(Counter))   # (template, skeleton) -> position -> slot votes
    for t, f in FILES.items():
        for text, prog in pairs(load_questions(os.path.join(ROOT, f % "train"))):
            st, slots = slot_text(text)
            sk, vals = skeleton(prog)
            key = json.dumps(sk)
            skel[st][key] += 1
            for pos, v in enumerate(vals):
                for si, (_, sv) in enumerate(slots):
                    if sv == v:
                        votes[(st, key)][pos][si] += 1
    tpl = {}
    for st, c in skel.items():
        key = c.most_common(1)[0][0]
        npos = json.loads(key).count("@")
        vt = votes[(st, key)]
        tpl[st] = {"skeleton": json.loads(key), "slots": [vt[p].most_common(1)[0][0] if vt[p] else None for p in range(npos)]}
    json.dump(tpl, open(OUT, "w"))
    print("templates %d (text templates with more than one skeleton: %d) -> %s"
          % (len(tpl), sum(1 for c in skel.values() if len(c) > 1), OUT))


_TPL = None


def parse(text):
    global _TPL
    if _TPL is None:
        _TPL = json.load(open(OUT))
    st, slots = slot_text(text)
    t = _TPL.get(st)
    if t is None:
        return None
    out, k = [], 0
    for tok in t["skeleton"]:
        if tok == "@":
            si = t["slots"][k]; k += 1
            if si is None or si >= len(slots):
                return None
            out.append(slots[si][1])
        elif tok == "FILTER":
            out.append(FILTER[TYPE[out[-1]]])
        else:
            out.append(tok)
    return out


def check(qset):
    part = {"val": "validation", "train": "train"}[qset]
    for t, f in FILES.items():
        n = ok = miss = 0
        for text, prog in pairs(load_questions(os.path.join(ROOT, f % part))):
            p = parse(text)
            n += 1
            ok += (p == prog)
            miss += (p is None)
        print("%-15s %7d texts  exact program %.3f%%  no template %d" % (t, n, 100.0 * ok / n, miss))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fit", "check"])
    ap.add_argument("--qset", default="val")
    a = ap.parse_args()
    fit() if a.mode == "fit" else check(a.qset)
