"""
paper_exp/score_butfor.py  —  score CLEVRER explanatory questions two ways on the same options
============================================================================================

  trace    CLEVRER's chain rule (filter_ancestor) run on our observed events
  but-for  remove the candidate cause, simulate, responsible if the target collision is gone;
           candidates that cannot be removed without removing the target (same pair, or a
           participant of the target) and unresolved ones keep the trace answer

Extension past the video end for the target check: chosen on VAL-A (most options), then
applied once to TEST.

    python paper_exp/score_butfor.py val_A            # prints every extension
    python paper_exp/score_butfor.py test_A test_B --ext E
"""

import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    recs = []
    for f in sorted(glob.glob(os.path.join(HERE, "butfor", name, "shard*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                r["set"] = name
                recs.append(r)
    return recs


def butfor(r, ext):
    if "target_after" not in r:
        return r["trace"]
    responsible = not r["target_after"][str(ext)]
    return "correct" if responsible != r["negate"] else "wrong"


def score(recs, fn):
    ok = [fn(r) == r["truth"] for r in recs]
    q = defaultdict(lambda: True)
    for r, o in zip(recs, ok):
        q[(r["set"], r["video"], r["qid"])] &= o
    return sum(ok), len(ok), sum(q.values()), len(q), ok


def mcnemar(a, b):
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    return n10, n01, (n10 - n01) / math.sqrt(n10 + n01) if n10 + n01 else 0.0


def main():
    argv = sys.argv[1:]
    ext = None
    if "--ext" in argv:
        i = argv.index("--ext")
        ext = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    args = argv
    recs = [r for n in args for r in load(n)]
    print("%s: %d options" % ("+".join(args), len(recs)))
    kinds = Counter(r["kind"] for r in recs)
    print("candidate kinds:", dict(kinds))
    t = score(recs, lambda r: r["trace"])
    print("  trace (chain rule)       %.1f%% (%d/%d)  questions %.1f%% (%d/%d)" % (100 * t[0] / t[1], t[0], t[1], 100 * t[2] / t[3], t[2], t[3]))
    res = {"n_options": t[1], "n_questions": t[3], "trace": [t[0], t[2]], "kinds": dict(kinds)}
    for e in ([ext] if ext is not None else (0, 10, 30)):
        b = score(recs, lambda r: butfor(r, e))
        n10, n01, z = mcnemar(b[4], t[4])
        print("  but-for, +%2d frames      %.1f%% (%d/%d)  questions %.1f%% (%d/%d)   vs trace: %d vs %d, z %+.2f"
              % (e, 100 * b[0] / b[1], b[0], b[1], 100 * b[2] / b[3], b[2], b[3], n10, n01, z))
        res["butfor_%d" % e] = [b[0], b[2], n10, n01, round(z, 2)]
        for k in sorted(kinds):
            sub = [r for r in recs if r["kind"] == k]
            bs = sum(butfor(r, e) == r["truth"] for r in sub)
            ts = sum(r["trace"] == r["truth"] for r in sub)
            print("      %-15s n %5d   but-for %.1f%%   trace %.1f%%" % (k, len(sub), 100 * bs / len(sub), 100 * ts / len(sub)))
            res["kind_%s_%d" % (k, e)] = [len(sub), bs, ts]
    json.dump(res, open(os.path.join(HERE, "butfor", "score_%s.json" % "+".join(args)), "w"), indent=1)


if __name__ == "__main__":
    main()
