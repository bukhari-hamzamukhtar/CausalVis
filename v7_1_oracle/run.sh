#!/usr/bin/env bash
# Ceiling probe for the temporal-GNN idea: v7.1 with ORACLE hand-off states (future frames) on VAL,
# paired against v7.1's own VAL records. Test split untouched; this number is never reportable.
cd "G:/CausalVis"
python v7_1_oracle/make_oracle.py || exit 1
mkdir -p v7_1_oracle/runs/val
for k in 0 1 2 3; do
  ORACLE_HANDOFF=1 python -u v7_1_oracle/evaluate.py --which val --shard $k --nshards 4 --out v7_1_oracle/runs/val > v7_1_oracle/runs/val/log$k.txt 2>&1 &
done
wait
python v7/report.py v7_1_oracle/runs/val > v7_1_oracle/runs/val/report.txt 2>&1
python - > v7_1_oracle/runs/paired_vs_v7_1_val.txt <<'PY'
import sys, json, glob, math
from collections import defaultdict
sys.path.insert(0, "v7")
import report as R
def load(d):
    out = {}
    for f in glob.glob(d + "/shard*.jsonl"):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); out[(r["video"], str(r["qid"]), r["choice"])] = r
    return out
def ok(r, e):
    return (("correct" if R.happens(r, "cal", None, e) != r["negate"] else "wrong") == r["truth"])
new, old = load("v7_1_oracle/runs/val"), load("v7_1/runs/val")
keys = [k for k in new if k in old]
for e in (30, 45, 60):
    a = {k: ok(new[k], e) for k in keys}; b = {k: ok(old[k], e) for k in keys}
    x = sum(a[k] and not b[k] for k in keys); y = sum(b[k] and not a[k] for k in keys)
    z = (x - y) / math.sqrt(x + y) if x + y else 0.0
    print("cal +%d: oracle %d vs v7.1 %d of %d | oracle-only right %d, v7.1-only right %d, z = %.2f"
          % (e, sum(a.values()), sum(b.values()), len(keys), x, y, z))
qa = defaultdict(lambda: [True, True])
for k in keys:
    qa[(k[0], k[1])][0] &= ok(new[k], 30)
    qa[(k[0], k[1])][1] &= ok(old[k], 30)
print("per question, cal +30: oracle %d vs v7.1 %d of %d"
      % (sum(v[0] for v in qa.values()), sum(v[1] for v in qa.values()), len(qa)))
PY
grep -h ORACLE_STATS v7_1_oracle/runs/val/log*.txt >> v7_1_oracle/runs/paired_vs_v7_1_val.txt
echo DONE > v7_1_oracle/runs/DONE
