#!/usr/bin/env bash
# Finish the hand-off sweep on VAL (1, 3, 9 frames), choose the best VAL run of the whole
# sweep by the usual rule, and only if it beats v7.1's VAL 3024 run TEST once, paired vs v7.1.
cd "G:/CausalVis"
run () {  # dir, env assignment, split
  mkdir -p v7_3/runs/$1
  for k in 0 1 2 3; do
    env $2 python -u v7_3/evaluate.py --which $3 --shard $k --nshards 4 --out v7_3/runs/$1 > v7_3/runs/$1/log$k.txt 2>&1 &
  done
  wait
}
run val_lb3 CF_LOOKBACK=3 val
run val_lb1 CF_LOOKBACK=1 val
run val_lb9 CF_LOOKBACK=9 val
python v7_3/choose.py > v7_3/runs/choice.txt 2>&1
BEATS=$(python -c "import json; print(json.load(open('v7_3/runs/choice.json'))['beats_bar'])")
if [ "$BEATS" = "True" ]; then
  ENVSTR=$(python -c "import json; c=json.load(open('v7_3/runs/choice.json')); print(' '.join('%s=%s' % kv for kv in c['env'].items()))")
  DET=$(python -c "import json; c=json.load(open('v7_3/runs/choice.json')); print(c['det'] if c['tol'] is None else 'voxel<=%d' % c['tol'])")
  EXT=$(python -c "import json; print(json.load(open('v7_3/runs/choice.json'))['ext'])")
  run test_chosen "$ENVSTR" test
  python v7/report.py v7_3/runs/test_chosen "$DET" "$EXT" > v7_3/runs/test_chosen/report_chosen.txt 2>&1
  python - "$DET" "$EXT" > v7_3/runs/paired_vs_v7_1_test.txt <<'PY'
import sys, json, glob, math
from collections import defaultdict
sys.path.insert(0, "v7")
import report as R
det_name, ext = sys.argv[1], int(sys.argv[2])
tol = float(det_name.split("<=")[1]) if det_name.startswith("voxel") else None
det = "voxel" if det_name.startswith("voxel") else det_name
def load(d):
    out = {}
    for f in glob.glob(d + "/shard*.jsonl"):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); out[(r["video"], str(r["qid"]), r["choice"])] = r
    return out
def ok(r, d, t, e):
    return (("correct" if R.happens(r, d, t, e) != r["negate"] else "wrong") == r["truth"])
new, old = load("v7_3/runs/test_chosen"), load("v7_1/runs/test")
keys = [k for k in new if k in old]
a = {k: ok(new[k], det, tol, ext) for k in keys}
b = {k: ok(old[k], "cal", None, 30) for k in keys}
qa, qb = defaultdict(lambda: True), defaultdict(lambda: True)
for k in keys:
    qa[(k[0], k[1])] &= a[k]; qb[(k[0], k[1])] &= b[k]
x = sum(a[k] and not b[k] for k in keys); y = sum(b[k] and not a[k] for k in keys)
print("matched options %d" % len(keys))
print("chosen (%s +%d): %d options, %d questions   v7.1 (cal +30): %d options, %d questions"
      % (det_name, ext, sum(a.values()), sum(qa.values()), sum(b.values()), sum(qb.values())))
print("chosen right & v7.1 wrong: %d   v7.1 right & chosen wrong: %d   McNemar z = %.2f"
      % (x, y, (x - y) / math.sqrt(x + y) if x + y else 0.0))
PY
  echo TESTED > v7_3/runs/DONE2
else
  echo "NOT TESTED: nothing beat v7.1 on VAL" > v7_3/runs/DONE2
fi
