#!/usr/bin/env bash
# v8 end to end: VAL -> choose setting -> pre-registered bar -> TEST once -> paired vs v7.
# Bar (fixed before seeing any v8 number): v8 must beat v7's best VAL score,
# 3017/3391 (cal, +30), to be scored on TEST at all.
cd "G:/CausalVis"
mkdir -p v8/runs/val
for k in 0 1 2 3; do
  python -u v8/evaluate.py --which val --shard $k --nshards 4 --out v8/runs/val > v8/runs/val/log$k.txt 2>&1 &
done
wait
python v7/report.py v8/runs/val > v8/runs/val/report.txt 2>&1

python - > v8/runs/choice.json <<'PY'
import sys, json, glob
sys.path.insert(0, "v7")
import report as R
recs = [json.loads(l) for f in sorted(glob.glob("v8/runs/val/shard*.jsonl")) for l in open(f, encoding="utf-8")]
best = None
for det, tol in R.DETECTORS:
    for e in R.EXTENDS:
        ok, n, qok, qn = R.score(recs, det, tol, e)
        if best is None or ok > best[0]:
            best = (ok, n, det, tol, e, qok, qn)
ok, n, det, tol, e, qok, qn = best
name = det if tol is None else "voxel<=%d" % tol
print(json.dumps({"det": name, "ext": e, "val_ok": ok, "val_n": n, "val_q_ok": qok, "val_q_n": qn,
                  "v7_val_ok": 3017, "beats_v7": ok > 3017}))
PY

BEATS=$(python -c "import json; print(json.load(open('v8/runs/choice.json'))['beats_v7'])")
DET=$(python -c "import json; print(json.load(open('v8/runs/choice.json'))['det'])")
EXT=$(python -c "import json; print(json.load(open('v8/runs/choice.json'))['ext'])")

if [ "$BEATS" = "True" ]; then
  mkdir -p v8/runs/test
  for k in 0 1 2 3; do
    python -u v8/evaluate.py --which test --shard $k --nshards 4 --out v8/runs/test > v8/runs/test/log$k.txt 2>&1 &
  done
  wait
  python v7/report.py v8/runs/test "$DET" "$EXT" > v8/runs/test/report_chosen.txt 2>&1
  python - "$DET" "$EXT" > v8/runs/paired_vs_v7.txt <<'PY'
import sys, json, glob, math
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
v8, v7 = load("v8/runs/test"), load("v7/runs/test_A")
keys = [k for k in v8 if k in v7]
a = {k: ok(v8[k], det, tol, ext) for k in keys}
b = {k: ok(v7[k], "cal", None, 30) for k in keys}
only8 = sum(a[k] and not b[k] for k in keys); only7 = sum(b[k] and not a[k] for k in keys)
print("matched options %d" % len(keys))
print("v8 (%s, +%d): %d correct   v7 (cal, +30): %d correct" % (det_name, ext, sum(a.values()), sum(b.values())))
print("v8 right & v7 wrong: %d   v7 right & v8 wrong: %d   McNemar z = %.2f"
      % (only8, only7, (only8 - only7) / math.sqrt(only8 + only7) if only8 + only7 else 0.0))
PY
  echo TESTED > v8/runs/DONE
else
  echo "NOT TESTED: v8 did not beat v7 on validation" > v8/runs/DONE
fi
