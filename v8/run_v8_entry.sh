#!/usr/bin/env bash
# v8 + entry correction: VAL -> choose setting -> pre-registered bar -> TEST once -> paired vs v8.
# Bar (fixed before seeing any number): must beat plain v8's best VAL score, 3055/3391.
cd "G:/CausalVis"
mkdir -p v8/runs/val_entry
for k in 0 1 2 3; do
  python -u v8/evaluate.py --which val --shard $k --nshards 4 --entry-fix --out v8/runs/val_entry > v8/runs/val_entry/log$k.txt 2>&1 &
done
wait
python v7/report.py v8/runs/val_entry > v8/runs/val_entry/report.txt 2>&1

python - > v8/runs/choice_entry.json <<'PY'
import sys, json, glob
sys.path.insert(0, "v7")
import report as R
recs = [json.loads(l) for f in sorted(glob.glob("v8/runs/val_entry/shard*.jsonl")) for l in open(f, encoding="utf-8")]
best = None
for det, tol in R.DETECTORS:
    for e in R.EXTENDS:
        ok, n, qok, qn = R.score(recs, det, tol, e)
        if best is None or ok > best[0]:
            best = (ok, n, det, tol, e, qok, qn)
ok, n, det, tol, e, qok, qn = best
name = det if tol is None else "voxel<=%d" % tol
print(json.dumps({"det": name, "ext": e, "val_ok": ok, "val_n": n, "val_q_ok": qok, "val_q_n": qn,
                  "v8_val_ok": 3055, "beats_v8": ok > 3055}))
PY

BEATS=$(python -c "import json; print(json.load(open('v8/runs/choice_entry.json'))['beats_v8'])")
DET=$(python -c "import json; print(json.load(open('v8/runs/choice_entry.json'))['det'])")
EXT=$(python -c "import json; print(json.load(open('v8/runs/choice_entry.json'))['ext'])")

if [ "$BEATS" = "True" ]; then
  mkdir -p v8/runs/test_entry
  for k in 0 1 2 3; do
    python -u v8/evaluate.py --which test --shard $k --nshards 4 --entry-fix --out v8/runs/test_entry > v8/runs/test_entry/log$k.txt 2>&1 &
  done
  wait
  python v7/report.py v8/runs/test_entry "$DET" "$EXT" > v8/runs/test_entry/report_chosen.txt 2>&1
  python - "$DET" "$EXT" > v8/runs/paired_entry_vs_v8.txt <<'PY'
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
new, old = load("v8/runs/test_entry"), load("v8/runs/test")
keys = [k for k in new if k in old]
a = {k: ok(new[k], det, tol, ext) for k in keys}
b = {k: ok(old[k], "cal", None, 30) for k in keys}
only_new = sum(a[k] and not b[k] for k in keys); only_old = sum(b[k] and not a[k] for k in keys)
print("matched options %d" % len(keys))
print("v8+entry (%s, +%d): %d correct   v8 (cal, +30): %d correct" % (det_name, ext, sum(a.values()), sum(b.values())))
print("v8+entry right & v8 wrong: %d   v8 right & v8+entry wrong: %d   McNemar z = %.2f"
      % (only_new, only_old, (only_new - only_old) / math.sqrt(only_new + only_old) if only_new + only_old else 0.0))
PY
  echo TESTED > v8/runs/DONE_ENTRY
else
  echo "NOT TESTED: the entry correction did not beat v8 on validation" > v8/runs/DONE_ENTRY
fi
