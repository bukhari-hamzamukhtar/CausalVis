"""Pick the best VAL run of the v7_3 sweep by the usual rule (most VAL options right over
every detector x extension), then say whether it beats the pre-registered bar: v7.1's
VAL 3024/3391. Writes v7_3/runs/choice.json.
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "v7"))
import report as R                                                      # noqa: E402

BAR = 3024
RUNS = {"lookback 12 (v7.1)": ("v7_1/runs/val", {}),
        "lookback 1": ("v7_3/runs/val_lb1", {"CF_LOOKBACK": "1"}),
        "lookback 3": ("v7_3/runs/val_lb3", {"CF_LOOKBACK": "3"}),
        "lookback 6": ("v7_3/runs/val_lb6", {"CF_LOOKBACK": "6"}),
        "lookback 9": ("v7_3/runs/val_lb9", {"CF_LOOKBACK": "9"}),
        "lookback 20": ("v7_3/runs/val_lb20", {"CF_LOOKBACK": "20"}),
        "lookback 30": ("v7_3/runs/val_lb30", {"CF_LOOKBACK": "30"}),
        "physics radius x1.2": ("v7_3/runs/val_phys12", {"PHYS_RADIUS_SCALE": "1.2"})}


def main():
    rows = []
    for name, (d, env) in RUNS.items():
        recs = [json.loads(l) for f in sorted(glob.glob(os.path.join(ROOT, d, "shard*.jsonl")))
                for l in open(f, encoding="utf-8")]
        if not recs:
            continue
        best = None
        for det, tol in R.DETECTORS:
            for e in R.EXTENDS:
                ok, n, qok, qn = R.score(recs, det, tol, e)
                if best is None or ok > best[0]:
                    best = (ok, n, det, tol, e, qok, qn)
        ok, n, det, tol, e, qok, qn = best
        rows.append({"name": name, "dir": d, "env": env, "det": det, "tol": tol, "ext": e,
                     "val_ok": ok, "val_n": n, "val_q_ok": qok, "val_q_n": qn})
        print("%-22s best %s +%d: %d/%d (%.1f%%), questions %d/%d (%.1f%%)"
              % (name, det if tol is None else "voxel<=%d" % tol, e, ok, n, 100.0 * ok / n, qok, qn, 100.0 * qok / qn))
    top = max(rows, key=lambda r: r["val_ok"])
    top["beats_bar"] = top["val_ok"] > BAR and top["name"] != "lookback 12 (v7.1)"
    top["bar"] = BAR
    json.dump(top, open(os.path.join(ROOT, "v7_3", "runs", "choice.json"), "w"), indent=1)
    print("\nchosen on VAL: %s (%d); bar %d; run TEST: %s" % (top["name"], top["val_ok"], BAR, top["beats_bar"]))


if __name__ == "__main__":
    main()
