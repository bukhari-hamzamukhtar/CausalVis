#!/usr/bin/env bash
# TEST, run once with the configuration chosen on VAL (A: detector cal, +30),
# plus the no-simulation baseline for attribution. 4 shards each, in parallel.
cd "G:/CausalVis"
mkdir -p v7/runs/test_A v7/runs/test_nosim
for k in 0 1 2 3; do
  python -u v7/evaluate.py --which test --shard $k --nshards 4 --out v7/runs/test_A          > v7/runs/test_A/log$k.txt 2>&1 &
  python -u v7/evaluate.py --which test --shard $k --nshards 4 --out v7/runs/test_nosim --no-sim > v7/runs/test_nosim/log$k.txt 2>&1 &
done
wait
python v7/report.py v7/runs/test_A cal 30      > v7/runs/test_A/report_cal30.txt 2>&1
python v7/report.py v7/runs/test_A             > v7/runs/test_A/grid.txt 2>&1
python v7/report.py v7/runs/test_nosim cal 0   > v7/runs/test_nosim/report_cal0.txt 2>&1
echo DONE > v7/runs/TEST_DONE
