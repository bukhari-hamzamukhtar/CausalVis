#!/usr/bin/env bash
# VAL only. Four single-variable runs on top of v7.1, then a paired comparison.
# Bar for any later TEST run (fixed now): beat v7.1's VAL 3024/3391 by the usual rule.
cd "G:/CausalVis"
python v7_3/make_v7_3.py || exit 1
run () {  # name, env assignment
  mkdir -p v7_3/runs/$1
  for k in 0 1 2 3; do
    env $2 python -u v7_3/evaluate.py --which val --shard $k --nshards 4 --out v7_3/runs/$1 > v7_3/runs/$1/log$k.txt 2>&1 &
  done
  wait
}
run val_phys12 PHYS_RADIUS_SCALE=1.2
run val_lb6 CF_LOOKBACK=6
run val_lb30 CF_LOOKBACK=30
run val_lb20 CF_LOOKBACK=20
python v7_3/compare.py > v7_3/runs/compare.txt 2>&1
echo DONE > v7_3/runs/DONE
