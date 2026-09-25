#!/usr/bin/env bash
# three world-building choices on VAL, 4 shards each, one config at a time
cd "G:/CausalVis"
run() {  # name, extra flags
  mkdir -p v7/runs/$1
  for k in 0 1 2 3; do
    python -u v7/evaluate.py --which val --shard $k --nshards 4 --out v7/runs/$1 $2 > v7/runs/$1/log$k.txt 2>&1 &
  done
  wait
  python v7/report.py v7/runs/$1 > v7/runs/$1/report.txt 2>&1
}
run val_A ""
run val_B "--no-lost-affected"
run val_S "--strict"
echo DONE > v7/runs/VAL_DONE
