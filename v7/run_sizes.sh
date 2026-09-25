#!/usr/bin/env bash
# Measured object sizes (v7/measured_sizes.json) on VAL, two ways, every detector logged.
cd "G:/CausalVis"
run() {
  mkdir -p v7/runs/$1
  for k in 0 1 2 3; do
    python -u v7/evaluate.py --which val --shard $k --nshards 4 --out v7/runs/$1 $2 > v7/runs/$1/log$k.txt 2>&1 &
  done
  wait
  python v7/report.py v7/runs/$1 > v7/runs/$1/report.txt 2>&1
}
run val_sizes     "--sizes v7/measured_sizes.json"
run val_sizes_dyn "--sizes v7/measured_sizes.json --dynamics-sizes"
echo DONE > v7/runs/SIZES_DONE
