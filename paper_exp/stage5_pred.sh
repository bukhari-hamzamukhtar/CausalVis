#!/usr/bin/env bash
# predictive questions with every simulator on TEST (VAL runs already done)
cd "G:/CausalVis"
qt () {
  for k in 0 1 2; do
    SIM=$4 PARSED=1 CF_LOOKBACK=3 python -u paper_exp/eval_qtypes.py --which $2 --qset $3 --shard $k --nshards 3 --only-predictive \
        --out paper_exp/qruns/$1 > paper_exp/qruns_$1_$k.log 2>&1 &
  done
  wait
}
for s in laws straight straight_friction; do qt test_A_pred_$s test val $s; qt test_B_pred_$s test train $s; done
echo "predictive TEST per simulator done $(date)" >> paper_exp/build.log
