#!/usr/bin/env bash
# restart after the power loss: finish the TEST queue (DONE jobs are skipped), then the
# other question types on TEST and the predictive runs per simulator
cd "G:/CausalVis"
python -u paper_exp/runq.py paper_exp/jobs_test.txt --parallel 2 >> paper_exp/queue_test.log 2>&1
echo "TEST counterfactual queue done $(date)" >> paper_exp/build.log
qt () {
  for k in 0 1 2; do
    SIM=$4 PARSED=1 CF_LOOKBACK=3 python -u paper_exp/eval_qtypes.py --which $2 --qset $3 --shard $k --nshards 3 $5 \
        --out paper_exp/qruns/$1 > paper_exp/qruns_$1_$k.log 2>&1 &
  done
  wait
}
qt test_A test val learned ""
qt test_B test train learned ""
for s in laws straight straight_friction; do qt val_A_pred_$s val val $s --only-predictive; done
echo "qtypes TEST + predictive VAL per simulator done $(date)" >> paper_exp/build.log
