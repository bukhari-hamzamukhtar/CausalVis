#!/usr/bin/env bash
# after VAL-A selection: apply the choices (make_test_jobs.py), run every pre-registered
# counterfactual system on TEST-A and TEST-B, then the other question types on TEST.
cd "G:/CausalVis"
until grep -q "VAL selection queue done" paper_exp/build.log 2>/dev/null; do sleep 60; done
python paper_exp/make_test_jobs.py > paper_exp/make_test_jobs.log 2>&1
python -u paper_exp/runq.py paper_exp/jobs_test.txt --parallel 2 > paper_exp/queue_test.log 2>&1
echo "TEST counterfactual queue done $(date)" >> paper_exp/build.log
qt () {  # out which qset sim extra
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
