#!/usr/bin/env bash
cd "G:/CausalVis"
r () { for k in 0 1 2 3; do PARSED=1 CF_LOOKBACK=3 python -u paper_exp/butfor_explanatory.py --which $2 --qset $3 --shard $k --nshards 4 --out paper_exp/butfor/$1 > paper_exp/butfor_$1_$k.log 2>&1 & done; wait; }
r val_A val val
r test_A test val
r test_B test train
echo "butfor done $(date)" >> paper_exp/build.log
