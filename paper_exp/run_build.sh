#!/usr/bin/env bash
# detection-only tracks for every evaluation video (paper_exp/detonly.py)
cd "G:/CausalVis"
b () { python -u paper_exp/detonly.py build --which $1 --questions $2 --shard $3 --nshards 2 >> paper_exp/build_$1_$2_$3.log 2>&1; }
b val cf-val 0 & b val cf-val 1 & b test cf-val 0 & b test cf-val 1 & wait
echo "A sets done $(date)" >> paper_exp/build.log
b test cf-train 0 & b test cf-train 1 & b val cf-train 0 & b val cf-train 1 & wait
echo "B sets done $(date)" >> paper_exp/build.log
