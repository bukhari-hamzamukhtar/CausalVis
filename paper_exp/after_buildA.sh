#!/usr/bin/env bash
# wait for the VAL-A / TEST-A tracks, then run the VAL-A selection queue (4 processes)
cd "G:/CausalVis"
until grep -q "A sets done" paper_exp/build.log 2>/dev/null; do sleep 60; done
python -u paper_exp/runq.py paper_exp/jobs_val_det.txt --parallel 1 > paper_exp/queue_val_det.log 2>&1
echo "VAL selection queue done $(date)" >> paper_exp/build.log
