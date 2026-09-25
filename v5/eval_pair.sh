#!/usr/bin/env bash
cd "G:/CausalVis"
while [ ! -f v5/V3_7_DONE ]; do sleep 60; done       # wait for the frozen-radius run
Q=zechennlp/counterfactual/validation-00000-of-00001.json
D=data/trajectories_3d_yaw
python -u v5/evaluate.py --model v5_noyaw.pt --spatial            --data $D --split split3d.json --which test --questions $Q > v5/cmp/v5_noyaw.log 2>&1
python -u v5/evaluate.py --model v5_yaw.pt   --spatial --yaw-known --data $D --split split3d.json --which test --questions $Q > v5/cmp/v5_yaw.log 2>&1
echo DONE > v5/PAIR_EVAL_DONE
