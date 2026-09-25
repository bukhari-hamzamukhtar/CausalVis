#!/usr/bin/env bash
cd "G:/CausalVis"
while [ ! -f v5/PAIR_EVAL_DONE ]; do sleep 60; done   # behind v3_7 and the first eval
Q=zechennlp/counterfactual/validation-00000-of-00001.json
D=data/trajectories_3d_yaw
S=split3d_yaw.json
# THE PROPER ORIENTATION TEST: radius frozen (no 30%-too-small cubes),
# 11k clips instead of 3.2k, identical data both arms, one variable.
python -u v5/train.py --data $D --split $S --freeze-radius --no-yaw --epochs 30 --threads 8 --out v5b_noyaw.pt > v5/train2_noyaw.log 2>&1
python -u v5/train.py --data $D --split $S --freeze-radius --yaw    --epochs 30 --threads 8 --out v5b_yaw.pt   > v5/train2_yaw.log 2>&1
python -u v5/evaluate.py --model v5b_noyaw.pt --spatial             --data $D --split $S --which test --questions $Q > v5/cmp/v5b_noyaw.log 2>&1
python -u v5/evaluate.py --model v5b_yaw.pt   --spatial --yaw-known --data $D --split $S --which test --questions $Q > v5/cmp/v5b_yaw.log 2>&1
echo DONE > v5/PAIR2_DONE
