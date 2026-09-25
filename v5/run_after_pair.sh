#!/usr/bin/env bash
# Queued behind the --yaw run so the CPU is not oversubscribed.
cd "G:/CausalVis"
while [ ! -f v5/PAIR_DONE ]; do sleep 60; done
Q=zechennlp/counterfactual/validation-00000-of-00001.json

# 1. THE FIX: radius frozen at the measured value, on 3D data.
#    v3_6_3d let radius_corr collapse to 0.7000, so its physics used objects
#    30% smaller than the ones drawn and evaluated. That mismatch is what makes
#    non-touching objects register as collisions.
python -u src2/dynamics.py --data data/trajectories_3d --split split3d.json \
  --epochs 30 --batch 16 --threads 8 --dissipative --legacy-loss --freeze-radius \
  --collision-frac 0.0 --contact-weight 0 --out v3_7_3d_frozen.pt > v5/train_v3_7.log 2>&1

# 2. the honest benchmark, same rule as every other number
python -u v5/evaluate.py --model v3_7_3d_frozen.pt --data data/trajectories_3d \
  --split split3d.json --which test --questions $Q > v5/cmp/v3_7.log 2>&1
echo DONE > v5/V3_7_DONE
