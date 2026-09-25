#!/usr/bin/env bash
cd "G:/CausalVis"
while [ ! -f v5/PAIR2_DONE ]; do sleep 60; done      # behind the orientation pair
Q=zechennlp/counterfactual/validation-00000-of-00001.json
# v3_8: identical to v3_7 (3D data, drag, radius frozen) plus the contact
# impulse. ONE variable. The potential can now learn to hand contact over to
# the impulse instead of fighting it.
PYTHONPATH=v5 python -u src2/dynamics.py --data data/trajectories_3d --split split3d.json \
  --epochs 30 --batch 16 --threads 8 --dissipative --legacy-loss --freeze-radius --impulse \
  --collision-frac 0.0 --contact-weight 0 --out v3_8_impulse.pt > v5/train_v3_8.log 2>&1
PYTHONPATH=v5 python -u v5/evaluate.py --model v3_8_impulse.pt --data data/trajectories_3d \
  --split split3d.json --which test --questions $Q > v5/cmp/v3_8.log 2>&1
echo DONE > v5/V3_8_DONE
