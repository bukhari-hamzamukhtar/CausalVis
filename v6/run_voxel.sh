#!/usr/bin/env bash
# Voxel contacts INSIDE training, one variable, then the full test benchmark.
cd "G:/CausalVis"
Q=zechennlp/counterfactual/validation-00000-of-00001.json
D=data/trajectories_3d_yaw
S=split3d_yaw.json
python -u v6/train_voxel.py --init v5b_noyaw.pt --voxel    --epochs 4 --threads 4 --out v6_voxel.pt   > v6/train_voxel.log   2>&1 &
P1=$!
python -u v6/train_voxel.py --init v5b_noyaw.pt --no-voxel --epochs 4 --threads 4 --out v6_control.pt > v6/train_control.log 2>&1 &
P2=$!
wait $P1 $P2
# 1 trained WITH voxels, evaluated with voxels     (the new model)
# 2 same fine-tune WITHOUT voxels                  (control: isolates "trained longer")
# 3 production model + fixed voxel normal at eval  (replaces the buggy-normal 74.5%)
OMP_NUM_THREADS=3 python -u v5/evaluate.py --model v6_voxel.pt   --spatial --voxel --data $D --split $S --which test --questions $Q > v6/cmp/v6_voxel_trained.log 2>&1 &
OMP_NUM_THREADS=2 python -u v5/evaluate.py --model v6_control.pt --spatial         --data $D --split $S --which test --questions $Q > v6/cmp/v6_control.log 2>&1 &
OMP_NUM_THREADS=3 python -u v5/evaluate.py --model v5b_noyaw.pt  --spatial --voxel --data $D --split $S --which test --questions $Q > v6/cmp/v5b_voxel_evalonly_fixed.log 2>&1 &
wait
echo DONE > v6/VOXEL_TRAIN_DONE
