#!/usr/bin/env bash
# v7.2: fine-tune v7's world model (v6_voxel.pt) on LONG simulations.
# Three arms in parallel, same init / clip order / seed / lr / 2000-batch budget / checkpoint rule;
# ONLY the rollout-length schedule differs (v7_2/train_long.py):
#   fixed       random 40-60 frames per batch
#   curriculum  40 -> 50 -> 60 -> 70 -> 80, next length only when val stops improving
#   control     20 frames (v6's length), isolates "trained more" from "trained longer"
# Then each arm's best checkpoint runs inside v7.1's world builder (entry fix on) on VAL, the setting is
# chosen by the usual rule, and the bar fixed now applies: beat v7.1's VAL 3024/3391 to reach TEST.
cd "G:/CausalVis"
mkdir -p v7_2/ckpt v7_2/runs
python -u v7_2/train_long.py --schedule fixed:40-60 --budget 2000 --threads 3 --out v7_2/ckpt/fixed.pt > v7_2/train_fixed.log 2>&1 &
python -u v7_2/train_long.py --schedule curriculum:40,50,60,70,80 --budget 2000 --threads 3 --out v7_2/ckpt/curriculum.pt > v7_2/train_curriculum.log 2>&1 &
python -u v7_2/train_long.py --schedule fixed:20-20 --budget 2000 --threads 2 --out v7_2/ckpt/control.pt > v7_2/train_control.log 2>&1 &
wait
echo TRAINED > v7_2/TRAIN_DONE

for arm in fixed curriculum control; do
  mkdir -p v7_2/runs/val_$arm
  for k in 0 1 2 3; do
    python -u v7_1/evaluate.py --model v7_2/ckpt/$arm.pt --which val --shard $k --nshards 4 --out v7_2/runs/val_$arm > v7_2/runs/val_$arm/log$k.txt 2>&1 &
  done
  wait
  python v7/report.py v7_2/runs/val_$arm > v7_2/runs/val_$arm/report.txt 2>&1
done
python v7_2/decide.py choose > v7_2/runs/choice.txt 2>&1

for arm in $(python -c "import json; print(' '.join(json.load(open('v7_2/runs/choice.json'))['test']))"); do
  mkdir -p v7_2/runs/test_$arm
  for k in 0 1 2 3; do
    python -u v7_1/evaluate.py --model v7_2/ckpt/$arm.pt --which test --shard $k --nshards 4 --out v7_2/runs/test_$arm > v7_2/runs/test_$arm/log$k.txt 2>&1 &
  done
  wait
  DET=$(python -c "import json; c=json.load(open('v7_2/runs/choice.json'))['arms']['$arm']; print(c['det'] if c['tol'] is None else 'voxel<=%d' % c['tol'])")
  EXT=$(python -c "import json; print(json.load(open('v7_2/runs/choice.json'))['arms']['$arm']['ext'])")
  python v7/report.py v7_2/runs/test_$arm "$DET" "$EXT" > v7_2/runs/test_$arm/report_chosen.txt 2>&1
done
python v7_2/decide.py paired > v7_2/runs/paired.txt 2>&1
echo DONE > v7_2/runs/DONE
