#!/usr/bin/env bash
# after every track is rebuilt: perception parity, then the question-type pass on VAL-A,
# then physics quality of every simulator (the audit's x-axis)
cd "G:/CausalVis"
until grep -q "B sets done" paper_exp/build.log 2>/dev/null; do sleep 60; done
python -u paper_exp/perception_parity.py > paper_exp/perception_parity.log 2>&1
for k in 0 1 2; do
  PARSED=1 python -u paper_exp/eval_qtypes.py --which val --qset val --shard $k --nshards 3 --out paper_exp/qruns/val_A > paper_exp/qruns_val_A_$k.log 2>&1 &
done
wait
echo "qtypes VAL-A done $(date)" >> paper_exp/build.log
mkdir -p paper_exp/quality
q () { python -u paper_exp/physics_quality.py "$@" > /dev/null 2>> paper_exp/quality/errors.log; }
q --sim learned --model v6_voxel.pt --out paper_exp/quality/learned.json &
q --sim laws --out paper_exp/quality/laws.json &
wait
q --sim straight --out paper_exp/quality/straight.json &
q --sim straight_friction --out paper_exp/quality/strfric.json &
wait
for m in v3_6_3d v3_7_3d_frozen v3_8_impulse v5_noyaw v5b_noyaw v6_control; do
  q --sim learned --model $m.pt --out paper_exp/quality/ckpt_$m.json
done &
for m in fixed curriculum control; do
  q --sim learned --model v7_2/ckpt/$m.pt --out paper_exp/quality/ckpt_v72_$m.json
done
q --sim learned --no-voxel --out paper_exp/quality/abl_novoxel.json
q --sim learned --no-force --out paper_exp/quality/abl_noforce.json
q --sim learned --drag-scale 0 --out paper_exp/quality/abl_nodrag.json
wait
echo "physics quality done $(date)" >> paper_exp/build.log
