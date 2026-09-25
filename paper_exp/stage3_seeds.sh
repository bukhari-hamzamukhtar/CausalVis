#!/usr/bin/env bash
# seeds 1 and 2 of the voxel fine-tune: same recipe as v6_voxel.pt (init v5b_noyaw, voxel
# contacts in training, 4 epochs, horizon 20, lr 3e-4, best of epochs on val_err@20)
cd "G:/CausalVis"
until grep -q "VAL selection queue done" paper_exp/build.log 2>/dev/null; do sleep 120; done
for s in 1 2; do
  python -u v6/train_voxel.py --init v5b_noyaw.pt --voxel --epochs 4 --threads 2 --seed $s \
      --out v6_voxel_seed$s.pt > paper_exp/train_seed$s.log 2>&1 &
done
wait
echo "seeds done $(date)" >> paper_exp/build.log
