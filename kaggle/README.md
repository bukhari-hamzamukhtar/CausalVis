# Extra training seeds on a free Kaggle CPU session

The paper reports the production checkpoint plus two extra seeds of the same fine-tune, and
both of those were stopped after two of the four epochs because the laptop had no time for
more. A free Kaggle CPU session runs for twelve hours, which is enough for all four epochs, so
the seed spread can be measured on the recipe as written.

One variable: the seed. Data, split, init, epochs, horizon, batch size, learning rate and the
voxel contacts are what `paper_exp/stage3_seeds.sh` used. No GPU is requested: the model has
43,460 parameters and the time goes into the voxel contact check, which is CPU work, so the
GPU quota stays free for the VLM baseline in `vlm_baseline/`.

Measured on the laptop with the same code copy: 2.6 s per batch, 699 batches per epoch, so
about half an hour per epoch and two to three hours for a full run.

## Build and push

```bash
python kaggle/prepare_seeds.py --user bukharihamzamukhtar --seeds 3,4,5,6 --clips
kaggle datasets create -p kaggle/build/code  --dir-mode zip     # 1.0 MB, once
kaggle datasets create -p kaggle/build/clips --dir-mode zip     # 178 MB, once
KAGGLE_API_TOKEN=... bash kaggle/run_kaggle.sh causalvis-train-seed3
```

`--clips` copies the 11,382 clips the trainer reads (the 11,182 train clips and the 200 val
clips it selects on) into the build folder. Leave it off on later builds; the clip dataset does
not change. Later builds of the code use `kaggle datasets version -p kaggle/build/code -m
"update" --dir-mode zip`.

Each seed is its own kernel, so they can run at the same time. `run_kaggle.sh` retries the push
until Kaggle accepts it, waits, and downloads the checkpoint into
`kaggle/outputs/causalvis-train-seed3/`.

## Score a seed that came back

Copy the checkpoint to the repository root, add two lines to `paper_exp/jobs_test.txt`, and run
the queue. The answers for a job that already has a `DONE` file are kept, so nothing is
recomputed:

```
A_seed3 | PARSED=1 CF_LOOKBACK=3 | --model v6_voxel_seed3.pt --data data/trajectories_3d_det | test | cf-val
B_seed3 | PARSED=1 CF_LOOKBACK=3 | --model v6_voxel_seed3.pt --data data/trajectories_3d_det | test | cf-train
```

```bash
python paper_exp/runq.py paper_exp/jobs_test.txt --parallel 2
python paper_exp/score.py one A_seed3 cal 30
```

Every seed is scored at the setting chosen on validation for the production checkpoint
(`cal`, +30 frames), which is what the existing seed rows in `paper_exp/RESULTS.md` use. Test
answers are read once, at that setting, and nothing is chosen from them.

## What is already known

| checkpoint            | epochs run | test A+B options / questions |
|-----------------------|-----------:|------------------------------|
| v6_voxel (production) | 4          | 90.4 / 72.3                  |
| seed 1                | 2 of 4     | 90.5 / 72.4                  |
| seed 2                | 2 of 4     | 90.1 / 71.3                  |

So the spread across seeds is about 0.4 points per option. New full-length seeds either
confirm that or widen it; either way the paper can say how many seeds it rests on.
