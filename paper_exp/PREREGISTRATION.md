# Pre-registration — final CausalVis evaluation (written 2026-09-18, before any TEST-B look)

Everything below is fixed BEFORE the fresh test set (TEST-B) is scored. Settings are
chosen on VAL-A only. TEST-A and TEST-B are then scored once per listed system; no
change of any kind is made after that. Anything not listed here, if reported, is
labelled exploratory.

## Inputs (identical for every system)
- Tracks: detection-only (paper_exp/detonly.py): NS-DR's released detections, no
  annotation at test time (object list, in/out frames and cube-orientation breakpoints
  all from detections). data/trajectories_3d_det.
- Questions: programs from paper_exp/qparser.py (fitted on CLEVRER TRAIN questions);
  the dataset's ground-truth programs are reported only as a check.
- Every option in the question file is scored; an option naming an object the
  detector did not find is answered "does not collide".

## Sets
- VAL-A: 500 CLEVRER-validation videos in our val split (941 counterfactual questions).
- TEST-A: 500 CLEVRER-validation videos in our test split (930). Consulted earlier.
- TEST-B: 898 CLEVRER-train videos in our test split (1,689). Never scored before.
  No trajectory of any val/test video was used to train anything; no question or
  answer was ever used for training.

## Selection rule on VAL-A (the usual one)
Maximise options right; ties -> questions right -> first listed.
- Simulators S in {learned (v6_voxel.pt), laws (v8/laws.json), straight, straight+friction}:
  hand-off look-ahead in {1, 3, 6, 12} x detector in {rule, cal, kick2, kick4, union} x
  extension in {0, 10, 20, 30, 45, 60}.
- No physics (delete the object, keep the recording): detector in the same set.
- Descriptive / explanatory / predictive executor settings (paper_exp/score_qtypes.py
  grid), per question type.

## Systems scored on TEST-A and TEST-B (once each, at their VAL-chosen setting)
1. CausalVis (learned simulator) — the main system.
2. Textbook laws; 3. straight lines; 4. straight lines + friction; 5. no physics.
6. Ablations of 1 at 1's setting, one change each: no voxel impulse; no learned pair
   energy; no learned friction (eval-time); no entry correction; no extension past the
   video end; 12-frame hand-off; distance rule only (cal) instead of the union.
7. Other trained simulators at 1's setting: v3_6_3d, v3_7_3d_frozen, v3_8_impulse,
   v5_noyaw, v5b_noyaw, v6_control, v7_2 fixed / curriculum / control.
8. Seeds: the voxel fine-tune repeated with seeds 1 and 2 (same recipe), at 1's setting.
9. Descriptive, explanatory, predictive with the executor (predictive with simulators
   1-4).
10. System 1 with the dataset's ground-truth programs (parser check).

## Reported statistics
Per option and per question; 95% bootstrap intervals resampling videos (2,000,
seed 0); paired McNemar z against system 1 on matched options. Primary number: TEST-B;
also TEST-A and A+B pooled. Subset table (ALOE App. C style) and error breakdown for 1.

## Addition, 2026-09-27: vision-language baseline and full-length seeds
Written before either run. Both use free compute (Kaggle sessions); nothing here changes a
setting of systems 1-10, and no number below is chosen on a test set.

11. VISION-LANGUAGE BASELINE. Qwen2.5-VL-7B-Instruct, 16 frames per clip spread evenly over
    the video, greedy decoding, seed 0, prompter as in VLP (text first, then images, at most
    224x224 pixels each). Exactly the options of TEST-A (3,332 options, 930 questions), scored
    by the same keys as system 1. Two conditions, both run: `video` (frames shown) and `blind`
    (question only, no frames). The model is asked whether each event happens under the
    intervention; the question's polarity is applied when scoring, so the number is not a test
    of how a language model handles the word "not". A reply that cannot be read counts as "no"
    and the count of those is reported. Nothing is tuned: the frame count, the prompt and the
    decoding are fixed before the run, and the validation set is used only to check that the
    job runs.
    PREDICTION: `video` lands between the two trivial policies (54.1% / 45.9% per option) and
    the no-physics baseline (81.0%), well below system 1 (89.9% on TEST-A), and `blind` lands
    within about 2 points of `video`, because the answer depends on what happens after an
    intervention that the frames do not show. If `blind` matches `video`, the baseline is
    reading the wording, not the video, and that is the finding to report.
    Code: `vlm_baseline/`. Scorer checked against answers whose score is known in advance
    (`vlm_baseline/selftest.py`: the true answers score 100%, and the two trivial policies
    split the options 1,803 / 1,529).

12. FULL-LENGTH SEEDS. The voxel fine-tune repeated with seeds 3, 4, 5, 6, same recipe as
    system 1 and all four epochs (seeds 1 and 2 stopped after two epochs for lack of laptop
    time). Each is scored once on TEST-A and TEST-B at system 1's setting.
    CORRECTION, written the same day, after seed 3 was scored: this line first named that
    setting "cal, +30", which is the wrong name for it. System 1's setting, chosen on VAL-A,
    is the UNION detector (distance rule OR path change) at +30 frames, and that is what the
    seed rows in the results table use. Both readings were computed for seed 3 and they give
    the same answer (union 90.5 / 72.4, cal 90.2 / 71.6 against the production checkpoint's
    90.4 / 72.3 and 90.1 / 71.5); the union numbers are the comparable ones.
    PREDICTION: the spread stays near the 0.4 points per option seen across the production
    checkpoint and seeds 1 and 2. Reported as the seed range in the results table, with the
    number of seeds and how many epochs each ran.
    Code: `kaggle/`.
