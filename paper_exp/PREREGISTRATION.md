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
