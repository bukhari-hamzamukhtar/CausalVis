# Settings, model size and compute (for the paper's appendix tables)

## Hardware
One laptop CPU: Intel i5-8350U (4 cores, 8 threads), 7.9 GB RAM. No GPU was used at any
stage. Every number in the paper was produced on this machine.

## World model (v6_voxel.pt)
| part | shape | parameters |
|---|---|---|
| attribute encoder | 8 -> 128 -> 64 | 9,408 |
| mass head | 64 -> 128 -> 1 | 8,449 |
| pair potential energy V | 68 -> 128 -> 128 -> 1 | 25,473 |
| translational friction head | 64 -> 1 | 65 |
| rotational friction head | 64 -> 1 | 65 |
| total used | | **43,460** |
| radius head (frozen at zero, kept only so old checkpoints load) | 64 -> 128 -> 1 | 8,449 |
Force = -grad V by automatic differentiation; leapfrog integration with exact friction
decay half-steps (Strang splitting); voxel contact impulse, voxel size 0.02; radius equals
the measured radius; rotation off (yaw_known = False).

## Training
| stage | data | schedule | optimiser | CPU time |
|---|---|---|---|---|
| base (v5b_noyaw) | 11,182 TRAIN clips, detection tracks | 30 epochs, rollout horizon 3 -> 20 (+1 per epoch) | AdamW, lr 1e-3, batch 16, gradient clip 1.0 | 5.2 h |
| voxel fine-tune (v6_voxel) | same | 4 epochs, horizon 20, voxel contacts in the rollout; best epoch on val rollout error (epoch 0) | AdamW, lr 3e-4, batch 16 | 2.9 h |
| seeds 1 and 2 | same | same recipe, stopped after 2 of 4 epochs by a power loss; best epoch 0 in both | same | 4 h each |
No question or answer is used in any training stage.

## Textbook-law simulator (v8/laws.json, fitted on TRAIN)
Cubes and cylinders slide with constant deceleration 0.00008 per frame; spheres roll with
0.3% (metal) or 0.2% (rubber) speed loss per frame; bounce along the line of centres,
restitution 0.2 for every material pair, metal/rubber mass ratio 1.00, contact friction 0.05,
contact when surfaces are within 0.01.

## Counterfactual engine (paper_exp/cf_world.py)
| setting | value | chosen on |
|---|---|---|
| hand-off look-ahead | 3 frames before a lost contact (laws 3; straight lines 1) | VAL-A sweep {1, 3, 6, 12} |
| velocity at hand-off | mean of the last 3 recorded frames | fixed |
| entry correction | TRAIN profile over the first 15 frames after entry | TRAIN |
| collision detector | calibrated distance rule (radius x 1.20, +0.03) OR path change (surfaces within 0.03 and a velocity change of at least 0.002 within 6 frames) | VAL-A |
| hysteresis release | 0.02 | fixed |
| extension past the video end | +30 frames (laws +60, straight lines +45) | VAL-A |
| object leaves the table | position beyond 0.95 | fixed |
| world units | positions normalised by 6.0 | fixed |

## Question-type executor (paper_exp/executor.py; settings chosen on VAL-A)
| type | detector | other settings |
|---|---|---|
| descriptive | calibrated rule, onset moved to closest approach within 6 frames | moving if speed > 0.001 (mean over +-4 frames at a given frame; 15 consecutive frames for "ever moving"); "end" = frame 125; one-collision tie-break by smallest gap |
| explanatory | calibrated rule, same onset refinement | official chain rule (filter_ancestor) |
| predictive | plain distance rule (radius x 1.00, +0.02) | +60 frames of simulation past the end |

## Perception (paper_exp/detonly.py)
NS-DR's released detections (Mask R-CNN masks with predicted colour, material, shape).
Non-maximum suppression within the same colour and shape at 8 px; two objects share colour
and shape only if two detections appear at least 20 px apart in 3 or more frames; tracks
shorter than 5 frames are dropped; gaps up to 5 frames are interpolated; material by
score-weighted majority. 3D position from the recovered camera (camera_fit.json, silhouette
IoU 0.93).

## Runtime (answering), paper_exp/runtime.py -> runtime.json
One process, torch 1 thread, nothing else running, 40 TEST-A questions, start-up excluded:
| simulator | seconds per counterfactual question |
|---|---|
| learned world model | 2.24 |
| textbook laws | 1.36 |
| straight lines | 1.40 |
| no physics | 1.04 |
(During the TEST runs, with 8 processes sharing the CPU, it was about 7 / 5 / 4 / 3 s.)
