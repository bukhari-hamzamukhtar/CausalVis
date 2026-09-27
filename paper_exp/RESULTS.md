# Results log (filled in as runs finish)

All CausalVis numbers: detection-only tracks, questions parsed from text, settings chosen
on the dev split named in each section, then scored once.

## CLEVRER counterfactual — VAL-A selection (941 q / 3,391 options), 2026-09-19
Every simulator got the same sweep (look-ahead 1/3/6/12 x detector x extension); best by the usual rule.
| simulator | look-ahead | detector, ext | options | questions |
|---|---|---|---|---|
| textbook laws (v8/laws.json) | 3 | union, +60 | 91.5% (3104) | 74.9% (705) |
| learned world model (v6_voxel) | 3 | union, +30 | 90.3% (3061) | 71.2% (670) |
| straight lines + friction | 1 | union, +45 | 89.9% (3047) | 71.0% (668) |
| straight lines | 1 | union, +45 | 89.3% (3028) | 69.2% (651) |
| no physics | – | union | 81.7% (2769) | 50.3% (473) |
FINDING: with the same hand-off, the fitted laws beat the learned model; straight lines +
friction are within 0.4 points of it. The earlier "learned beats laws" came from comparing
look-ahead 3 (learned) with 12 (laws). Removing the annotation leak cost the learned model
15 options (3076 -> 3061).

## CLEVRER counterfactual — TEST (pre-registered, run once), 2026-09-19
TEST-A = 500 CLEVRER-validation videos (930 q / 3,332 options); TEST-B = 898 fresh
CLEVRER-train videos (1,689 q / 6,018 options). Options % / questions % [95% video bootstrap].
| system | TEST-A | TEST-B (fresh) | A+B pooled | paired z vs learned (A+B) |
|---|---|---|---|---|
| textbook laws | 91.7 / 75.3 | 91.3 / 74.8 | 91.4 [90.8-92.1] / 75.0 [73.2-76.8] | +5.03 / +4.67 |
| learned world model | 90.2 / 71.4 | 90.5 / 72.8 | 90.4 [89.7-91.1] / 72.3 [70.5-74.2] | – |
| straight lines + friction | 90.1 / 71.4 | 90.2 / 72.2 | 90.2 / 71.9 | -0.88 / -0.49 (n.s.) |
| straight lines | 89.7 / 70.2 | 89.6 / 70.2 | 89.7 / 70.2 | -2.78 / -2.86 |
| no physics | 81.3 / 49.4 | 81.5 / 50.6 | 81.4 / 50.1 | -22.5 / -20.7 |
| learned, no extension past the end | 86.2 / 58.9 | 85.7 / 57.5 | 85.9 / 58.0 | -16.0 / -16.6 |
| learned, distance rule only | 89.9 / 70.8 | 90.2 / 71.9 | 90.1 / 71.5 | -4.52 / -3.66 |
TEST-A ablations (TEST-B still running): 12-frame hand-off 88.1 / 66.1 (z -5.16); no voxel
89.8 (z -1.17); no learned pair energy 90.5 (z +1.34); no learned friction 90.9 (z +2.14);
no entry correction 90.3 (z +0.77). Checkpoints (TEST-A): all 89.2-90.8.
Audit (4 simulators with both sets done): Pearson r(err@40, options) = -0.91, but options
span only 89.7-91.4 while err@40 spans 0.050-0.112 (2.2x).
Subsets (A+B): E 48.2% of questions (key says "collide" with none in the video on 10.1% of
their options, in 34.6% of E questions -- replicates on the fresh set); H 28.9%: learned
83.6 / 56.6 vs no physics 60.0 / 5.3.
Errors (896 wrong options, learned): invented during video 40%, missed after the end 31%,
invented after the end 16%, missed visible 10%, object not found by the detector 4%.

### Full TEST A+B table (paper_exp/final_report.txt), options / questions, z vs learned
| one change to the learned system | A+B | z |
|---|---|---|
| no extension past the video end | 85.9 / 58.0 | -16.0 |
| 12-frame hand-off (instead of 3) | 89.2 / 69.1 | -5.4 |
| distance rule only (no path change) | 90.1 / 71.5 | -4.5 |
| no voxel contact impulse | 89.6 / 70.3 | -3.7 |
| no learned pair energy | 90.8 / 73.4 | +3.2 (removing it HELPS) |
| no learned friction (eval-time) | 90.5 / 72.2 | +0.6 n.s. |
| no entry correction | 90.3 / 72.0 | -1.3 n.s. |
| seed 1 / seed 2 of the final fine-tune | 90.5 / 72.4 ; 90.1 / 71.3 | +0.8 ; -2.2 |
Other trained simulators: 89.2 (v5b_noyaw) .. 90.9 (v3_8_impulse).

### Seeds of the final fine-tune, six of them, 2026-09-27
Seeds 1 and 2 stopped after two of four epochs (laptop time); seeds 3 to 6 ran all four on a
free Kaggle CPU session (15 min per epoch there, `kaggle/`). Same recipe otherwise, scored
once each on TEST-A and TEST-B at the production system's setting (union, +30).
| checkpoint | epochs | val err@20 | TEST-A | TEST-B | A+B options | A+B questions | paired z vs production |
|---|---|---|---|---|---|---|---|
| v6_voxel (production) | 4 | 0.0230 | 90.2 / 71.4 | 90.5 / 72.8 | 90.4% (8454) | 72.3% | – |
| seed 1 | 2 of 4 | – | 90.3 / 71.6 | 90.6 / 72.8 | 90.5% (8464) | 72.4% | +0.8 |
| seed 2 | 2 of 4 | – | 90.1 / 71.0 | 90.1 / 71.5 | 90.1% (8424) | 71.3% | -2.2 |
| seed 3 | 4 | **0.0214** | 90.3 / 71.3 | 90.6 / 73.0 | 90.5% (8463) | 72.4% | +0.85 n.s. |
| seed 4 | 4 | 0.0253 | 90.1 / 71.1 | 90.2 / 71.9 | 90.2% (8430) | 71.6% | -1.60 n.s. |
| seed 5 | 4 | 0.0231 | 90.3 / 71.0 | 90.4 / 72.6 | 90.4% (8448) | 72.0% | -0.54 n.s. |
| seed 6 | 4 | 0.0239 | 90.2 / 71.0 | 89.9 / 70.8 | 90.0% (8414) | 70.8% | -2.63 |
Seven checkpoints: 90.3% +- 0.2 per option, range 90.0-90.5; questions 71.8% +- 0.6.
CAVEAT on the val column: `v6/train_voxel.py` draws its validation windows from seed+1, so
each seed's val error is measured on its own window draw. Relative to its own starting point
seed 3 improves 22% where the production checkpoint improved 13%, so it is the strongest
simulator this project has trained, but 0.0214 against 0.0230 is not a like-for-like pair.
FINDING: seed 3 trains clearly better and answers +9 options of 9,350 (z = +0.85). Across the
five checkpoints that have a val number, r(val rollout error, options) = -0.76 with the
benchmark spanning 0.5 points while the rollout error spans 18%. One more confirmation, with
freshly trained checkpoints rather than modified ones, that this benchmark barely sees the
simulator's accuracy.

## Vision-language baseline — TEST-A, run once, 2026-09-27
Qwen2.5-VL-7B-Instruct, 16 frames per clip, greedy, seed 0, asked directly whether each event
happens under the intervention (polarity applied when scoring). Same 3,332 options as the
engine, `vlm_baseline/`. 1,859 prompts (two videos give a byte-identical blind prompt, which
the answer cache correctly answers once). 4.4 s per answer on one T4.
| system | per option | per question | says "yes" |
|---|---|---|---|
| always answer "no" | 54.1% | 0.0% | 0% |
| Qwen2.5-VL, 16 frames | 54.4% (1811) [53.1-55.6] | 2.9% (27) | 9% |
| Qwen2.5-VL, blind (no frames) | 48.8% (1625) [47.4-50.1] | 4.2% (39) | 78% |
| always answer "yes" | 45.9% | 2.6% | 100% |
| CausalVis, learned world model | 89.9% (2996) | 70.8% (658) | – |
Paired on the 3,332 options: engine right where the VLM is wrong 1,362 times, the reverse 177,
z = -30.2 (questions 634 vs 3, z = -25.0).
FINDING: with the frames the model is indistinguishable from never predicting a collision
(54.4 against 54.1). It answers "no" to every event of 86% of questions when it sees frames
and "yes" to every event of 70% of them when it cannot, so it is reacting to whether images
are present rather than to the intervention. Two replies of 1,859 could not be read.
PRE-REGISTERED PREDICTION, half wrong: "well below the engine" held; "between the trivial
policies and the no-physics baseline (81.0)" did not, it landed at the trivial floor; and the
video-blind gap was 5.6 points where 2 was predicted.
AUDIT, 13 simulators: Pearson r(err@40, options) = -0.59; options span 89.2-91.4 while
err@40 spans 0.050-0.112.

## Other CLEVRER question types — TEST (parsed programs; settings from VAL-A)
| type | TEST-A | TEST-B | A+B | NS-DR | DCL | ALOE | VRDP†‡ |
|---|---|---|---|---|---|---|---|
| descriptive | 90.7 | 91.0 | 90.9 | 88.1 | 90.7 | 94.0 | 93.4 |
| explanatory opt / q | 88.9 / 80.0 | 88.1 / 79.5 | 88.4 / 79.7 | 87.6 / 79.6 | 89.6 / 82.8 | 98.5 / 96.0 | 96.3 / 91.9 |
| predictive opt / q | 91.3 / 83.7 | 90.1 / 82.0 | 90.5 / 82.6 | 82.9 / 68.7 | 90.5 / 82.0 | 93.5 / 87.5 | 95.7 / 91.4 |
Dataset programs instead of parsed: descriptive TEST-A 90.9 (parsed 90.7), others identical.

### Predictive questions with every simulator — TEST A+B (998 q / 1,996 options; settings per simulator on VAL)
| simulator | physics err@40 | options | questions | paired z vs learned |
|---|---|---|---|---|
| straight lines | 0.113 (worst) | 91.4 | 84.4 | +1.94 |
| straight + friction | 0.095 | 90.9 | 83.4 | +0.82 |
| learned | 0.063 | 90.5 | 82.6 | – |
| textbook laws | 0.050 (best) | 88.1 | 77.8 | -3.80 |
(VAL order was learned 90.7 > strfric 90.2 > straight 89.5 > laws 88.4.) No simulator is best on
both question types: laws win counterfactuals (+1.0, z 5.0) and lose predictive (-2.4, z -3.8).

## CLEVRER other question types — VAL-A (settings chosen here; TEST pending)
| type | CausalVis | NS-DR | DCL | ALOE | VRDP†‡ |
|---|---|---|---|---|---|
| descriptive | 90.7 | 88.1 | 90.7 | 94.0 | 93.4 |
| explanatory opt / q | 87.4 / 78.8 | 87.6 / 79.6 | 89.6 / 82.8 | 98.5 / 96.0 | 96.3 / 91.9 |
| predictive opt / q | 90.7 / 82.3 | 82.9 / 68.7 | 90.5 / 82.0 | 93.5 / 87.5 | 95.7 / 91.4 |
(published: official test set; ours: VAL-A. Settings: paper_exp/qruns/val_A/chosen_settings_v2.json)

## Physics quality (200 VAL-A videos) — the audit's x-axis
| simulator | err@20 | err@40 | replay 12 | replay 30 |
|---|---|---|---|---|
| laws | 0.0197 | 0.0502 | 0.973 | 0.696 |
| learned v6_voxel | 0.0237 | 0.0629 | 0.976 | 0.683 |
| v7_2 fixed | 0.0237 | 0.0621 | 0.976 | 0.644 |
| v7_2 curriculum | 0.0242 | 0.0640 | 0.976 | 0.626 |
| v3_6_3d | 0.0246 | 0.0673 | 0.963 | 0.569 |
| v3_7_3d_frozen | 0.0275 | 0.0714 | 0.961 | 0.574 |
| straight + friction | 0.0342 | 0.0945 | 0.949 | 0.512 |
| straight | 0.0382 | 0.1125 | 0.947 | 0.525 |

## CLEVRER-Humans (human causal judgments; 216 q / 432 options, validation)
Rule chosen on 102 grounded TRAIN videos (240 options): but-for 64.6, trace 58.8, yes 52.1.
| method | per option | per question |
|---|---|---|
| **CausalVis but-for (chosen on dev)** | **64.4** | **43.5** |
| CLEVRER's causal-chain rule on our events | 60.9 | 39.4 |
| always "responsible" | 50.7 | 30.1 |
| best published model (ALOE pretrain / CNN+LSTM scratch) | 54.0 | 34.2 |
| humans | 84.5 | 71.4 |
Caveats: event sentences grounded by a language model (Groq gpt-oss-120b, temperature 0,
cached in paper_exp/humans_ground_*.json); 150 of 183 videos lie in the world model's
training split (trajectories only, no labels). Published numbers are the paper's test.

## ComPhy (validation, 2,000 scenes; test server closed 2026-01-31)
Input: object tracks from ComPhy's annotation (ORACLE perception, visible frames only, first
5 s); masses and charges INFERRED from the target + 4 reference videos (never read). Laws
(friction, charge constant 0.138, contact, restitution) fitted on TRAIN annotations.
Settings chosen on 300 TRAIN scenes: reference videos on, CF +0 frames, predictive +50.
| method | CF per option | predictive per option | notes |
|---|---|---|---|
| **CausalVis engine (oracle tracks, inferred properties)** | **79.5** | **86.9** | val |
| CPL-Gt (ground-truth properties AND attributes/dynamics) | 74.0 | 87.6 | test |
| CPL | 68.3 | 75.3 | test |
| NS-DR+ (given ground-truth properties) | 61.1 | 73.3 | test |
| ALOE (Ref) | 67.9 | 67.9 | test |
| humans | 80.0 | 88.0 | test, sampled |
Per QUESTION is not comparable (val has 2 options per question, test mostly 4); ours:
CF 66.7, predictive 74.1. Hidden properties exactly right: mass scenes 98%, charge scenes 11%
(charged vs not detected in 99.6%; exact signs are hard), yet charge-scene CF is 79.7%.
Of 15,968 options, 371 name an object we could not resolve (answered "no").

## CoPhy BallsCF (test splits, 1,000 scenes each; laws fitted on the 4-ball TRAIN split)
Input: CoPhy's object states (the paper's state-based setting); confounders (mass,
restitution) INFERRED from the observed run AB, never read. Metric: squared 2D position error,
summed over x,y, averaged over time steps and balls (same metric for every row below).
| balls | CausalVis engine (inferred) | same, TRUE confounders | copy B | copy C |
|---|---|---|---|---|
| 2 | 0.89 | 0.42 | 7.69 | 12.58 |
| 3 | 1.56 | 0.72 | 6.81 | 12.22 |
| 4 | 2.03 | 1.09 | 6.18 | 11.88 |
| 5 | 2.39 | 1.28 | 5.93 | 11.65 |
| 6 | 2.60 | 1.55 | 5.74 | 11.26 |
CAVEAT: the released CoPhy files (Zenodo 3674790, md5 checked) differ from the paper's
description (masses 1/2/5, restitution 0.1/0.5/1 vs the paper's masses 1/10, friction 0.5/1);
no metric variant reproduces the paper's copy baselines (paper 4->4 copy B 2.688, ours 6.18), so
published CoPhyNet numbers are NOT comparable. Relative to copy B: ours 0.12-0.45 of it,
CoPhyNet (paper data) 0.71-0.76 of it -- indicative only.
CollisionCF / BlocktowerCF not run: objects fall and topple (gravity, 3D rotation), which the
planar engine does not model.

## Evidence audit: are answers right for the right reason? (TEST A+B, learned system)
paper_exp/audit_traces.py; breakdown in audit_breakdown_testAB.json. An answer's deciding event
can be checked against CLEVRER's annotation when both asked objects still follow the recorded
video at that moment (the annotation is never used to answer).
| correct answers (8,454) | count | share |
|---|---|---|
| decided by a recorded collision, annotation has it within 15 frames | 2,567 | 30.4% |
| decided by "no collision" over the whole recorded video, annotation agrees | 2,314 | 27.4% |
| recorded, not confirmed (40 annotated >15 frames away, 52 none) | 92 | 1.1% |
| decided in simulation (no ground truth exists) | 3,426 | 40.5% |
| asked object removed or not found | 55 | 0.7% |
Checkable correct answers resting on a real event: 4,881 / 4,973 = **98.2%** (15-frame
tolerance); 93.4% at 6 frames. Timing: our deciding frame is a median of 4 frames BEFORE the
annotated contact (p10 -7, p90 -2; n = 2,607), because the calibrated rule fires as surfaces
enter its margin. Compare Bourigault (2026): 35% of correct answers of mid-tier VLMs rest on
invalid traces.
| wrong answers (896) | count |
|---|---|
| detector reports a recorded collision the annotation lacks | 188 |
| detector misses an annotated collision | 37 |
| key needs a collision after the video end (both objects unaffected in the video) | 72 |
| other recorded cases | 8 |
| decided in simulation | 557 |
| object not found by the detector | 34 |
Observed-video events against the annotation (all collisions, TEST A+B): precision 0.788,
recall 0.951, median timing offset 0 frames after onset refinement (audit_events_observed.json).

## Explanatory questions by a counterfactual test (butfor_explanatory.py, score_butfor.py)
Remove what the candidate cause adds, simulate, call it responsible if the target collision
is gone. Extension chosen on VAL-A (+0: 76.8% vs +10: 76.4%, +30: 75.3%). TEST once:
| rule | TEST A+B options | questions |
|---|---|---|
| CLEVRER's chain rule (filter_ancestor) on our events | **88.4%** (7572/8566) | **79.7%** (1886/2366) |
| but-for test with the engine | 77.7% (6653/8566) | 61.5% (1455/2366) |
Paired: 111 vs 1030, z = -27.2. By candidate kind (but-for / chain): collision 78.3 / 98.1,
entrance 88.2 / 97.2, presence 85.7 / 95.6, unresolved 51.0 / 51.0 (same answer).
Direction (VAL-A): when the key says "responsible", removing the cause leaves the target
collision in our simulation 31% of the time; when it says "not responsible", removal kills the
target only 4.6% of the time.

## CLEVRER's "responsible" is not counterfactual dependence (key_consistency.py, model-free)
Same video, same objects: an explanatory option "Z (presence or entrance) is responsible for
the collision of X and Y" matched with a counterfactual option "without Z, X and Y collide".
No simulator involved; both answers are CLEVRER's own.
| set | matched pairs | key says responsible, yet X and Y still collide without Z | key says not responsible, yet X and Y do not collide without Z |
|---|---|---|---|
| VAL-A | 446 | 37 / 87 (42.5%) | 1 / 359 (0.3%) |
| TEST-A | 423 | 27 / 85 (31.8%) | 0 / 338 (0.0%) |
| TEST-B | 817 | 49 / 148 (33.1%) | 0 / 669 (0.0%) |
| all | 1,686 | **113 / 320 (35.3%)** | **1 / 1,366 (0.1%)** |
Reading: CLEVRER labels an object responsible when it is an ancestor in the event chain, and
in about a third of those cases its own counterfactual answer says the collision pair still
collides without that object. Caveat: CLEVRER's counterfactual answer does not say WHEN the
pair collides, so some of these may be a later, different collision of the same pair (checked
next with the row-level output). Together with CLEVRER-Humans (but-for 64.4 vs chain rule 60.9
per option, where humans are the reference), the rule that matches CLEVRER's key does not match
human judgments, and the rule that matches humans loses 10.7 points on CLEVRER's key.

## CoPhy event audit (cophy_balls.py audit): predicted vs true counterfactual collisions
BallsCF test, 300 scenes each, confounders inferred, same pair within +-1 step.
| balls | precision | recall | tp / fp / fn |
|---|---|---|---|
| 2 | 0.853 | 0.646 | 64 / 11 / 35 |
| 4 | 0.608 | 0.487 | 279 / 180 / 294 |
| 6 | 0.492 | 0.409 | 631 / 651 / 910 |

## Why physics barely moves the score (sim_agreement.py, TEST A+B, 9,350 options)
The four simulators (learned, laws, straight, straight + friction) at their VAL-chosen
settings give the SAME answer on 8,544 options (91.4%); all four are right on 93.7% of those.
All differences between simulators come from the other 806 options (8.6%): laws right on
67.5%, learned 55.7%, straight + friction 53.1%, straight 47.0%. At least one asked object is
simulated before the video ends on 47.8% of options, yet the answers rarely depend on which
simulator was used. No physics gives the same answer as the four on 87.2% of the agreed options.

## ComPhy predictive evidence audit (comphy.py audit + comphy_audit_score.py, 400 val scenes)
ComPhy annotates the future (frames 125-174), so every predictive answer is checkable.
1,597 options with both objects resolved; 1,411 correct. Of the correct answers, 97.7% rest on
a real event (589 simulated collisions matching a real one within 5 frames, 789 absences that
match; 33 simulated collisions with no real counterpart). Within 3 frames 95.1%, within 10
98.7%. Wrong 186: missed a real future collision 152, invented 8, key says collide but the
annotated future window has none 26. First simulated future collision per asked pair:
precision 0.935 (589/630), recall 0.769 (589/766).

## Is the "responsible, yet still collides" collision the same one? (key_consistency_timing.py)
All 113 cases: the pair collides exactly ONCE in the recorded video. In the learned engine's
world without Z: first collision within 10 frames of the recorded one 56, >10 frames earlier 23,
>10 frames later 14, no collision 20. So at least half are the same collision surviving the
removal of the object CLEVRER calls responsible.

## Runtime, clean (runtime.py; one process, one thread, idle CPU, 40 TEST-A questions)
learned 2.24 s per question, laws 1.36, straight 1.40, no physics 1.04 (start-up excluded).
