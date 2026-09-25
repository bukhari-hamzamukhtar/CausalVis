# CausalVis v5 — state as of 2026-09-09

## What is here

| file | what it does | status |
|---|---|---|
| `world.py` | `SpatialWorld`: v3 physics + shape-aware contact (support functions) + rotational state with torque by autograd + rotational drag | built, 5 structural checks pass |
| `train.py` | trains `SpatialWorld` on yaw-fitted 3D data; `--yaw` / `--no-yaw` for a fair pair on the same clips | built, smoke-tested (NaN-free after two fixes) |
| `evaluate.py` | benchmark with **no copied answers** — ground truth is initial conditions only, every choice is simulated | **77.7% per-choice, 100% model-decided** (full test split) |
| `pipeline.py` | one question end to end: VLM parse → taint rollout → `RolloutStore` → VLM answer → GIF | built; VLM steps blocked on `ANTHROPIC_API_KEY` |
| `demo/` | question 40 / video 10020: store JSON+text, GIF (observed vs predicted) | produced |

## Measured

**Taint-design benchmark, full test split (930 q / 3332 choices):**

| | per-choice | model-decided | n model-decided |
|---|---|---|---|
| v3_2, pixel data | 86.1% | 70.9% | 595 |
| v3_5 drag, pixel data | — | 69.9% | 581 |
| **v3_6_3d drag, 3D data** | **86.9%** | **81.2%** | 320 |

Caveat: 3D geometry changes which pairs get tainted, so the model-decided
subset is a different (smaller) set of choices. The +10 pp is real on each
model's own subset but not a matched comparison. `simulated hit` went
74.5% → 88.1%.

**Demo (video 10020, "which will NOT happen if the cube is removed"):**
model simulated 262 object-frames; predicted future contains exactly one
collision (blue cylinder → brown sphere, frame 108) and no green–blue
collision, which matches both ground-truth labels. The VLM answer itself is
blocked on credentials.

## Three structural guarantees in `SpatialWorld` (all measured)
- linear momentum of the interaction: 4.2e-07 over 300 steps
- **total angular momentum (orbital + spin): 3.6e-07** — new, free from the autograd design
- yaw unknown → torque identically zero (no orientation is asserted)

## Two NaN bugs found in `V()` with yaw known, both r = 0 in `atan2`
absent–absent padded pairs, and the diagonal (i,i). The presence mask zeroes
the forward, not the gradient: NaN·0 = NaN. Fixed by giving masked pairs a
fixed unit direction *including the diagonal*.

## Gates
1. **API key** for the two VLM steps — `console.anthropic.com` → API Keys →
   `$env:ANTHROPIC_API_KEY = "..."`. Pro is not API.
2. **Yaw data** for the orientation-aware model — `build_trajectories_3d.py
   --with-yaw` running → `data/trajectories_3d_yaw` (1.35 s/clip, ~7.5 h total;
   the extractor now stores the averaged cube support directly, no post-pass).
   When ~4000 clips exist run the fair pair:
   ```
   python v5/train.py --data data/trajectories_3d_yaw --split split3d.json --limit 3200 --no-yaw --out v5_noyaw.pt
   python v5/train.py --data data/trajectories_3d_yaw --split split3d.json --limit 3200 --yaw    --out v5_yaw.pt
   ```
   Same clips, same epochs, one variable.

## What "cheating" means here, precisely
v3's evaluator used ground truth two ways with one mechanism: as the initial
state of simulated objects (legitimate) and as the *answer* for pairs outside
the cone (the crutch — 100% on 831 choices, model never asked). `evaluate.py`
keeps the first and removes the second: the cone only decides *when* each
simulation starts; every asked-about pair is simulated by the model from a
short observed window. Expect the score to drop; that drop is the crutch's
size.

## CORRECTED HONEST BENCHMARK, three models, clean logs (2026-09-09)
Rule: ground truth = initial conditions only. Factual pairs: window around
observed closest approach. Counterfactual (tainted) pairs: from cone entry to
END of clip. Full test split, 930 questions / 3328 choices, 100% model-decided.

| model     | data  | per-choice | in-cone (counterfactual) | outside-cone (factual) | per-question |
|-----------|-------|-----------:|-------------------------:|-----------------------:|-------------:|
| v3_2      | pixel |  68.0%     | 64.5% (1385/2148)        | 74.5% (879/1180)       | 21.7%        |
| v3_5 drag | pixel |  67.3%     | 63.5% (1363/2148)        | 74.4% (878/1180)       | 20.4%        |
| v3_6 drag | 3D    | **74.2%**  | 65.6% (935/1425)         | **80.6%** (1533/1903)  | 29.7%        |

- **3D data helps: +6.2 pp, z = 5.5** (p << 0.001). The gain is on the FACTUAL
  pairs (74.5 -> 80.6) -- short windows where geometry decides; the
  counterfactual long-horizon pairs are flat (64.5 vs 65.6).
- Composition caveat: better geometry also shrinks the cone (2148 -> 1425
  in-cone pairs), and outside-cone is the easier category. Re-weighted to
  v3_2's composition v3_6 scores 70.9%: **+2.9 pp is the
  like-for-like gain**, the rest is the cone being more accurate on 3D data.
- **Drag does nothing on the benchmark, again: z = -0.60.** Confirmed under
  both evaluators now. Binary collision questions do not see trajectory
  accuracy. The 25%/28% rollout/collision-error improvements are real and this
  benchmark is blind to them.
- The earlier 77.7% (flawed fixed-30-frame window) was INFLATED: tainted pairs
  could not see late events, defaulted to "no", and "no" is the majority label.
  Corrected in-cone went 74.0 -> 65.6. **Report 74.2%.**

Reportable headline: v3 light-cone pipeline 86.9% (18% model-decided) vs the
same model with no copied answers 74.2% (100% model-decided). The 12.7-point
gap is the crutch; the 74.2% is the world model.

## The honest windowing rule (settled 2026-09-09, three iterations)
For a pair the question asks about:
- **neither member tainted** (factual = counterfactual for them): simulate the
  whole scene from observed state at `closest_approach - 15` to
  `closest_approach + 15`. The recording says when they get near; the model
  says whether they touch.
- **a member is tainted**: the event time is UNKNOWABLE from the recording --
  the observed closest approach lies on a path the intervention erased -- so
  simulate from the tainted member's cone entry (its last true state) to the
  END of the clip. This is exactly the July taint design.
Two earlier versions were wrong: a fixed 30-frame horizon (could not reach
events later than the cone entry -- biased toward "no collision"), then
`closest_approach + 15` for tainted pairs (anchored on an erased path).

`pipeline.py` now stores BOTH a single continuous rollout (for the GIF) and a
focused re-simulation of EVERY surviving pair (no ground-truth pair
selection). The VLM is told the focused table is the more reliable evidence.

Demo under the honest rule: row 0 (video 10000) 3/3 vs ground truth, the
focused window catching a frame-84 collision the long rollout drifted past;
row 40 (video 10020) 1/2 -- the frame-108 collision is a genuine miss.

## Data facts that cost time to learn (2026-09-09)
- **Counterfactual questions reference videos 10000-14999 only.** Trajectory
  clips sim_00000-09999 have no questions. Any benchmark run restricted to
  clips < 10000 scores zero questions -- not a bug, no questions exist there.
- The yaw extractor walks files in order, so a question-range instance was
  launched separately: `--start 10000 --end 15000` -> `build3d_yaw_q.log`.
  Writes are atomic (tmp + os.replace) so two instances can meet safely.
- `v5/evaluate.py --spatial [--yaw-known]` is the ONLY path that exercises
  orientation-aware physics. `model.step` (used by the non-spatial evaluator
  and by `lightcone_rollout`) forces yaw off for v3 compatibility.
- The training pair's own `val_err@20` (rollout error, `rollout_spatial`) is a
  valid trajectory-level comparison on its own and needs no questions.

## Pending (in flight)
- `v5_noyaw.pt` then `v5_yaw.pt`: 3200 train clips (< 4000), 30 epochs each ->
  `v5/train_noyaw.log`, `v5/train_yaw.log`, `v5/PAIR_DONE`.
- then: `python v5/evaluate.py --model v5_noyaw.pt --spatial             --data data/trajectories_3d_yaw --split split3d.json --which test --questions <val json>`
        `python v5/evaluate.py --model v5_yaw.pt   --spatial --yaw-known --data data/trajectories_3d_yaw --split split3d.json --which test --questions <val json>`

## Orientation pair result: NEGATIVE on trajectory error (2026-09-09)
Identical 3200 clips (< 4000), 30 epochs, one conceptual change.

| arm                       | best val err @20 |
|---------------------------|-----------------|
| `v5_noyaw.pt` control     | **0.0249**      |
| `v5_yaw.pt` orientation   | 0.0270 (8% worse)|

Shape-aware contact + rotational state + torque did NOT improve trajectory
prediction. Three caveats, all stated BEFORE the numbers came in:

1. **Both arms carry the collapsed radius** (`r/r_meas 0.7000`). The yaw arm
   derives a cube's half-side from that radius, so it reasons about cubes 30%
   too small -- the orientation signal is applied to the wrong geometry.
2. **Trajectory error is the wrong metric for this change.** Shape-aware
   contact only acts during contact; averaging position error over every frame
   dilutes it. The collision benchmark is the right test and is queued.
3. The yaw arm adds a yaw loss term, so strictly this bundles three things.
   train.py argues they are inseparable (no torque without a yaw-dependent
   potential, no yaw supervision without a yaw loss) -- defensible, but it is
   still not a single knob.

Do not conclude "orientation does not help" from this alone. Conclude
"orientation did not help TRAJECTORY ERROR, on collapsed-radius geometry."

## The three-radius bug (2026-09-09) -- found by looking at a GIF
`v3_6_3d` was trained with the radius correction free and it collapsed to
0.7000, as every such run does. Result, measured on sim_10020:

| quantity                          | objects touch at (world units) |
|-----------------------------------|-------------------------------|
| what render3d draws               | 0.700                         |
| what the PHYSICS uses             | **0.490** (30% smaller)       |
| what the collision test fires at  | 0.820 (contact + 0.02 thresh) |

So a collision is recorded with 0.330 world units of clear air between the
objects -- 0.94x an object's radius. The render, the physics and the scoring
were describing three different objects. Accuracy had already been measured as
insensitive to the collapse; VISUAL CONSISTENCY was never checked, and that is
what exposed it. `v3_7_3d_frozen.pt` (--freeze-radius on 3D data) is the fix
and is training.

Lesson: "benign for the metric" is not "benign". Look at the picture.


## THE REAL BUG: no collision mechanics (2026-09-10)
The bounce angle was measured against the recorded trajectory, split by shape:

    sphere-sphere 23.5 deg | cyl-sphere 26.0 | cube-sphere 28.5
    cyl-cyl       29.9     | cube-cyl   35.9 | cube-cube   37.9   (speed 0.81-1.02)

For two SPHERES the line of centres IS the exact contact normal, so 23.5 deg
there proves the error is not about shape. The model has no collision
mechanics: the learned potential is a SOFT SPRING whose force builds over many
frames while the line of centres rotates, so the accumulated impulse points
between where the normal started and ended.

`v5/impulse.py` adds the textbook rigid-body impulse
    J = -(1+e)(v_rel . n) / (1/m_i + 1/m_j)
applied at contact along the normal. Verified against textbook results: head-on
equal masses swap velocities exactly; a glancing hit on a stationary equal ball
separates at EXACTLY 90.0 deg; momentum conserved to 2.4e-07 over 300 random
contacts; separating pairs correctly ignored; restitution retains 100/25/0% of
kinetic energy at e = 1.0/0.5/0.0.

Measured on 224 replayed real collisions (model NOT retrained for it):

| contact model                | angle err | p90   | speed |
|------------------------------|-----------|-------|-------|
| soft spring (v3_7 as shipped)| 30.3 deg  | 111   | 0.95  |
| **impulse, line-of-centres** | **21.0**  | **79**| 0.86  |
| impulse + cube face normals  | 23.5      | --    | --    |

Cube face normals REJECTED -- see the docstring in impulse.py. Restitution
finally does something: attrs[:,1] (metal 0.8 / rubber 0.5) had fed only the
attribute encoder and removed no energy since the project began.

Speed 0.86 is the potential and the impulse both removing energy; `v3_8_impulse`
retrains so the potential can hand contact over instead of fighting it.

**This is the thing four days of experiments circled without touching.** Drag,
orientation, radius freezing and 3D data are all refinements of a contact model
that was structurally a spring. A distance-only potential is a sphere
approximation on ANY dataset -- which is exactly the generalisation concern.


## FIVE FOR FIVE: the benchmark cannot see physics quality (2026-09-10)
Every physics improvement, measured on its own terms and on the benchmark:

| change            | physical measurement                | benchmark      | z     |
|-------------------|-------------------------------------|----------------|-------|
| drag / friction   | rollout err -25%, collision err -28%| 69.9 vs 70.9   | -0.60 |
| 3D vs pixel data  | contact geometry 0.740 -> 1.081     | 74.2 vs 68.0   | +5.52 |
| radius frozen     | physics matches the render          | 74.1 vs 74.2   | -0.03 |
| orientation       | (shape-aware contact distance)      | 73.1 vs 73.9   | -0.78 |
| **contact impulse** | **bounce angle 30.3 -> 21.0 deg**  | **74.4 vs 74.1** | **+0.28** |

Only the 3D data moved it, and that gain was on the FACTUAL pairs where exact
geometry decides a short window -- not on the counterfactual reasoning.

The mechanism is not mysterious. Each choice asks "do these two touch, yes or
no". A collision either happens or it does not. Being 9 degrees more accurate
about the direction of a bounce flips a yes into a no only for pairs sitting
exactly on the boundary, and there are few of those. Measured directly: of 861
wrong answers only 97 (11%) were within 0.02 of the opposite verdict.

PREDICTION, recorded before v3_8 finishes: retraining WITH the impulse will
improve the bounce angle further and leave the benchmark flat.

This is the project's central result and it is defensible: the physics is
measurably better on five independent axes, and the benchmark everyone quotes
is blind to all five.


## ORIENTATION: SETTLED, NEGATIVE, SIGNIFICANT (2026-09-10)
Definitive test -- radius frozen (confound removed), 11,182 clips (3.5x the
first attempt), identical data both arms, one variable.

| measurement                     | control    | orientation | verdict          |
|---------------------------------|------------|-------------|------------------|
| val rollout err @20 (first pair)| 0.0249     | 0.0270      | worse            |
| val rollout err @20 (this pair) | 0.0264     | 0.0275      | worse            |
| collision benchmark (first)     | 73.9%      | 73.1%       | worse, z=-0.78   |
| **collision benchmark (this)**  | **74.5%**  | **71.5%**   | **z = -2.82**    |
| counterfactual pairs            | 66.6%      | 62.5%       | z = -2.27        |
| training stability              | stable     | **DIVERGED**| loss 0.0003 -> 5.8 |

The yaw arm blew up at epoch 23 (loss 5.81, val 1.22) and never recovered; its
0.0275 is a checkpoint from before the divergence, not where it finished.

Likely cause: torque enters as dV/dyaw and the cube support function
s*(|cos a| + |sin a|) has corners at 45 degrees. The smoothed version still has
a near-kink there, so a small yaw change swings contact distance sharply and
compounds over a 20-frame rollout. Cube face normals failed for the same
underlying reason (26.8 vs 21.4 deg) -- both are smooth approximations of
genuinely non-smooth geometry.

DO NOT re-attempt orientation without real contact generation (GJK/EPA giving
the true closest-point normal and distance). Smooth approximations of corners
are worse than the sphere approximation they replace.

**v5b_noyaw.pt is the production model: 74.5%, the best honest score reached.**
3D geometry, radius frozen, drag, 11,182 clips, nothing copied in evaluation.


## FAILURE PROFILE -- unchanged by five fixes (2026-09-10, v5b_noyaw)
| kind                    | count | share |
|-------------------------|-------|-------|
| MISSED a collision      | 800   | 24.0% |
| invented one            | 47    | 1.4%  |

17:1. Split by question type:

| pairs                        | wrong  | FN  | FP |
|------------------------------|--------|-----|----|
| counterfactual (long chains) | 33.4%  | 459 | 20 |
| factual (short window)       | 19.4%  | 341 | 27 |

Counterfactual reasoning fails at nearly DOUBLE the rate of factual
prediction. Those questions need a chain of collisions to stay correct 40+
frames past an intervention.

**The VLM question is settled.** 79% of missed collisions were nowhere near
(median gap +0.115 -- a full contact distance of air); only 9% were within a
threshold tweak. No reader can recover them; the information is not in the
store. Conversely 100% of invented collisions were within 0.02, so the false
positives are a threshold artefact and there are only 47.
-> The language model is worth having for GROUNDED EXPLANATION, not for
   accuracy. If asked "does the LLM improve the score", the answer is no, and
   this is the measurement.

## WHERE THE MODEL IS LACKING (assessment, 2026-09-10)
It reproduces 89.8% of real collisions when started 15 frames out and aimed at
one, but misses 800 on the benchmark. Those reconcile one way: it can do a
collision, it cannot do a CHAIN. One bounce 21-30 deg wrong sends the object
elsewhere and every collision downstream never happens.

Three structural gaps, in order of estimated cost:

1. **NO CONTACT POINT.** The model knows object CENTRES and nothing else. Real
   collision resolution needs the point where surfaces meet -- that sets the
   true normal and the torque. Everything here approximates it from centres.
   This is also why orientation failed twice: without a contact point a cube's
   face normal is a guess. Fixing it means real contact generation (GJK/EPA).
   Dataset-independent, which is the property that matters for generalisation.
2. **COLLISIONS DETECTED AT FRAME BOUNDARIES.** Objects move 0.0025/frame;
   contact is continuous. Discrete checks catch it late, at a rotated normal,
   with wrong velocities -- a plausible chunk of the residual 21 deg. Standard
   fix: continuous collision detection.
3. **NOTHING TRAINS THE CHAIN.** The loss is mean position error over a 20-frame
   rollout. A missed SECOND collision costs almost nothing in that average.
   No term says the sequence of events must be right.

Five fixes (drag, 3D, radius freeze, orientation, impulse) each improved a
physical measurement and left this profile identical. That is one structural
limitation showing through five different patches.


## FINAL RESULTS (2026-09-10) -- seven experiments, one moved the benchmark

**Trivial baselines on the 3332 test choices** (this is the context everything
else must be read against):

| policy                          | score  |
|---------------------------------|--------|
| never predict any collision     | 54.1%  |
| always predict a collision      | 45.9%  |
| **best world model (v5b_noyaw)**| **74.5%** |

The world model is worth **+20.4 points over doing nothing**. That is the
honest value of the physics on this benchmark.

**Every intervention, its physical effect, and its benchmark effect:**

| change              | physics                          | benchmark        | z     |
|---------------------|----------------------------------|------------------|-------|
| drag / friction     | rollout -25%, collision err -28% | 69.9 vs 70.9     | -0.60 |
| 3D vs pixel data    | contact geometry 0.740 -> 1.081  | 74.2 vs 68.0     | +5.52 |
| radius frozen       | physics matches the render       | 74.1 vs 74.2     | -0.03 |
| orientation         | DIVERGED in training             | 71.5 vs 74.5     | -2.82 |
| impulse (eval only) | bounce 30.3 -> 21.0 deg          | 74.4 vs 74.1     | +0.28 |
| impulse (retrained) | **val err 0.0221, best ever**    | 73.0 vs 74.1     | -0.85 |
| contact calibration | **ceiling 79.3% -> 98.3%**       | 73.9 vs 74.5     | -0.62 |

Only the 3D data moved it, and that gain was on FACTUAL pairs where exact
geometry decides a short window -- not on counterfactual reasoning.

**The decisive one is the last.** We raised what is even physically DETECTABLE
by 19 points (measured on ground-truth trajectories, no model involved) and the
score moved -0.6. If the benchmark were measuring collision prediction, that
could not happen.

## THE CONTACT-RULE CALIBRATION FINDING
Sweeping the rule on VAL against annotated collisions, using the RECORDED
trajectory (no model):

| rule            | detected | false pos |
|-----------------|----------|-----------|
| r x1.00, +0.02  | 79.3%    | 2.1%      |
| r x1.20, +0.03  | 98.3%    | 5.8%      |

Radii come from sqrt(silhouette area / pi) and detectors under-segment mask
boundaries, so they run ~20% small. 20.7% of CLEVRER's annotated collisions
were INVISIBLE to our measurement -- no simulator could ever have been scored
correct on them. Use `--contact-scale 1.20 --thresh 0.03` when the question is
"can we see the collision"; it does not improve the benchmark (the extra true
positives and false positives cancel).

TRUE surface-to-surface geometry was tried and is WORSE (75.9% vs 79.3%): the
stored cube radius is the direction-AVERAGED reach (1.27x its half-side), which
is more generous than the true square in the face direction. Exact shape
geometry shrinks cubes and detects fewer contacts.

## PRODUCTION MODEL: v5b_noyaw.pt (74.5%)
3D geometry, radius frozen, drag on, 11,182 training clips, nothing copied at
evaluation. v3_8_impulse.pt has strictly better PHYSICS (val 0.0221 vs 0.0264)
and a slightly lower benchmark score -- pick by which you need to defend.


## v6: VOXEL CONTACT RESOLUTION (2026-09-10) -- correct, practical, invisible
Hamza's design: put an origin in the scene, lay a 3D grid over it, record which
object presses which voxels each frame. Contact is where the pressed sets meet;
the normal comes from the pressed geometry; the velocity splits along it.

**Validated where the right answer is not debatable** -- a ball striking a heavy
cube's flat face must return along the face normal at ANY offset:

| offset from face centre | voxel normal | line-of-centres | truth |
|-------------------------|--------------|-----------------|-------|
| 0.00                    | 180.0        | 180.0           | 180   |
| 0.10                    | **180.0**    | 162.6           | 180   |
| 0.20                    | **180.0**    | 146.0           | 180   |
| 0.28 (near the edge)    | 164.5        | 133.6           | ~     |

Line-of-centres -- what every earlier version used -- is wrong by up to 34 deg.

**Made practical**: 77 ms -> 15.5 ms per call. Two changes: keep only SURFACE
voxels (interior voxels can never be first contact; 22368 -> 3072 for a sphere)
and cache each object's shell in its own frame so placing it is integer
addition. Both verified to give identical contact verdicts.

**Benchmark: 74.5% (2479/3328) vs 74.5% (2481/3328) baseline. Two choices.**

The reason is structural, not a tuning failure. `simulate_pair` returns the
moment the asked pair touches -- the question is "did these two make contact",
and the BOUNCE only determines what happens after that. The contact normal can
be exact or nonsense and the answer is the same. It matters only through chains
(A hits B, deflects correctly, therefore reaches C), and contacts fire on just
2.7% of simulated frames, so the effect washes out.

**Instrumentation note.** The first voxel benchmark returned EXACTLY the
baseline (2481/3328, same in/out splits). That was not a null result -- the
flag printed "VOXEL OCCUPANCY" but --spatial routes through
simulate_pair_spatial and only simulate_pair had been hooked. Zero references
in the function that ran. If the numbers had differed by even one choice the
dead path would have been reported as a finding. When a change produces
digit-identical output, verify it executed.

Files: v6/voxel.py (occupancy, surface, shell cache, contact normal),
v6/resolve.py (impulse from pressed voxels), v6/demo.py (one question ->
stored future + contacts + orbiting 960x640 gif).
Enable with `v5/evaluate.py --voxel`.
