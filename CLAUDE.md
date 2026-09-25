# CausalVis — project context

## What this is
A physics-structured world model that answers CLEVRER counterfactual questions
("which event would still happen if object X were removed"). Pixel-free: it
works from object tracks, not video frames.

The model learns exactly one thing: the pairwise potential energy V(r) between
two objects. Force is its negative gradient (autograd, not hand-derived), and
motion advances by a symplectic leapfrog integrator. Because force comes from a
shared energy that depends only on the difference of positions, every push is
automatically equal and opposite, so momentum is conserved as an arithmetic
identity rather than a trained behaviour. This guarantee is the core
contribution and must not be broken.

## Layout
- `src/` original generation (checkpoint `v3_1_dynamics.pt`, 7 physical attrs,
  learned `radius_head`)
- `src2/` current generation (`v3_2_dynamics.pt` = old/broken baseline,
  `v3_3_dynamics.pt` = fixed objective; 8 physical attrs, measured radius in
  attrs column 16 / index 15)
- pipeline: `build_trajectories.py` -> `make_split.py` -> `dynamics.py` (train)
  -> diagnostics (`collision_check.py`, `find_chains.py`, `ccd_eval.py`,
  `benchmark_eval.py`, `dynamics_check.py`) -> `viewer2.py`
- every diagnostic runs with `--split split.json --which test`
- data: `data/trajectories_v2/*.npz` with keys positions [T,N,2],
  velocities [T,N,2], presence [T,N], attrs [N,16], obj_keys, collisions [K,3]
- T = 128 frames, N up to 8 objects
- deterministic 16k/2k/2k train/val/test clip split, written once to
  `split.json`, read by everything

## Key constants
contact = r_i + r_j from measured radii (scalar fallback 0.119 for old npz
without the radius column), collision threshold 0.02, hysteresis release 0.02,
chain margin 0.06, taint lookback 12 frames, velocity smoothing 3 frames, dt = 1

## Measured facts about this dataset (don't re-derive, don't guess)
- median per-frame displacement 0.0025, p90 0.0087, max 0.21
- mean contact distance r_i+r_j = 0.117
- 2.42 annotated collisions per clip, 128 frames per clip
- **only 6.8% of pair-frames are actually overlapping (gap < 0)**; 10.8% are
  within 0.02 of touching
- with uniform start frames at horizon 20, 40.7% of windows already contain an
  annotated collision — window-level coverage was never the problem, the
  frame-level 6.8% was
- dt=1 resolves the wall fine (step 0.0025 vs feature widths 0.02–0.08)
- hardware: i5-8350U, 4 cores / 8 threads, 7.9 GB, CPU only.
  ~0.43 s per batch of 16 at horizon 20 => ~7 min/epoch => ~3.6 h for 30 epochs

## The bug that was found and fixed (2026-09-08)
An earlier diagnosis said the training objective was one-step next-state
prediction and prescribed multi-step rollout loss plus collision-weighted
sampling. **That diagnosis was wrong.** The loop in `dynamics.py` has always
used a multi-step rollout loss with a horizon curriculum 3 -> 20, and
collision coverage was already 40.7%. Both prescribed fixes were non-fixes.

The actual defect: **`radius_corr` collapsed.** The formula was
`radius = r_meas * (1 + 0.3*tanh(radius_corr(e)))`, and training drove tanh to
-1 for essentially every object — the ratio measured 0.7000 across 200 clips,
i.e. pinned at the floor with dead gradient. The model shrank every object by
30%, so its internal contact distance was 0.082 while the world's is 0.117.
Objects interpenetrated by ~0.035 (more than half a radius) before feeling any
force. This is a loophole, not a subtlety: shrinking radius weakens collisions
without touching the potential, and that lowers the average loss because only
6.8% of pair-frames are in contact.

It also explains the misleading diagnostic output. `dynamics_check.py` probed
force at the MEASURED contact (0.101) while the model integrates at its own
(0.0707), which made the force look like it vanished at contact. In the model's
own gap-space the wall was roughly correctly shaped, just applied 0.035 late.

Ablating `radius_corr` at eval time makes things WORSE (0.0898 vs 0.0682 at 40
frames), because the potential was co-trained in the shrunken gap-space. The
fix required retraining, not a patch.

## The fix that was applied
1. `radius_corr` is zero-initialised, excluded from the optimiser, and
   short-circuited by `model.freeze_radius`. Radius is now exactly the measured
   radius. The module is left in the class only so old checkpoints still load.
   Verify with the per-epoch `r/r_meas` column: it must read 1.0000.
2. `contact_weights()` reweights the loss by frame: ~1 while coasting, 1+lam
   (default 10x) on frames where objects are touching. Computed from ground
   truth under `no_grad`, so it changes the objective only and cannot affect
   momentum conservation. This lifts contact from ~7% to ~40% of the signal.
3. `--collision-frac 0.5` centres half the training windows on a real
   annotated collision (frame indices are in `z["collisions"][:,0]`).
4. `dynamics_check.py` now reports BOTH contact distances and `|F|` at the
   model's own contact, and warns when the model's radii have shrunk.
5. Training now tracks val rollout error at horizon 20 and saves the best
   checkpoint rather than the last.

## RESULT OF THE FIX: it did not work (2026-09-08)
`v3_3_dynamics.pt` = radius frozen + contact-weighted loss + collision-centred
windows. Measured against `v3_2` on identical settings, test split:

| metric                     | v3_2     | v3_3     |
|----------------------------|----------|----------|
| rollout err @20            | 0.0202   | 0.0227   |
| rollout err @40            | 0.0684   | 0.0755   |
| vs const-velocity @40      | 1.18x    | 1.07x    |
| \|F\| at its own contact   | 1.69e-04 | 6.26e-05 |
| force localisation         | 6322x    | 163x     |
| collisions reproduced      | 95.5%    | 94.3%    |
| collision err @+15         | 0.0816   | 0.0843   |
| median min-gap at collision| -0.0530  | -0.0473  |

Worse on every axis except interpenetration, which improved slightly (the one
thing the radius fix should do). The pre-registered defence -- "average error
gets worse but collisions get better" -- was TESTED AND FALSIFIED.

**Methodological error to learn from:** three changes went into one run
(radius freeze, contact weighting, loss renormalisation), so nothing can be
attributed. The renormalisation is the prime suspect: switching from
`.mean()` over 8 padded slots to a weighted mean over ~3.5 live objects makes
the loss ~2.3x larger before the contact weights multiply it further, against
a gradient clip still fixed at 1.0 -- so the whole run clipped far harder.
`--legacy-loss` now reproduces v3_2's exact objective so one variable can be
moved at a time. Do not bundle changes again.

### The single-variable run settled it: the radius collapse was BENIGN
`v3_4_radiusonly.pt` = frozen radius, legacy loss, no contact weighting. One
variable moved. Test split (nothing was ever selected on it):

| metric                | v3_2       | v3_3     | v3_4     |
|-----------------------|------------|----------|----------|
| rollout @20           | **0.0202** | 0.0227   | 0.0222   |
| rollout @40           | **0.0684** | 0.0755   | 0.0739   |
| vs const-vel @40      | **1.18x**  | 1.07x    | 1.09x    |
| collisions reproduced | **95.5%**  | 94.3%    | 94.9%    |
| collision err @+15    | **0.0816** | 0.0843   | 0.0845   |
| interpenetration      | -0.0530    |**-0.0473**| -0.0499 |

`radius_corr` really was pinned at 0.7000 with dead gradient, but the
potential co-adapted around it and removing it does not help. v3_2 remains the
best checkpoint on held-out data. Keep v3_2 as the production model.

**BEWARE SELECTION BIAS (this nearly produced a false positive).** v3_4 scores
0.0187 on VAL versus v3_2's 0.0199 and looks like a win. It is not: v3_4's
checkpoint was chosen as best-of-30-epochs on val, while v3_2 was a single
unselected measurement. The test split reverses the ranking. Always compare on
test, and never compare a val-selected checkpoint against an unselected one.

### What is left, by elimination
Three training objectives could not move the model past ~1.2x better than
straight-line motion. The remaining measured defect is structural, not an
optimisation problem: the data is DISSIPATIVE (0.85% speed loss per frame) and
force = -grad V is CONSERVATIVE by construction. That is the leading
hypothesis for the residual gap and it is still UNTESTED. The experiment is an
explicit drag term, run as a single variable against v3_2 with --legacy-loss.

## Where things stand
Baseline to beat, val rollout position error at horizon 20: **0.0199**
(`v3_2_dynamics.pt`). Old test-split numbers for reference: 0.0202 @20,
0.0682 @40 versus a constant-velocity baseline of 0.0805 @40 — only 1.18x
better than assuming straight lines.

Run:
    python src2/dynamics.py --data data/trajectories_v2 --split split.json \
        --epochs 30 --batch 16 --threads 8 --out v3_3_dynamics.pt
Then:
    python src2/dynamics_check.py --model v3_3_dynamics.pt --split split.json
Targets: `r/r_meas` exactly 1.0000; meaningful `|F|` at the model's own contact
(the impulse budget says ~5e-4 per frame is what it takes to reverse a typical
object); rollout error at 40 frames well below 0.03; clearly more than 2x
better than constant velocity. Then re-run `benchmark_eval.py` and read the
model-decided subset — that is the number that reflects the dynamics.

## 3D reconstruction: what was learned (2026-09-08)
The published CLEVR base-scene camera is **NOT** the CLEVRER camera. Silhouette
recovery in `fit3d.py` moved it from loc=[7.48, -6.51, 5.34] to
loc=[2.27, -6.36, 5.01], lens 35.0 -> 34.6, and median silhouette IoU rose
0.5745 -> 0.9340. Do not use the published CLEVR numbers; use `camera_fit.json`.
That wrong camera is exactly why the first analytic lift (`lift3d.py`) scored
worse than the 2D pipeline.

`fit3d.py` recovers 3D by INVERTING THE RENDERER, not by estimating depth.
CLEVRER is synthetic: 3 known shapes, ONE size class (the ground truth carries
no size attribute; the 3x spread in silhouette area is entirely depth), all
resting on a flat table. So pose has 2 DOF for a sphere/cylinder and 3 for a
cube (X, Y, yaw), and all three shapes are convex, so the silhouette is the
convex hull of the projected vertices -- no rasteriser, no network, no GPU.
Recovered object size: cube 0.305, sphere 0.35, cylinder 0.35.

Two measurements that came out of the same work:
- **CLEVRER has friction.** Over 620 collision-free segments objects lose
  0.85% of their speed per frame (34% over a 40-frame rollout); 70% of
  segments decelerate. A conservative potential (force = -grad V) CANNOT
  represent dissipation. The model needs an explicit, separately-audited drag
  term; the pairwise interaction stays momentum-conserving and the drag is
  reported as momentum transferred to the table (which is physically correct,
  the table is an external body).
- **The "linearity in time" test in `lift3d.py` is confounded by that
  friction** and cannot be used to judge perspective correction. A grid search
  over the entire 2-parameter projective family only improved it from 0.03903
  to 0.03886, because the residual is real deceleration, not projection.

## Object identity: uniqueness is over (colour, material, shape)
NOT over colour. Measured on 500 clips: **21% contain two objects sharing
colour AND shape**, differing only by material (sim_00000 has a blue rubber
sphere id 0 and a blue metal sphere id 4). Zero clips share all three.

Consequences:
- Any renderer or UI must show material, or a correct reconstruction looks
  like it duplicated an object. `render3d.py` now gives metal a specular
  highlight and rubber a matte finish.
- `explain.py` is fine: it names objects from `obj_keys`
  ("blue_rubber_sphere"), so narration is unambiguous. Do not "simplify" that
  to colour + shape.
- `read_clip`/`build_trajectories` match on (colour, material, shape) first
  and REFUSE ambiguous (colour, shape) fallbacks. That refusal is load-bearing
  in 21% of clips -- dropping a detection is correct, mis-assigning is not.

## THE DISSIPATION FIX WORKED (2026-09-09) -- v3_5_drag.pt is the new best
`v3_5_drag.pt` = v3_2's exact config plus a learned per-object drag term,
integrated conformal-symplectically (Strang splitting: half a step of exact
decay p <- p*exp(-c*dt/2), the usual kick-drift-kick, then the other half).
ONE variable moved. Test split, nothing selected on it:

| metric                | v3_2     | v3_3   | v3_4   | **v3_5**   |
|-----------------------|----------|--------|--------|------------|
| rollout @20           | 0.0202   | 0.0227 | 0.0222 | **0.0181** |
| rollout @40           | 0.0684   | 0.0755 | 0.0739 | **0.0516** |
| vs const-vel @40      | 1.18x    | 1.07x  | 1.09x  | **1.57x**  |
| \|F\| at own contact  | 1.69e-04 | 6.3e-05| --     | **8.20e-04**|
| collision err @+15    | 0.0816   | 0.0843 | 0.0845 | **0.0584** |
| interpenetration      | -0.0530  |-0.0473 |-0.0499 | **-0.0429**|
| collisions reproduced | **95.5%**| 94.3%  | 94.9%  | 94.5%      |

The pre-registered prediction in NEXT_PHASE.md ("the const-velocity ratio at
40 frames should move well past 1.2x") was CONFIRMED: 1.18x -> 1.57x. Rollout
error at 40 frames fell 25%, collision error 28%. The only regression is
collision reproduction, down 1 point, which is expected: objects now decelerate
so a few marginal collisions no longer quite reach.

**USE v3_5_drag.pt AS THE PRODUCTION MODEL** (it replaces v3_2).

**The guarantee survived, and got better.** With drag off, the pairwise
interaction alone still conserves momentum to 1.6e-07 over 500 steps. With
drag on, momentum legitimately leaves the object system (the table is an
external body) and `ledger()["momentum_to_table"]` accounts for it. Conservation
is now an auditable balance sheet rather than a constant -- a stronger claim.

**The model recovered physics nobody told it.** Learned drag per frame:
sphere 1.33-1.64%, cube 1.61-1.96%, cylinder 1.85-2.27%; metal always above
rubber. Spheres lowest is correct -- a sphere ROLLS while cubes and cylinders
slide -- and it matches VRDP's published description of CLEVRER's physics
(same friction for everything except the rolling sphere). Magnitudes run ~2x
the 0.0085/frame measured in pixel space, which is itself a perspective-biased
estimate.

## THE BENCHMARK CANNOT SEE THE IMPROVEMENT (2026-09-09)
Ran the full CLEVRER counterfactual benchmark on the test split for both
checkpoints. v3_5 is 25% better at rollout and 28% better at collision error.
The benchmark cannot tell them apart.

| decision path            | v3_2         | v3_5         |
|--------------------------|--------------|--------------|
| outside cone (no model)  | 100.0% (831) | 100.0% (831) |
| never happens (no model) | 84.8% (1906) | 84.4% (1920) |
| simulated hit            | 74.5% (482)  | 74.1% (463)  |
| **MODEL-DECIDED subset** | **70.9%**    | **69.9%**    |

Difference +1.0pp, z = 0.39, **p = 0.69 -- statistically indistinguishable.**

Two reasons, and together they are the finding:
1. Only **595 of 3332 choices (18%)** consult the dynamics model at all. The
   other 82% are settled by copying the observed record or by "this pair never
   meets in either world".
2. Those 18% ask a BINARY question -- did the collision happen? -- which is far
   coarser than position error. Halving trajectory error does not flip a
   yes/no answer unless the pair sat exactly on the boundary.

**So the headline 86.1% is not a measurement of the world model.** A large,
real, independently-verified improvement in the physics moves it by nothing.
That is a property of the benchmark, not of the model, and it is the strongest
defensible claim this project has produced: CLEVRER's counterfactual metric is
insensitive to the physics it appears to be testing.

Report BOTH numbers and this analysis. Do not report the benchmark alone --
it hides the improvement -- and do not report the rollout numbers alone, which
would overclaim end-to-end impact.

## v5: THE HONEST BENCHMARK NUMBER (2026-09-09)
`v5/evaluate.py` enforces one rule: ground truth is INITIAL CONDITIONS only,
never an answer. The light cone survives only to choose WHEN each simulation
starts (a few frames before the event, from a true state -- the design Hamza
specified in July). Every pair a question asks about is simulated.

| evaluator                              | per-choice | model-decided share |
|----------------------------------------|-----------|---------------------|
| v3 light-cone, copied outside the cone | 86.9%     | 18%                 |
| **v5 no-copied-answers**, same model   | **77.7%** | **100%**            |
| v4 cone-free from frame 0              | 62.2%     | 100%                |

(all v3_6_3d on data/trajectories_3d, full test split, 930 q / 3328 choices)

- The crutch was worth **9.2 points**.
- Short-horizon initialisation is worth **15 points** over blind rollout. That
  was never a "horizon problem"; it is the method, and a full-scene rollout
  from frame 0 was already measured at 44% in July.
- simulated_in_cone 74.0% (1054/1425), simulated_outside_cone 80.6% (1533/1903).

REPORT 77.7%, NOT 86.9%. 86.9% is the pipeline with a lookup in it.

## v5 layout
`v5/world.py` SpatialWorld (shape-aware contact + rotational state, torque by
autograd; angular momentum conserved 3.6e-07); `v5/train.py`; `v5/evaluate.py`;
`v5/pipeline.py` (question -> VLM -> rollout -> RolloutStore -> VLM -> GIF;
VLM steps need ANTHROPIC_API_KEY); `v5/README.md` has the full state.
Yaw-fitted data: `data/trajectories_3d_yaw` (extractor stores averaged cube
support at source; no fix_cube_radius pass needed on that folder).

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

## CONTACT IMPULSE -- the real bug (2026-09-10)
The learned potential is a SOFT SPRING: force builds over many frames while the
line of centres rotates, so the impulse points the wrong way. Measured bounce
direction error vs the recorded trajectory: sphere-sphere 23.5 deg, cube-cube
37.9 deg. Sphere-sphere is the proof -- the line of centres IS the exact normal
there, so the error cannot be about shape.

`v5/impulse.py` applies the textbook rigid-body impulse
J = -(1+e)(v_rel . n)/(1/m_i + 1/m_j) at contact. Momentum exactly conserved
(equal and opposite); angle correct by construction; restitution (attrs[:,1],
metal 0.8 / rubber 0.5) finally removes energy after being dead all project.
Verified: head-on swaps velocities, glancing hit separates at EXACTLY 90.0 deg.

Bounce angle 30.3 -> 21.0 deg. Benchmark 74.1 -> 74.4% (z=0.28, nothing).
Enable with `--impulse` on dynamics.py or v5/evaluate.py.
Cube face normals were built, tested and REJECTED (26.8 vs 21.4 deg) -- snapping
to a face is wrong near an edge; needs real GJK contact generation.

RESTITUTION MUST BE SET BY properties(), NOT BY THE CALLER. step() only gets
mass/radius/e/present, so an externally stashed restitution goes stale the
moment a different caller steps a different object count (a 4-object tensor met
a 5-object rollout and crashed). Same class of bug as the drag= argument.

## Leveraging the 3D: what it actually buys (2026-09-09)
Measured, not assumed:

**`r_i + r_j` is EXACT for spheres and cylinders.** Their horizontal cross
section is a circle, so contact distance does not depend on direction. There
is nothing to improve there and no information is being thrown away.

**It is wrong only for CUBES**, understating contact by up to 19% against a
sphere/cylinder and 41% against another cube. So "use the 3D geometry" has one
precise target: cube orientation. `exact_contact()` in fit3d.py computes the
true surface-to-surface distance via support functions.

**Cube yaw is weakly identifiable per frame.** A dense sweep of the full 90
degrees moves silhouette IoU by only 0.09-0.19, so per-frame estimates are
noise: measured jitter 4.37 deg median, 16.5 deg p90, wandering 76.9 -> 0.0.

**The physics prior fixes it.** A cube spins at constant angular velocity
between collisions, so yaw(t) is piecewise linear with breakpoints only at
annotated collisions. `fit_yaw_track()` fits (yaw0, omega) per segment against
10 spread frames. Result on sim_00000: jitter 4.37 deg -> **0.00 deg**, omega
= 0 in both segments (the cube slides without spinning) and orientation
changes only at the frame-40 collision. That is physically exactly right.
Cost: 1.3 s/clip without yaw, 3.8 s/clip with (21 h for 20k).

When yaw is NOT recovered, `support_radius` returns the direction-AVERAGED
cube support (4/pi ~ 1.273 x s) rather than assuming yaw=0. Assuming zero
would assert an orientation that was never measured.

**Occlusion is a non-problem in CLEVRER.** 7.4% of pair-frames have masks
touching at all but only 0.019% overlap by >30%. The occlusion flag is
computed and is ~always zero here. It would matter on real video; it earns
nothing on this dataset. (Built before measuring -- same mistake as bundling
three changes in one run.)

## Order of work
1. Training fix (in progress) — the bottleneck for everything else.
2. 3D lift on Colab, in parallel — independent, removes a stated limitation.
3. Re-check the benchmark script, then re-run it last, so there is one clean
   before/after.

## Hard constraints
- Never break momentum conservation. In particular never clamp forces per
  object: clipping one member of a force pair and not the other destroys the
  cancellation. If forces need bounding, rescale ALL forces by one shared
  scalar.
- Forces are computed by autograd, so gradients must stay enabled during
  rollouts. Do not wrap simulation in `@torch.no_grad()`. (Use
  `create_graph=False` for evaluation instead.)
- Radius is measured from mask pixel area, not learned. Do not reintroduce a
  learned radius head or a "small bounded correction" on top of it — that is
  what just collapsed. It is unidentifiable and it collapses to the bound.
- Only bounded feature shapes near contact. Exponential gap features overflow
  when tracks briefly overlap and produce NaN losses. Use
  `sqrt(r^2 + 1e-12)` for distance rather than `norm().clamp_min()`, so the
  gradient is smooth at zero separation.
- CPU only. No GPU available locally, and no remote compute — every command
  runs on this machine.
- Everything is graded on the test split. Never tune against it. Tune on val.

## Recurring bug classes in this codebase
- gradients disabled where forces are needed (forces are a derivative)
- tensor vs bool comparisons (use `.item()` on single-element tensors)
- `None`-default contact distance
- **a measurement reported against the wrong reference frame** (the diagnostic
  probing measured contact while the model used its own) — when a number looks
  impossible, check what it is being compared against before believing it
- a defect found on one line usually exists in several places: scan the whole
  file for the pattern, not just the crashing line

## Working style
Plain English, no publication framing, honest caveats stated openly. Complete
verified files rather than snippets. Measure before and after every change.
Show the current code and explain what it does before changing it — the last
diagnosis survived three weeks because nobody opened the training loop.


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

## v6 VOXEL NORMAL WAS BROKEN -- the 74.5% voxel benchmark is INVALID (2026-09-11)
The old `contact_normal` took the difference of the two objects' local patch
centroids. On 39 known-answer cases (sphere-sphere at 5 angles, ball on a cube
face off-centre, rotated cube, cube-cube, cylinder-sphere; touching AND
overlapping) it was >10 deg wrong in 22 and FLIPPED 180 deg in 7. Cause: at a
real collision the objects overlap by a voxel or two, the patches
interpenetrate, their centroids cross (normal flips) or coincide (vector
vanishes into a hard-coded [1,0,0] fallback). A flipped normal makes an
approaching pair look separating (no impulse) and a separating pair look
approaching (pulled together). The earlier "180.0 deg at every offset"
validation only used exact-touch placements that happened to work.

Fix: normal = smallest principal component of the surface voxels within 5
voxels of the contact point (fitted in the horizontal plane), sign from the
voxel centroids (valid for convex bodies). Suite: median 0.9 deg, worst 7.5,
0/39 over 10. Batch check: head-on e=1 masses 1,2 gives exactly (-0.01667,
+0.00333); ball off heavy cube face rebounds at 0.6 deg; separating overlap
untouched; momentum drift 0.0.

Also: `touching` now uses searchsorted against cached sorted keys instead of
intersect1d (0 disagreements on 107 real contacts, 4.7 -> 3.8 ms/frame).

## v6 TRAINING WITH VOXELS (launched 2026-09-11)
Voxel contacts were EVAL-ONLY until now. `v6/train_voxel.py` puts them inside
every training rollout via `v6/batch.py` (batched; normal detached as geometry,
impulse magnitude differentiable in momentum). Warm start from v5b_noyaw.pt,
4 epochs, horizon 20, lr 3e-4. Control arm = identical fine-tune with
--no-voxel. `v6/run_voxel.sh` then benchmarks: v6_voxel.pt (--voxel),
v6_control.pt, and v5b_noyaw.pt with the FIXED voxel normal. Logs: v6/cmp/.

## LM answering from the store (2026-09-11)
Groq free tier works (Llama 3.3 retired there; lm.py auto-resolves to
openai/gpt-oss-120b). Key goes in GROQ_API_KEY on the command line only.

NEGATION BUG in ANSWER_SYSTEM: "correct if the record supports it" made the LM
label every event 'wrong' on "which will NOT happen" questions when the record
showed no collisions. Fixed with an explicit two-step instruction (decide
HAPPENS, then apply polarity) and a `happens` field in the JSON so the
reasoning is auditable. Verified on q40/video 10020: both labels flipped to
the right reading; the remaining miss there is the world model's (it did not
simulate the blue cylinder-sphere collision).

`v6/lm_eval.py`: LM parses the intervention (no program), world model
simulates with voxel contacts (continuous rollout + focused pair windows),
store rendered as text with voxel contact events, LM answers every choice.
Scores the LM and the literal reading of the SAME store side by side, so any
gap is the reader. Smoke (2 questions, v5b_noyaw): parse 2/2, LM 5/6, literal
4/6 -- the LM's extra point cited a real voxel contact from the continuous
rollout that the focused window missed. Too small to mean anything.

Queued: `v6/after_voxel.sh` waits for v6/VOXEL_TRAIN_DONE, then runs
lm_eval on v6_voxel.pt (60 test questions, resumable) and the final orbiting
GIF (demo index 5, video 10002) into v6/demo_trained/.

## v6 RESULTS -- voxels trained in, full test benchmark (2026-09-11)
Full test split, 930 questions / 3328 choices, nothing copied.

| model                                   | val@20 | per-choice      | per-question | in-cone | outside |
|-----------------------------------------|--------|-----------------|--------------|---------|---------|
| v5b_noyaw (production, no voxels)       | 0.0264 | 74.5% (2481)    | 30.3%        | 66.6%   | 80.6%   |
| v5b_noyaw + FIXED voxels at eval only   |   --   | 74.5% (2479)    | 30.0%        | 66.6%   | 80.4%   |
| v6_control (same fine-tune, no voxels)  | 0.0250 | 74.3% (2474)    | 30.0%        | 66.9%   | 80.0%   |
| **v6_voxel (voxels inside training)**   |**0.0230**| **74.8% (2488)** | **30.9%**  | 67.0%   | 80.7%   |

- Voxel-trained beats its control on val rollout (0.0230 vs 0.0250, both
  best-of-4 on val, same selection) and by +14 choices on the benchmark
  (unpaired z ~ 0.4 -- NOT significant). Best honest score reached, but
  within noise of every other 3D model.
- Both arms peaked at epoch 0 and got worse after (loss spike at epoch 1 in
  both): lr 3e-4 is too high for fine-tuning from v5b. Not voxel-specific.
- Fixed voxel normal at eval only: 74.5%, same as without. The bounce is
  still invisible to "did they touch" questions.

LM from the stored future (v6_voxel, Groq gpt-oss-120b, 60 test questions,
203 choices, LM also parsed the intervention: 60/60 correct):
LM 74.9% per-choice / 31.7% per-question vs literal reading of the SAME store
70.0% / 23.3%. The LM agrees with the literal reading on 94.1% of choices;
the gap comes from it also using the continuous rollout's collision list and
voxel contact events, which the literal reading ignores. Paired stats in
the next line. Final GIF: v6/demo_trained/q5_video10002_voxel.gif (LM parse
ok, 2 voxel contacts, 4/4 correct).
LM paired: LM right/literal wrong 11  literal right/LM wrong 1  McNemar z=2.89

## WHERE v6_voxel ACTUALLY FAILS -- full per-choice diagnosis (2026-09-11)
`v6/diagnose.py` (4 shards) + `v6/diagnose_report.py` -> v6/cmp/diagnosis.txt.
Reproduces the benchmark exactly (2488/3328). Every wrong answer assigned to
ONE cause, first-applies order.

The model is right 97.7% when the pair truly does NOT collide and 47.7% when
it does. 799 of 840 errors are missed collisions. Polarity, shapes, removed
shape: all flat (73-77%). Per question 30.9%; 461 of 643 wrong questions miss
by exactly ONE choice.

MISSED COLLISIONS (799) BY CAUSE:
| cause                                                        | n   | share |
|--------------------------------------------------------------|-----|-------|
| BEFORE START: pair collided in video before the sim start    | 239 | 29.9% |
| CONE: factual pair, video shows no collision, truth says yes | 188 | 23.5% |
| REFUSED: one object not on table at start -> "no" unsimulated| 183 | 22.9% |
| FAR MISS: simulation never came close (real physics error)   |  94 | 11.8% |
| LEFT TABLE in simulation                                     |  34 |  4.3% |
| NOT SEEN: never within 3 voxels even on the REAL recording   |  30 |  3.8% |
| RULE: voxel sensor saw contact, centre rule did not count it |  21 |  2.6% |
| NEAR MISS: simulation within 10 voxels                       |  10 |  1.3% |

610 of 799 (76%) are EVALUATOR LOGIC, not physics or sensing -- the frames or
objects where the collision happens were never simulated, so no detector can
see it:
- BEFORE START: in-cone pairs are simulated from cone entry to clip end. The
  docstring says s = min(cone entry, closest approach - WINDOW); the CODE uses
  only the cone entry. Accuracy on pairs whose only video collision precedes
  the start: 1.6% (4/248).
- REFUSED: simulate_pair_spatial returns None before its loop if i or j is
  absent at frame s, although the loop already injects late entries. Pairs with
  a late object score 55.7% vs 77.5%.
- CONE: factual windows are centred on the OBSERVED closest approach; when the
  removal changes the pair the event is elsewhere. 4.9% (11/224) right.

VOXEL SENSOR AS THE DETECTOR IS WORSE (57.5% at 0 voxels, 70.5% at 3, 76.1%
at 6 vs rule 74.8%) because the reconstruction sits a few voxels short: on the
REAL recorded motion only 29.1% of annotated collisions share a voxel, 55.4%
within 3, 72.1% within 6, 88.5% within 10. Same ~20% size/position shortfall
as the contact calibration finding; the rule's 0.02 tolerance (= 6 voxels)
hides it. Sensor ceiling on real motion, factual pairs: 62.4% (0 vox) ..
78.0% (6 vox) vs model 80.7%.

## CLEVRER COUNTERFACTUAL ANSWERS INCLUDE EVENTS AFTER THE VIDEO ENDS (2026-09-11)
Of the 188 v6 "cone" misses, 129 were in scenes where the removal affected NO
object at all, and 167 had their observed closest approach at the clip's last
frames (window s >= 112). Test: roll those pairs past frame 127 with the world
model from a true state 8 frames before the end.
| group (window at clip end)          | n   | collide past frame 127 | median frames past end |
|-------------------------------------|-----|------------------------|------------------------|
| missed pairs, truth = collide       | 167 | 146 (87%)              | 14                     |
| control pairs, truth = no collision | 167 | 29 (17%)               | 49                     |
CLEVRER clips are 127 frames and its predictive questions are about events
after the video ends; the counterfactual truth evidently covers that tail too.
Every evaluator so far stopped at frame 127. -> v7 simulates past the end;
extension length is chosen on VAL (short extensions separate the groups).

Also measured: 31 of the 188 were pairs knocked in the recording by an object
that was already being simulated -- v5's cone only handled lost contact with
the REMOVED object (and read it from the annotation list).

VRDP (NeurIPS 2021) answers counterfactuals by removing the object and
re-running the simulation. v7/cf_world.py does the same: one counterfactual
world per intervention, all choices read off it.

## v7 VALIDATION (2026-09-11) -- one counterfactual world per intervention
VAL split, 941 questions / 3391 choices, v6_voxel.pt, nothing read from
annotations. Per-choice [per-question]:
| config                                        | best detector / extension | per-choice | per-question |
|-----------------------------------------------|---------------------------|------------|--------------|
| A  full: lost contact with ANY changed object | cal, +30 frames           | 89.0%      | 67.4%        |
| B  lost contact with REMOVED object only      | cal, +30 frames           | 88.7%      | 66.7%        |
| S  strict: every object simulated from entry  | cal, +0 frames            | 65.3%      | 17.3%        |
Grid for A (per-choice): rule 80.0 (+0) -> 84.6 (+30); cal 84.8 (+0) -> 89.0
(+30) -> 88.2 (+60); voxel<=1 66.1-67.2; voxel<=3 72.5-76.0; voxel<=6
78.4-83.2. Extension past the video end is worth ~4 points; past +30 false
positives start to cost. The voxel sensor at MEASURED size still trails the
calibrated rule by ~6 points -- the reconstruction is ~20% small.
Strict (Hamza's "initial conditions only") loses 24 points: long free
rollouts drift. Following the recording until the removal can reach an object
is causality, not a crutch, and is what makes the difference.
SELECTED ON VAL: config A, detector cal, extension +30. Test run once.

## v7 TEST RESULT (2026-09-11) -- run once, config chosen on VAL
Config A (lost contact with any changed object), detector cal (radius x1.20,
+0.03), +30 frames past the video end, v6_voxel.pt. 930 q / 3332 choices.
| evaluator                                             | per-choice       | per-question |
|-------------------------------------------------------|------------------|--------------|
| v6 per-pair windows (previous best)                   | 74.8%            | 30.9%        |
| NO-SIMULATION baseline (delete object, keep recording)| 81.2% (2704)     | 48.7%        |
| **v7 one counterfactual world, world model on**       | **88.1% (2936)** | **66.6%**    |
Val was 89.0 / 67.4 -> no sign of overfitting the selection. (Test grid peaks
at cal +45 = 88.3; +30 was fixed on val and is the number to report.)
Paired McNemar: v7 vs v6 z = 15.3 (637 vs 195); v7 vs no-sim z = 10.5 (356 vs 126).

HONEST ATTRIBUTION: most of the jump from 74.8 is the EVALUATION LOGIC fix --
reading the recording correctly with the object deleted already scores 81.2,
above every previous evaluator. The world model adds +6.9 per-choice and +17.9
per-question on top, significant. Recording-only pairs 92.9% (1616), simulated
pairs 83.7% (1709).

Remaining errors 393: missed 204 (158 in simulated pairs; 153 have no video
collision -> new collisions caused by the removal that the simulation does not
produce), invented 189 (120 simulated pairs, 69 recording-only; 51 only past
the video end). FP rose 41 -> 189: the calibrated detector and the extension
are generous; FN fell 799 -> 204.

LM (Groq gpt-oss-120b) on the first 60 test questions from the v7 store:
86.2% per-choice / 60.0% per-question, parse 60/60, agrees with the literal
reading on 203/203 choices. Same questions from the v6 store: 74.9%. With one
consistent world stored, the reader adds nothing beyond the store -- the
store is the thing that got better.

## OBJECT SIZE: MEASURED AT THE MOMENT OF CONTACT (2026-09-12)
At 7,340 annotated collisions in 3,000 TRAIN clips, the closest centre distance
(+-2 frames) was compared with the stored r_i + r_j. Stored sizes are one
constant per shape (sphere/cylinder 0.05833, cube 0.06472 normalised).
| pair              | n    | dist / stored sum: median (p25-p75) |
|-------------------|------|-------------------------------------|
| sphere+sphere     | 868  | 0.999 (0.960-1.071)                 |
| cylinder+sphere   | 1841 | 1.014 (0.968-1.137)                 |
| cylinder+cylinder | 916  | 1.028 (0.973-1.148)                 |
| cube+sphere       | 1610 | 1.064 (0.983-1.173)                 |
| cube+cylinder     | 1541 | 1.081 (1.003-1.204)                 |
| cube+cube         | 564  | 1.090 (1.031-1.188)                 |
Least squares -> v7/measured_sizes.json: sphere 0.05845 (x1.00), cylinder
0.06021 (x1.03), cube 0.07179 (x1.11). So the "~20% small" was NOT a uniform
size error: spheres are exact, cubes 11% small, and the wide p25-p75 spread
(+-8%) is POSITION WOBBLE in the reconstruction. Averaging silhouettes over
videos could not have found this (the bias is constant); contact distance can.

VAL recorded motion, 700 clips / 1,705 annotated collisions, no model:
- voxel sensor, surfaces sharing a voxel within +-3 frames: 25.3% -> 41.9% with
  measured sizes; within 1 voxel 34.4 -> 49.9; within 6 voxels 68.7 -> 78.5.
- VELOCITY-CHANGE confirmation (Hamza: "a collision changes someone's path"):
  surfaces within 0.03 AND either object's velocity changes >= 0.002 nearby:
  found 84.9% with 332 made-up vs 84.5% / 407 without it; >= 0.004: 82.5% / 251.
  Logged in cf_world as detectors kick2 / kick4 (for simulated objects the
  velocity change is the world model's own bounce).
Running: v7/run_sizes.sh -> v7/runs/val_sizes (detection only) and
val_sizes_dyn (world model also simulates with measured sizes). Pick on VAL.

## MEASURED SIZES + VELOCITY-CHANGE DETECTOR ON VAL -- tested, NOT adopted (2026-09-12)
VAL, 941 q / 3391 choices, v6_voxel.pt, config A. Per-choice at +30 frames [per-question]:
| detector   | stored sizes (val_A) | measured sizes, detection only | measured sizes in dynamics too |
|------------|----------------------|--------------------------------|--------------------------------|
| cal        | **89.0% [67.4]**     | 88.4% [66.0]                   | 88.2% [65.7]                   |
| rule       | 84.6% [55.0]         | 85.4% [57.9]                   | 85.4% [58.1]                   |
| kick2      | --                   | 87.4% [63.7] (+45: 87.8)       | 87.3% [63.7]                   |
| kick4      | --                   | 85.9% [58.9]                   | 86.2% [59.8]                   |
| voxel<=1   | 67.0% [16.0]         | **74.5% [31.1]**               | 69.7% [20.9]                   |
| voxel<=3   | 75.6% [33.6]         | 80.6% [45.5]                   | 77.9% [39.3]                   |
| voxel<=6   | 83.0% [51.4]         | 85.2% [57.6]                   | 85.1% [57.3]                   |
- Measured sizes make Hamza's voxel sensor far better (+7.5 at 1 voxel, +5.0 at
  3) and the plain rule better (+0.8), but the best detector is still the
  generous calibrated rule on stored sizes (89.0). Not adopted; test untouched.
- Velocity change ("a collision changes someone's path") beats the plain
  distance rule by +2.0 on the same sizes, but not the calibrated rule.
- Putting measured sizes into the world model's dynamics without retraining
  hurts slightly (the potential was trained on stored sizes). A retrain with
  measured sizes is the only untested variant.
The app (app/server.py) serves the unchanged 88.1% configuration.

## TEXTBOOK LAWS BEAT THE LEARNED MODEL (probe/physics_laws.py, 2026-09-12)
Laws fitted on TRAIN (1,500 clips), system picks the best law per object kind,
scored on VAL (700 clips). Same protocols for both systems.
A. Free motion, 7,417 val segments: 40-frame error laws 0.0502 vs learned
   0.0566 (-11%); 20-frame 0.0239 vs 0.0250. Chosen laws: cubes and cylinders
   SLIDE (constant deceleration 0.00008/frame), spheres ROLL (exponential
   damping 0.2-0.3%/frame) -- physically right, found without being told.
B. Bounce given the recorded contact, 1,345 isolated val collisions: line-of-
   centres impulse median 9.9 deg / p75 21.1; voxel normal WORSE (11.7 / 23.8)
   on reconstructed data; contact friction changes nothing (10.0). Fitted:
   mass ratio metal/rubber 1.00, bounciness 0.2 for every pair, mu 0.05.
C. Head-to-head, both start 8 frames before a real collision and run 16:
   law engine 12.1 deg median / 23.8 p75, contact 95.5%; learned v6_voxel
   17.6 / 34.7, contact 93.6%.
-> v8 = v7's world builder with the learned simulator swapped for the law
   engine (ONE variable: detectors, influence rules, sizes for detection all
   unchanged). Constants from v8/fit_laws.py -> v8/laws.json (TRAIN only).

## v8: TEXTBOOK-LAW SIMULATOR (built 2026-09-12)
v8/fit_laws.py -> v8/laws.json (TRAIN only): cubes/cylinders slide at constant
deceleration 0.00008/frame; spheres roll with 0.3% (metal) / 0.2% (rubber)
damping per frame; bounce along the line of centres, bounciness 0.2 for every
material pair, metal/rubber mass ratio 1.00, contact friction 0.05; contact
when surfaces (measured sizes) are within 0.01.
v8/make_v8.py GENERATES v8/cf_world.py and v8/evaluate.py from v7's files by
inserting at anchors that must each occur once: the ONLY change is the
simulator (law_step replaces the learned model + voxel bounce). Influence
rules, entries, +30 frames, detectors and detection sizes are v7's exactly.
Smoke (12 val questions, 42 options): laws changed events on 7 options; cal
+30 v8 39/42 vs v7 37/42 (too small to mean anything).
v8/run_v8.sh: full VAL -> best detector/extension (same rule as v7) ->
pre-registered bar: must beat v7 VAL 3017/3391 -> only then TEST once +
paired McNemar vs v7 test_A. Results: v8/runs/.

## v8 RESULT: THE LAW ENGINE WINS (2026-09-12)
VAL (941 q / 3391 options), same selection rule as v7: best = cal, +30 frames
-> 90.1% (3055) [69.9% per question] vs v7 89.0% (3017) [67.4%]. Cleared the
pre-registered bar, so TEST was run once with cal +30:
| system                                   | options          | questions        |
|------------------------------------------|------------------|------------------|
| v7 learned world model                   | 88.1% (2936)     | 66.6% (619/930)  |
| **v8 textbook laws, fitted on TRAIN**    | **89.4% (2980)** | **69.8% (649/930)** |
Paired on 3325 matched options: v8 right & v7 wrong 79, v7 right & v8 wrong 36,
McNemar z = 4.01 (real). Gains are on SIMULATED pairs (83.7 -> 86.1%);
recording-only pairs unchanged (92.9%). Invented collisions 189 -> 153, missed
204 -> 199. "No collision in video, truth collides" 68.6 -> 71.0; "collision
after the video ends" 85.2 -> 88.2.
Approximate comparison with papers (their official test set, our held-out
validation questions): VRDP without extra labels 89.9 / 75.7 -> v8 is -0.5 /
-5.9; ALOE 91.4 / 75.6 -> -2.0 / -5.8.
NOTE: in v8 the neural world model is no longer in the simulation loop. The
simulator is explicit laws whose constants were fitted from data (VRDP-style
system identification). Momentum is still conserved (equal and opposite
impulses; friction is momentum given to the table). The app still serves v7.

## SHADOW REPLAY: WHERE THE SIMULATION BREAKS ON REAL VIDEO (2026-09-12)
probe/shadow_replay.py: for every annotated collision in 200 VAL clips, start
the simulator from the true state L frames before and run it alone (entering
objects start when they appear). Reproduced = pair within the benchmark's
collision check +-5 frames. Ceiling (check on the real recording) 97.9%.
| engine            | L=12  | L=30  | top cause of L=30 misses (first thing that went wrong) |
|-------------------|-------|-------|--------------------------------------------------------|
| v8 laws           | 96.0% | 57.4% | bad entry 60%, slow drift 24%, wrong bounce 12%        |
| v7 learned        | 96.7% | 55.0% | bad entry 59%, slow drift 25%, wrong bounce 13%        |
| v8 + 6-frame past velocity fit | 94.4% | 55.0% | REJECTED (worse)                        |
Both simulators break the same way -> a DATA problem, not a physics one.

ENTRY STATE IS WRONG: 346 entering objects (VAL, no collision in first 16
frames): speed at the entry frame off by 46% median (p75 63%), direction 15.9
deg; position 0.035 off the clean path (~1/3 contact distance). On v8 TEST,
287 of 1715 simulated pairs involve an object simulated within 10 frames of
entering: 69.3% accuracy vs 89.5% for the rest; they hold 72 of 153 missed
collisions in simulated pairs.

ROOT CAUSE (hypothesis under test): build_trajectories_3d.py places objects at
the CENTRE OF THE VISIBLE MASK back-projected to mid-height. Entering objects
are cut off by the image edge, so the centre sits inside and moves too fast.
fit3d.fit_pose (renderer inversion, silhouette clipped to the frame) handles cut
masks correctly but was only used for cube yaw. probe/edge_fix_test.py refits
only border-touching entry frames and measures entry errors before/after.
EDGE FIX FALSIFIED (probe/edge_fix_test.py, 150 VAL clips, 258 entries, 60% cut
off by the image edge at entry, 1,105 edge frames refitted with fit_pose):
entry speed error 47% -> 99% (WORSE), direction 15.9 -> 16.7 deg, position off
clean path 0.0391 -> 0.0394 (unchanged). With most of the object outside the
frame the silhouette match is ambiguous and jumps frame to frame. The mask
centre is not the cause of the bad entry state; do not re-run this fix.

## ENTRY CORRECTION WORKS IN THE REPLAY (2026-09-12)
Measured cause of "bad entry": an object sliding into view is measured at ~70%
of its real speed for ~8 frames (steps 0.75 0.71 0.69 0.70 0.71 0.73 0.77 0.81
0.83 0.90 0.92 0.97 0.99 1.01 1.04 of the later steady step) and its visible-
part centre starts ~2.5 steady steps AHEAD of the true centre. Wobble is not
the issue (0.0019 at entry vs 0.0015 mid-track). Profile fitted on 1,660 TRAIN
entries -> v8/entry_profile.json (ratio[tau], lag[tau]).
Correction when an object is handed to the simulator tau < 15 frames after
entering: v /= ratio[tau]; start position -= lag[tau] * v. No future frames.
Shadow replay, v8 engine, same 200 VAL clips:
| start before collision | v8     | v8 + entry correction |
|------------------------|--------|-----------------------|
| 12 frames              | 96.0%  | 96.4%                 |
| 30 frames              | 57.4%  | **65.3%** (+7.9)      |
Bad-entry misses 107 -> 92, slow drift 42 -> 30. Next: v8 world builder with
entry_fix (generator option, off by default), VAL -> bar 3055 -> TEST once.

## v8 + ENTRY CORRECTION: SMALL GAIN, NOT YET SIGNIFICANT (2026-09-12)
VAL: best setting moved to cal +60 -> 90.3% (3062) [70.9%] vs plain v8 90.1%
(3055) [69.9%]; cleared the bar by 7 options (cal +30 was 90.2%). TEST once:
| system               | options          | questions        | missed | invented |
|----------------------|------------------|------------------|--------|----------|
| v7 learned           | 88.1% (2936)     | 66.6%            | 204    | 189      |
| v8 laws              | 89.4% (2980)     | 69.8%            | 199    | 153      |
| **v8 + entry fix**   | **89.8% (2991)** | **70.9% (659)**  | 168    | 173      |
Paired vs v8 on 3325 options: 41 vs 30, McNemar z = 1.31 -> NOT significant.
Two things changed at once (entry correction AND the val-chosen extension
30 -> 60), so the paired number is not a clean single-variable attribution.
Missed collisions fell 199 -> 168 ("no video collision, truth collides" 71.0
-> 76.5%; "after the video ends" 88.2 -> 100%), invented rose 153 -> 173 (the
longer extension). The replay's +7.9 points (57.4 -> 65.3 at 30 frames) mostly
does not reach the benchmark: few benchmark decisions hinge on an object handed
to the simulator in its first frames. Entry DIRECTION error (~12 deg) is not
corrected yet; bad entry is still 63% of replay misses at 30 frames.

## CAMERA ALIGNMENT CHECK (professor's point, 2026-09-12)
If the camera-world alignment were off, measured contact distances would drift
across the image. 3,625 round-object (sphere/cylinder, exact r_i + r_j)
collisions in TRAIN, closest centre distance / stored contact sum:
- by distance from camera: near 1.032, middle 1.005, far 0.977 (a mild ~5%
  near-to-far trend)
- left / centre / right: 1.013 / 1.009 / 1.021 (flat)
- top / middle / bottom of image: 0.982 / 1.023 / 1.055 (bottom n=109)
The alignment is right on average (median 1.013; silhouette IoU 0.95 with the
recovered camera). A small depth-dependent bias of about +-3% remains; the
spread within each bin (p25-p75 about 0.97-1.12) is several times larger, so
alignment is not the main source of position error.

## v7.1 = v7 + ENTRY CORRECTION (launched 2026-09-12)
Hamza chose to keep v7's learned world model (not v8's laws). v7_1/make_v7_1.py
generates v7_1/cf_world.py and evaluate.py from v7 with ONE change: the entry-
state correction (v7_1/entry_profile.json, TRAIN). Verified: fix OFF gives
records identical to v7 (42/42); fix ON changes events on 4/42.
v7_1/run.sh: VAL -> bar 3017 -> TEST once -> paired vs v7 test_A.
v7.1 RESULT: VAL cal +30 89.2% (3024) [67.8%] vs v7 89.0% (3017) -> cleared the
bar by 7. TEST once, cal +30 (same setting as v7, so ONE variable):
| system          | options          | questions        | missed | invented |
|-----------------|------------------|------------------|--------|----------|
| v7              | 88.1% (2936)     | 66.6% (619)      | 204    | 189      |
| v7.1 (+entry)   | 88.4% (2944)     | 66.9% (622)      | 199    | 189      |
Paired on 3325 options: 24 vs 17, McNemar z = 1.09 -> NOT significant. The
entry correction helps the learned model a little, less than it helped the
laws engine (v8: 199 -> 168 missed), because the learned model's long-span
drift, not the entry state, limits it once simulation starts.

## v7.2 = v7.1 + WORLD MODEL FINE-TUNED ON LONG SIMULATIONS (launched 2026-09-12)
v6_voxel.pt was trained on 20-frame rollouts; in a what-if world it runs alone
for 30-100 frames. v7_2/train_long.py fine-tunes it, three arms in parallel,
same init / clip order / seed / lr 1e-4 / 2000-batch budget / checkpoint rule;
ONLY the rollout-length schedule differs:
- fixed:40-60 (my proposal) -- random length 40-60 every batch
- curriculum:40,50,60,70,80 (Hamza's) -- next length only when val error over
  frames 1-H fails to improve by 1% for 2 checks in a row (or 400 batches);
  stops when 80 stops improving
- fixed:20-20 -- control, isolates "trained more" from "trained longer"
DATA FIX NEEDED FOR LONG WINDOWS: v5's SpatialDataset keeps only objects
present for the whole window. On 600 TRAIN clips, windows where a kept object
collides with a DROPPED one: 1.9% at 20 frames, 20.5% at 50, 46.7% at 80. All
three arms therefore hand entering objects over at their first frame (v7's
to_sim velocity + v7.1 entry correction), skip their first 15 frames in the
loss, and drop objects that disappear.
Checkpoint rule (fixed before running): every 50 batches, one fixed set of 200
val clips x 80 frames; select on mean position error over frames 1-80.
Init v6_voxel on those 200 windows: mean1-80 0.1012, @20 0.0347, @40 0.0968,
@60 0.1619, @80 0.2211 (a full contact distance is ~0.117, so by frame 80 the
model is off by nearly two). Speed with all three running: fixed ~7.1 s/batch,
curriculum at 40 ~5.8, control ~3.2. Peak memory 1.2 / 1.0 / 0.6 GB, free RAM
~0.5 GB, so the curriculum's 80-frame stage may page until the control ends.
Then v7_2/run.sh: each arm's best checkpoint in v7.1's builder (entry fix on)
on VAL, setting by the usual rule, bar = beat v7.1 VAL 3024/3391 -> TEST once;
control tested too if any long arm is; v7_2/decide.py pairs vs v7.1 test (cal
+30) and vs control at the same setting. Logs: v7_2/train_*.log, v7_2/runs/.

## v7.2 RESULT: LONGER TRAINING SIMULATIONS DID NOT HELP (2026-09-13)
Val rollout (200 clips x 80 frames, mean error frames 1-80; init 0.1012):
| arm        | best  | at batch | @20    | @80    | at the end of training       |
|------------|-------|----------|--------|--------|------------------------------|
| fixed 40-60| 0.0979| 50       | 0.0347 | 0.2107 | 0.1089 at 2000 (WORSE than init) |
| curriculum | 0.0980| 100      | 0.0350 | 0.2091 | stopped at 500: no stage gained 1% |
| control 20 | 0.0983| 400      | 0.0346 | 0.2118 | 0.1015 at 2000               |
All three gain the same ~3%, so the gain is "trained a bit more", not length.
Long-window training then degraded steadily (fixed 0.0979 -> 0.1089); the
curriculum advanced every 100 batches because nothing at 50/60/70/80 improved
by 1%. Error at frame 80 stays ~0.21 (almost two contact distances) in every arm.
VAL benchmark in v7.1's builder, cal +30 [per question]: v7.1 89.2% (3024)
[67.8]; fixed 88.9% (3014) [67.6]; curriculum 88.5% [66.6]; control 89.0%
[67.7]. Best settings: fixed 3014, curriculum 3004 (cal +45), control 3020
(cal +60). NONE beat the bar -> TEST NOT RUN, untouched. v7.1 stays best.
Reading: the drift over 30-80 frames is not something more gradient steps on
longer windows can remove -- consistent with the earlier finding that three
training objectives and five physics fixes left the benchmark flat. Do not
re-run horizon/curriculum fine-tunes of this model.

## APP: GENERAL WHAT-IF REQUESTS + A 3D VIEW THE USER TURNS (2026-09-12)
Hamza: removal-only is too narrow, and a fixed menu of change types is too
("it shouldnt be limited to js this, thats the whole purpose of kinda generalised
model"). The GIFs must not auto-rotate; the user turns the view.

WORLD BUILDER. v7/cf_world.build_world(edits=...) (v7_1 regenerated with
make_v7_1.py): per-object change EVENTS at any frame (pos, vel, aim, heading,
turn, speed, still_speed, speed_factor, push, vanish, replace_track), ADDED
objects (indices N, N+1, ...), mass_scale / radius_scale applied AFTER the
attribute encoder (the learned embedding keeps seeing trained attributes; only
inertia and contact geometry change). Influence spreads from changed and added
objects by the same lost-contact / new-contact rules. Returns yaw per frame.
IDENTITY VERIFIED: 28 removal worlds (14 VAL clips, with and without a removal)
identical before/after in events, taint, voxel near-lists, positions and modes.
Every benchmark number above stands.

REQUESTS (app/intervene.py). The LM writes a PLAN of composable events: remove
(or disappear at a frame), add (standing, or rolling in from a side; labels A, B
for later events), move (speed_factor, slow/normal/fast, turn, toward a place or
direction, from_side, arrive_frame, push, stop), place, shift_time, set
(material, shape, size_factor, mass_factor, bounciness, friction_factor), pin
(mass x1e4 + stop), world {friction_factor, predict_frames <= 120}. Places are
relational (marked spot, object at a frame +- a side, between two, a recorded
collision, camera-aligned (right, far), centre). Side entries start just outside
the real camera picture; arrivals are timed with the world model's own friction
(single-object free rollouts). intervene.friction(model, scale) patches
drag_coeff / rot_drag_coeff on the model instance for one simulation and always
restores them. A world friction change hands EVERY object to the simulator from
its entry (friction shapes whole paths); with friction 0 speeds after the video
stay constant (checked 0.0069 -> 0.0067).

MEASURED: A HAND-OFF ON THE ENTRY FRAME IS UNRELIABLE. Video 10002, green cube
handed to the simulator at its entry frame with NO change (speed x1): its
entry-frame velocity was ~50% too fast and ~45 deg off, and the simulation sent it
off the table, missing its real collision at frame 49. App fix:
Scene.steady_state(k, f) takes the velocity from recorded frames entry+4 ..
entry+12 (before the first collision) and extrapolates the position; used for
every hand-off within 12 frames of entry that sets no state of its own, and for
shift_time. After it: 2x speed stays on the table (hits the gray rubber sphere at
79); material/shape and size changes keep the real first collision (49-50). The
BENCHMARK path (to_sim + entry profile) is unchanged; steady_state there is an
untested candidate.

Event tests, video 10002, 15 plans including a combination: all run, 3-14 s per
world on CPU, a pinned object moves <= 1e-5, removal through the plan path is
identical to build_world removal. Only removals have dataset answers: the page
labels every other change an unscored prediction and flags extrapolations (size,
mass, friction are outside the training data).

PAGE (app/static/index.html). No pre-drawn GIFs: tracks go to the page as JSON
and a canvas renderer ported from src2/render3d draws them (same pinhole camera,
painter's order, Lambert / Blinn shading). Drag = turn, wheel = zoom, sliders,
"video camera" view = camera_fit.json, both panels share camera and timeline,
play / scrub / speed, faint ghosts of removed or changed objects, click to mark a
spot ("here"), per-object buttons (remove, 2x, stop, glue down), examples. "Save
as GIF (this view)" renders server-side with render_worker at the chosen angle.

CONTENTION: v7.2's three training arms ran during these tests; their batches went
~7 -> ~12 s while two extra model processes ran. Test the app one process at a
time while training runs.

LANGUAGE MODEL PLANS (Groq gpt-oss-120b; 17 sentences on test videos 10002 and
10020). All came back as valid plans. Right first time: removal, 2x speed, a cube
at the marked spot, metal + 3x heavy, no friction, glued down, "the 3 seconds
after the video" (75 frames), enter 50 frames earlier + 2x size, every object 2x
heavy. Wrong, each measured, fixed, retested:
- "came in from the right": the LM copied the entry frame as a start, so the cube
  arrived at 137 instead of 49. Resolver: arrive on time by default; a given frame
  is a start only if it is not the real entry frame. The LM then once imitated it
  with place + a 180 deg turn; prompt rule against that. Retest: move from_side,
  the cube hits the gray sphere at 44.
- "a sphere rolls in and hits the brown cylinder": the LM filled both 'at' (a
  guessed start) and 'toward'; arrival was timed to the target's centre; a target
  that never moves gave arrival frame 0 (it started on top of the cylinder).
  Resolver: 'toward' wins, arriving = touching (contact distance), default
  arrival frame 40 unless the place's frame is a real moment (>= 25), an added
  object's start step is replaced, not repeated.
- "stopped at frame 40 and a cylinder appeared behind it": the LM used place (a
  jump). Rules: stop = move stop, place only when the user says it is elsewhere,
  places named from the scene. Retest correct.
- "gravity pulls everything left": the LM faked it with one push per object. Added
  world slope = a steady pull (cf_world edits["accel"], an external force like
  friction; every object simulated from entry). Retest: slope left; "tilted
  towards the camera" -> slope near. Removal identity re-checked: 28/28.
- "a red sphere rolled in from the right at frame 30" (10020): frame read as the
  arrival. 'frame' on an add is now when it appears. Retest: enters at 30.
The LM's plan for the same sentence varies between runs; the resolver's guards
(arrival default, copied entry frame, 'toward' over 'at') are what keep results
stable, not the prompt alone.

## TEMPORAL-GNN IDEA ON CLEVRER: CEILING PROBE SAYS NO (2026-09-14)
Hamza proposed reusing the jet project's temporal graph network
(Desktop/DNN/neural_evasion, V3 pilot) on CLEVRER. Context from DNN/crosstest/
CROSSTEST.md, all inside our own what-if builder, TEST: jet world model = straight
lines 86.8% options / 64.4% questions; learned v7 88.1% / 66.6% (z = 2.79 vs straight
lines); jet model + 3 constants fitted on TRAIN 88.0%; v8 laws 89.4%. The "86%" was
never the TGN.
In the jet game the TGN infers per-missile HIDDEN parameters (true values 42/48 matches
vs no GNN 22/48). CLEVRER objects hide almost nothing (laws fit: one bounciness, mass
ratio 1.0); the only hidden per-object quantity is the true motion at hand-off.
v7_1_oracle/make_oracle.py generates cf_world/evaluate from v7_1; with ORACLE_HANDOFF=1
the hand-off state is a least-squares line through recorded frames t-3..t+8 (FUTURE
frames, clear of recorded contacts, skipping 10 biased frames after entry), used on
2,784 of 3,691 hand-offs (75%). CEILING ONLY, never reportable.
VAL cal +30: oracle 3022/3391 [68.0% per question] vs v7.1 3024 [67.8%]; paired on
3,387 options 34 vs 36, z = -0.24; +45 z = -1.01; +60 z = -0.43; questions 640 vs 638.
-> Even perfect hand-off states do not move the benchmark, so any learned state
estimator (TGN or otherwise) is capped at ~0 here. Not built; test untouched.
Answer added to presentation/CausalVis_Presentation_Explainer.docx section 5.

## v7.3: LATER HAND-OFF + PATH-CHANGE DETECTOR -- NEW BEST (2026-09-18)
v7_3/make_v7_3.py generates cf_world/evaluate from v7_1 with two env knobs, both off
by default (records identical to v7.1 with them off: 19/19). PHYS_RADIUS_SCALE scales
only the physics radius; CF_LOOKBACK replaces the 12-frame hand-off look-ahead.
Detector sizes question (Hamza: "use one size for both"), all VAL, 3,387 options:
- detector at the model's size (x1.0): 84.9% [55.3] (TEST 84.7 / 55.8) -- much worse
- physics at the detector's size (x1.2, no retrain): 89.1% [67.4] vs 89.2% [67.8] -- tie
- exact sizes only for simulated pairs / voxel gating / 2-of-3 votes: all worse
  (they cut invented collisions but lose more real ones: the x1.2 margin absorbs drift)
Hand-off look-ahead sweep, VAL best setting (options / questions):
  lookback 1: 90.2% / 71.4 | 3: 90.4% / 70.7 | 6: 89.4% / 69.2 | 9: 89.9% / 69.9 |
  12 (v7.1): 89.2% / 67.8 | 20: 87.2% / 63.7 | 30: 85.4% / 59.0
  Following the real recording until just before the lost contact beats handing over
  early: every extra frame of real path is drift avoided.
Detector: "distance rule OR path change" (cal events UNION kick2 events), chosen on VAL
(lb12: z=+3.13; lb3: 3076 vs 3061, z=+3.27). Catches real hits the distance rule skips.
TEST, run once per pre-registered step (bar 3024 cleared by lookback 3 with 3064):
| system                              | options          | questions      | missed | invented |
|-------------------------------------|------------------|----------------|--------|----------|
| v7.1 (lookback 12, cal)             | 88.4% (2939/3325)| 66.9% (622)    | 197    | 189      |
| v7.3 lookback 3, cal                | 90.2% (3004/3332)| 71.4% (664)    | 143    | 184      |
| v7.3 lookback 3, cal OR path change | 90.4% (3007/3325)| 72.0% (670)    | 129    | 189      |
Paired vs v7.1: lookback 3 + cal 118 vs 59, z = 4.43; + union 131 vs 63, z = 4.88.
Union vs cal on the same test world: 14 vs 5, z = +2.06. The union was chosen on VAL
before the test look; test was consulted twice for this world (cal, then union).
The learned world model now beats the v8 law engine on test (89.4 / 69.8; +entry 89.8 /
70.9). Approx. vs papers: VRDP no extra labels 89.9 / 75.7 -> +0.5 / -3.7.
The app (app/server.py) still serves v7.1 (lookback 12, cal) until switched.

## UNCERTAINTY-AWARE HYBRID DETECTOR -- tested, NOT adopted (2026-09-18)
Hamza: combine the strict voxel check (VAL invented 65) with the union's low missed.
Idea tested: node memory = frames each object has been simulated; a simulated pair's
candidate hit counts only if its voxel gap <= g0 + k * (summed simulated age), else
the distance rule / path change as in v7.3. Offline on v7.3 VAL records (lookback 3),
fitted on half the videos, checked on the other half, both folds:
  fold 0: 1543 vs union 1542 of 1691 (+1, z=+0.58); fold 1: 1526 vs 1534 of 1696 (z=-1.89)
Diagnostic (first distance-rule hit per option, real vs invented, gap > 6 voxels): at
every simulated age real collisions with a gap outnumber invented ones (16-40 frames:
102 real vs 19 invented; 41-80: 73 vs 28; 81+: 37 vs 32). Real and invented near-misses
look the same to every signal we record (gap, age, path change), so no geometric or
age-based rule separates them. The jet project's temporal-GNN strength (inferring
hidden per-object behaviour) has nothing to infer in CLEVRER (perfect hand-off states
+0, age memory +0); it belongs in a domain with hidden intent (road users).

## APP NOW SERVES v7.3 (2026-09-18)
app/server.py loads v7_3/cf_world.py with CF_LOOKBACK=3 and answers with
collision_frames() = calibrated rule OR path change (kick2), hits within 6 frames
merged. Verified: the app's answers equal the evaluated v7.3 VAL records on 33/33
options across 10 worlds.

## PAPER READINESS + LITERATURE CHECK (2026-09-18)
Full review: https://claude.ai/artifact/8eLGgw5iwuvCh2kLdRFRU1 (private). Key facts for any paper:
- PRIOR ART: CRCG (Ishay et al., WACV 2024, arXiv 2506.10753) already keeps unaffected objects on
  their perceived states and simulates only reached objects from the first affected frame (+ new
  collisions from simulated objects). That is v7's core idea -> cite it; our part is the hand-off
  timing study (single variable: lookback 3 vs 12 on TEST +1.8 / +4.5, z = 4.43) and +30 frames.
  Rovai (arXiv 2605.15967, 2026) replays the ANNOTATION log with no physics: 86.7 / 59.9.
- FAIR COMPARISON GROUP IS †‡: we read NS-DR's Mask R-CNN masks + predicted attributes (trained
  with labels). VRDP †‡ 94.8 / 84.3, CRCG_VRDP 96.1 / 87.8, ALOE (no labels) 91.4 / 75.6.
  "VRDP no extra labels 89.9 / 75.7" (above) is VRDP with UNSUPERVISED slot objects -- the wrong
  reference class for us; do not use it as the headline comparison.
- CLEVRER's EvalAI test server (challenge 667) closed 2026-01-16. Our TEST = 500 CLEVRER
  validation videos (10000-14999) never trained or tuned on; VAL = another 500 of them.
- v7_3/subsets.py (TEST, 3,325 options / 930 q), per option / per question:
  | subset (ALOE App. C style)                       | share | v7.3        | no physics  |
  | removed object never collides (E)                | 47.2% | 93.0 / 77.2 | 87.7 / 59.9 |
  | collides, video answers it (M)                   | 22.5% | 94.7 / 83.7 | 94.9 / 83.3 |
  | collides, needs counterfactual reasoning (H)     | 30.3% | 83.2 / 55.3 | 61.2 /  5.7 |
  Answer as if nothing was removed (annotated video events): 82.9 / 53.0 overall.
  In E the answer key says "collide" for 168/1,630 options (10.3%, 35.3% of E questions) where
  the video shows no collision (1 the other way) -> post-video events, model-free evidence.
  v7.3's 318 wrong options: invented during video 138, missed after video end 103, invented
  after end 51, missed visible 26.
- Missing before submission: v8 laws at lookback 3 + same detector (fair learned-vs-laws),
  one ablation table, seeds, predictive questions.

## PAPER EXPERIMENTS: CLEAN PIPELINE + FAIR COMPARISON (2026-09-19) -- see paper_exp/
Everything below uses paper_exp/ (single generated builder, PREREGISTRATION.md, RESULTS.md).
- ANNOTATION LEAK FOUND AND REMOVED: the old tracks took the object list, in/out frames and
  cube-yaw breakpoints from the scene annotation. paper_exp/detonly.py rebuilds tracks from
  NS-DR detections only (data/trajectories_3d_det; inventory recall 99.6%, precision 99.4-99.7%
  on every set). Questions are parsed from text (paper_exp/qparser.py, TRAIN templates; CF
  100% exact). Cost of the fix: 15 VAL options.
- FRESH TEST-B: 898 CLEVRER-train videos in our test split (1,689 CF q), never trained/scored.
- FAIR SIMULATOR COMPARISON (same hand-off sweep for each), TEST A+B (9,350 options):
  textbook laws 91.4 / 75.0 > learned 90.4 / 72.3 (z +5.0) ~ straight+friction 90.2 / 71.9
  (n.s.) > straight 89.7 / 70.2 > no physics 81.4 / 50.1. The earlier "learned beats laws"
  compared lookback 3 vs 12 -- WRONG, superseded. Learned-model parts do not matter here
  (no learned energy +1.3, no friction +2.1, no voxel -1.2 on TEST-A); what matters is the
  evaluation logic: extension past the end +4.5 (z 16), hand-off 3 vs 12 +2.1 (z 5.2).
- Other CLEVRER types (VAL-A, settings chosen there): descriptive 90.7, explanatory
  87.4 / 78.8, predictive 90.7 / 82.3 (paper_exp/executor.py, NS-DR semantics).
- CLEVRER-Humans valid: counterfactual but-for rule 64.4 / 43.5 (best published 54.0 / 34.2).
- ComPhy (oracle tracks, inferred mass/charge): dev CF ~78.5 / 65.4 (CPL 68.3 / 29.1).
- Predictive per simulator (TEST A+B): straight 91.4 > strfric 90.9 > learned 90.5 > laws 88.1
  (laws z -3.8). No simulator is best on both question types.
- Other question types TEST A+B: descriptive 90.9, explanatory 88.4 / 79.7, predictive 90.5 / 82.6.
- ComPhy val (oracle tracks, inferred mass/charge): CF 79.5 per option, predictive 86.9
  (CPL-Gt 74.0 / 87.6; humans 80.0 / 88.0). Test servers of CLEVRER and ComPhy are closed.
- CoPhy BallsCF test: engine 0.89-2.60 vs copy B 5.74-7.69 (released data differs from paper).
- Seeds of the final fine-tune: 90.1-90.5 options on TEST A+B (spread ~0.4).

## AUDITS FOR THE PAPER (2026-09-19)
- Evidence audit (paper_exp/audit_traces.py, TEST A+B, learned): of correct answers whose
  deciding event can be checked against the annotation, 98.2% rest on a real event (15-frame
  tolerance; 93.4% at 6). The detector fires a median 4 frames BEFORE annotated contact (the
  calibrated margin). 40.5% of correct answers are decided in simulation: uncheckable in CLEVRER.
- Explanatory by but-for (paper_exp/butfor_explanatory.py): 77.7% vs chain rule 88.4% (TEST
  A+B, z -27). CLEVRER's key follows the chain rule.
- MODEL-FREE (paper_exp/key_consistency.py): when CLEVRER's explanatory key says Z is
  responsible for the X-Y collision, CLEVRER's own counterfactual key says X and Y still
  collide without Z in 113/320 (35.3%); the reverse 1/1,366. With CLEVRER-Humans (but-for 64.4
  vs chain 60.9): CLEVRER's "responsible" is chain ancestry, not counterfactual dependence.
  Caveat: the counterfactual answer does not say when the pair collides.
- CoPhy BallsCF event audit (+-1 step, same pair): 2 balls P 0.85 / R 0.65; 4 balls 0.61 / 0.49.
- Figures fully specified in paper_exp/FIGURES.md (data in paper_exp/figures/data). Hamza draws
  them. Venue order in paper_exp/VENUES.md (arXiv -> CVPR 2027 -> NeurIPS 2027 E&D -> TMLR).
- (2026-09-20) Key-consistency caveat checked: in all 113 "responsible, yet still collides"
  cases the pair collides once in the video; engine world without Z collides within 10 frames
  of it in 56. ComPhy predictive evidence audit: 97.7% of correct answers rest on a real future
  event (future is annotated there). CoPhy 6 balls P 0.49 / R 0.41. Simulators agree on 91.4%
  of CLEVRER CF options (sim_agreement.py). Clean runtime: learned 2.24 s, laws 1.36 s per
  question. Paper draft in paper/ (main.tex, supp.tex, refs.bib); all numbers final.

## PUBLIC RELEASE (2026-09-25)
Pushed to https://github.com/bukhari-hamzamukhtar/CausalVis on top of the layout Hamza had
made there (legacy/v1, legacy/v2, legacy/v3, v3_2/src). Three commits add the new work:
the engine (src2, v4 to v8, probe, checkpoints, splits, camera), the paper experiments
(paper_exp, including records/test_records.zip with one line per answered option of every
test run), and the paper, app, report and README.
Kept out of git on purpose: data/ and external/ datasets, zechennlp questions, the bulk run
records under paper_exp/runs and the version folders, node_modules, and the 2.6 GB second
copy of the detections at legacy/v2/data/processed_proposals (legacy/v1 has them already).
Local main now tracks origin/main; the older local-only commits were superseded.
