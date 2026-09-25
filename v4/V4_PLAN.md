# CausalVis v4 — cone-free, 3D-native, answers from a stored future

## Why v4 exists

v3 has three structural problems that no amount of training fixes.

**1. The light cone does the causal work, not the model.**
`benchmark_eval.py` computes which objects an intervention can reach, freezes
everything else to its OBSERVED trajectory, and simulates only the rest. The
benchmark reports `outside cone (observed, no model): 100.0% (831/831)` -- 831
choices answered perfectly because the cone handed them over. The model is told
the answer to "who is affected", which is most of the causal question.

**2. The predicted future is thrown away.**
The rollout collapses into a list of `{frame, i, j}` collision events. Every
position, velocity and orientation the model computed is discarded frame by
frame. Answering is then `if pair in cf_pairs` -- set membership, not reasoning.
This is why a model 25% better at trajectories scored identically (z=0.39,
p=0.69): the answer path cannot see trajectories.

**3. The 3D annotations are not used.**
`data/trajectories_3d/` has metric position, size, volume, depth, contact
geometry for all 20k clips. Every benchmark number so far was run on the 2D
pixel data.

## What the literature says (checked 2026-09-09)

- **GNS** (Sanchez-Gonzalez et al. 2020, arXiv 2002.09405) recomputes graph
  edges by nearest neighbour EVERY timestep. Interactions are discovered, never
  pre-pruned. Cone-free is the standard, not the experiment.
- **VRDP** (NeurIPS 2021, the 94.8% SOTA) simulates the whole scene with a
  differentiable rigid-body engine and fits mass/restitution per object. No cone.
- **ContPhy** (arXiv 2402.06119) parses a question into a program with an LLM,
  then executes it symbolically over simulated state. This is exactly the
  "store the future, then answer from it" design.
- **Sim2Reason** (2604.11805) turns simulator traces into QA.

## Scope discipline

CLEVRER is rigid bodies on a table. There is no diffusion, no relativity, no
thermodynamics in this data. Adding those terms would be weight the optimiser
must learn to ignore. Three physics gaps are REAL and measured:

- **angular momentum** -- cubes rotate; measured yaw jumps 74.1 -> 87.4 deg at
  a collision. v3 has no rotational state at all.
- **restitution** -- present in `attrs[:,1]`, feeds the attribute encoder, and
  removes ZERO energy. It is dead code today.
- **contact geometry** -- `r_i + r_j` treats every object as a sphere. Exact
  for spheres/cylinders, understates cube contact by 19-41%.

## Architecture

```
v4/
  world.py       cone-free world model: full-scene rollout, no teacher forcing,
                 conservative pairwise potential + drag + restitution + torque
  store.py       RolloutStore -- the FULL predicted future as structured 3D
                 annotation (pose, velocity, orientation, contacts, ledger)
  answer.py      question -> answer FROM the stored annotation
                 tier 1 deterministic program executor
                 tier 2 LLM reading the annotation as text (never pixels)
  evaluate.py    benchmark with NO cone; reports how much the model decided
  train.py       trains world.py on data/trajectories_3d
```

## Phases, each measured before the next

**P1. Cone-free rollout + store the future.** The model simulates every object
from the intervention frame. Nothing is teacher-forced. `RolloutStore` keeps
the whole predicted trajectory. Success = the store round-trips and the factual
rollout reproduces observed collisions above chance.

**P2. Answer from the store.** Deterministic executor reads the stored future.
Success = benchmark runs end-to-end with `outside cone` path GONE and
model-decided fraction far above 18%.

**P3. Rotational dynamics + restitution.** Angular momentum, torque from
off-centre contact, restitution that actually dissipates. One variable at a time
against P1.

**P4. Shape-aware contact at scale.** Run `fit_yaw_track` over 20k (21 h) so
contact geometry is direction-exact rather than direction-averaged.

**P5. LLM readout.** Structured annotation -> text -> answer. Compared against
the deterministic executor on the same stored futures.

## Non-negotiables carried from v3

- Pairwise interaction stays exactly momentum conserving (measured 1.6e-07).
  Drag is external and separately logged.
- Forces via autograd; never wrap rollouts in `no_grad` (use create_graph=False).
- Radius measured, never learned -- it collapses to the bound.
- ONE variable per run. Three-changes-in-one-run cost a full day.
- Report on the TEST split. Never compare a val-selected checkpoint against an
  unselected one.

## Expected outcome, stated before running

Removing the cone will make the benchmark score DROP. The cone was contributing
831 free correct choices. That is the point: the resulting number is the first
honest measurement of what the world model can actually do, and the gap between
it and 86.1% is the size of the crutch.


## MEASURED 2026-09-09: what removing the cone revealed

Cone-free benchmark, test split, 400 questions / 1425 choices:

| model | per-choice | model-decided share |
|-------|-----------|---------------------|
| v3_2  | 62.2%     | 100%                |
| v3_5  | 61.9%     | 100%                |

(v3 WITH cone on the same data: 92.6% per choice, 18% model-decided.)

Two findings.

**1. The 30-point drop is the crutch, exactly as predicted.** 100% of choices
are now the model's own work instead of 18%. 62% against a 50% chance baseline
is the first honest measurement this project has produced.

**2. Better physics STILL does not show, and now we know why.** Full-scene
rollout error from frame 0:

    10 frames  0.0034   3% of contact distance
    20 frames  0.0103   9%
    40 frames  0.0416   36%
    80 frames  0.1119   96%  -- error EQUALS the contact distance

The cone was not only handing over answers, it was keeping rollouts SHORT:
tainted objects were initialised at their observed state a few frames before
the event. Cone-free means predicting 128 frames from frame 0, and the model's
usable horizon is 20-40. Past frame 80 every collision prediction is a coin
flip, so v3_2 and v3_5 tie because both are equally far outside their range.

**The bottleneck is now long-horizon rollout stability.** Not the answer path,
not perception. A VLM reading the stored future would be reading noise after
frame 80. P3 (rotational dynamics, restitution) and long-horizon training are
therefore the critical path, ahead of P5.
