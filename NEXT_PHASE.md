# CausalVis — next phase plan

Written after checking the literature. Two things are now confirmed from
outside our own measurements, and they change the priority order.

---

## CONFIRMED FINDING 1: the data is dissipative, and it is documented

We measured 0.85% speed loss per frame. That is not a measurement artefact.
Two independent confirmations from the literature:

1. The CLEVRER paper states the videos are simulated with the **Bullet**
   physics engine on a flat tabletop. Bullet applies ground friction by
   default.

2. The VRDP paper (Ding et al.), the 94.8% state of the art on this
   benchmark, describes CLEVRER's physics explicitly: all objects have the
   same size and the same friction coefficient (except the sphere rolling on
   the ground), and each object has a different mass and a different
   restitution coefficient.

So there are **two** dissipation channels in the data:

- **ground friction** — continuous drag, our measured 0.85%/frame
- **restitution < 1** — energy lost at every collision, and it is per-object,
  not global

Our architecture represents neither. `F = -grad V` is a conservative force
field; conservative forces cannot dissipate energy. This is not "hard to
learn", it is **structurally impossible**. Three training fixes failed
because they were all fighting the architecture rather than changing it.

This also explains the specific error signature: objects that do not lose
speed travel too far, which produces BOTH missed collisions (overshoot the
partner) AND invented collisions (reach something they should not have).
That is exactly our 554 missed / 448 invented split.

It also explains part of why VRDP wins: it explicitly fits mass, restitution
and friction per object.

## CONFIRMED FINDING 2: the fix is an established formulation

Dissipative Hamiltonian dynamics is solved in the literature. Three relevant
lines, all applicable:

- **Dissipative HNN (D-HNN)**, Sosanya & Greydanus — adds a Rayleigh
  dissipation function alongside the Hamiltonian
- **Port-Hamiltonian Neural Networks (pHNN)**, Desai et al. — generalises
  Hamilton's equations with a positive-semidefinite damping matrix D(q) plus
  an external port; reports ~23% accuracy improvement on damped systems
- **Dissipative SymODEN**, Zhong et al. — same idea for mechanical systems

Port-Hamiltonian form:

    [q̇]   ( [ 0   I ]          )  [∂H/∂q]     [ 0  ]
    [ṗ] = ( [-I   0 ]  +  D(q) )  [∂H/∂p]  +  [G(q)] u

with D positive semidefinite. Our current model is the D = 0, u = 0 special
case. This is not abandoning the architecture, it is completing it.

**Integration matters.** A symplectic integrator on a dissipative system is
no longer symplectic. The correct scheme for linear damping is **conformal
symplectic**: split the step into the conservative part (existing
kick-drift-kick) followed by an exact exponential decay of momentum. For
linear drag `p <- p * exp(-c*dt)` is exact, cheap and structure preserving.
Do NOT simply subtract a drag force inside the existing leapfrog — that
breaks the scheme and will look like the drag term "failed".

---

## TASK 1 (do this first): the dissipation term

**Single variable. Nothing else changes in this run.** Start from the v3_2
training configuration exactly, including the legacy loss.

1. **Ground drag.** Learn a per-material damping coefficient c >= 0
   (parameterise with softplus so it cannot go negative). Two materials, so
   two scalars, plus optionally a third for spheres since VRDP's description
   says spheres roll differently. Apply as a conformal symplectic half step:
   `p <- p * exp(-c*dt/2)` before and after the existing kick-drift-kick.

2. **Restitution at collision.** Per-object restitution `e` already exists in
   the `properties` head. Check whether it actually removes energy at contact
   or only shapes the potential. If the latter, it must remove energy: damp
   the relative normal velocity of an approaching pair by the restitution
   factor.

3. **The audit ledger gets better, not worse.** Momentum is no longer
   conserved for the object system alone, which is physically correct because
   the table is an external body. The pairwise interaction stays exactly
   momentum conserving. Log the momentum removed by drag per frame so the
   ledger reports where it went. Conservation becomes a balance sheet rather
   than a constant. Print it: "momentum transferred to table: X", "momentum
   lost to numerics: ~1e-7".

**Sanity check before training:** with the potential switched off entirely, a
single object launched at CLEVRER's typical speed should lose ~34% of its
speed over 40 frames. If the drag term alone cannot reproduce the measured
0.85%/frame, the implementation is wrong and training will not fix it.

**Report on the test split only**, and report all of:
- `dynamics_check.py` rollout error at 20 and 40, and the ratio vs constant
  velocity — this is the headline
- `collision_eval.py` reproduction rate, error at +15, interpenetration
- momentum selftest (pairwise interaction alone should still be ~1e-7)
- the learned drag coefficients, and whether they match the measured
  0.85%/frame

**Pre-registered prediction:** if dissipation is the bottleneck, the
constant-velocity ratio at 40 frames should move well past 1.2x, because
constant velocity cannot decelerate either and the model now can. If it does
not move, dissipation is falsified as the bottleneck and we look elsewhere.

---

## TASK 2: rebuild trajectories in 3D

`fit3d.py` is validated at 0.95 median silhouette IoU. Use it to regenerate
the trajectory files with metric 3D quantities:

- exact position on the table plane (removes perspective distortion)
- exact radius per object (no longer inferred from pixel area, which is
  depth dependent)
- orientation for cubes, which the current pipeline does not have at all

Write these to a NEW data directory and keep the old one intact, so the two
can be compared under identical training.

**One check first:** the raw CLEVRER annotation files may already contain
ground-truth 3D motion traces — the paper says videos are "associated with
the ground-truth motion traces and histories of all objects". We have been
working from derived proposals containing only 2D masks. If the raw
annotations have 3D, use them to VALIDATE fit3d.py's output, which turns a
0.95 IoU claim into a direct positional error in metric units. That is a far
stronger statement. If they do not, fit3d.py is the only source and the IoU
number stands as the validation.

---

## TASK 3: retrain on 3D plus dissipation

Only after Tasks 1 and 2 have reported separately. Then one combined run so
the final model has both. Same metric set.

---

## TASK 4: the orbit viewer (the demo)

`fit3d.py` outputs real meshes at real poses, so free viewpoint rendering is
a rendering task rather than a research problem. Strongest single visual
available: take a 2D video, show the reconstructed 3D scene, orbit it. It is
also the validation criterion made visible.

---

## TASK 5: the grounded explainer (last)

The language model reads the **audit ledger only** — per object energy,
momentum, drag losses, the factual versus counterfactual event diff. Never
pixels, never frames. It cannot invent a collision because the collision list
is an input. Structure: the model writes a query against the event record,
the record returns facts, a deterministic renderer speaks them.

---

## Rules that still hold

- The **pairwise interaction** must stay exactly momentum conserving. Drag is
  an explicit, separately logged external force. Never clamp forces per
  object; if bounding is needed, rescale all forces by one shared scalar.
- Forces are computed by autograd, so gradients stay enabled during rollouts.
- One variable per run. The three-changes-in-one-run mistake cost a full day
  of attribution.
- Report on the test split. `v3_4` scored better on val purely because it was
  selected as best of 30 epochs on val while `v3_2` was a single unselected
  measurement. That is selection bias, and the test split reversed it.
- `v3_2_dynamics.pt` stays the production checkpoint until something beats it
  on held out data.
