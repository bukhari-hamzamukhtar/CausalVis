"""
intervene.py  —  CausalVis V3, Phase 2a: the do(.) operator
===========================================================

This is the piece V1/V2 could never have. A world model can be EDITED: we
change the scene, then RE-SIMULATE it. Each function here takes a scene (the
start state the physics engine reads) and returns a NEW start state with one
surgical change applied. Nothing else is touched.

    do_nothing(scene)     -> the factual scene, unchanged  (the baseline)
    do_remove(scene, k)   -> object k is gone from the world
    do_freeze(scene, k)   -> object k is pinned where it is (still pushes, never moves)

A "scene" is just the tuple the engine already uses:
    q0  [N,2]  positions        v0 [N,2]  velocities
    attrs [N,15] attributes     present [N]  on-screen flags

REMOVE is the important one: it answers CLEVRER's core counterfactual,
"what happens without object X?"  We flip X's present flag to 0, and the
physics engine then excludes it from every force and from the potential
entirely -- exactly as if it had never been in the scene.
"""

import torch


def _clone(scene):
    q0, v0, attrs, present = scene[0], scene[1], scene[2], scene[3]
    return [q0.clone(), v0.clone(), attrs.clone(), present.clone()]


def do_nothing(scene):
    """The factual world, untouched. Every counterfactual is compared to this."""
    return _clone(scene)


def do_remove(scene, k):
    """
    Delete object k. Its present flag goes to 0 (the engine drops it from all
    forces and from the potential), and its position/velocity are zeroed so
    nothing can accidentally read a ghost.
    """
    q0, v0, attrs, present = _clone(scene)
    present[k] = 0.0
    q0[k] = 0.0
    v0[k] = 0.0
    return [q0, v0, attrs, present]


def do_freeze(scene, k):
    """
    Pin object k: it stays on screen and still pushes others, but never moves
    itself. We zero its start velocity and return a 'frozen' mask so the roller
    can re-pin it every step. (Use simulate(..., frozen=mask) in readout.py.)
    """
    q0, v0, attrs, present = _clone(scene)
    v0[k] = 0.0
    frozen = torch.zeros_like(present)
    frozen[k] = 1.0
    return [q0, v0, attrs, present, frozen]
