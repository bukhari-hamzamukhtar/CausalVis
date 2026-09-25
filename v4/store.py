"""
v4/store.py  —  keep the predicted future instead of throwing it away
=====================================================================

WHAT WAS WRONG IN v3
--------------------
`benchmark_eval.py` rolls the world forward, and as each frame is computed it
tests a few pair distances, appends {frame, i, j} to an event list, and drops
the state. By the time a question is answered the only surviving artefact is a
set of colliding pairs, and answering is `if pair in cf_pairs` -- set
membership, not reasoning over a predicted world.

That is why a model measured 25% better at trajectories and 28% better at
collision error scored IDENTICALLY on the benchmark (70.9% vs 69.9%, z = 0.39,
p = 0.69). The answer path physically could not see the improvement. Fixing the
physics could never have moved that number.

WHAT THIS DOES
--------------
Keeps the whole predicted future as a structured 3D annotation: every object's
pose, velocity, orientation and presence at every frame, plus derived contact
events, plus the conservation ledger. The same object a renderer would need to
draw the predicted world, and the same object a language model can read as text.

The store is deliberately PLAIN DATA -- arrays and dicts, JSON-serialisable.
Nothing downstream should need the model to answer a question; if it does, the
store is missing a field.
"""

import json
import numpy as np


class RolloutStore:
    """The predicted future of one (video, intervention) pair.

    positions  [T, N, 2]   metric, world units / world_scale
    velocities [T, N, 2]
    yaw        [T, N]      radians; 0 for shapes with no visible orientation
    present    [T, N]      1 while the object exists and is on the table
    source     [T, N]      0 = observed, 1 = simulated by the model
    """

    def __init__(self, video, removed, obj_keys, attrs, T, N, world_scale=6.0):
        self.video = int(video)
        self.removed = sorted(int(k) for k in removed)
        self.obj_keys = [str(k) for k in obj_keys]
        self.attrs = np.asarray(attrs, np.float32)
        self.T, self.N = int(T), int(N)
        self.world_scale = float(world_scale)
        self.positions = np.full((T, N, 2), np.nan, np.float32)
        self.velocities = np.zeros((T, N, 2), np.float32)
        self.yaw = np.zeros((T, N), np.float32)
        self.present = np.zeros((T, N), np.float32)
        self.source = np.zeros((T, N), np.int8)
        self.ledger = {"momentum_to_table": 0.0, "steps": 0}
        self._events = None

    # -- writing ------------------------------------------------------------
    def write(self, t, q, v, present, yaw=None, simulated=True):
        q = np.asarray(q, np.float32).reshape(self.N, 2)
        v = np.asarray(v, np.float32).reshape(self.N, 2)
        pr = np.asarray(present, np.float32).reshape(self.N)
        self.positions[t] = np.where(pr[:, None] > 0, q, np.nan)
        self.velocities[t] = np.where(pr[:, None] > 0, v, 0.0)
        self.present[t] = pr
        self.source[t] = 1 if simulated else 0
        if yaw is not None:
            self.yaw[t] = np.asarray(yaw, np.float32).reshape(self.N)

    # -- derived events -----------------------------------------------------
    def contacts(self, contact_fn=None, thresh=0.02, release=0.02):
        """Collision events derived FROM the stored trajectory.

        Same approach gate and hysteresis v3 used, but computed after the fact
        from the stored future rather than mutated during the rollout. That
        matters: events are now a VIEW of the trajectory, so a question can be
        re-answered without re-simulating, and a different contact rule can be
        tried without touching the physics.

        contact_fn(i, j, t) -> contact distance; defaults to r_i + r_j from
        attrs[:,15]. Pass a shape-aware function to use true geometry.
        """
        if contact_fn is None:
            r = self.attrs[:, 15]

            def contact_fn(i, j, t):
                return float(r[i] + r[j])

        events, in_contact, prev = [], {}, {}
        for t in range(self.T):
            for i in range(self.N):
                if self.present[t, i] <= 0:
                    continue
                for j in range(i + 1, self.N):
                    if self.present[t, j] <= 0:
                        continue
                    pi, pj = self.positions[t, i], self.positions[t, j]
                    if not (np.isfinite(pi).all() and np.isfinite(pj).all()):
                        continue
                    d = float(np.linalg.norm(pi - pj))
                    gap = d - contact_fn(i, j, t)
                    key = (i, j)
                    if key not in prev:
                        prev[key] = d
                        in_contact[key] = gap <= thresh
                        continue
                    approaching = d < prev[key]
                    if gap <= thresh and approaching and not in_contact.get(key):
                        in_contact[key] = True
                        events.append({"frame": int(t), "i": i, "j": j,
                                       "gap": round(gap, 5),
                                       "approach_speed": round(prev[key] - d, 6)})
                    elif gap > thresh + release:
                        in_contact[key] = False
                    prev[key] = d
        self._events = events
        return events

    def exits(self):
        """Frames where an object leaves (present 1 -> 0 and never returns)."""
        out = []
        for k in range(self.N):
            live = np.flatnonzero(self.present[:, k] > 0)
            if live.size and live[-1] < self.T - 1:
                out.append({"frame": int(live[-1] + 1), "i": int(k)})
        return out

    # -- reading ------------------------------------------------------------
    def name(self, k):
        if k < len(self.obj_keys):
            return self.obj_keys[k].replace("_", " ")
        return "object " + str(k)

    def to_dict(self):
        ev = self._events if self._events is not None else self.contacts()
        return {
            "video": self.video,
            "removed": self.removed,
            "objects": [{"index": k, "name": self.name(k),
                         "radius": round(float(self.attrs[k, 15]), 5),
                         "mass": round(float(self.attrs[k, 0]), 4)}
                        for k in range(self.N)],
            "frames": self.T,
            "collisions": ev,
            "exits": self.exits(),
            "ledger": self.ledger,
        }

    def to_text(self, max_events=40):
        """The store as prose, for a language model. Never pixels -- only
        quantities the simulator actually computed."""
        d = self.to_dict()
        lines = ["Scene from video " + str(d["video"]) + "."]
        if d["removed"]:
            names = ", ".join(self.name(k) for k in d["removed"])
            lines.append("Intervention: removed " + names + ".")
        else:
            lines.append("No intervention (factual world).")
        kept = [o for o in d["objects"] if o["index"] not in d["removed"]]
        lines.append("Objects present: " +
                     "; ".join(o["name"] + " (radius " + str(o["radius"]) + ")"
                               for o in kept))
        lines.append("Simulated " + str(d["frames"]) + " frames.")
        if d["collisions"]:
            lines.append("Collisions in this world:")
            for e in d["collisions"][:max_events]:
                lines.append("  frame " + str(e["frame"]) + ": " +
                             self.name(e["i"]) + " hits " + self.name(e["j"]))
        else:
            lines.append("No collisions occur in this world.")
        if d["exits"]:
            lines.append("Objects that leave: " +
                         ", ".join(self.name(e["i"]) + " at frame " +
                                   str(e["frame"]) for e in d["exits"]))
        foc = getattr(self, "focused", None)
        if foc:
            lines.append("Focused re-simulations (each pair restarted from its true state "
                         "a few frames before its closest approach; more reliable than the "
                         "long rollout for whether that pair touches):")
            for f in foc:
                verdict = ("collide at frame " + str(f["frame"])) if f["frame"] is not None                     else "do not collide"
                lines.append("  " + self.name(f["i"]) + " & " + self.name(f["j"]) +
                             " (from frame " + str(f["start"]) + "): " + verdict)
        mt = d["ledger"].get("momentum_to_table", 0.0)
        if mt:
            lines.append("Momentum transferred to the table by friction: "
                         + format(mt, ".5f") + ".")
        return "\n".join(lines)

    def save(self, path):
        """Write BOTH halves of the predicted future.

        .json  -- the summary a language model reads: objects, events, ledger.
        .npz   -- the full per-frame trajectory, in the SAME schema as the
                  original clip's annotation, so the predicted world can be
                  re-rendered, re-analysed or diffed against the real one
                  without re-running the simulator.

        The JSON alone was 477 bytes of events; the arrays were computed, used
        for the GIF, then dropped. A "record of the simulated future" that
        cannot reproduce the future is not a record.
        """
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)
        npz = path[:-5] + ".npz" if path.endswith(".json") else path + ".npz"
        np.savez_compressed(
            npz,
            positions=self.positions, velocities=self.velocities,
            yaw=self.yaw, presence=self.present,
            source=self.source,                 # 0 = observed, 1 = model's own
            attrs=self.attrs,
            obj_keys=np.asarray(self.obj_keys),
            removed=np.asarray(self.removed, np.int64),
            collisions=np.asarray([[e["frame"], e["i"], e["j"]]
                                   for e in (self._events or self.contacts())],
                                  np.int64).reshape(-1, 3),
            exits=np.asarray([[e["frame"], e["i"]] for e in self.exits()],
                             np.int64).reshape(-1, 2),
            world_scale=np.float32(self.world_scale),
            video_name="sim_%05d" % self.video,
            momentum_to_table=np.float32(self.ledger.get("momentum_to_table", 0.0)),
        )
        return npz
