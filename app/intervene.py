"""
app/intervene.py  —  from any "what if" sentence to changes the world model can simulate
=======================================================================================

The world model (v7.1) simulates; it does not read English. A request goes
through three steps, and only the middle one uses the language model:

  1. describe_scene()  the video as text: each object's path in plain
                       coordinates, when it enters and from which side, the
                       dataset's recorded collisions, the spot the user clicked
  2. PLAN_SYSTEM       the language model writes a PLAN: a list of EVENTS made
                       of a few general operations that combine freely --
                       remove, add, move, place, shift_time, set (properties),
                       pin -- each at any frame, on any object (including ones
                       it adds), plus world settings (friction everywhere, how
                       far past the video to predict). Places are named relative
                       to the scene, never as invented physics numbers
  3. resolve()         turns the events into start frames, positions,
                       velocities and property changes, using the real camera
                       (camera_fit.json), the recorded tracks and the world
                       model's own friction; cf_world.build_world(edits=...)
                       then simulates

What can be changed is exactly what the world model has a handle on: which
objects exist and when, where each one is, how it moves, and its physical
properties (material, shape, size, mass, bounciness, friction). The language
model never decides what HAPPENS: every collision comes from the world model.

HONEST LIMITS. The world model was trained on CLEVRER (3 shapes x 2 materials,
one size) and scored only on removals (88.4%). Material and shape changes stay
inside what it saw. Mass, size and friction factors act on the physics directly
(inertia, contact distance, the learned drag term) and are extrapolations; the
page says so.
"""

import contextlib
import json
import math
import os

import numpy as np

from lift3d import Camera                                   # noqa: E402  (src2 is on sys.path)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

WORLD = 6.0                                                 # track units -> render units
SHAPES = ("cube", "sphere", "cylinder")                     # attrs column order (build_trajectories_3d.py)
COLORS = ("gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow")
MATERIALS = ("metal", "rubber")
DENSITY = {"metal": 1.5, "rubber": 1.0}                    # reproduces the stored masses exactly
RESTITUTION = {"metal": 0.8, "rubber": 0.5}
# per frame. Objects entering CLEVRER videos move at a median 0.0135 (p10 0.0068, p90 0.0185)
SPEED = {"slow": 0.008, "normal": 0.0135, "fast": 0.02}
MOVING = 0.004                                              # slower than this counts as standing still
INSIDE = 0.9                                                # places stay on the simulated table (cf_world drops > 0.95)
REACH = 0.03                                                # "arrived" = centre within this of the target
DEFAULT_EXTEND, MAX_EXTEND = 30, 120                        # frames predicted past the video (30 = the tested setting)
MAX_OBJECTS = 10
PIN_MASS = 1e4                                              # "glued to the table": nothing measurable moves it
SLOPE = {"gentle": 0.00008, "medium": 0.0002, "steep": 0.0005}  # pull per frame^2 (a normal roll is 0.0135 per frame)

SIDE_WORDS = {"left": "the left", "right": "the right", "near": "the front (the camera side)", "far": "the back"}
BESIDE = {"left": "left of", "right": "right of", "near": "in front of", "far": "behind"}
SIDE_SYNONYMS = {"front": "near", "bottom": "near", "camera": "near", "camera side": "near",
                 "toward the camera": "near", "towards the camera": "near",
                 "back": "far", "top": "far", "behind": "far", "away": "far", "away from the camera": "far"}
WORD_SYNONYMS = {"grey": "gray", "ball": "sphere", "block": "cube", "box": "cube",
                 "metallic": "metal", "shiny": "metal", "matte": "rubber", "rubbery": "rubber"}

with open(os.path.join(ROOT, "camera_fit.json"), encoding="utf-8") as _fh:
    _BLOB = json.load(_fh)
CAM = Camera(_BLOB["camera"])                               # the CLEVRER camera, recovered by fit3d
METRIC = _BLOB["sizes"]                                     # centre height = half-size of each shape
_fwd = np.asarray(CAM.cfg["target"], float)[:2] - CAM.loc[:2]
FAR = _fwd / np.linalg.norm(_fwd)                           # away from the camera, on the table
RIGHT = np.array([FAR[1], -FAR[0]])                         # the picture's right, on the table
SIDE_VEC = {"left": -RIGHT, "right": RIGHT, "near": -FAR, "far": FAR}


class PlanError(ValueError):
    """A request that cannot be turned into a simulation. The message is shown to the user."""


# ----------------------------------------------------------------------------- geometry and physics helpers
def to_view(p):
    return float(p[0] * RIGHT[0] + p[1] * RIGHT[1]), float(p[0] * FAR[0] + p[1] * FAR[1])


def from_view(right, far):
    return right * RIGHT + far * FAR


def image_uv(p, h):
    """Track position (centre at height h, render units) -> pixel in the real video."""
    rel = np.array([p[0] * WORLD, p[1] * WORLD, h]) - CAM.loc
    cp = rel @ CAM.R
    d = max(-cp[2], 1e-6)
    return CAM.cx + CAM.fx * cp[0] / d, CAM.cy - CAM.fy * cp[1] / d


def ground_point(u, v, h):
    """Pixel in the real video -> track position on the plane at height h."""
    X, Y, _ = CAM.intersect_plane(np.array(u, float), np.array(v, float), h)
    X, Y = float(X), float(Y)
    if not (math.isfinite(X) and math.isfinite(Y)):
        return None
    return np.array([X, Y]) / WORLD


def _inside(p):
    p = np.asarray(p, np.float64)
    m = float(np.abs(p).max())
    return p * (INSIDE / m) if m > INSIDE else p


def _pull_inside(target, start):
    """Slide `start` towards `target` (which is inside) until it is on the table."""
    d = start - target
    s = 1.0
    for c in range(2):
        if abs(target[c] + d[c]) > INSIDE and abs(d[c]) > 1e-9:
            s = min(s, (math.copysign(INSIDE, d[c]) - target[c]) / d[c])
    return target + max(s, 0.0) * d


def attrs_row(shape, color, material):
    """The 16 stored attributes of an object, built the way build_trajectories_3d.py builds them."""
    s = float(METRIC[shape])
    vol = {"sphere": 4.0 / 3.0 * math.pi * s ** 3, "cube": (2 * s) ** 3, "cylinder": math.pi * s * s * 2 * s}[shape]
    row = np.zeros(16, np.float32)
    row[0] = DENSITY[material] * vol / 0.1
    row[1] = RESTITUTION[material]
    row[2 + SHAPES.index(shape)] = 1.0
    row[5 + COLORS.index(color)] = 1.0
    row[13 + MATERIALS.index(material)] = 1.0
    row[15] = s / WORLD * (4.0 / math.pi if shape == "cube" else 1.0)
    return row


@contextlib.contextmanager
def friction(model, scale):
    """Scale the world model's learned friction (its separate drag term) for one
    simulation. scale: None (unchanged), one number, or one factor per object."""
    if scale is None:
        yield
        return
    import torch
    s = torch.tensor(np.asarray(scale, np.float32)).view(1, -1) if np.ndim(scale) else float(scale)
    cls = type(model)

    def drag(e, present):
        c = cls.drag_coeff(model, e, present)
        return None if c is None else c * s

    def rot(e, present):
        c = cls.rot_drag_coeff(model, e, present)
        return None if c is None else c * s
    model.drag_coeff, model.rot_drag_coeff = drag, rot
    try:
        yield
    finally:
        del model.drag_coeff, model.rot_drag_coeff


def free_path(model, row, p0, v0, steps, fscale=1.0):
    """Roll one object alone with the world model (its learned friction, nothing to
    hit). Used only to TIME a start so it arrives when asked; the simulation with
    every object happens in cf_world."""
    import torch
    from world import strip_colour
    phys = strip_colour(torch.from_numpy(np.asarray(row, np.float32)).view(1, 1, 16))
    with torch.no_grad():
        mass, radius, e = model.properties(phys)
    q = torch.tensor(np.asarray(p0, np.float32)).view(1, 1, 2)
    p = mass.unsqueeze(-1) * torch.tensor(np.asarray(v0, np.float32)).view(1, 1, 2)
    yaw, L, act = torch.zeros(1, 1), torch.zeros(1, 1), torch.ones(1, 1)
    P, V = [np.asarray(p0, np.float64)], [np.asarray(v0, np.float64)]
    with friction(model, None if fscale == 1.0 else float(fscale)):
        for _ in range(max(int(steps), 0)):
            q, p, yaw, L = model.step_spatial(q, p, yaw, L, mass, radius, e, act, phys, 1.0, create_graph=False)
            q, p, yaw, L = q.detach(), p.detach(), yaw.detach(), L.detach()
            P.append(q[0, 0].numpy().astype(np.float64))
            V.append((p[0, 0] / mass[0, 0]).numpy().astype(np.float64))
    return np.array(P), np.array(V)


def apply_motion(ed, p0, v0):
    """cf_world.build_world's set_state rule, for an added object's starting velocity."""
    v0 = np.asarray(v0, np.float64)
    sp = float(np.linalg.norm(v0))
    u = v0 / sp if sp > 1e-3 else None
    if ed.get("aim") is not None:
        dd = np.asarray(ed["aim"], np.float64) - np.asarray(p0, np.float64)
        if float(np.linalg.norm(dd)) > 1e-6:
            u = dd / float(np.linalg.norm(dd))
    if ed.get("heading") is not None:
        u = np.array([math.cos(ed["heading"]), math.sin(ed["heading"])])
    if ed.get("turn") and u is not None:
        c, s = math.cos(ed["turn"]), math.sin(ed["turn"])
        u = np.array([c * u[0] - s * u[1], s * u[0] + c * u[1]])
    if ed.get("speed") is not None:
        sp = float(ed["speed"])
    elif sp <= 1e-3 and ed.get("still_speed") is not None:
        sp = float(ed["still_speed"])
    v1 = u * sp if u is not None else np.zeros(2)
    if ed.get("speed_factor") is not None:
        v1 = v1 * float(ed["speed_factor"])
    if ed.get("push") is not None:
        v1 = v1 + np.asarray(ed["push"], np.float64)
    return v1


# ----------------------------------------------------------------------------- the recorded video
class Scene:
    def __init__(self, z):
        self.z = z
        self.T = int(z["positions"].shape[0])
        self.N = min(8, int(z["positions"].shape[1]))
        self.pos = z["positions"][:, :self.N].astype(np.float64)
        self.vel = z["velocities"][:, :self.N].astype(np.float64)
        self.pres = z["presence"][:, :self.N] > 0
        self.attrs = z["attrs"][:self.N].astype(np.float32)
        self.keys = [str(k) for k in z["obj_keys"]][:self.N]
        self.names = [k.replace("_", " ") for k in self.keys]
        self.shapes = [k.split("_")[2] for k in self.keys]
        self.r = self.attrs[:, 15]
        self.coll = sorted((int(f), int(i), int(j)) for f, i, j in z["collisions"]
                           if int(i) < self.N and int(j) < self.N and int(i) != int(j))

    def on(self, k):
        return np.where(self.pres[:, k])[0]

    def at(self, k, t):
        """Recorded position at the frame nearest t on which the object is on the table."""
        on = self.on(k)
        if not on.size:
            raise PlanError("The %s is never on the table." % self.names[k])
        tt = int(on[np.argmin(np.abs(on - t))])
        return self.pos[tt, k].copy(), tt

    def speed(self, k, t):
        t = min(max(int(t), 0), self.T - 1)
        return float(np.linalg.norm(self.vel[t, k])) if self.pres[t, k] else 0.0

    def steady_speed(self, k):
        on = self.on(k)
        s = np.linalg.norm(self.vel[on, k], axis=-1) if on.size else np.zeros(0)
        s = s[s > MOVING]
        return float(np.median(s)) if s.size else 0.0

    def first_moving(self, k):
        for t in self.on(k):
            if np.linalg.norm(self.vel[t, k]) > MOVING:
                return int(t)
        return None

    def first_collision(self, k):
        for f, i, j in self.coll:
            if k in (i, j):
                return f, (j if i == k else i)
        return None

    def default_frame(self, k):
        """When an object is named as a place without a time: just before its first
        collision if it has one, else a little after it starts moving."""
        on = self.on(k)
        if not on.size:
            return 0
        e0, e1 = int(on[0]), int(on[-1])
        fc = self.first_collision(k)
        if fc is not None:
            return max(e0, fc[0] - 6)
        fm = self.first_moving(k)
        return e0 if fm is None else min(fm + 20, e1)

    def height(self, k):
        return float(METRIC[self.shapes[k]])

    def steady_state(self, k, f):
        """Position and velocity at frame f taken from the object's STEADY early motion
        (frames entry+4 .. entry+12, before its first collision), not from the frame
        itself. Near the entry the per-frame velocity is unreliable: measured on video
        10002 the green cube's entry-frame velocity was ~50% too fast and ~45 degrees
        off, and a hand-off there sent it off the table even with no change at all."""
        on = self.on(k)
        if not on.size:
            return None
        e0, e1 = int(on[0]), int(on[-1])
        t1 = min(e0 + 12, e1)
        fc = self.first_collision(k)
        if fc is not None:
            t1 = min(t1, fc[0] - 2)
        t0 = max(e0 + 4, t1 - 8)
        if t1 - t0 < 3 or not (self.pres[t0, k] and self.pres[t1, k]):
            return None
        v = (self.pos[t1, k] - self.pos[t0, k]) / (t1 - t0)
        return self.pos[t1, k] + v * (f - t1), v


def entry_side(p, h):
    W, H = CAM.cfg["width"], CAM.cfg["height"]
    u, v = image_uv(p, h)
    d = {"left": u, "right": W - u, "far": v, "near": H - v}
    return min(d, key=d.get)


def describe_scene(sc, marked=None):
    W, H = CAM.cfg["width"], CAM.cfg["height"]
    h = float(METRIC["sphere"])

    def vw(p):
        return "(%.2f, %.2f)" % to_view(p)

    left, right, bottom = ground_point(0, H / 2, h), ground_point(W, H / 2, h), ground_point(W / 2, H, h)
    lines = ["VIDEO: frames 0-%d, about 25 frames per second. After the video the world model predicts %d more "
             "frames unless the plan asks for more (up to %d)." % (sc.T - 1, DEFAULT_EXTEND, MAX_EXTEND),
             "COORDINATES (right, far): right grows to the right of the camera picture, far grows away from the "
             "camera. One object is about 0.12 across. Across the middle of the picture right runs from %.2f to "
             "%.2f; the bottom edge of the picture is at far=%.2f. Keep places within about 0.8 of (0, 0)."
             % (to_view(left)[0], to_view(right)[0], to_view(bottom)[1]),
             "", "OBJECTS (index: name -- facts. Path = frame(right, far)):"]
    for k in range(sc.N):
        on = sc.on(k)
        if not on.size:
            lines.append("  %d: %s -- never on the table" % (k, sc.names[k]))
            continue
        e0, e1 = int(on[0]), int(on[-1])
        bits = ["on the table frames %d-%d" % (e0, e1)]
        bits.append("enters from %s at frame %d" % (SIDE_WORDS[entry_side(sc.pos[e0, k], sc.height(k))], e0)
                    if e0 > 0 else "there from the start")
        fm = sc.first_moving(k)
        bits.append("never moves" if fm is None else ("moving from the start" if fm == e0
                                                      else "starts moving at frame %d" % fm))
        ts = list(range(e0, e1 + 1, 16))
        if ts[-1] != e1:
            ts.append(e1)
        path = " ".join("f%d%s" % (sc.at(k, t)[1], vw(sc.at(k, t)[0])) for t in ts)
        lines.append("  %d: %s -- %s. Path: %s" % (k, sc.names[k], "; ".join(bits), path))
    lines += ["", "RECORDED COLLISIONS in the real video (index: frame, objects, place):"]
    for c, (f, i, j) in enumerate(sc.coll):
        mid = (sc.at(i, f)[0] + sc.at(j, f)[0]) / 2
        lines.append("  %d: frame %d, %s (%d) and %s (%d) at %s" % (c, f, sc.names[i], i, sc.names[j], j, vw(mid)))
    if not sc.coll:
        lines.append("  none")
    lines += ["", "MARKED SPOT: " + ("%s (the user clicked it)" % vw(marked) if marked is not None
                                     else "none (the user has not clicked a spot)")]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- the language model's part
PLAN_SYSTEM = """You turn a user's question about a CLEVRER video into a PLAN for a physics world model. Objects slide or roll on a flat table filmed by a fixed camera. The world model simulates; you never predict what happens. You only write down what is DIFFERENT from the real video, as a list of EVENTS. Events combine freely: several objects, several events on one object at different frames, new objects that later change too.

EVENTS ("frame": null means the natural default)
  {"do": "remove", "object": O, "frame": null}
      gone from that frame on; null = it is never in the video
  {"do": "add", "label": "A", "shape": S, "color": C, "material": M, "size_factor": null, "at": PLACE, "motion": null, "frame": null}
      a new object appears at 'at' on 'frame' (null = frame 0); motion null = it stands still. Later events call it "A", "B", ...
  {"do": "move", "object": O, "frame": null, "motion": MOTION}
      from that frame it moves differently
  {"do": "place", "object": O, "frame": null, "at": PLACE, "motion": null}
      it is suddenly at PLACE on that frame; frame null = it STARTS at PLACE instead of its real starting point
  {"do": "shift_time", "object": O, "frames": n}
      it comes into the video n frames later (negative = earlier), the same way
  {"do": "set", "object": O, "frame": null, "material": null, "shape": null, "size_factor": null, "mass_factor": null, "bounciness": null, "friction_factor": null}
      it has other physical properties (bounciness 0-1; metal is 0.8, rubber 0.5; friction_factor 0 = no friction)
  {"do": "pin", "object": O, "frame": null}
      glued to the table from that frame: nothing can move it
  O is an object index from the scene, or the label of an added object.

MOTION (every field optional)
  {"stop": false, "speed_factor": null, "speed": null, "turn_degrees": null, "toward": null, "from_side": null, "arrive_frame": null, "push": false}
  speed_factor: 2 = twice as fast, 0.5 = half. speed: "slow" | "normal" | "fast".
  turn_degrees: positive = to its own left, negative = to its own right.
  toward: a PLACE, or {"direction": "left" | "right" | "near" | "far"}.
  from_side: it comes into the picture from that side (near = camera side / bottom of the picture, far = back / top) instead of its real path. Without 'toward' it still heads for the spot of its first real collision at the same time.
  arrive_frame: with 'toward' a place (or with from_side), time it so it gets there at this frame.
  push: true = a push added to how it already moves, instead of replacing its motion.

PLACE
  {"marked": true}  the spot the user clicked
  {"object": O, "frame": null, "side": null}  where O is at that frame; side left|right|near|far = just beside it
  {"between": [O1, O2], "frame": null}
  {"collision": c}  where recorded collision c happened
  {"right": x, "far": y}
  {"center": true}

WORLD settings: {"friction_factor": null, "predict_frames": null, "slope": null}
  friction_factor: friction for every object (0 = none). predict_frames: frames to predict after the video (default 30, at most 120).
  slope: {"direction": "left" | "right" | "near" | "far", "strength": "gentle" | "medium" | "steep"} = the table is tilted, so everything is steadily pulled that way (use it for 'gravity pulls left', 'a tilted table', 'wind blowing right').

RULES
- Leave every "frame" null unless the user names a time; the system picks sensible frames itself.
- 'here', 'this spot', 'there' mean the marked spot.
- Name places from the scene (objects, collisions, the marked spot) rather than raw (right, far) numbers. 'behind X', 'in front of X', 'left of X' = {"object": X, "side": "far" | "near" | "left"}.
- 'in the way of X' / 'blocking X' = the place {"object": X, "frame": null}.
- 'hit X' / 'crash into X' = toward {"object": X, "frame": f} with arrive_frame f, where f is a frame when X is on the table.
- 'X stops' = a move with {"stop": true}. Use "place" only when the user says an object is somewhere else.
- 'X comes in / came in / enters from the left (right, front, back)' = {"do": "move", "object": X, "motion": {"from_side": "left"}}. Never imitate it with place and turn_degrees.
- A new object that rolls in: put from_side (and toward / arrive_frame) in the add event's own motion, not in a separate move; 'toward' is what it heads for, and 'at' can then be null.
- A question that changes nothing ('what happens next?', 'what if the video were longer?') is a plan with no events and a predict_frames value.
- New objects: if the user gives no colour, material or shape, pick ones easy to tell apart from the scene. Colours: gray red blue green brown purple cyan yellow. Materials: metal rubber. Shapes: cube sphere cylinder.
- Anything these events cannot express (walls, magnets, objects sticking together, ...): give no events and explain in 'cannot'."""

PLAN_FORMAT = ('Return JSON: {"understood": "<the change, in one short plain sentence>", "events": [...], '
               '"world": {"friction_factor": null, "predict_frames": null, "slope": null}, "cannot": null}')


def ask_plan(lm, cfg, prov, request, sc, marked=None):
    user = "SCENE\n%s\n\nREQUEST: %s\n\n%s" % (describe_scene(sc, marked), request, PLAN_FORMAT)
    return lm.ask_json(cfg, prov, PLAN_SYSTEM, user)


# ----------------------------------------------------------------------------- small parsers
def _frame(x, default, lo=0, hi=10 ** 6):
    try:
        f = int(round(float(x)))
    except (TypeError, ValueError):
        f = int(default)
    return int(min(max(f, lo), hi))


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _given(x):
    return x is not None and x is not False and str(x).strip().lower() not in ("", "none", "null")


def _side(x):
    s = str(x or "").strip().lower().replace("_", " ")
    s = SIDE_SYNONYMS.get(s, s)
    if s not in SIDE_VEC:
        raise PlanError("'%s' is not a side the system knows (left, right, front, back)." % x)
    return s


def _pick(x, allowed, default):
    s = WORD_SYNONYMS.get(str(x or "").strip().lower(), str(x or "").strip().lower())
    return s if s in allowed else default


def _times(f):
    return "%g" % round(f, 2)


def _arrival(given, place_frame, lo=0, hi=10 ** 6):
    """When a rolling object should reach its target. A frame the user gave wins. A place's
    own frame is used only if it is a real moment: for an object that never moves it is
    just its first frame (often 0), and aiming to arrive then made a new sphere start on top
    of the brown cylinder in video 10002. Otherwise frame 40."""
    if _given(given):
        return _frame(given, 40, lo, hi)
    if place_frame is not None and place_frame >= 25:
        return int(min(max(place_frame, lo), hi))
    return int(min(max(40, lo), hi))


def _at(words):
    """'at the spot where ...', but 'just left of ...' and 'halfway between ...' as they are."""
    return words if words.startswith(("just ", "halfway ")) else "at " + words


def aim_from_side(model, row, target, side, speed, h, last, fscale, arrive=None, start=None, reach=REACH):
    """Start just outside the camera picture on `side`, aimed at `target`.

    With `arrive`, the start frame is chosen so that, rolling on its own under the
    world model's friction, the object reaches the target at that frame. If friction
    would stop it short it is pushed faster, up to 3x."""
    W, H = CAM.cfg["width"], CAM.cfg["height"]
    target = _inside(target)
    u, v = image_uv(target, h)
    m = 30.0
    uv = {"left": (-m, v), "right": (W + m, v), "near": (u, H + m), "far": (u, -m)}[side]
    p0 = ground_point(uv[0], uv[1], h)
    if p0 is None or np.linalg.norm(p0 - target) < 0.15:
        p0 = target + SIDE_VEC[side] * 0.5
    p0 = _pull_inside(target, p0)
    gap = float(np.linalg.norm(target - p0))
    if gap < 0.03:
        raise PlanError("That place is already at the edge of the table on %s, so nothing can come in from there."
                        % SIDE_WORDS[side])
    u_dir = (target - p0) / gap
    best = None
    for boost in (1.0, 1.5, 2.25, 3.0):
        P, V = free_path(model, row, p0, u_dir * speed * boost, last, fscale)
        d = np.linalg.norm(P - target, axis=1)
        n = int(np.argmax(d <= reach)) if (d <= reach).any() else int(np.argmin(d))
        best = (n, P, V, boost, float(d[n]))
        if d[n] <= reach:
            break
    n, P, V, boost, miss = best
    out = {"boost": boost, "reached": miss <= reach, "rolling_at_start": False}
    if start is not None:
        s = _frame(start, 0, 0, last)
        out.update(frame=s, pos=p0, vel=u_dir * speed * boost, arrive=min(s + n, last))
        return out
    s = int(arrive) - n
    if s < 0:
        # it would have to set off before the video starts: at frame 0 it is
        # already part of the way along the same path
        out.update(frame=0, pos=P[-s], vel=V[-s], arrive=int(arrive), rolling_at_start=True)
        return out
    out.update(frame=s, pos=p0, vel=u_dir * speed * boost, arrive=int(arrive))
    return out


def _aim_notes(notes, name, a, twords):
    if a["boost"] > 1.0:
        notes.append("The %s was given %sx its speed so that friction does not stop it before it arrives."
                     % (name, _times(a["boost"])))
    if not a["reached"]:
        notes.append("Even so, friction stops the %s before it fully reaches %s." % (name, twords))
    if a["rolling_at_start"]:
        notes.append("To arrive in time the %s is already rolling when the video starts." % name)


def clear_of(sc, p, r_new, frame, others, removed, skip=None):
    """Nudge an object placed standing still off anything already at that spot."""
    p = np.asarray(p, np.float64).copy()
    t = min(max(frame, 0), sc.T - 1)
    blockers = [(sc.names[k], sc.pos[t, k], float(sc.r[k])) for k in range(sc.N)
                if sc.pres[t, k] and k not in removed and k != skip] + list(others)
    moved = []
    for _ in range(8):
        hit = False
        for name, q, rq in blockers:
            need = r_new + rq + 0.01
            d = p - q
            dist = float(np.linalg.norm(d))
            if dist < need:
                p = q + (d / dist if dist > 1e-6 else RIGHT) * need
                hit = True
                if name not in moved:
                    moved.append(name)
        if not hit:
            break
    return _inside(p), moved


# ----------------------------------------------------------------------------- events -> cf_world edits
class Plan:
    ORDER = ("add", "set", "remove", "shift_time", "place", "move", "pin")

    def __init__(self, sc, model, marked, entry, extend, world_friction):
        self.sc, self.model, self.entry = sc, model, entry
        self.marked = None if marked is None else np.asarray(marked, np.float64)
        self.extend, self.last = extend, sc.T - 1 + extend
        self.N0 = sc.N
        self.attrs = sc.attrs.copy()
        self.keys, self.names = list(sc.keys), list(sc.names)
        self.removed, self.change, self.add, self.labels, self.paths = set(), {}, [], {}, {}
        self.mass, self.radius, self.drag = [1.0] * sc.N, [1.0] * sc.N, [1.0] * sc.N
        self.world_friction = world_friction
        self.steps, self.notes, self.standing = [], [], []
        self.add_step, self.accel = {}, None

    # -- facts about any object, original or added ------------------------------
    def obj(self, ref):
        s = "" if ref is None else str(ref).strip()
        for cand in (s, s.upper()):
            if cand in self.labels:
                return self.labels[cand]
        try:
            k = int(float(s))
        except ValueError:
            raise PlanError("The plan refers to an object '%s' that is not in the scene." % ref)
        if 0 <= k < self.N0:
            return k
        raise PlanError("The plan names object %s, which is not in this video." % ref)

    def spec(self, k):
        return self.add[k - self.N0]

    def row(self, k):
        return self.attrs[k] if k < self.N0 else np.asarray(self.spec(k)["attrs"], np.float32)

    def set_row(self, k, row):
        if k < self.N0:
            self.attrs[k] = row
        else:
            self.spec(k)["attrs"] = np.asarray(row, np.float32).tolist()

    def r(self, k):
        return float(self.row(k)[15]) * self.radius[k]

    def h(self, k):
        return float(METRIC[self.keys[k].split("_")[2]]) * self.radius[k]

    def fscale(self, k):
        return self.world_friction * self.drag[k]

    def span(self, k):
        if k < self.N0:
            on = self.sc.on(k)
            if not on.size:
                raise PlanError("The %s is never on the table." % self.names[k])
            return int(on[0]), int(on[-1])
        return int(self.spec(k)["frame"]), self.last

    def where(self, k, t):
        """(position, frame): recorded for a video object, the free path for an added one."""
        if k < self.N0:
            return self.sc.at(k, t)
        s, P = self.paths[k]
        t = max(int(t), s)
        return P[min(t - s, len(P) - 1)].copy(), t

    def default_frame(self, k):
        return self.sc.default_frame(k) if k < self.N0 else min(int(self.spec(k)["frame"]) + 10, self.last)

    def event_frame(self, k, x, default):
        """A frame on which object k is on the table (so there is a state to change)."""
        e0, e1 = self.span(k)
        hi = e1 if (k < self.N0 and e1 < self.sc.T - 1) else self.last
        return _frame(x, default, e0, hi)

    def move_frame(self, k):
        if k >= self.N0:
            return int(self.spec(k)["frame"])
        e0, e1 = self.span(k)
        fm = self.sc.first_moving(k)
        if fm is None or fm == e0:
            return e0
        return min(fm + 3, e1 if e1 < self.sc.T - 1 else self.last)      # it was pushed: just after it starts

    def moving_at(self, k, t):
        if k < self.N0:
            return self.sc.speed(k, t) >= MOVING
        return float(np.linalg.norm(self.spec(k)["vel"])) >= MOVING

    def base_speed(self, k):
        s = self.sc.steady_speed(k) if k < self.N0 else float(np.linalg.norm(self.spec(k)["vel"]))
        return s if s > MOVING else SPEED["normal"]

    def refresh_path(self, k):
        sp = self.spec(k)
        if float(np.linalg.norm(sp["vel"])) < 1e-9:
            self.paths[k] = (int(sp["frame"]), np.asarray([sp["pos"]], np.float64))
        else:
            P, _ = free_path(self.model, sp["attrs"], sp["pos"], sp["vel"], self.last - int(sp["frame"]), self.fscale(k))
            self.paths[k] = (int(sp["frame"]), P)

    def say_start(self, k, text):
        """An added object's start was redefined: replace its first step instead of adding one."""
        if k in self.add_step:
            self.steps[self.add_step[k]] = text
        else:
            self.steps.append(text)

    # -- places and motions ---------------------------------------------------
    def place(self, spec, new_r):
        """PLACE -> (track position, the frame it refers to or None, words for the user)."""
        sc = self.sc
        if not isinstance(spec, dict):
            raise PlanError("A place in the plan was not understood.")
        if spec.get("marked"):
            if self.marked is None:
                raise PlanError("Your request points at a spot. Press 'Mark a spot', click the table in the video, "
                                "then run it again.")
            return _inside(self.marked), None, "the spot you marked"
        if _given(spec.get("collision")):
            c = _frame(spec["collision"], -1, -1)
            if not 0 <= c < len(sc.coll):
                raise PlanError("The plan names a collision that is not in this video.")
            f, i, j = sc.coll[c]
            return (_inside((sc.at(i, f)[0] + sc.at(j, f)[0]) / 2), f,
                    "the spot where the %s hit the %s in the video (frame %d)" % (sc.names[i], sc.names[j], f))
        if _given(spec.get("between")):
            pair = spec["between"]
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise PlanError("'between' needs exactly two objects.")
            i, j = self.obj(pair[0]), self.obj(pair[1])
            f = _frame(spec.get("frame"), min(self.default_frame(i), self.default_frame(j)), 0, self.last)
            return (_inside((self.where(i, f)[0] + self.where(j, f)[0]) / 2), f,
                    "halfway between the %s and the %s (frame %d)" % (self.names[i], self.names[j], f))
        if _given(spec.get("object")):
            k = self.obj(spec["object"])
            p, f = self.where(k, _frame(spec.get("frame"), self.default_frame(k), 0, self.last))
            words = "the spot where the %s is at frame %d" % (self.names[k], f)
            if _given(spec.get("side")):
                s = _side(spec["side"])
                p = p + SIDE_VEC[s] * (self.r(k) + new_r + 0.02)
                words = "just %s the %s (as it is at frame %d)" % (BESIDE[s], self.names[k], f)
            return _inside(p), f, words
        if _given(spec.get("right")) and _given(spec.get("far")):
            r, fa = _num(spec["right"]), _num(spec["far"])
            if r is None or fa is None:
                raise PlanError("A place in the plan has no usable coordinates.")
            return _inside(from_view(r, fa)), None, "the point (right %.2f, far %.2f)" % (r, fa)
        if spec.get("center"):
            return np.zeros(2), None, "the middle of the table"
        raise PlanError("A place in the plan was not understood.")

    def speed_to_arrive(self, k, p0, target, frames):
        """Friction under the world model's drag is proportional to speed, so the
        distance covered in n frames scales linearly with the starting speed."""
        d = np.asarray(target, np.float64) - p0
        dist = float(np.linalg.norm(d))
        if dist < 1e-6:
            return 0.0
        P, _ = free_path(self.model, self.row(k), p0, d / dist * 0.01, frames, self.fscale(k))
        covered = float(np.linalg.norm(P[frames] - p0))
        return float(min(0.01 * dist / max(covered, 1e-6), 0.08))

    def motion(self, k, f, m, ed, words, p_from=None):
        """Write a MOTION into a change event. Returns True if it changes anything."""
        toward = m.get("toward") if isinstance(m.get("toward"), dict) else None
        fac, turn = _num(m.get("speed_factor")), _num(m.get("turn_degrees"))
        p0 = np.asarray(p_from, np.float64) if p_from is not None else self.where(k, f)[0]
        u, target, twords, pwords = None, None, "", ""
        if toward is not None:
            if _given(toward.get("direction")):
                s = _side(toward["direction"])
                u, twords = SIDE_VEC[s].copy(), "towards " + SIDE_WORDS[s]
                pwords = twords
            else:
                target, _, pw = self.place(toward, self.r(k))
                twords, pwords = "for " + pw, "towards " + pw
                dd = target - p0
                if float(np.linalg.norm(dd)) > 1e-6:
                    u = dd / float(np.linalg.norm(dd))
        sp, sp_words = None, ""
        word = str(m.get("speed") or "").strip().lower()
        if word in SPEED:
            sp, sp_words = SPEED[word], " at %s speed" % word
        if _given(m.get("arrive_frame")) and target is not None:
            a = _frame(m["arrive_frame"], f, 0, self.last)
            if a > f:
                sp, sp_words = self.speed_to_arrive(k, p0, target, a - f), ", arriving at about frame %d" % a
            else:
                self.notes.append("Arrival frame %d is not after frame %d, so it was ignored." % (a, f))
        if m.get("push"):
            if u is None:
                raise PlanError("A push needs a direction or a place to push towards.")
            push = u * (sp if sp is not None else SPEED["normal"]) * (fac if fac is not None and fac > 0 else 1.0)
            ed["push"] = push.tolist()
            words.append("gets pushed %s%s" % (pwords, sp_words))
            return True
        if u is not None:
            if target is not None:
                ed["aim"] = target.tolist()
            else:
                ed["heading"] = math.atan2(u[1], u[0])
            ed["still_speed"] = SPEED["normal"]
            words.append("heads %s%s" % (twords, sp_words))
            sp_words = ""
        if turn:
            ed["turn"] = math.radians(turn)
            words.append("turns %d degrees to its %s" % (round(abs(turn)), "left" if turn > 0 else "right"))
        if sp is not None:
            ed["speed"] = sp
            if sp_words:
                words.append("moves" + sp_words)
        if m.get("stop"):
            ed["speed_factor"] = 0.0
            words.append("stops")
        elif fac is not None:
            ed["speed_factor"] = max(0.0, fac)
            if fac <= 0:
                words.append("stops")
            elif fac >= 1:
                words.append("moves %s times as fast" % _times(fac))
            else:
                words.append("moves at %d%% of its speed" % round(100 * fac))
        return bool(words)

    # -- the events ---------------------------------------------------------------
    def ev_add(self, ev):
        if self.N0 + len(self.add) >= MAX_OBJECTS:
            raise PlanError("The world model takes at most %d objects in one scene." % MAX_OBJECTS)
        shape = _pick(ev.get("shape"), SHAPES, "cube")
        color = _pick(ev.get("color"), COLORS, "gray")
        mat = _pick(ev.get("material"), MATERIALS, "rubber")
        sf = _num(ev.get("size_factor"))
        sf = sf if sf is not None and sf > 0 else 1.0
        k = self.N0 + len(self.add)
        row = attrs_row(shape, color, mat)
        name, n = "new %s %s %s" % (color, mat, shape), 2
        while name in self.names:
            name, n = "new %s %s %s %d" % (color, mat, shape, n), n + 1
        label = str(ev.get("label") or chr(ord("A") + len(self.add))).strip()
        spec = {"frame": 0, "pos": [0.0, 0.0], "vel": [0.0, 0.0], "attrs": row.tolist(), "yaw": 0.0}
        self.add.append(spec)
        self.labels[label] = k
        self.add_step[k] = len(self.steps)
        self.keys.append("%s_%s_%s" % (color, mat, shape))
        self.names.append(name)
        self.mass.append(sf ** 3)
        self.radius.append(sf)
        self.drag.append(1.0)
        what = ("%s-times-as-big " % _times(sf) if sf != 1.0 else "") + "%s %s %s" % (color, mat, shape)
        if sf != 1.0:
            self.notes.append("The world model never saw objects of other sizes, so the %s is an estimate." % name)
        r_new, h = self.r(k), self.h(k)
        m = ev.get("motion") if isinstance(ev.get("motion"), dict) else {}
        at = ev.get("at") if isinstance(ev.get("at"), dict) else None

        if _given(m.get("from_side")):
            side = _side(m["from_side"])
            toward = m.get("toward") if (isinstance(m.get("toward"), dict)
                                         and not _given(m["toward"].get("direction"))) else None
            # measured: asked for "a sphere rolls in from the left and hits the brown cylinder", the
            # language model gave 'toward' the cylinder AND 'at' a guessed start point; aiming at 'at'
            # missed the cylinder. The target is 'toward' whenever it is given.
            goal = toward or at or {"center": True}
            target, pf, pw = self.place(goal, r_new)
            arrive = _arrival(m.get("arrive_frame"), pf, 0, self.last)
            # measured on video 10020: "a red sphere rolled in from the right at frame 30" was read as
            # ARRIVING at frame 30, so it was already rolling at frame 0. 'frame' on an add is when it appears.
            start = ev.get("frame") if (_given(ev.get("frame")) and not _given(m.get("arrive_frame"))) else None
            word = str(m.get("speed") or "normal").strip().lower()
            fac = _num(m.get("speed_factor"))
            sp = SPEED.get(word, SPEED["normal"]) * (fac if fac is not None and fac > 0 else 1.0)
            reach = (self.r(self.obj(goal["object"])) + r_new) if (_given(goal.get("object"))
                                                                   and not _given(goal.get("side"))) else REACH
            a = aim_from_side(self.model, row, target, side, sp, h, self.last, self.fscale(k),
                              arrive=arrive, start=start, reach=reach)
            _aim_notes(self.notes, "new " + what, a, pw)
            spec.update(frame=a["frame"], pos=a["pos"].tolist(), vel=a["vel"].tolist())
            self.refresh_path(k)
            self.steps.append("A %s rolls in from %s from frame %d, aimed to reach %s at about frame %d."
                              % (what, SIDE_WORDS[side], a["frame"], pw, a["arrive"]))
            return
        f = _frame(ev.get("frame"), 0, 0, self.last)
        if at is not None and _given(at.get("object")) and not _given(at.get("frame")) and _given(ev.get("frame")):
            at = dict(at, frame=f)
        p, _, pw = self.place(at or {"center": True}, r_new)
        spec.update(frame=f, pos=p.tolist())
        self.paths[k] = (f, np.asarray([p], np.float64))
        ed, words = {}, []
        if m and self.motion(k, f, m, ed, words, p_from=p):
            spec["vel"] = apply_motion(ed, p, np.zeros(2)).tolist()
        if float(np.linalg.norm(spec["vel"])) < 1e-9:
            p, moved = clear_of(self.sc, p, r_new, f, self.standing, self.removed)
            spec["pos"] = p.tolist()
            self.standing.append((name, p, r_new))
            if moved:
                self.notes.append("The new %s was moved slightly so it does not overlap the %s."
                                  % (what, " and the ".join(moved)))
            self.steps.append("A %s stands still %s from frame %d." % (what, _at(pw), f))
        else:
            self.steps.append("A %s appears %s at frame %d and %s." % (what, _at(pw), f, " and ".join(words)))
        self.refresh_path(k)

    def ev_set(self, ev):
        k = self.obj(ev.get("object"))
        if k in self.removed:
            self.notes.append("The %s is removed, so its properties were not changed." % self.names[k])
            return
        col, mat, shp = self.keys[k].split("_")[:3]
        words, guess = [], []
        mat2, shp2 = _pick(ev.get("material"), MATERIALS, mat), _pick(ev.get("shape"), SHAPES, shp)
        if (mat2, shp2) != (mat, shp):
            self.set_row(k, attrs_row(shp2, col, mat2))
            self.keys[k] = "%s_%s_%s" % (col, mat2, shp2)
            base = self.names[k].split(" (now ")[0]
            self.names[k] = "%s (now %s %s)" % (base, mat2, shp2)
            words.append("becomes a %s %s" % (mat2, shp2))
        b = _num(ev.get("bounciness"))
        if b is not None:
            row = self.row(k).copy()
            row[1] = min(max(b, 0.0), 1.0)
            self.set_row(k, row)
            words.append("has bounciness %.2f" % row[1])
        mf = _num(ev.get("mass_factor"))
        if mf is not None and mf > 0 and mf != 1.0:
            self.mass[k] *= mf
            words.append("is %s times as heavy" % _times(mf))
            guess.append("mass")
        sf = _num(ev.get("size_factor"))
        if sf is not None and sf > 0 and sf != 1.0:
            self.radius[k] *= sf
            self.mass[k] *= sf ** 3
            words.append("is %s times as big (and as heavy as that size makes it)" % _times(sf))
            guess.append("size")
        ff = _num(ev.get("friction_factor"))
        if ff is not None and ff >= 0 and ff != 1.0:
            self.drag[k] *= ff
            words.append("has no friction" if ff == 0 else "has %s times the friction" % _times(ff))
            guess.append("friction")
        if not words:
            self.notes.append("The plan asked for no actual property change to the %s." % self.names[k])
            return
        if guess:
            self.notes.append("The world model never saw %s changes, so those results are estimates."
                              % " or ".join(guess))
        if k < self.N0:
            f = self.event_frame(k, ev.get("frame"), self.span(k)[0])
            self.change.setdefault(k, []).append({"frame": f})
            self.steps.append("The %s %s (simulated this way from frame %d)." % (self.sc.names[k], " and ".join(words), f))
        else:
            if ff is not None:
                self.refresh_path(k)
            self.steps.append("The %s %s." % (self.names[k], " and ".join(words)))

    def ev_remove(self, ev):
        k = self.obj(ev.get("object"))
        if _given(ev.get("frame")):
            f = _frame(ev["frame"], 0, self.span(k)[0], self.last)
            self.change.setdefault(k, []).append({"frame": f, "vanish": True})
            self.steps.append("At frame %d the %s disappears." % (f, self.names[k]))
        elif k < self.N0:
            if k not in self.removed:
                self.removed.add(k)
                self.steps.append("The %s is removed." % self.names[k])
        else:
            raise PlanError("An object cannot be added and removed in the same request.")

    def ev_shift_time(self, ev):
        k = self.obj(ev.get("object"))
        if k >= self.N0:
            raise PlanError("To change when a new object appears, give its 'frame'.")
        n = _num(ev.get("frames"))
        n = int(round(n)) if n is not None else 0
        if n == 0 or k in self.removed:
            return
        e0, _ = self.span(k)
        if e0 == 0 and n < 0:
            self.notes.append("The %s is already there when the video starts, so it cannot come earlier." % self.names[k])
            return
        s = max(0, e0 + n)
        if s > self.last:
            raise PlanError("The %s would come in after the end of the prediction." % self.names[k])
        st = self.sc.steady_state(k, e0)
        v, p = (st[1], st[0]) if st is not None else (self.sc.vel[e0, k].copy(), self.sc.pos[e0, k].copy())
        if st is None and e0 > 0 and self.entry:
            # the same entry-state correction v7.1 applies (entry_profile.json)
            v = v / max(self.entry["ratio"][0], 0.3)
            p = p - self.entry["lag"][0] * v
        self.change.setdefault(k, []).append({"frame": s, "replace_track": True, "pos": p.tolist(), "vel": v.tolist()})
        if e0 == 0:
            self.steps.append("The %s only appears at frame %d, where it really started." % (self.names[k], s))
        else:
            self.steps.append("The %s comes in %d frames %s (frame %d instead of %d), the same way it really came in."
                              % (self.names[k], abs(n), "later" if n > 0 else "earlier", s, e0))

    def ev_place(self, ev):
        k = self.obj(ev.get("object"))
        if k in self.removed:
            self.notes.append("The %s is removed, so it was not moved." % self.names[k])
            return
        m = ev.get("motion") if isinstance(ev.get("motion"), dict) else {}
        p, _, pw = self.place(ev.get("at"), self.r(k))
        words = []
        if k >= self.N0:
            sp = self.spec(k)
            if _given(ev.get("frame")):
                sp["frame"] = _frame(ev["frame"], 0, 0, self.last)
            sp["pos"] = p.tolist()
            ed = {}
            if m and self.motion(k, sp["frame"], m, ed, words, p_from=p):
                sp["vel"] = apply_motion(ed, p, np.asarray(sp["vel"])).tolist()
            self.refresh_path(k)
            self.say_start(k, "The %s starts at %s at frame %d%s." % (self.names[k], pw, sp["frame"],
                                                                     ", and " + " and ".join(words) if words else ""))
            return
        e0, _ = self.span(k)
        if _given(ev.get("frame")):
            f = self.event_frame(k, ev["frame"], e0)
            ed = {"frame": f, "pos": p.tolist()}
            text = "At frame %d the %s is suddenly at %s" % (f, self.names[k], pw)
        else:
            f = e0
            if self.sc.speed(k, e0) < MOVING:
                p, moved = clear_of(self.sc, p, self.r(k), f, self.standing, self.removed, skip=k)
                if moved:
                    self.notes.append("The %s was moved slightly so it does not overlap the %s."
                                      % (self.names[k], " and the ".join(moved)))
            ed = {"frame": f, "replace_track": True, "pos": p.tolist()}
            text = "The %s starts from %s at frame %d instead of its real starting point" % (self.names[k], pw, f)
        if m:
            self.motion(k, f, m, ed, words, p_from=p)
        self.change.setdefault(k, []).append(ed)
        self.steps.append(text + (", and " + " and ".join(words) if words else "") + ".")

    def ev_move(self, ev):
        k = self.obj(ev.get("object"))
        if k in self.removed:
            self.notes.append("The %s is removed, so its motion was not changed." % self.names[k])
            return
        m = ev.get("motion") if isinstance(ev.get("motion"), dict) else {}
        if _given(m.get("from_side")):
            return self.move_from_side(k, ev, m)
        if k >= self.N0:
            f = _frame(ev.get("frame"), self.spec(k)["frame"], int(self.spec(k)["frame"]), self.last)
        else:
            f = self.event_frame(k, ev.get("frame"), self.move_frame(k)) if _given(ev.get("frame")) else self.move_frame(k)
        ed, words = {"frame": f}, []
        if not self.motion(k, f, m, ed, words):
            self.notes.append("The plan asked for no actual change to how the %s moves." % self.names[k])
            return
        if not self.moving_at(k, f) and not any(key in ed for key in ("aim", "heading", "push")):
            self.notes.append("The %s is not moving at frame %d, so that change does nothing there." % (self.names[k], f))
        self.change.setdefault(k, []).append(ed)
        self.steps.append("From frame %d, the %s %s." % (f, self.names[k], " and ".join(words)))

    def move_from_side(self, k, ev, m):
        side = _side(m["from_side"])
        toward = m.get("toward") if isinstance(m.get("toward"), dict) else None
        if toward is not None and not _given(toward.get("direction")):
            target, tf, pw = self.place(toward, self.r(k))
            tf = _arrival(None, tf, 0, self.last)
        elif k < self.N0:
            e0, e1 = self.span(k)
            fc = self.sc.first_collision(k)
            if fc is not None:
                tf, pw = fc[0], "the spot where it hit the %s in the video" % self.sc.names[fc[1]]
            else:
                tf, pw = min(e0 + 25, e1), "the spot it had reached by frame %d" % min(e0 + 25, e1)
            target = self.sc.at(k, tf)[0]
        else:
            target, tf = self.where(k, self.default_frame(k))
            pw = "where it would have been at frame %d" % tf
        if _given(m.get("arrive_frame")):
            tf = _frame(m["arrive_frame"], tf, 0, self.last)
        word = str(m.get("speed") or "").strip().lower()
        sp = SPEED[word] if word in SPEED else self.base_speed(k)
        fac = _num(m.get("speed_factor"))
        if fac is not None and fac > 0:
            sp *= fac
        reach = REACH
        if toward is not None and _given(toward.get("object")) and not _given(toward.get("side")):
            reach = self.r(self.obj(toward["object"])) + self.r(k)      # arriving = touching it
        # arrive on time by default. A given frame is a START only when the user aimed it at a place
        # without saying when it gets there: measured, the language model otherwise copies the entry
        # frame here, and the green cube of video 10002 then arrived at frame 137 instead of 49
        aimed = toward is not None and not _given(toward.get("direction"))
        given = ev.get("frame") if (_given(ev.get("frame")) and not _given(m.get("arrive_frame"))) else None
        copied_entry = given is not None and k < self.N0 and _frame(given, -1) == self.span(k)[0]
        start = given if (given is not None and (aimed or not copied_entry)) else None
        a = aim_from_side(self.model, self.row(k), target, side, sp, self.h(k), self.last, self.fscale(k),
                          arrive=tf, start=start, reach=reach)
        _aim_notes(self.notes, self.names[k], a, pw)
        if k < self.N0:
            self.change.setdefault(k, []).append({"frame": a["frame"], "replace_track": True,
                                                  "pos": a["pos"].tolist(), "vel": a["vel"].tolist()})
            self.steps.append("The %s comes in from %s at frame %d instead, heading for %s (arriving at about frame %d)."
                              % (self.names[k], SIDE_WORDS[side], a["frame"], pw, a["arrive"]))
        else:
            self.spec(k).update(frame=a["frame"], pos=a["pos"].tolist(), vel=a["vel"].tolist())
            self.refresh_path(k)
            self.say_start(k, "The %s rolls in from %s from frame %d, heading for %s (arriving at about frame %d)."
                           % (self.names[k], SIDE_WORDS[side], a["frame"], pw, a["arrive"]))

    def ev_pin(self, ev):
        k = self.obj(ev.get("object"))
        if k in self.removed:
            return
        f = self.event_frame(k, ev.get("frame"), self.span(k)[0])
        self.mass[k] *= PIN_MASS
        self.change.setdefault(k, []).append({"frame": f, "speed_factor": 0.0})
        self.steps.append("From frame %d the %s is fixed to the table, so nothing can move it." % (f, self.names[k]))

    # -- output -------------------------------------------------------------------
    def result(self, plan):
        sc = self.sc
        N = self.N0 + len(self.add)
        self.notes = list(dict.fromkeys(self.notes))
        if self.world_friction != 1.0 or self.accel is not None:
            # friction or a tilt shapes every object's whole path, so no recorded motion stays valid
            for k in range(self.N0):
                if k not in self.removed and sc.on(k).size:
                    self.change.setdefault(k, []).append({"frame": int(sc.on(k)[0])})
            self.notes.append("Because this changes how everything moves, every object is simulated from the moment it "
                              "enters, so small errors have longer to grow.")
        for k, eds in self.change.items():
            if k >= self.N0 or not sc.on(k).size:
                continue
            e0 = int(sc.on(k)[0])
            for ed in eds:
                f = int(ed["frame"])
                if ed.get("vanish") or ed.get("vel") is not None or f - e0 >= 12 or sc.speed(k, f) < MOVING:
                    continue
                st = sc.steady_state(k, f)
                if st is not None:
                    if ed.get("pos") is None:
                        ed["pos"] = st[0].tolist()
                    ed["vel"] = st[1].tolist()
        for k, eds in self.change.items():
            # an object whose track is replaced has nothing to change before its new
            # start, and the replacement must come first among events on that frame
            starts = [int(ed["frame"]) for ed in eds if ed.get("replace_track")]
            if starts:
                for ed in eds:
                    if not ed.get("replace_track"):
                        ed["frame"] = max(int(ed["frame"]), min(starts))
            eds.sort(key=lambda ed: (int(ed["frame"]), 0 if ed.get("replace_track") else 1))
        drag = [self.world_friction * d for d in self.drag]
        props = [k for k in range(self.N0) if self.keys[k] != sc.keys[k] or not np.array_equal(self.attrs[k], sc.attrs[k])]
        scaled = [k for k in range(N) if self.mass[k] != 1.0 or self.radius[k] != 1.0 or self.drag[k] != 1.0]
        if not (self.removed or self.change or self.add or props or scaled
                or self.world_friction != 1.0 or self.extend != DEFAULT_EXTEND or self.accel is not None):
            raise PlanError("Nothing in that request could be simulated. " + " ".join(self.notes))
        attrs_file = np.asarray(sc.z["attrs"], np.float32).copy()
        attrs_file[:self.N0] = self.attrs
        zz = {key: sc.z[key] for key in ("positions", "velocities", "presence")}
        zz["attrs"] = attrs_file
        if "yaw" in sc.z:
            zz["yaw"] = sc.z["yaw"]
        render = np.array([self.row(k) for k in range(N)], np.float32)
        render[:, 15] *= np.asarray(self.radius, np.float32)
        pure = (bool(self.removed) and not self.change and not self.add and not props and not scaled
                and self.world_friction == 1.0 and self.extend == DEFAULT_EXTEND and self.accel is None)
        return {"z": zz, "removed": sorted(self.removed),
                "edits": {"change": self.change, "add": self.add,
                          "mass_scale": list(self.mass) if any(x != 1.0 for x in self.mass) else None,
                          "radius_scale": list(self.radius) if any(x != 1.0 for x in self.radius) else None,
                          "accel": self.accel},
                "drag_scale": drag if any(x != 1.0 for x in drag) else None,
                "extend": self.extend, "names": self.names, "keys": self.keys, "render_attrs": render,
                "changed": sorted(set(self.change) | set(scaled) | set(props)),
                "steps": self.steps, "notes": self.notes,
                "understood": str(plan.get("understood") or "").strip(), "pure_removal": pure}


def resolve(plan, sc, model, marked=None, entry=None):
    """PLAN (the language model's JSON, or one built by the page) -> everything
    cf_world and the page need. Raises PlanError with a message for the user."""
    if not isinstance(plan, dict):
        raise PlanError("The language model did not return a plan. Try saying it another way.")
    events = [e for e in plan.get("events", []) if isinstance(e, dict)] if isinstance(plan.get("events"), list) else []
    world = plan.get("world") if isinstance(plan.get("world"), dict) else {}
    ff = _num(world.get("friction_factor"))
    ff = 1.0 if ff is None or ff < 0 else ff
    extend = _frame(world.get("predict_frames"), DEFAULT_EXTEND, 0, MAX_EXTEND)
    slope = world.get("slope") if isinstance(world.get("slope"), dict) and _given(world["slope"].get("direction")) else None
    if not events and ff == 1.0 and extend == DEFAULT_EXTEND and slope is None:
        why = plan.get("cannot")
        raise PlanError(("This can't be simulated: %s" % why) if _given(why)
                        else "I couldn't find a change to simulate in that request.")
    p = Plan(sc, model, marked, entry, extend, ff)
    if ff != 1.0:
        p.steps.append("Friction is %s for every object." % ("switched off" if ff == 0 else "%s times the real friction" % _times(ff)))
        p.notes.append("The world model never saw other friction, so the result is an estimate.")
    if slope is not None:
        s_ = _side(slope["direction"])
        word = str(slope.get("strength") or "medium").strip().lower()
        word = word if word in SLOPE else "medium"
        p.accel = (SIDE_VEC[s_] * SLOPE[word]).tolist()
        p.steps.append("The table is tilted, so everything is steadily pulled towards %s (%s slope)."
                       % (SIDE_WORDS[s_], word))
        p.notes.append("A tilted table was never in the training videos, so the result is an estimate.")
    if extend != DEFAULT_EXTEND:
        p.steps.append("The world model predicts %d frames past the end of the video." % extend)
    kinds = [str(ev.get("do", "")).strip().lower() for ev in events]
    for kind in Plan.ORDER:
        for ev, kd in zip(events, kinds):
            if kd == kind:
                getattr(p, "ev_" + kind)(ev)
    unknown = sorted(set(kinds) - set(Plan.ORDER))
    if unknown:
        p.notes.append("Skipped steps the system does not know: %s." % ", ".join(u or "?" for u in unknown))
    return p.result(plan)
