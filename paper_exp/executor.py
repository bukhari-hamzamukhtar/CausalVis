"""
paper_exp/executor.py  —  CLEVRER's program language over CausalVis scenes
=========================================================================

A re-implementation of the official NS-DR symbolic executor (chuangg/CLEVRER,
executor/executor.py) for the program vocabulary used in CLEVRER's question files
(start/end, get_frame, get_col_partner, get_object, unseen_events, all_events,
get_counterfact). Same stack semantics, same 'error' behaviour, same causal-trace
definition of "responsible for". What changes is the scene it reads:

  objects      detection-only inventory (paper_exp/detonly.py)
  in / out     first / last detected frame
  collisions   events of a CausalVis collision detector on the recorded tracks
  moving       recorded speed above a threshold (chosen on VAL)
  unseen       collisions the world model produces after the video ends
  counterfact  collisions in a CausalVis counterfactual world

A multiple-choice option is executed as choice program + question program, and
'yes' means the option is correct -- as in the official run_mc.py.
"""

import numpy as np

COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'yellow', 'cyan', 'purple']
MATERIALS = ['metal', 'rubber']
SHAPES = ['sphere', 'cylinder', 'cube']
ERR = 'error'


def merge_frames(frames, gap=6):
    out = []
    for f in sorted(set(int(x) for x in frames)):
        if not out or f - out[-1] > gap:
            out.append(f)
    return out


def pair_frames(events, det, pair):
    if det in ("union", "conf"):
        fr = events["cal"].get(pair, []) + events["kick2"].get(pair, [])
    else:
        fr = events[det].get(pair, [])
    return merge_frames(fr)


def refine(P, i, j, f, span):
    """Move a detector onset to the closest approach within `span` frames."""
    if span <= 0:
        return f
    best, bf = None, f
    for t in range(f, min(len(P), f + span + 1)):
        if np.isfinite(P[t, i]).all() and np.isfinite(P[t, j]).all():
            d = float(np.linalg.norm(P[t, i] - P[t, j]))
            if best is None or d < best:
                best, bf = d, t
    return bf


RADIUS = {"sphere": 0.058333, "cylinder": 0.058333, "cube": 0.064723}


def features(P, i, j, f, contact, lo=3, hi=8):
    """Smallest surface gap and largest velocity change of the pair around frame f."""
    g, kmax = 1e9, 0.0
    for t in range(max(0, f - lo), min(len(P), f + hi + 1)):
        if np.isfinite(P[t, i]).all() and np.isfinite(P[t, j]).all():
            g = min(g, float(np.linalg.norm(P[t, i] - P[t, j])) - contact)
        if 3 <= t < len(P) - 3:
            for k in (i, j):
                a, b = P[t + 3, k] - P[t, k], P[t, k] - P[t - 3, k]
                if np.isfinite(a).all() and np.isfinite(b).all():
                    kmax = max(kmax, float(np.linalg.norm(a - b)) / 3.0)
    return g, kmax


class Scene:
    """Everything a program can ask about one video."""

    def __init__(self, keys, presence, speed, obs_world, fut_world=None, cf_worlds=None,
                 det="union", moving_th=0.004, end_frame=125, refine_span=0, fut_ext=30,
                 min_run=1, smooth=0, gap_th=0.01, kick_th=0.003, tiebreak=False):
        self.attrs = [dict(zip(("color", "material", "shape"), str(k).split("_")[:3])) for k in keys]
        self.N = len(self.attrs)
        self.pres = presence > 0
        self.T = int(obs_world["T"])
        self.speed = speed
        self.th, self.min_run, self.smooth, self.tiebreak = moving_th, min_run, smooth, tiebreak
        ev = [{'type': 'start', 'frame': 0}, {'type': 'end', 'frame': end_frame}]
        for k in range(self.N):
            on = np.where(self.pres[:self.T, k])[0]
            if on.size == 0:
                continue
            if on[0] > 0:
                ev.append({'type': 'in', 'object': [k], 'frame': int(on[0])})
            if on[-1] < self.T - 1:
                ev.append({'type': 'out', 'object': [k], 'frame': int(on[-1]) + 1})
        P = obs_world["positions"]
        rad = np.array([RADIUS[a["shape"]] for a in self.attrs])
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if det == "conf":
                    # every detector's onsets are candidates; keep those whose surfaces came
                    # within gap_th, or within 2*gap_th with a clear change of velocity
                    cands = merge_frames(sum((obs_world["events"][d].get((i, j), []) for d in ("rule", "cal", "kick2", "kick4")), []))
                    for f in cands:
                        if f >= self.T:
                            continue
                        g, kmax = features(P, i, j, f, rad[i] + rad[j])
                        if g <= gap_th or (g <= 2 * gap_th and kmax >= kick_th):
                            ev.append({'type': 'collision', 'object': [i, j], 'frame': refine(P, i, j, f, refine_span), 'conf': -g})
                    continue
                for f in pair_frames(obs_world["events"], det, (i, j)):
                    if f < self.T:
                        g, _ = features(P, i, j, f, rad[i] + rad[j])
                        ev.append({'type': 'collision', 'object': [i, j], 'frame': refine(P, i, j, f, refine_span), 'conf': -g})
        self.existing = ev
        self.traces = [[] for _ in range(self.N)]
        for e in ev:
            if e['type'] not in ('start', 'end'):
                for o in e['object']:
                    self.traces[o].append(e)
        self.traces = [sorted(tr, key=lambda e: e['frame']) for tr in self.traces]
        self.unseen = []
        if fut_world is not None:
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    for f in pair_frames(fut_world["events"], det, (i, j)):
                        if self.T <= f < self.T + fut_ext:
                            self.unseen.append({'type': 'collision', 'object': [i, j], 'frame': f})
        self.cf_worlds = cf_worlds or {}
        self.cf_det, self.fut_ext = det, fut_ext

    # ---------------- helpers ----------------
    def visible(self, o, frame=None):
        if frame is None:
            return bool(self.pres[:self.T, o].any())
        return 0 <= frame < self.T and bool(self.pres[frame, o])

    def moving(self, o, frame=None):
        sp = self.speed[:self.T, o]
        if frame is None:
            run = 0
            for t in range(self.T):
                run = run + 1 if (self.pres[t, o] and sp[t] > self.th) else 0
                if run >= self.min_run:
                    return True
            return False
        lo, hi = max(0, frame - self.smooth), min(self.T, frame + self.smooth + 1)
        w = [sp[t] for t in range(lo, hi) if self.pres[t, o]]
        return bool(w and np.mean(w) > self.th)

    # ---------------- program runner ----------------
    def run(self, pg):
        out = self.run_raw(pg)
        return ERR if out is ERR else str(out)

    def run_raw(self, pg):
        """Like run, but returns the top of the stack itself (an event, an object id, ...)."""
        stack = []
        for m in pg:
            if m in ('<END>', '<NULL>'):
                break
            if m == '<START>':
                continue
            if m not in MODULES:
                stack.append(m)
                continue
            fn, nargs = MODULES[m]
            if len(stack) < nargs:
                return ERR
            argv = stack[len(stack) - nargs:] if nargs else []
            del stack[len(stack) - nargs:]
            try:
                out = fn(self, *argv)
            except Exception:
                return ERR
            if isinstance(out, str) and out == ERR:
                return ERR
            stack.append(out)
        return stack[-1] if stack else ERR


def _objs(x):
    return isinstance(x, list) and (len(x) == 0 or isinstance(x[0], int))


def _evs(x):
    return isinstance(x, list) and (len(x) == 0 or isinstance(x[0], dict))


def m_objects(s):
    return list(range(s.N))


def m_events(s):
    return sorted(s.existing, key=lambda e: e['frame'])


def m_unique(s, x):
    if isinstance(x, list) and len(x) == 1:
        return x[0]
    if s.tiebreak and isinstance(x, list) and len(x) > 1 and all(isinstance(e, dict) and e.get('type') == 'collision' for e in x):
        # the question presumes ONE such collision: keep the one whose surfaces came closest
        return max(x, key=lambda e: e.get('conf', 0.0))
    return ERR


def m_count(s, x):
    return len(x) if isinstance(x, list) else ERR


def m_exist(s, x):
    return ('yes' if x else 'no') if isinstance(x, list) else ERR


def m_negate(s, b):
    return {'yes': 'no', 'no': 'yes'}.get(b, ERR)


def m_belong_to(s, entry, events):
    if not isinstance(events, list):
        return ERR
    if isinstance(entry, dict):
        for e in events:
            if e['type'] not in ('start', 'end') and entry['type'] == e['type'] and set(entry['object']) == set(e['object']):
                return 'yes'
        return 'no'
    if isinstance(entry, int):
        for e in events:
            if e['type'] not in ('start', 'end') and entry in e['object']:
                return 'yes'
        return 'no'
    return ERR


def _filter_attr(field, vocab):
    def f(s, objs, val):
        if not _objs(objs) or val not in vocab:
            return ERR
        return [o for o in objs if s.attrs[o][field] == val]
    return f


def _frame(frame):
    if frame == 'null':
        return None, True
    if isinstance(frame, (int, np.integer)):
        return int(frame), True
    return None, False


def m_filter_moving(s, objs, frame):
    fr, ok = _frame(frame)
    if not isinstance(objs, list) or not ok:
        return ERR
    return [o for o in objs if s.visible(o, fr) and s.moving(o, fr)]


def m_filter_stationary(s, objs, frame):
    fr, ok = _frame(frame)
    if not isinstance(objs, list) or not ok:
        return ERR
    return [o for o in objs if s.visible(o, fr) and not s.moving(o, fr)]


def m_start(s):
    return next(e for e in s.existing if e['type'] == 'start')


def m_end(s):
    return next(e for e in s.existing if e['type'] == 'end')


def _filter_io(kind):
    def f(s, events, objs):
        if not _evs(events):
            return ERR
        objs = objs if isinstance(objs, list) else [objs]
        if not _objs(objs):
            return ERR
        return [e for e in events if e['type'] == kind and e['object'][0] in objs]
    return f


def m_filter_collision(s, events, objs):
    if not _evs(events):
        return ERR
    objs = objs if isinstance(objs, list) else [objs]
    if not _objs(objs):
        return ERR
    return [e for e in events if e['type'] == 'collision' and (e['object'][0] in objs or e['object'][1] in objs)]


def m_filter_order(s, events, order):
    if not _evs(events) or order not in ('first', 'second', 'last'):
        return ERR
    idx = {'first': 0, 'second': 1, 'last': -1}[order]
    if len(events) == 0 or idx >= len(events):
        return ERR
    return events[idx]


def m_filter_before(s, events, event):
    if not _evs(events) or not isinstance(event, dict):
        return ERR
    return [e for e in events if e['frame'] < event['frame']]


def m_filter_after(s, events, event):
    if not _evs(events) or not isinstance(event, dict):
        return ERR
    return [e for e in events if e['frame'] > event['frame']]


def _query(field):
    def f(s, o):
        return s.attrs[o][field] if isinstance(o, int) else ERR
    return f


def m_get_frame(s, e):
    return e['frame'] if isinstance(e, dict) else ERR


def m_get_object(s, e):
    if not isinstance(e, dict) or e['type'] not in ('in', 'out'):
        return ERR
    return e['object'][0]


def m_get_col_partner(s, e, o):
    if not isinstance(e, dict) or not isinstance(o, int) or e['type'] != 'collision' or o not in e['object']:
        return ERR
    a, b = e['object']
    return b if o == a else a


def m_filter_ancestor(s, events, event):
    if not _evs(events) or not isinstance(event, dict) or 'type' not in event:
        return ERR
    causes = []

    def search(target):
        nxt = []
        for tr in s.traces:
            if target in tr:
                i = tr.index(target)
                if i > 0 and tr[i - 1] not in nxt:
                    nxt.append(tr[i - 1])
                    if tr[i - 1] not in causes:
                        causes.append(tr[i - 1])
        for e in nxt:
            search(e)
    search(event)
    return [e for e in events if e in causes]


def m_unseen_events(s):
    return s.unseen


def m_all_events(s):
    return [{'type': 'collision', 'object': [i, j]} for i in range(s.N) for j in range(i + 1, s.N)]


def m_get_counterfact(s, events, obj):
    if not _evs(events) or not isinstance(obj, int):
        return ERR
    w = s.cf_worlds.get(obj)
    if w is None:
        return ERR
    pairs = set()
    for i in range(s.N):
        for j in range(i + 1, s.N):
            if i == obj or j == obj:
                continue
            fr = pair_frames(w["events"], s.cf_det, (i, j))
            if any(f < s.T + s.fut_ext for f in fr):
                pairs.add(frozenset((i, j)))
    return [e for e in events if frozenset(e['object']) in pairs]


MODULES = {
    'objects': (m_objects, 0), 'events': (m_events, 0), 'unique': (m_unique, 1),
    'count': (m_count, 1), 'exist': (m_exist, 1), 'negate': (m_negate, 1),
    'belong_to': (m_belong_to, 2),
    'filter_color': (_filter_attr('color', COLORS), 2),
    'filter_material': (_filter_attr('material', MATERIALS), 2),
    'filter_shape': (_filter_attr('shape', SHAPES), 2),
    'filter_moving': (m_filter_moving, 2), 'filter_stationary': (m_filter_stationary, 2),
    'start': (m_start, 0), 'end': (m_end, 0),
    'filter_in': (_filter_io('in'), 2), 'filter_out': (_filter_io('out'), 2),
    'filter_collision': (m_filter_collision, 2), 'filter_order': (m_filter_order, 2),
    'filter_before': (m_filter_before, 2), 'filter_after': (m_filter_after, 2),
    'query_color': (_query('color'), 1), 'query_material': (_query('material'), 1),
    'query_shape': (_query('shape'), 1),
    'get_frame': (m_get_frame, 1), 'get_object': (m_get_object, 1),
    'get_col_partner': (m_get_col_partner, 2),
    'filter_ancestor': (m_filter_ancestor, 2),
    'unseen_events': (m_unseen_events, 0), 'all_events': (m_all_events, 0),
    'get_counterfact': (m_get_counterfact, 2),
}
