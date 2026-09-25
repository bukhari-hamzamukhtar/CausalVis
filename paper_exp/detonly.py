"""
paper_exp/detonly.py  —  object tracks from DETECTIONS ONLY (no annotation at test time)
=======================================================================================

The original 3D build (src2/build_trajectories_3d.py) took three things from the
dataset's scene annotation: the list of objects and their attributes, each object's
in/out frames, and the annotated collision frames used to segment cube orientation.
For the paper every test-time input must come from the detector alone. This module
rebuilds the same arrays from NS-DR's released detections (masks + predicted colour,
material, shape) and nothing else:

  inventory    (colour, shape) groups of detections; two objects share a colour and
               shape only if two such detections appear in the same frame far apart
               (CLEVRER never repeats all three attributes, 21% of clips repeat two);
               each track's material is its score-weighted majority label (the
               detector flips material between frames).
  presence     frames with an assigned detection; gaps of <= 5 frames interpolated,
               exactly as before.
  orientation  cube yaw still piecewise linear, but its breakpoints are contact
               onsets DETECTED on the recorded tracks (calibrated rule), not annotated
               collisions.
Annotated collisions are still stored (mapped by attributes) for DIAGNOSIS only; the
evaluator never answers from them.

    python paper_exp/detonly.py report --which train --limit 1000     # tracker vs annotation
    python paper_exp/detonly.py build --which test --questions cf-train --out data/trajectories_3d_det
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src2"))
import lift3d                                                            # noqa: E402
import fit3d                                                             # noqa: E402
import build_trajectories_3d as bt3                                     # noqa: E402

PROPOSALS = os.path.join(ROOT, "legacy", "v1", "data", "processed_proposals")
NMS_PX = 8.0          # two same-(colour, shape) detections closer than this are one object
SPLIT_PX = 20.0       # ... farther apart than this in one frame are two objects
SPLIT_FRAMES = 3      # frames with such a pair needed to declare two objects
MIN_FRAMES = 5        # a track seen in fewer frames is a false detection (checked on TRAIN)
CAL_SCALE, CAL_THRESH = 1.2, 0.03


def _dets(z):
    """Per frame: detections as dicts with centroid and area, NMS within (colour, shape)."""
    T = len(z["frames"])
    out = [[] for _ in range(T)]
    for f in z["frames"]:
        t = int(f["frame_index"])
        if not (0 <= t < T):
            continue
        cand = []
        for d in f.get("objects", []):
            st = lift3d.mask_stats(d["mask"])
            if st is None:
                continue
            cand.append({"c": d.get("color"), "m": d.get("material"), "s": d.get("shape"),
                         "u": st[0], "v": st[1], "area": st[2], "score": float(d.get("score", 1.0)),
                         "rle": d["mask"]})
        cand.sort(key=lambda d: -d["score"])
        kept = []
        for d in cand:
            if any(k["c"] == d["c"] and k["s"] == d["s"] and
                   np.hypot(k["u"] - d["u"], k["v"] - d["v"]) < NMS_PX for k in kept):
                continue
            kept.append(d)
        out[t] = kept
    return out


def _track_pair(frames_dets):
    """Two objects with the same colour and shape: assign each frame's detections to
    two tracks by position (nearest last position), with the material label as the
    tie-breaker. Returns two lists of (t, det)."""
    T = len(frames_dets)
    seed = next(t for t in range(T) if len(frames_dets[t]) >= 2)
    a, b = frames_dets[seed][0], frames_dets[seed][1]
    if a["m"] == b["m"]:
        pass
    tracks = [[(seed, a)], [(seed, b)]]
    last = [(a["u"], a["v"]), (b["u"], b["v"])]
    for order in (range(seed + 1, T), range(seed - 1, -1, -1)):
        pos = [last[0], last[1]]
        for t in order:
            ds = frames_dets[t][:2]
            if not ds:
                continue
            if len(ds) == 2:
                c_same = (np.hypot(ds[0]["u"] - pos[0][0], ds[0]["v"] - pos[0][1]) +
                          np.hypot(ds[1]["u"] - pos[1][0], ds[1]["v"] - pos[1][1]))
                c_swap = (np.hypot(ds[1]["u"] - pos[0][0], ds[1]["v"] - pos[0][1]) +
                          np.hypot(ds[0]["u"] - pos[1][0], ds[0]["v"] - pos[1][1]))
                pair = (ds[0], ds[1]) if c_same <= c_swap else (ds[1], ds[0])
                for k in (0, 1):
                    tracks[k].append((t, pair[k])); pos[k] = (pair[k]["u"], pair[k]["v"])
            else:
                d = ds[0]
                dist = [np.hypot(d["u"] - pos[k][0], d["v"] - pos[k][1]) for k in (0, 1)]
                k = 0 if dist[0] <= dist[1] else 1
                tracks[k].append((t, d)); pos[k] = (d["u"], d["v"])
    return [sorted(tr, key=lambda x: x[0]) for tr in tracks]


def _majority(track, field):
    w = {}
    for _, d in track:
        w[d[field]] = w.get(d[field], 0.0) + d["score"]
    return max(w, key=w.get)


def read_clip_det(path, keep_masks=False, min_frames=MIN_FRAMES):
    with open(path, "r", encoding="utf-8") as fh:
        z = json.load(fh)
    fd = _dets(z)
    T = len(fd)
    groups = {}
    for t in range(T):
        for d in fd[t]:
            groups.setdefault((d["c"], d["s"]), [[] for _ in range(T)])[t].append(d)
    tracks = []
    for (c, s), per in groups.items():
        two = sum(1 for t in range(T) if len(per[t]) >= 2 and
                  np.hypot(per[t][0]["u"] - per[t][1]["u"], per[t][0]["v"] - per[t][1]["v"]) > SPLIT_PX)
        if two >= SPLIT_FRAMES:
            trs = _track_pair(per)
        else:
            trs = [[(t, per[t][0]) for t in range(T) if per[t]]]
        for tr in trs:
            if len(tr) < min_frames:
                continue
            tracks.append({"c": c, "s": s, "m": _majority(tr, "m"), "tr": tr})
    # two tracks that ended with the SAME material: give the less certain one the other label
    by_cs = {}
    for k in tracks:
        by_cs.setdefault((k["c"], k["s"]), []).append(k)
    for grp in by_cs.values():
        if len(grp) == 2 and grp[0]["m"] == grp[1]["m"]:
            other = "rubber" if grp[0]["m"] == "metal" else "metal"
            share = [sum(d["score"] for _, d in g["tr"] if d["m"] == other) / max(1e-9, sum(d["score"] for _, d in g["tr"]))
                     for g in grp]
            grp[int(np.argmax(share))]["m"] = other
    tracks.sort(key=lambda k: (k["tr"][0][0], k["tr"][0][1]["u"]))
    N = len(tracks)
    objs = [{"color": k["c"], "material": k["m"], "shape": k["s"], "id": i} for i, k in enumerate(tracks)]
    det = [[None] * N for _ in range(T)]
    raw = [[None] * N for _ in range(T)] if keep_masks else None
    for i, k in enumerate(tracks):
        for t, d in k["tr"]:
            det[t][i] = (d["u"], d["v"], d["area"], d["score"])
            if keep_masks:
                raw[t][i] = lift3d.rle_decode(d["rle"]["counts"], d["rle"]["size"]).astype(bool)
    # annotation, mapped by attributes: DIAGNOSIS ONLY
    gt = z.get("ground_truth", {})
    key2ours = {(o["color"], o["material"], o["shape"]): o["id"] for o in objs}
    gt2ours = {o["id"]: key2ours.get((o["color"], o["material"], o["shape"])) for o in gt.get("objects", [])}
    cols = []
    for c in gt.get("collisions", []):
        if len(c.get("object", [])) == 2:
            a, b = gt2ours.get(c["object"][0]), gt2ours.get(c["object"][1])
            if a is not None and b is not None:
                cols.append((int(c["frame"]), int(a), int(b)))
    exits = [(int(e["frame"]), gt2ours[e["object"]]) for e in gt.get("in_outs", [])
             if e.get("type") == "out" and gt2ours.get(e["object"]) is not None]
    clip = {"video_name": z.get("video_name", os.path.basename(path).split(".")[0]),
            "objects": objs, "det": det, "inside": np.ones((T, N), dtype=bool),
            "collisions": cols, "exits": exits, "T": T, "N": N,
            "gt_objects": gt.get("objects", []), "gt_in_outs": gt.get("in_outs", [])}
    if keep_masks:
        clip["raw"] = raw
    return clip


def detected_onsets(rec):
    """Contact onsets on the recorded (detected) tracks, calibrated rule -- used only as
    breakpoints for the cube orientation fit."""
    P, pres, r = rec["positions"], rec["presence"] > 0, rec["attrs"][:, 15]
    T, N = pres.shape
    out = []
    for i in range(N):
        for j in range(i + 1, N):
            both = pres[:, i] & pres[:, j]
            d = np.linalg.norm(P[:, i] - P[:, j], axis=-1)
            hit = both & (d - CAL_SCALE * (r[i] + r[j]) <= CAL_THRESH)
            on = np.where(hit & ~np.concatenate([[False], hit[:-1]]))[0]
            out += [(int(t), i, j) for t in on]
    return out


_CACHE = {}


def build_det(path, cam, sizes):
    """Two passes: tracks without orientation -> detected contact onsets -> tracks with
    cube orientation segmented at those onsets. Same arrays as build_trajectories_3d."""
    _CACHE.clear()
    bt3.read_clip = lambda p: _CACHE.setdefault(("plain", p), read_clip_det(p))
    first = bt3.build_one(path, cam, sizes, with_yaw=False)
    if first is None:
        return None
    onsets = detected_onsets(first)
    clip = read_clip_det(path, keep_masks=True)
    gt_cols = clip["collisions"]
    clip["collisions"] = onsets                          # what fit_yaw_track segments on
    bt3.load_one = lambda p: clip
    rec = bt3.build_one(path, cam, sizes, with_yaw=True)
    if rec is None:
        return None
    rec["collisions"] = np.asarray(gt_cols, np.int64).reshape(-1, 3)   # diagnosis only
    rec["detected_onsets"] = np.asarray(onsets, np.int64).reshape(-1, 3)
    rec["inventory_source"] = np.asarray("detections")
    return rec


def compare_inventory(clip):
    """Ours vs the annotation for one clip."""
    gt = {(o["color"], o["material"], o["shape"]): o["id"] for o in clip["gt_objects"]}
    ours = {(o["color"], o["material"], o["shape"]): o["id"] for o in clip["objects"]}
    matched = set(gt) & set(ours)
    ins = {e["object"]: e["frame"] for e in clip["gt_in_outs"] if e.get("type") == "in"}
    outs = {e["object"]: e["frame"] for e in clip["gt_in_outs"] if e.get("type") == "out"}
    d_in, d_out = [], []
    T = clip["T"]
    for key in matched:
        g, o = gt[key], ours[key]
        fr = [t for t in range(T) if clip["det"][t][o] is not None]
        d_in.append(fr[0] - ins.get(g, 0))
        d_out.append((fr[-1] + 1) - outs.get(g, T))
    return {"gt": len(gt), "ours": len(ours), "matched": len(matched), "d_in": d_in, "d_out": d_out}


def _videos(which, questions):
    sys.path.insert(0, os.path.join(ROOT, "src2"))
    from benchmark_eval import load_questions, load_split_set, _vid_num
    allowed = load_split_set(os.path.join(ROOT, "split3d_yaw.json"), which)
    if questions is None:
        return sorted(allowed)
    rows = load_questions(questions)
    return sorted(set(_vid_num(r["video"]) for r in rows) & allowed)


def main():
    qf = {"cf-val": "zechennlp/counterfactual/validation-00000-of-00001.json",
          "cf-train": "zechennlp/counterfactual/train-00000-of-00001.json"}
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["report", "build"])
    ap.add_argument("--which", default="train")
    ap.add_argument("--questions", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "trajectories_3d_det"))
    ap.add_argument("--camera", default=os.path.join(ROOT, "camera_fit.json"))
    a = ap.parse_args()
    q = qf.get(a.questions, a.questions)
    vids = _videos(a.which, os.path.join(ROOT, q) if q else None)
    vids = vids[a.shard::a.nshards][:a.limit]
    if a.mode == "report":
        tot = {"gt": 0, "ours": 0, "matched": 0, "clips": 0, "perfect": 0}
        d_in, d_out = [], []
        for v in vids:
            c = read_clip_det(os.path.join(PROPOSALS, "sim_%05d.json" % v))
            r = compare_inventory(c)
            for k in ("gt", "ours", "matched"):
                tot[k] += r[k]
            tot["clips"] += 1
            tot["perfect"] += int(r["gt"] == r["ours"] == r["matched"])
            d_in += r["d_in"]; d_out += r["d_out"]
        d_in, d_out = np.abs(np.array(d_in)), np.abs(np.array(d_out))
        print(json.dumps({"which": a.which, "clips": tot["clips"],
                          "objects_annotated": tot["gt"], "objects_found": tot["ours"], "matched": tot["matched"],
                          "recall": round(tot["matched"] / max(1, tot["gt"]), 4),
                          "precision": round(tot["matched"] / max(1, tot["ours"]), 4),
                          "clips_inventory_exact": round(tot["perfect"] / max(1, tot["clips"]), 4),
                          "entry_frame_abs_err_median": float(np.median(d_in)), "entry_err_p90": float(np.percentile(d_in, 90)),
                          "exit_frame_abs_err_median": float(np.median(d_out)), "exit_err_p90": float(np.percentile(d_out, 90))}))
        return
    with open(a.camera, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    cam, sizes = lift3d.Camera(blob["camera"]), blob["sizes"]
    os.makedirs(a.out, exist_ok=True)
    done = 0
    for v in vids:
        outp = os.path.join(a.out, "sim_%05d.npz" % v)
        if os.path.exists(outp):
            continue
        rec = build_det(os.path.join(PROPOSALS, "sim_%05d.json" % v), cam, sizes)
        if rec is None:
            print("skip", v, flush=True)
            continue
        np.savez_compressed(outp, **rec)
        done += 1
        if done % 25 == 0:
            print("shard %d built %d / %d" % (a.shard, done, len(vids)), flush=True)
    print("shard %d finished: %d built" % (a.shard, done), flush=True)


if __name__ == "__main__":
    main()
