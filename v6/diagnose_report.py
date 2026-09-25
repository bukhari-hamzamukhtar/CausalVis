"""
v6/diagnose_report.py  —  read v6/cmp/diag/shard*.jsonl and say where it fails

Every wrong answer is assigned to ONE cause, checked in this order (the first
that applies wins, so the buckets add up to the total):

  missed collisions (truth: they touch, model: no)
    1 REFUSED        one of the pair was not on the table yet at the start frame,
                     so the simulation returned "no collision" without running
    2 LEFT TABLE     one of the pair slid off the table in the simulation
    3 CONE           (factual pairs) the video shows NO collision between them,
                     yet they collide after the removal: the removal did affect
                     this pair, but the cone treated it as unaffected
    4 NOT SEEN       (factual pairs) even replaying the REAL recorded motion, our
                     3D objects never come within TOL voxels: a reconstruction /
                     size problem, no simulator could fix it
    5 RULE           the voxel sensor on the simulated motion saw contact (within
                     TOL voxels) but the centre-distance rule did not count it
    6 NEAR MISS      simulation came within NEAR voxels but did not touch
    7 FAR MISS       simulation never came within NEAR voxels

  invented collisions (truth: no touch, model: yes)
    1 CONE           (factual pairs) the video shows them colliding, the removal
                     prevented it, and the cone treated the pair as unaffected
    2 PREVENTED      (counterfactual pairs) they collided in the video and the
                     model failed to see the removal prevents it
    3 SENSOR SAYS NO the voxel sensor never saw contact but the rule counted one
    4 GRAZE          everything else
"""

import glob
import json
import os
import sys
from collections import Counter, defaultdict

TOL = 3.0      # voxels: "touching" for the sensor (3 voxels = 0.06 world = 0.01 normalised)
NEAR = 10.0    # voxels: a near miss


def pct(a, b):
    return "%5.1f%% (%d/%d)" % (100.0 * a / b, a, b) if b else "   --"


def table(title, groups):
    print("\n" + title)
    for name, rs in groups:
        if not rs:
            continue
        ok = sum(r["ok"] for r in rs)
        fn = sum(1 for r in rs if not r["ok"] and r["truth_collides"] and not r["model_happens"])
        fp = sum(1 for r in rs if not r["ok"] and not r["truth_collides"] and r["model_happens"])
        print("   %-44s %-20s missed %4d  invented %3d" % (name, pct(ok, len(rs)), fn, fp))


def sensor_touch(r, tol=TOL, key="sim_near"):
    return any(g <= tol for f, g in r.get(key, []) if f > r.get("s", -1))


def fn_cause(r):
    if r.get("refused"):
        return "1 REFUSED (object not on table at start)"
    if r["path"] == "in_cone" and any(f < r["s"] for f in r["obs_frames"]):
        return "1b BEFORE START (video collision earlier than the sim start)"
    if r.get("left_table") is not None:
        return "2 LEFT TABLE in simulation"
    if r["path"] == "outside_cone":
        if not r["obs_frames"]:
            return "3 CONE (removal changed this pair, cone said unaffected)"
        if r.get("rec_vgap_min", 99) > TOL:
            return "4 NOT SEEN even in the real recording"
    if sensor_touch(r):
        return "5 RULE (sensor saw contact, rule did not count it)"
    if r.get("sim_vgap_min", 99) <= NEAR:
        return "6 NEAR MISS (simulation within %d voxels)" % NEAR
    return "7 FAR MISS (simulation never came close)"


def fp_cause(r):
    if r["path"] == "outside_cone" and r["obs_frames"]:
        return "1 CONE (removal prevented it, cone said unaffected)"
    if r["path"] == "in_cone" and r["obs_frames"]:
        return "2 PREVENTED (model missed that removal stops it)"
    if not sensor_touch(r):
        return "3 SENSOR SAYS NO, rule said yes"
    return "4 GRAZE / invented"


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "v6/cmp/diag"
    recs = []
    for f in sorted(glob.glob(os.path.join(d, "shard*.jsonl"))):
        recs += [json.loads(l) for l in open(f, encoding="utf-8")]
    sim = [r for r in recs if r["path"] != "participant_removed"]
    n = len(recs); ok = sum(r["ok"] for r in recs)
    print("=" * 78)
    print("CHOICES %d   CORRECT %s   (benchmark log says 74.8%% = 2488/3328)" % (n, pct(ok, n)))
    print("=" * 78)

    # ---------------- what kinds of question it does well / badly on --------------
    table("BY QUESTION POLARITY", [
        ("'which event WILL happen'", [r for r in recs if not r["negate"]]),
        ("'which event will NOT happen'", [r for r in recs if r["negate"]])])
    table("BY WHAT THE TRUE ANSWER IS (physically)", [
        ("pair really collides after removal", [r for r in recs if r["truth_collides"]]),
        ("pair really does NOT collide", [r for r in recs if not r["truth_collides"]])])
    table("BY HOW THE PAIR RELATES TO THE REMOVED OBJECT", [
        ("a removed object is in the pair (logic)", [r for r in recs if r["path"] == "participant_removed"]),
        ("factual: pair unaffected by removal", [r for r in recs if r["path"] == "outside_cone"]),
        ("counterfactual: pair affected by removal", [r for r in recs if r["path"] == "in_cone"])])
    cf = [r for r in recs if r["path"] == "in_cone"]
    table("COUNTERFACTUAL PAIRS: what the removal does to them", [
        ("collided in video, still collide", [r for r in cf if r["obs_frames"] and r["truth_collides"]]),
        ("collided in video, removal PREVENTS it", [r for r in cf if r["obs_frames"] and not r["truth_collides"]]),
        ("no collision in video, removal CAUSES one", [r for r in cf if not r["obs_frames"] and r["truth_collides"]]),
        ("no collision in video, still none", [r for r in cf if not r["obs_frames"] and not r["truth_collides"]])])
    table("COUNTERFACTUAL PAIRS: when their video collision happened vs the simulation start", [
        ("video collision BEFORE sim start (only)", [r for r in cf if r["obs_frames"] and all(f < r["s"] for f in r["obs_frames"])]),
        ("video collision at/after sim start", [r for r in cf if any(f >= r["s"] for f in r["obs_frames"])]),
        ("no video collision", [r for r in cf if not r["obs_frames"]])])
    fa = [r for r in recs if r["path"] == "outside_cone"]
    table("FACTUAL PAIRS: video vs truth", [
        ("collided in video, collide in truth", [r for r in fa if r["obs_frames"] and r["truth_collides"]]),
        ("collided in video, NOT in truth (cone wrong)", [r for r in fa if r["obs_frames"] and not r["truth_collides"]]),
        ("no collision in video, collide in truth (cone wrong)", [r for r in fa if not r["obs_frames"] and r["truth_collides"]]),
        ("no collision in video, none in truth", [r for r in fa if not r["obs_frames"] and not r["truth_collides"]])])
    table("COUNTERFACTUAL PAIRS BY SIMULATION LENGTH", [
        ("<= 30 frames", [r for r in cf if r.get("horizon", 0) <= 30]),
        ("31-60 frames", [r for r in cf if 30 < r.get("horizon", 0) <= 60]),
        ("61-90 frames", [r for r in cf if 60 < r.get("horizon", 0) <= 90]),
        ("> 90 frames", [r for r in cf if r.get("horizon", 0) > 90])])
    table("BY PAIR SHAPES (simulated pairs)",
          [(" + ".join(k), [r for r in sim if tuple(r["shapes"]) == k])
           for k in sorted(set(tuple(r["shapes"]) for r in sim))])
    table("BY SHAPE OF THE REMOVED OBJECT",
          [(k, [r for r in recs if r["removed_shape"] and r["removed_shape"][0] == k])
           for k in ("cube", "sphere", "cylinder")])
    table("BY NUMBER OF OBJECTS IN THE SCENE",
          [(str(k), [r for r in recs if r["n_objects"] == k]) for k in sorted(set(r["n_objects"] for r in recs))])
    table("BY WHETHER BOTH OBJECTS WERE ON THE TABLE WHEN SIMULATION STARTED", [
        ("both present", [r for r in sim if r.get("present_i_at_s") and r.get("present_j_at_s")]),
        ("one NOT yet present", [r for r in sim if not (r.get("present_i_at_s") and r.get("present_j_at_s"))])])

    # ---------------- causes of every wrong answer --------------------------------
    fn = [r for r in sim if not r["ok"] and r["truth_collides"] and not r["model_happens"]]
    fp = [r for r in sim if not r["ok"] and not r["truth_collides"] and r["model_happens"]]
    other = [r for r in recs if not r["ok"] and r not in fn and r not in fp]
    print("\n" + "=" * 78)
    print("WRONG ANSWERS: %d   missed a collision %d   invented one %d   other %d"
          % (n - ok, len(fn), len(fp), len(other)))
    print("=" * 78)
    for title, rs, fnc in (("MISSED COLLISIONS", fn, fn_cause), ("INVENTED COLLISIONS", fp, fp_cause)):
        print("\n%s -- cause (first that applies)" % title)
        by = defaultdict(list)
        for r in rs:
            by[fnc(r)].append(r)
        for k in sorted(by):
            sub = by[k]
            print("   %-58s %4d  (%4.1f%%)   factual %d / counterfactual %d"
                  % (k, len(sub), 100.0 * len(sub) / max(len(rs), 1),
                     sum(r["path"] == "outside_cone" for r in sub), sum(r["path"] == "in_cone" for r in sub)))
        for k in sorted(by):
            print("\n   examples: " + k)
            for r in by[k][:3]:
                print("      video %d  \"%s\"  -> \"%s\"  truth=%s model=%s  path=%s  s=%s end=%s  "
                      "sim voxel gap min=%s  rec voxel gap min=%s  video collisions at %s  entered at %s/%s"
                      % (r["video"], r["question"], r["choice"], r["truth"], r["model_label"], r["path"],
                         r.get("s"), r.get("end"), r.get("sim_vgap_min"), r.get("rec_vgap_min"),
                         r["obs_frames"], r.get("enter_i"), r.get("enter_j")))

    # ---------------- would the voxel sensor as the detector help? ---------------
    print("\n" + "=" * 78)
    print("IF THE VOXEL SENSOR MADE THE 'DID THEY TOUCH' CALL INSTEAD OF THE CENTRE RULE")
    print("=" * 78)
    base = sum(r["ok"] for r in recs)
    for tol in (0.0, 1.0, 2.0, 3.0, 6.0):
        right = 0
        for r in recs:
            if r["path"] == "participant_removed" or r.get("refused"):
                h = False
            else:
                h = sensor_touch(r, tol)
            right += (("correct" if h != r["negate"] else "wrong") == r["truth"])
        print("   touch = surfaces within %.0f voxels : %s   (centre rule: %s)" % (tol, pct(right, n), pct(base, n)))

    print("\nCEILING: sensor / rule on the REAL RECORDED motion, factual pairs only (no model)")
    have = [r for r in fa if r.get("rec_rule") is not None]
    for tol in (0.0, 1.0, 3.0, 6.0):
        right = sum((("correct" if (r.get("rec_vgap_min", 99) <= tol) != r["negate"] else "wrong") == r["truth"])
                    for r in have)
        print("   recording + sensor within %.0f voxels: %s" % (tol, pct(right, len(have))))
    right = sum((("correct" if bool(r["rec_rule"]) != r["negate"] else "wrong") == r["truth"]) for r in have)
    print("   recording + centre rule          : %s" % pct(right, len(have)))
    print("   model (same pairs)               : %s" % pct(sum(r["ok"] for r in have), len(have)))
    col = [r for r in fa if r["obs_frames"] and r.get("rec_vgap_min") is not None]
    print("\n   annotated collisions in factual windows: how close do our 3D objects get in the REAL motion?")
    for tol in (0, 1, 3, 6, 10, 20):
        print("      within %2d voxels: %s" % (tol, pct(sum(r["rec_vgap_min"] <= tol for r in col), len(col))))

    # ---------------- per question ----------------------------------------------------
    byq = defaultdict(list)
    for r in recs:
        byq[(r["video"], r["qid"])].append(r)
    wrongq = [v for v in byq.values() if not all(x["ok"] for x in v)]
    print("\nPER QUESTION: %s fully right. Of the %d wrong questions, %d have exactly ONE wrong choice."
          % (pct(len(byq) - len(wrongq), len(byq)), len(wrongq), sum(1 for v in wrongq if sum(not x["ok"] for x in v) == 1)))


if __name__ == "__main__":
    main()
