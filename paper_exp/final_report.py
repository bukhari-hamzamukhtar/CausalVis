"""
paper_exp/final_report.py  —  every counterfactual table of the paper from the TEST runs
======================================================================================

Reads paper_exp/chosen.json (VAL-A choices) and the runs named A_<system> (TEST-A) and
B_<system> (TEST-B). For each system: options / questions with 95% video-bootstrap
intervals on TEST-A, TEST-B and both pooled; paired McNemar z against CausalVis on the
same options. Then: ablations, the simulator-quality audit (score vs physics error), the
ALOE-style subsets and the error breakdown for CausalVis. Writes paper_exp/final_report.txt
and final_report.json.

    python paper_exp/final_report.py
"""

import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import score as S                                                        # noqa: E402

LINES = []


def out(s=""):
    LINES.append(s)
    print(s, flush=True)


def done(tag, name):
    """Records of a FINISHED run only (a DONE marker), else {}."""
    p = os.path.join(HERE, "runs", "%s_%s" % (tag, name))
    return S.load(p) if os.path.exists(os.path.join(p, "DONE")) else {}


def pooled(name):
    """TEST-A and TEST-B together; the set tag goes INSIDE the video id so that
    (video, question) grouping and the video bootstrap stay correct."""
    a, b = done("A", name), done("B", name)
    if not a or not b:
        return {}
    r = {}
    for tag, recs in (("A", a), ("B", b)):
        r.update({((tag, k[0]),) + k[1:]: v for k, v in recs.items()})
    return r


def setting(chosen, name):
    base = name.split("_")[0] if name in ("learned", "laws", "straight", "strfric", "nosim") else "learned"
    c = chosen.get(base, chosen["learned"])
    return c["det"], c["ext"]


def row(label, recs, det, ext, ref=None, ref_set=None):
    s = S.summary(recs, det, ext)
    (ol, oh), (ql, qh) = S.boot(recs, det, ext)
    z = ""
    if ref is not None:
        p = S.paired(recs, det, ext, ref, ref_set[0], ref_set[1])
        z = "  z %+.2f / %+.2f" % (p["z"], p["z_q"])
    out("  %-34s %5.1f [%4.1f-%4.1f]   %5.1f [%4.1f-%4.1f]   n %d/%d%s" % (
        label, 100.0 * s["options"] / s["n_options"], ol, oh, 100.0 * s["questions"] / s["n_questions"], ql, qh,
        s["n_options"], s["n_questions"], z))
    return s


def subsets(recs, tag_of):
    """E: the removed object never collides in the video; M: it collides but the video's
    own events answer every option; H: needs counterfactual reasoning (ALOE App. C)."""
    gtc = {}
    qs = defaultdict(list)
    for k in recs:
        qs[k[:-2]].append(k)
    groups = defaultdict(list)
    for q, ks in qs.items():
        r0 = recs[ks[0]]
        v = r0["video"]
        if v not in gtc:
            z = np.load(os.path.join(ROOT, "data", "trajectories_3d_det", "sim_%05d.npz" % v), allow_pickle=True)
            gtc[v] = [tuple(int(x) for x in c[1:]) for c in z["collisions"]]
        rem = set(r0["removed"])
        if not any(set(c) & rem for c in gtc[v]):
            groups["E"].append(q); continue
        desc = all((("correct" if bool(recs[k]["diag_obs_frames"]) != recs[k]["negate"] else "wrong") == recs[k]["truth"]) for k in ks)
        groups["M" if desc else "H"].append(q)
    return qs, groups


def main():
    chosen = json.load(open(os.path.join(HERE, "chosen.json")))
    report = {"chosen": chosen}
    det, ext = chosen["learned"]["det"], chosen["learned"]["ext"]
    ref = {tag: done(tag, "learned") for tag in ("A", "B")}
    ref["A+B"] = pooled("learned")
    out("VAL-A choices: " + "; ".join("%s lb %s %s +%s" % (k, v["lookback"], v["det"], v["ext"]) for k, v in chosen.items()))

    systems = [("learned", "CausalVis (learned world model)"), ("laws", "textbook laws"),
               ("strfric", "straight lines + friction"), ("straight", "straight lines"), ("nosim", "no physics")]
    abl = [("abl_novoxel", "- voxel contact impulse"), ("abl_noforce", "- learned pair energy"),
           ("abl_nodrag", "- learned friction (eval-time)"), ("abl_noentry", "- entry correction"),
           ("abl_lb12", "12-frame hand-off (v7.1)")]
    ckpts = ["v3_6_3d", "v3_7_3d_frozen", "v3_8_impulse", "v5_noyaw", "v5b_noyaw", "v6_control",
             "v72_fixed", "v72_curriculum", "v72_control", "seed1", "seed2"]
    for tag in ("B", "A", "A+B"):
        out("\n=== TEST-%s   (options %% [95%% CI]   questions %% [95%% CI]   paired z vs CausalVis: options / questions)" % tag)
        for name, label in systems:
            recs = pooled(name) if tag == "A+B" else done(tag, name)
            if not recs:
                continue
            d, e = chosen[name]["det"], chosen[name]["ext"]
            s = row(label, recs, d, e, None if name == "learned" else ref[tag], (det, ext))
            report.setdefault(tag, {})[name] = s
        out("  -- one change to CausalVis at a time:")
        r = ref[tag]
        row("- extension past the video end", r, det, 0, r, (det, ext))
        row("- path-change detector (cal only)", r, "cal", ext, r, (det, ext))
        for name, label in abl:
            recs = pooled(name) if tag == "A+B" else done(tag, name)
            if recs:
                report.setdefault(tag, {})[name] = row(label, recs, det, ext, r, (det, ext))
        out("  -- other trained simulators / seeds, same pipeline:")
        for c in ckpts:
            nm = "ckpt_" + c if not c.startswith("seed") else c
            recs = pooled(nm) if tag == "A+B" else done(tag, nm)
            if recs:
                report.setdefault(tag, {})[nm] = row(c, recs, det, ext, r, (det, ext))

    # the audit: benchmark score vs physics quality
    out("\n=== AUDIT: physics quality (VAL-A, 200 videos) vs TEST A+B score")
    qual = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(HERE, "quality", "*.json"))
            if "_smoke" not in f}
    name_of = {"learned": "learned", "laws": "laws", "straight": "straight", "strfric": "strfric",
               **{"ckpt_" + c: "ckpt_" + c for c in ckpts if not c.startswith("seed")}}
    xs, ys = [], []
    for run, qk in name_of.items():
        if qk not in qual or run not in report.get("A+B", {}):
            continue
        s = report["A+B"][run]
        acc = 100.0 * s["options"] / s["n_options"]
        xs.append(qual[qk]["rollout_err_40"]); ys.append(acc)
        out("  %-24s err@40 %.4f  replay30 %.3f   ->  options %.1f%%  questions %.1f%%" % (
            run, qual[qk]["rollout_err_40"], qual[qk]["replay_30"], acc, 100.0 * s["questions"] / s["n_questions"]))
    if len(xs) >= 3:
        r = np.corrcoef(xs, ys)[0, 1]
        out("  Pearson r(err@40, options) = %.2f over %d simulators; options range %.1f-%.1f while err@40 ranges %.3f-%.3f"
            % (r, len(xs), min(ys), max(ys), min(xs), max(xs)))

    # subsets and error breakdown for CausalVis, pooled
    out("\n=== CausalVis subsets (TEST A+B)")
    recs = ref["A+B"]
    qs, groups = subsets(recs, None)
    nos = pooled("nosim")
    for g in ("E", "M", "H"):
        ks = [k for q in groups[g] for k in qs[q]]
        a = S.summary(recs, det, ext, ks)
        b = S.summary(nos, chosen["nosim"]["det"], 0, [k for k in ks if k in nos])
        out("  %s  %4d questions (%.1f%%)   CausalVis %.1f / %.1f   no physics %.1f / %.1f" % (
            g, len(groups[g]), 100.0 * len(groups[g]) / len(qs), 100.0 * a["options"] / a["n_options"],
            100.0 * a["questions"] / a["n_questions"], 100.0 * b["options"] / max(1, b["n_options"]),
            100.0 * b["questions"] / max(1, b["n_questions"])))
    ek = [k for q in groups["E"] for k in qs[q]]
    post = sum(1 for k in ek if recs[k]["truth_collides"] and not recs[k]["diag_obs_frames"])
    qpost = sum(any(recs[k]["truth_collides"] and not recs[k]["diag_obs_frames"] for k in qs[q]) for q in groups["E"])
    out("  E: answer key says 'collide' with no collision in the video on %d / %d options (%.1f%%), in %d / %d questions (%.1f%%)"
        % (post, len(ek), 100.0 * post / max(1, len(ek)), qpost, len(groups["E"]), 100.0 * qpost / max(1, len(groups["E"]))))
    cat = Counter()
    for k, r in recs.items():
        if S.right(r, det, ext):
            continue
        h = S.happens(r, det, ext)
        if r.get("unresolved"):
            cat["object not found by the detector"] += 1
        elif r["truth_collides"] and not h:
            cat["missed: collision after the video ends" if not r["diag_obs_frames"] else "missed: collision visible in the video"] += 1
        elif h and not r["truth_collides"]:
            fr = [f for f in (r["cal"] + r["kick2"] if det == "union" else r[det]) if f < r["T"] + ext]
            cat["invented: during the video" if min(fr) < r["T"] else "invented: after the video ends"] += 1
    tot = sum(cat.values())
    out("  error breakdown (%d wrong options): " % tot + "; ".join("%s %d (%.0f%%)" % (k, v, 100.0 * v / tot) for k, v in cat.most_common()))
    report["errors"] = dict(cat)
    json.dump(report, open(os.path.join(HERE, "final_report.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, "final_report.txt"), "w", encoding="utf-8").write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
