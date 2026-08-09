"""
explain.py  (src2)  —  grounded counterfactual explainer for CausalVis V3
=========================================================================

Reads the trace written by  benchmark_eval.py --trace-out traces.jsonl  and
narrates each question in plain English.

GROUNDING GUARANTEE: every sentence below is rendered from a template whose
slots are filled ONLY with fields of the trace record -- object names, frame
numbers, taint causes, event sources, and decision paths that the pipeline
actually computed.  The explainer has no access to the model, the video, or
any physics; it cannot state a collision, a cause, or a frame that is not in
the trace.  (An optional LLM polish layer can later rewrite these paragraphs
for fluency, validated against the same trace -- the facts come from here.)

RUN
    python src2/benchmark_eval.py --model v3_2_dynamics.pt --split split.json \\
           --which test --questions zechennlp/counterfactual/train-....json \\
           --trace-out traces.jsonl
    python src2/explain.py traces.jsonl                      # everything
    python src2/explain.py traces.jsonl --video 4213         # one clip
    python src2/explain.py traces.jsonl --misses-only        # errors only
    python src2/explain.py traces.jsonl --out explains.md    # save as markdown
"""

import argparse
import json


def load_traces(path):
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def narrate(rec):
    """One grounded narration (list of lines) for one question record."""
    keys = rec["obj_keys"]

    def nm(idx):
        return keys[idx].replace("_", " ") if idx < len(keys) else f"obj{idx}"

    removed = rec["removed"]
    lines = []
    if rec.get("question"):
        lines.append(f'QUESTION (video {rec["video"]:05d}): "{rec["question"]}"')

    # ---- the intervention --------------------------------------------------
    rm_names = ", the ".join(nm(k) for k in removed)
    note = ""
    if len(removed) > 1 and rec.get("match_mode", "description") == "description":
        note = ("  (the question named a description that matched more than "
                "one object, so every match is removed)")
    lines.append(f"INTERVENTION: remove the {rm_names}.{note}")

    # ---- who is affected, and why ------------------------------------------
    taints = rec["taints"]
    n_total = len(keys)
    n_free = n_total - len(removed) - len(taints)
    if not taints:
        lines.append(
            f"EFFECT: of {n_total} tracked objects, {len(removed)} are removed "
            f"and the remaining {n_free} never interact with them, so every "
            f"trajectory in the edited world is identical to the original "
            f"video.")
    else:
        tail = (f"and {n_free} keep their observed trajectories exactly"
                if n_free else
                "and no object is left untouched, so the intervention reaches "
                "the whole scene")
        lines.append(
            f"EFFECT: of {n_total} tracked objects, {len(removed)} are "
            f"removed, {len(taints)} are re-simulated by the physics engine, "
            f"{tail}.")
        for t in taints:
            c = t.get("cause", {})
            kind = c.get("kind")
            if kind == "lost_contact":
                why = (f'its recorded frame-{c["col_frame"]} collision with '
                       f'the removed {nm(c["other"])} can no longer happen')
            elif kind == "touch_removed":
                why = (f"the video shows it touching the removed "
                       f"{nm(c['other'])} here")
            elif kind == "chain":
                why = (f"the re-simulated {nm(c['other'])} comes into contact "
                       f"range with it (chain effect)")
            else:
                why = "it enters the intervention's causal cone"
            lines.append(f"   - from frame {t['frame']}, the {nm(t['idx'])} "
                         f"is re-simulated: {why}.")

    # ---- the edited-world event log ----------------------------------------
    evs = rec["cf_events"]
    gt_pairs = set(frozenset((g["i"], g["j"])) for g in rec["gt_collisions"])
    if evs:
        lines.append("EDITED-WORLD EVENTS:")
        for ev in evs:
            pr = frozenset((ev["i"], ev["j"]))
            if ev["source"] == "observed":
                tag = "observed -- outside the intervention's influence, carries over exactly"
            elif pr in gt_pairs:
                tag = "simulated -- re-derived by the physics engine inside the cone"
            else:
                tag = "simulated -- NEW collision, exists only in the edited world"
            lines.append(f"   frame {ev['frame']:3d}: {nm(ev['i'])} <-> "
                         f"{nm(ev['j'])}   ({tag})")
    else:
        lines.append("EDITED-WORLD EVENTS: none -- no collision occurs in the "
                     "edited world.")

    # recorded collisions that no longer happen
    cf_pairs = set(frozenset((ev["i"], ev["j"])) for ev in evs)
    gone = []
    for g in rec["gt_collisions"]:
        pr = frozenset((g["i"], g["j"]))
        if pr in cf_pairs:
            continue
        if g["i"] in removed and g["j"] in removed:
            gone.append(f"   frame {g['frame']:3d}: {nm(g['i'])} <-> "
                        f"{nm(g['j'])}   (both participants were removed)")
        elif g["i"] in removed or g["j"] in removed:
            rm_one = nm(g["i"]) if g["i"] in removed else nm(g["j"])
            gone.append(f"   frame {g['frame']:3d}: {nm(g['i'])} <-> "
                        f"{nm(g['j'])}   (the {rm_one} was removed)")
        else:
            gone.append(f"   frame {g['frame']:3d}: {nm(g['i'])} <-> "
                        f"{nm(g['j'])}   (VANISHES -- the simulation shows it "
                        f"never happens without the removed object)")
    if gone:
        lines.append("RECORDED COLLISIONS THAT NO LONGER HAPPEN:")
        lines.extend(gone)

    # ---- the answers, each justified by its decision path ------------------
    if not rec.get("choices"):
        return lines
    polarity = "will NOT happen" if rec.get("negate") else "will happen"
    lines.append(f"ANSWERS (question asks which event {polarity}):")
    for ch in rec["choices"]:
        i, j = ch["pair"]
        head = f'   "{ch["text"]}"' if ch["text"] else \
               f"   {nm(i)} <-> {nm(j)}"
        path = ch["path"]
        if path == "participant_removed":
            because = ("a participant of this collision is removed by the "
                       "intervention itself, so it cannot occur")
        elif path == "observed_exact":
            f0 = ch["cf_frames"][0] if ch["cf_frames"] else None
            at = f" at frame {f0}" if f0 is not None else ""
            because = ("neither object is ever influenced by the removal; "
                       f"their observed collision{at} carries over exactly")
        elif path == "sim_hit":
            f0 = ch["cf_frames"][0] if ch["cf_frames"] else None
            at = f" at frame {f0}" if f0 is not None else ""
            new = "" if ch["gt_frames"] else \
                  " -- a new collision that does not exist in the original video"
            because = (f"the physics engine produces this collision{at} in "
                       f"the edited world{new}")
        elif path == "vanished":
            f0 = ch["gt_frames"][0] if ch["gt_frames"] else None
            at = f" at frame {f0}" if f0 is not None else ""
            because = (f"this collision is recorded{at} in the original "
                       f"video, but the re-simulation shows it never happens "
                       f"without the removed object")
        else:  # never
            because = ("this collision occurs in neither the original video "
                       "nor the edited world")
        verdict = "HAPPENS" if ch["happens"] else "does NOT happen"
        mark = "RIGHT" if ch["ok"] else "WRONG"
        lines.append(head)
        lines.append(f"      -> {verdict}: {because}.")
        lines.append(f"      -> model answer: {ch['model']}   "
                     f"(ground truth: {ch['truth']})   [{mark}]")

    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", help="traces.jsonl from benchmark_eval --trace-out")
    ap.add_argument("--video", type=int, default=None,
                    help="only questions from this video number")
    ap.add_argument("--misses-only", action="store_true",
                    help="only questions with at least one wrong choice")
    ap.add_argument("--limit", type=int, default=None,
                    help="max questions to narrate")
    ap.add_argument("--out", default=None,
                    help="also write the narration to this file (markdown)")
    a = ap.parse_args()

    recs = load_traces(a.trace)
    if a.video is not None:
        recs = [r for r in recs if r["video"] == a.video]
    if a.misses_only:
        recs = [r for r in recs if any(not c["ok"] for c in r["choices"])]
    if a.limit is not None:
        recs = recs[:a.limit]

    blocks = []
    for rec in recs:
        blocks.append("\n".join(narrate(rec)))
    text = ("\n\n" + "=" * 68 + "\n\n").join(blocks)
    print(text)
    print(f"\n[{len(recs)} question(s) narrated]")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write("```\n" + text + "\n```\n")
        print(f"[written to {a.out}]")


if __name__ == "__main__":
    main()