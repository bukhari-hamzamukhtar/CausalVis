# src/reasoning/gallery.py
# ─────────────────────────────────────────────────────────────
# Runs the full pipeline across multiple videos, auto-generating
# real natural-language questions from each video's actual objects
# (not hand-typed), and collects a curated gallery of results.
#
# TWO USES:
#   1. Paper's qualitative results section needs several example
#      transcripts, not just one video.
#   2. Open house demo needs pre-tested working examples ready to
#      show live, not live improvisation on an unknown video.
#
# Uses causal_gnn_v2.pt (plain PD-GNN, no masking) — the model
# with the best overall accuracy/F1/precision balance, chosen as
# the production/demo model after comparing V1/V2/V3/V2+Masking.
# ─────────────────────────────────────────────────────────────

import os, sys, json, random

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from src.data.loader import load_scene
from src.models.pd_gnn import PD_GNN
from src.reasoning.pipeline import ask


def get_object_name(obj):
    return f"{obj['color']} {obj['material']} {obj['shape']}"


def generate_questions_for_event(objects, event):
    """
    Builds three natural-language questions for one collision event,
    using the REAL object names present in this specific video, in
    the exact phrasing query_parser.py's regex templates expect.

    Returns [(question_text, expected_mode), ...]
    """
    obj0 = objects[event['object'][0]]
    obj1 = objects[event['object'][1]]
    name0 = get_object_name(obj0)
    name1 = get_object_name(obj1)

    return [
        (f"What would happen to the {name0} if the {name1} didn't collide with it?",
         'prevent_collision'),
        (f"Would the {name0} have exited if the {name1} hadn't been there?",
         'remove_object'),
        (f"What if the {name1} had no velocity when it hit the {name0}?",
         'zero_velocity'),
    ]


def run_gallery(data_dir, model, max_videos=15, questions_per_video=1, seed=42):
    """
    Loops over available videos, generates real questions per video,
    runs the full pipeline quietly, and collects structured results.

    questions_per_video: how many collision EVENTS per video to
    generate questions for (each event yields 3 questions, one per
    mode). Keep this small — most videos only have 2-4 collisions.
    """
    random.seed(seed)

    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])
    if not files:
        print(f"No JSON files found in {data_dir}")
        return []

    random.shuffle(files)
    files = files[:max_videos]
    print(f"Found {len(os.listdir(data_dir))} total files, "
          f"sampling {len(files)} for the gallery run.\n")

    gallery = []

    for fname in files:
        json_path = os.path.join(data_dir, fname)
        try:
            objects, collisions, _ = load_scene(json_path)
        except Exception as e:
            print(f"  Skipped {fname}: {e}")
            continue

        if not collisions:
            continue

        events_to_use = collisions[:questions_per_video]

        for event in events_to_use:
            questions = generate_questions_for_event(objects, event)
            for question, expected_mode in questions:
                try:
                    result = ask(question, json_path, model, verbose=False)
                    r = result['intervention_result']
                    gallery.append({
                        'video': fname,
                        'frame': event['frame'],
                        'question': question,
                        'expected_mode': expected_mode,
                        'parsed_mode': result['parsed']['intervention_type'],
                        'factual_outcome': r['factual_outcome'],
                        'cf_outcome': r['cf_outcome'],
                        'is_causal': r['is_causal'],
                        'delta_prob': abs(r['cf_prob'] - r['factual_prob']),
                        'answer': result['answer'],
                        'rule': r['rule'],
                    })
                except Exception as e:
                    print(f"  Failed on {fname} / \"{question[:50]}...\": {e}")

    return gallery


def print_summary_table(gallery):
    print(f"\n{'='*100}")
    print(f"GALLERY SUMMARY — {len(gallery)} question(s) across "
          f"{len(set(g['video'] for g in gallery))} video(s)")
    print(f"{'='*100}")
    print(f"{'Video':<16} {'Mode':<18} {'Factual':<8} {'CF':<8} "
          f"{'Causal?':<9} {'Δprob':<7}")
    print(f"{'-'*100}")
    for g in gallery:
        print(f"{g['video']:<16} {g['parsed_mode']:<18} "
              f"{g['factual_outcome']:<8} {g['cf_outcome']:<8} "
              f"{'YES' if g['is_causal'] else 'no':<9} {g['delta_prob']:.3f}")
    print(f"{'='*100}")


def pick_best_demo_examples(gallery, n=5):
    """
    Curates the N most demo-worthy examples: prioritizes CAUSAL
    results with large delta_prob (clear, dramatic causal flips) —
    these read the clearest to a non-technical open-house visitor.
    Also includes at least one NOT-CAUSAL example for contrast.
    """
    causal    = sorted([g for g in gallery if g['is_causal']],
                       key=lambda g: -g['delta_prob'])
    not_causal = sorted([g for g in gallery if not g['is_causal']],
                        key=lambda g: -g['delta_prob'])

    picks = causal[:max(n-1, 1)]
    if not_causal:
        picks.append(not_causal[0])
    return picks[:n]


if __name__ == "__main__":
    model = PD_GNN(node_dim=17, edge_dim=4, hidden=64)
    model.load_state_dict(torch.load(
        os.path.join(ROOT_DIR, 'src', 'models', 'causal_gnn_v2.pt'),
        map_location='cpu'))

    DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed_proposals')

    gallery = run_gallery(DATA_DIR, model, max_videos=15, questions_per_video=1)
    print_summary_table(gallery)

    # ── Save full results for later use (paper writing, demo prep) ──
    out_path = os.path.join(ROOT_DIR, 'data', 'gallery_results.json')
    with open(out_path, 'w') as f:
        json.dump(gallery, f, indent=2)
    print(f"\nFull results saved to: {out_path}")

    # ── Print curated best examples for the paper / demo ──────────
    best = pick_best_demo_examples(gallery, n=5)
    print(f"\n{'='*100}")
    print("TOP 5 CURATED EXAMPLES — use these for the paper and open house")
    print(f"{'='*100}")
    for i, g in enumerate(best, 1):
        print(f"\n[{i}] Video: {g['video']}  |  Frame: {g['frame']}")
        print(f"    Q: {g['question']}")
        print(f"    A: {g['answer']}")
        print(f"    {'✅ CAUSAL' if g['is_causal'] else '❌ NOT SOLE CAUSE'} "
              f"(Δprob = {g['delta_prob']:.3f})")
