# src/reasoning/pipeline.py
# ─────────────────────────────────────────────────────────────
# CausalVis Unified Pipeline — V2 (PD_GNN)
# Ties all four modules together: Data Ingestion → Query Parser
# → Subgraph Dynamics Engine → Neuro-Symbolic + Counterfactual.
#
# Entry point: ask(question, json_path, model)
# ─────────────────────────────────────────────────────────────

import os, sys
import torch

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data.loader import load_scene, build_event_subgraph
from src.models.pd_gnn import PD_GNN
from src.reasoning.query_parser import parse_question
from src.reasoning.counterfactual import intervene, get_object_name
from src.reasoning.composer import compose_video_narrative


def _find_collision_for_objects(collisions, objects, subject, intervention):
    """Finds the collision event matching the parsed query's objects."""
    def matches(obj_dict, spec):
        if spec is None:
            return False
        c, m, s = spec
        return ((not c or obj_dict['color'] == c) and
                (not m or obj_dict['material'] == m) and
                (not s or obj_dict['shape'] == s))

    for i, event in enumerate(collisions):
        o0 = objects[event['object'][0]]
        o1 = objects[event['object'][1]]
        if matches(o0, subject) or matches(o1, subject):
            if intervention is None:
                return i, 0
            if matches(o0, intervention):
                return i, 0
            if matches(o1, intervention):
                return i, 1
    return 0, 0   # fallback: first collision


def ask(question: str, json_path: str, model) -> dict:
    """Full pipeline: natural language question → auditable causal answer."""
    print("\n" + "█"*65)
    print("CAUSALVIS PIPELINE — FULL QUERY (V2 / PD_GNN)")
    print("█"*65)

    # ── Module I: Data Ingestion ──────────────────────────────
    print(f"\n[MODULE I] Loading: {os.path.basename(json_path)}")
    objects, collisions, positions = load_scene(json_path)
    print(f"  Objects: {len(objects)} | Collisions: {len(collisions)}")

    # ── Module III: Query Parser ──────────────────────────────
    print(f"\n[MODULE III] Parsing question...")
    print(f"  Q: \"{question}\"")
    parsed = parse_question(question)
    print(f"  Intervention type : {parsed['intervention_type']}")
    print(f"  Subject object    : {parsed['subject_object']}")
    print(f"  Intervention obj  : {parsed['intervention_object']}")
    print(f"  Functional tokens : {parsed['tokens']}")

    # ── Module II: Subgraph Dynamics Engine ───────────────────
    print(f"\n[MODULE II] Running PD-GNN on {len(collisions)} collision subgraphs...")
    narrative = compose_video_narrative(json_path, model, verbose=False)

    col_idx, target_idx = _find_collision_for_objects(
        collisions, objects, parsed['subject_object'], parsed['intervention_object'])
    event = collisions[col_idx]
    subgraph = build_event_subgraph(objects, positions, event)

    print(f"  Matched collision: Event {col_idx+1}, Frame {event['frame']}")
    print(f"  Target object idx: {target_idx} "
          f"({get_object_name(objects[event['object'][target_idx]])})")

    # ── Module IV: Neuro-Symbolic Reasoner + Counterfactual ───
    mode_map = {
        'zero_velocity':     'zero_velocity',
        'prevent_collision': 'prevent_collision',
        'remove_object':     'remove_object',
        'query_cause':       'prevent_collision',
    }
    mode = mode_map.get(parsed['intervention_type'], 'zero_velocity')
    print(f"\n[MODULE IV] Running {mode} intervention...")

    result = intervene(model, subgraph, objects, target_idx, mode=mode)

    # ── Final Answer ───────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("ANSWER")
    print(f"{'─'*65}")
    affected = result['affected']
    if result['is_causal']:
        answer = (f"{affected} would {result['cf_outcome'].lower()} "
                  f"(instead of {result['factual_outcome'].lower()}) — "
                  f"the intervention changed the outcome.")
    else:
        answer = (f"{affected} would still {result['cf_outcome'].lower()} "
                  f"— the intervention did NOT change the outcome.")

    print(f"  {answer}")
    print(f"\nAuditable Causal Rule:")
    for line in result['rule'].split('\n'):
        print(f"  {line}")

    return {
        'question': question, 'parsed': parsed,
        'collision_event': event, 'intervention_result': result,
        'answer': answer, 'narrative': narrative,
    }


if __name__ == "__main__":
    model = PD_GNN(node_dim=17, edge_dim=4, hidden=64)
    model.load_state_dict(torch.load(
        os.path.join(ROOT_DIR, 'src', 'models', 'causal_gnn_v2.pt'),
        map_location='cpu'))

    JSON = os.path.join(ROOT_DIR, 'data', 'processed_proposals', 'sim_00000.json')

    questions = [
        "What would happen to the gray rubber sphere if the blue rubber sphere didn't collide with it?",
        "Would the cyan metal cube have exited if the gray rubber sphere hadn't been there?",
        "What if the blue rubber sphere had no velocity when it hit the gray sphere?",
    ]

    for q in questions:
        ask(q, JSON, model)
        print("\n")
