"""
CausalVis Unified Pipeline
Ties all four modules together exactly as shown in the architecture diagram:

  I.  Data Ingestion   (loader.py + build_dataset.py)
  II. Subgraph Dynamics Engine (GNN)
  III. Logic Query Parser
  IV. Neuro-Symbolic Reasoner + Counterfactual Verification

Entry point: ask(question, json_path, model)
"""

import os, sys, torch
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT_DIR)

from src.data.loader import load_scene
from src.data.build_dataset import process_video
from src.models.gnn import CausalGNN
from src.reasoning.query_parser import parse_question
from src.reasoning.counterfactual import intervene, get_object_name
from src.reasoning.composer import compose_video_narrative


def _find_collision_for_objects(collisions, objects, subject, intervention):
    """
    Find the collision event most relevant to the parsed query.
    subject and intervention are (color, mat, shape) tuples.
    Returns (event_idx, which_idx_is_target)
    """
    def matches(obj_dict, spec):
        if spec is None: return False
        c, m, s = spec
        return (not c or obj_dict['color'] == c) and \
               (not m or obj_dict['material'] == m) and \
               (not s or obj_dict['shape'] == s)

    for i, event in enumerate(collisions):
        o0 = objects[event['object'][0]]
        o1 = objects[event['object'][1]]
        # Subject in collision, intervention object also in collision
        if matches(o0, subject) or matches(o1, subject):
            if intervention is None:
                return i, 0
            if matches(o0, intervention):
                return i, 0
            if matches(o1, intervention):
                return i, 1
    return 0, 0  # fallback to first collision


def ask(question: str, json_path: str, model) -> dict:
    """
    Full pipeline: natural language question → auditable causal answer.

    Returns dict with all intermediate outputs for transparency.
    """
    print("\n" + "█"*65)
    print("CAUSALVIS PIPELINE — FULL QUERY")
    print("█"*65)

    # ─── Module I: Data Ingestion ───────────────────────────────────
    print(f"\n[MODULE I] Loading: {os.path.basename(json_path)}")
    objects, collisions, positions = load_scene(json_path)
    samples = process_video(json_path)
    print(f"  Objects: {len(objects)} | Collisions: {len(collisions)}")

    # ─── Module III: Query Parser ────────────────────────────────────
    print(f"\n[MODULE III] Parsing question...")
    print(f"  Q: \"{question}\"")
    parsed = parse_question(question)
    print(f"  Intervention type : {parsed['intervention_type']}")
    print(f"  Subject object    : {parsed['subject_object']}")
    print(f"  Intervention obj  : {parsed['intervention_object']}")
    print(f"  Functional tokens : {parsed['tokens']}")

    # ─── Module II: Subgraph Dynamics Engine ─────────────────────────
    print(f"\n[MODULE II] Running GNN on {len(collisions)} collision subgraphs...")
    narrative = compose_video_narrative(json_path, model)

    # Find the most relevant collision for this query
    col_idx, target_idx = _find_collision_for_objects(
        collisions, objects,
        parsed['subject_object'],
        parsed['intervention_object']
    )
    event = collisions[col_idx]
    data  = samples[col_idx]
    print(f"  Matched collision: Event {col_idx+1}, Frame {event['frame']}")
    print(f"  Target object idx: {target_idx} "
          f"({get_object_name(objects[event['object'][target_idx]])})")

    # ─── Module IV: Neuro-Symbolic Reasoner + Counterfactual ─────────
    print(f"\n[MODULE IV] Running {parsed['intervention_type']} intervention...")
    mode_map = {
        'zero_velocity':     'zero_velocity',
        'prevent_collision': 'prevent_collision',
        'remove_object':     'remove_object',
        'query_cause':       'prevent_collision',  # causal query → use prevent
    }
    mode = mode_map.get(parsed['intervention_type'], 'zero_velocity')

    result = intervene(model, data, objects, event, target_idx, mode=mode)

    # ─── Final Answer ────────────────────────────────────────────────
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
        'question': question,
        'parsed': parsed,
        'collision_event': event,
        'intervention_result': result,
        'answer': answer,
        'narrative': narrative,
    }


if __name__ == "__main__":
    model = CausalGNN()
    model.load_state_dict(torch.load(
        os.path.join(ROOT_DIR, 'src', 'models', 'causal_gnn_weighted.pt'),
        map_location='cpu'
    ))

    JSON = r"G:\CausalVis\data\processed_proposals\sim_00000.json"

    # Test all three intervention types
    questions = [
        "What would happen to the gray rubber sphere if the blue rubber sphere didn't collide with it?",
        "Would the cyan metal cube have exited if the gray rubber sphere hadn't been there?",
        "What if the blue rubber sphere had no velocity when it hit the gray sphere?",
    ]

    for q in questions:
        ask(q, JSON, model)
        print("\n")