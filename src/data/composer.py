# src/reasoning/composer.py
# ─────────────────────────────────────────────────────────────
# Causal Graph Composition — V2 (PD_GNN) version.
# Chains per-collision GNN predictions into a human-readable
# causal narrative for the whole video.
# ─────────────────────────────────────────────────────────────

import os, sys
import torch

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data.loader import load_scene, build_event_subgraph
from src.models.pd_gnn import PD_GNN


def get_object_name(obj):
    return f"{obj['color']} {obj['material']} {obj['shape']}"


def compose_video_narrative(json_path, model, verbose=True):
    """
    Runs the PD-GNN on every collision subgraph in a video and
    chains the per-event predictions into a causal narrative.
    """
    objects, collisions, positions = load_scene(json_path)

    if not collisions:
        if verbose:
            print("No collisions detected in this video.")
        return []

    per_event = []

    for event in collisions:
        subgraph = build_event_subgraph(objects, positions, event)
        x, edge_index, edge_attr = (subgraph['x'], subgraph['edge_index'],
                                     subgraph['edge_attr'])
        obj_ids = subgraph['obj_ids']

        model.eval()
        with torch.no_grad():
            batch = torch.zeros(x.size(0), dtype=torch.long)
            exit_logit, _ = model(x, edge_index, edge_attr, batch)
            prob = torch.sigmoid(exit_logit).item()

        # Momentum proxy to identify the "dominant" object in the collision
        v0 = (x[0, 15]**2 + x[0, 16]**2).sqrt().item()
        v1 = (x[1, 15]**2 + x[1, 16]**2).sqrt().item()
        dominant_id  = obj_ids[0] if v0 > v1 else obj_ids[1]
        affected_id  = obj_ids[1] if v0 > v1 else obj_ids[0]

        per_event.append({
            'frame': event['frame'],
            'dominant': get_object_name(objects[dominant_id]),
            'affected': get_object_name(objects[affected_id]),
            'outcome': "EXIT" if prob > 0.5 else "STAY",
            'prob': prob,
            'obj_ids': obj_ids,
        })

    if verbose:
        print("\n" + "="*60)
        print("CAUSAL CHAIN NARRATIVE")
        print("="*60)
        for i, ev in enumerate(per_event):
            prefix = f"[Event {i+1} | Frame {ev['frame']}]"
            if ev['outcome'] == "EXIT":
                print(f"{prefix} {ev['dominant']} collided with {ev['affected']} "
                      f"→ predicted EXIT (confidence: {ev['prob']:.2f})")
            else:
                print(f"{prefix} {ev['dominant']} collided with {ev['affected']} "
                      f"→ insufficient force, predicted STAY "
                      f"(confidence: {1-ev['prob']:.2f})")

            if i < len(per_event) - 1:
                next_ids = per_event[i+1]['obj_ids']
                if any(oid in next_ids for oid in ev['obj_ids']):
                    print(f"  ↓ {ev['affected']} carries momentum into next collision...")
        print("="*60)

    return per_event


if __name__ == "__main__":
    model = PD_GNN(node_dim=17, edge_dim=4, hidden=64)
    model.load_state_dict(torch.load(
        os.path.join(ROOT_DIR, 'src', 'models', 'causal_gnn_v2.pt'),
        map_location='cpu'))

    json_path = os.path.join(
        ROOT_DIR, 'data', 'processed_proposals', 'sim_00000.json')
    compose_video_narrative(json_path, model)
